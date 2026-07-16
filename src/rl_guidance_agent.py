"""
Reinforcement Learning Agent for Guidance Law Learning

This module implements a PPO (Proximal Policy Optimization) agent
for learning guidance policies from experience.

The agent learns to map states (position, velocity, FIM metrics) to
actions (velocity directions) that maximize observability and mission success.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class MLPNetwork:
    """
    Simple Multi-Layer Perceptron for policy and value networks.
    
    Uses a 2-layer architecture with ReLU activations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, is_policy: bool = False):
        """
        Initialize network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            is_policy: If True, output represents policy (tanh activation)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_policy = is_policy
        
        # Initialize weights with Xavier/He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input, shape (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Output, same batch shape as input
        """
        # Handle single sample
        single_sample = (x.ndim == 1)
        if single_sample:
            x = x.reshape(1, -1)
        
        # Layer 1
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        
        # Output layer
        out = h2 @ self.W3 + self.b3
        
        if self.is_policy:
            out = np.tanh(out)  # Bound policy output to [-1, 1]
        
        if single_sample:
            out = out.flatten()
        
        return out
    
    def get_params(self) -> Dict:
        """Get network parameters."""
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }
    
    def set_params(self, params: Dict) -> None:
        """Set network parameters."""
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.W3 = params['W3']
        self.b3 = params['b3']


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for guidance learning.
    
    PPO is a policy gradient method that:
    - Learns from on-policy data
    - Uses clipped objective to prevent large policy updates
    - Is sample efficient and stable
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_clip: float = 0.2,
        epochs_per_update: int = 10
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension for networks
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_clip: PPO clipping parameter
            epochs_per_update: Number of epochs per policy update
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs_per_update = epochs_per_update
        
        # Policy network (actor)
        self.policy = MLPNetwork(state_dim, hidden_dim, action_dim, is_policy=True)
        
        # Value network (critic)
        self.value = MLPNetwork(state_dim, hidden_dim, 1, is_policy=False)
        
        # Action noise for exploration
        self.action_noise = 0.1
        
        # Training buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: If True, use mean action without noise
        
        Returns:
            Tuple of (action, log_probability)
        """
        # Get policy output (mean action)
        mean_action = self.policy.forward(state)
        
        if deterministic:
            action = mean_action
            log_prob = 0.0
        else:
            # Add Gaussian noise for exploration
            noise = np.random.randn(self.action_dim) * self.action_noise
            action = mean_action + noise
            
            # Compute log probability (Gaussian)
            log_prob = -0.5 * np.sum(noise**2) / (self.action_noise**2)
            log_prob -= self.action_dim * 0.5 * np.log(2 * np.pi * self.action_noise**2)
        
        # Clip action to reasonable range
        action = np.clip(action, -1, 1)
        
        return action, log_prob
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float
    ) -> None:
        """Store transition in buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
    
    def clear_buffer(self) -> None:
        """Clear experience buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
        }
    
    def update(self) -> Dict[str, float]:
        """
        Update policy and value networks using collected experience.
        
        Returns:
            Dictionary of training statistics
        """
        # Convert buffer to arrays
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        old_values = np.array(self.buffer['values'])
        old_log_probs = np.array(self.buffer['log_probs'])
        
        if len(states) == 0:
            return {}
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        
        # Multiple epochs of updates
        for _ in range(self.epochs_per_update):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            for idx in indices:
                state = states[idx]
                action = actions[idx]
                advantage = advantages[idx]
                return_val = returns[idx]
                old_log_prob = old_log_probs[idx]
                
                # Update policy
                policy_loss = self._update_policy(
                    state, action, advantage, old_log_prob
                )
                policy_losses.append(policy_loss)
                
                # Update value
                value_loss = self._update_value(state, return_val)
                value_losses.append(value_loss)
        
        # Clear buffer after update
        self.clear_buffer()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'mean_return': np.mean(returns),
            'mean_advantage': np.mean(advantages)
        }
    
    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns.
        
        Args:
            rewards: Array of rewards
        
        Returns:
            Array of discounted returns
        """
        returns = np.zeros_like(rewards)
        running_return = 0.0
        
        # Compute returns backward
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _update_policy(
        self,
        state: np.ndarray,
        action: np.ndarray,
        advantage: float,
        old_log_prob: float
    ) -> float:
        """
        Update policy network using PPO clipped objective.
        
        Returns:
            Policy loss value
        """
        # Get current policy output
        mean_action = self.policy.forward(state)
        
        # Compute new log probability
        noise = action - mean_action
        new_log_prob = -0.5 * np.sum(noise**2) / (self.action_noise**2)
        new_log_prob -= self.action_dim * 0.5 * np.log(2 * np.pi * self.action_noise**2)
        
        # Probability ratio
        ratio = np.exp(new_log_prob - old_log_prob)
        
        # PPO clipped objective
        clipped_ratio = np.clip(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
        policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
        
        # Gradient descent (simplified - using finite differences)
        self._gradient_step_policy(state, action, advantage, old_log_prob)
        
        return policy_loss
    
    def _update_value(self, state: np.ndarray, target_value: float) -> float:
        """
        Update value network.
        
        Returns:
            Value loss
        """
        # Get current value estimate
        predicted_value = self.value.forward(state).item()
        
        # MSE loss
        value_loss = 0.5 * (predicted_value - target_value)**2
        
        # Gradient descent (simplified - using finite differences)
        self._gradient_step_value(state, target_value)
        
        return value_loss
    
    def _gradient_step_policy(
        self,
        state: np.ndarray,
        action: np.ndarray,
        advantage: float,
        old_log_prob: float
    ) -> None:
        """Perform gradient step for policy (using finite differences)."""
        # This is a simplified implementation
        # In practice, would use automatic differentiation
        
        eps = 1e-5
        
        # Update each parameter
        for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            param = getattr(self.policy, param_name)
            grad = np.zeros_like(param)
            
            # Estimate gradient via finite differences (only for a few samples)
            n_samples = min(10, param.size)
            indices = np.random.choice(param.size, n_samples, replace=False)
            
            for idx in indices:
                # Flatten index
                idx_tuple = np.unravel_index(idx, param.shape)
                
                # Perturb parameter
                param[idx_tuple] += eps
                mean_action_plus = self.policy.forward(state)
                noise_plus = action - mean_action_plus
                log_prob_plus = -0.5 * np.sum(noise_plus**2) / (self.action_noise**2)
                ratio_plus = np.exp(log_prob_plus - old_log_prob)
                loss_plus = -min(ratio_plus * advantage, np.clip(ratio_plus, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage)
                
                param[idx_tuple] -= 2 * eps
                mean_action_minus = self.policy.forward(state)
                noise_minus = action - mean_action_minus
                log_prob_minus = -0.5 * np.sum(noise_minus**2) / (self.action_noise**2)
                ratio_minus = np.exp(log_prob_minus - old_log_prob)
                loss_minus = -min(ratio_minus * advantage, np.clip(ratio_minus, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage)
                
                param[idx_tuple] += eps  # Restore
                
                grad[idx_tuple] = (loss_plus - loss_minus) / (2 * eps)
            
            # Update parameter
            param -= self.learning_rate * grad
    
    def _gradient_step_value(self, state: np.ndarray, target_value: float) -> None:
        """Perform gradient step for value network (using finite differences)."""
        eps = 1e-5
        
        for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            param = getattr(self.value, param_name)
            grad = np.zeros_like(param)
            
            # Estimate gradient via finite differences
            n_samples = min(10, param.size)
            indices = np.random.choice(param.size, n_samples, replace=False)
            
            for idx in indices:
                idx_tuple = np.unravel_index(idx, param.shape)
                
                param[idx_tuple] += eps
                value_plus = self.value.forward(state).item()
                loss_plus = 0.5 * (value_plus - target_value)**2
                
                param[idx_tuple] -= 2 * eps
                value_minus = self.value.forward(state).item()
                loss_minus = 0.5 * (value_minus - target_value)**2
                
                param[idx_tuple] += eps  # Restore
                
                grad[idx_tuple] = (loss_plus - loss_minus) / (2 * eps)
            
            # Update parameter
            param -= self.learning_rate * grad
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'policy': self.policy.get_params(),
            'value': self.value.get_params(),
            'action_noise': self.action_noise,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.policy.set_params(data['policy'])
        self.value.set_params(data['value'])
        self.action_noise = data['action_noise']
        
        logger.info(f"Agent loaded from {filepath}")
