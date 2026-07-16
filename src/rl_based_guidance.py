"""
RL-Based Guidance Law

This module provides reinforcement learning-based guidance that can be used
as a drop-in replacement for heuristic-based guidance laws.

Key advantage: The RL agent learns optimal policies from data rather than
relying on hand-crafted heuristics.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
import os
from rl_guidance_env import GuidanceEnvironment
from rl_guidance_agent import PPOAgent

logger = logging.getLogger(__name__)


class RLGuidanceLaw:
    """
    RL-based guidance law for bearing-only cooperative localization.
    
    This class provides the same interface as the heuristic GuidanceLaw
    but uses a learned policy instead of gradient-based optimization.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        state_dim: int = 16,
        action_dim: int = 3,
        hidden_dim: int = 64
    ):
        """
        Initialize RL guidance law.
        
        Args:
            model_path: Path to trained model. If None, uses untrained model.
            state_dim: State space dimension
            action_dim: Action space dimension  
            hidden_dim: Hidden layer dimension
        """
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            logger.info(f"Loaded RL model from {model_path}")
        else:
            logger.warning("No trained model loaded. Using untrained policy.")
    
    @staticmethod
    def compute_optimal_direction(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        model_path: Optional[str] = None,
        step_size: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal motion direction using RL policy.
        
        This provides the same interface as GuidanceLaw.compute_optimal_direction
        but uses a learned policy instead of FIM gradient.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            model_path: Path to trained model
            step_size: Not used (kept for interface compatibility)
            
        Returns:
            Tuple of (optimal_direction, dummy_objective_value)
        """
        # Create temporary environment to get state
        env = GuidanceEnvironment(
            landmarks=uvw,
            initial_rotation=R,
            target_position=t + np.array([10.0, 0.0, 0.0]),  # Dummy target
            target_velocity=np.zeros(3),
            max_steps=1
        )
        
        # Reset with current position
        state = env.reset(agent_position=t)
        
        # Create agent and get action
        guidance = RLGuidanceLaw(model_path=model_path)
        action, _ = guidance.agent.select_action(state, deterministic=True)
        
        # Normalize to unit vector
        direction = action / (np.linalg.norm(action) + 1e-10)
        
        # Return dummy objective value (for interface compatibility)
        objective_value = 0.0
        
        return direction, objective_value
    
    @staticmethod
    def compute_guidance_command(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        current_velocity: np.ndarray,
        model_path: Optional[str] = None,
        max_acceleration: float = 1.0,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Compute guidance acceleration command using RL policy.
        
        This provides the same interface as GuidanceLaw.compute_guidance_command.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            current_velocity: Current velocity vector, shape (3,)
            model_path: Path to trained model
            max_acceleration: Maximum allowed acceleration magnitude
            dt: Time step for guidance update
            
        Returns:
            Acceleration command vector, shape (3,)
        """
        # Get optimal direction from RL policy
        optimal_dir, _ = RLGuidanceLaw.compute_optimal_direction(
            uvw, R, t, model_path
        )
        
        # Compute desired velocity
        speed = np.linalg.norm(current_velocity)
        if speed < 1e-6:
            speed = 1.0
        desired_velocity = optimal_dir * speed
        
        # Compute acceleration command
        acceleration = (desired_velocity - current_velocity) / dt
        
        # Limit acceleration
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > max_acceleration:
            acceleration = acceleration * (max_acceleration / accel_magnitude)
        
        return acceleration


class RLTwoAgentPursuitGuidance:
    """
    RL-based two-agent pursuit guidance.
    
    Learns to balance pursuit and observability objectives through
    reinforcement learning rather than manual weight tuning.
    """
    
    def __init__(
        self,
        pursuer_speed: float = 10.0,
        model_path: Optional[str] = None,
        state_dim: int = 16,
        action_dim: int = 3,
        hidden_dim: int = 64
    ):
        """
        Initialize RL pursuit guidance.
        
        Args:
            pursuer_speed: Constant speed of pursuer
            model_path: Path to trained model
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
        """
        self.pursuer_speed = pursuer_speed
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            logger.info(f"Loaded RL pursuit model from {model_path}")
        else:
            logger.warning("No trained model loaded. Using untrained policy.")
    
    def compute_guidance_velocity(
        self,
        uvw: np.ndarray,
        R: np.ndarray,
        pursuer_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute pursuer velocity command using RL policy.
        
        Args:
            uvw: Landmark positions in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            pursuer_position: Current pursuer position, shape (3,)
            target_position: Current target position, shape (3,)
            target_velocity: Target velocity, shape (3,)
            
        Returns:
            Velocity command for pursuer, shape (3,)
        """
        # Create environment to get state
        env = GuidanceEnvironment(
            landmarks=uvw,
            initial_rotation=R,
            target_position=target_position,
            target_velocity=target_velocity,
            agent_speed=self.pursuer_speed,
            max_steps=1
        )
        
        # Get state
        state = env.reset(agent_position=pursuer_position)
        
        # Get action from policy
        action, _ = self.agent.select_action(state, deterministic=True)
        
        # Convert to velocity
        direction = action / (np.linalg.norm(action) + 1e-10)
        velocity = direction * self.pursuer_speed
        
        return velocity
    
    def simulate_pursuit(
        self,
        uvw: np.ndarray,
        R: np.ndarray,
        pursuer_position_0: np.ndarray,
        target_position_0: np.ndarray,
        target_velocity: np.ndarray,
        duration: float,
        dt: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Simulate two-agent pursuit using RL policy.
        
        Args:
            uvw: Landmark positions in global frame, shape (3, n)
            R: Rotation estimate (assumed constant), shape (3, 3)
            pursuer_position_0: Initial pursuer position, shape (3,)
            target_position_0: Initial target position, shape (3,)
            target_velocity: Target constant velocity, shape (3,)
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            
        Returns:
            Dictionary containing simulation results
        """
        # Create environment
        env = GuidanceEnvironment(
            landmarks=uvw,
            initial_rotation=R,
            target_position=target_position_0,
            target_velocity=target_velocity,
            agent_speed=self.pursuer_speed,
            max_steps=int(duration / dt),
            dt=dt
        )
        
        # Initialize
        state = env.reset(agent_position=pursuer_position_0)
        
        # Storage
        n_steps = int(duration / dt)
        time = np.linspace(0, duration, n_steps)
        pursuer_trajectory = np.zeros((n_steps, 3))
        target_trajectory = np.zeros((n_steps, 3))
        pursuer_velocities = np.zeros((n_steps, 3))
        distances = np.zeros(n_steps)
        
        # Initial state
        pursuer_trajectory[0] = pursuer_position_0
        target_trajectory[0] = target_position_0
        distances[0] = np.linalg.norm(pursuer_position_0 - target_position_0)
        
        # Simulate
        for i in range(1, n_steps):
            # Get action from policy
            action, _ = self.agent.select_action(state, deterministic=True)
            
            # Step environment
            state, reward, done, info = env.step(action)
            
            # Store results
            pursuer_trajectory[i] = info['agent_position']
            target_trajectory[i] = info['target_position']
            pursuer_velocities[i] = env.agent_velocity
            distances[i] = np.linalg.norm(info['agent_position'] - info['target_position'])
            
            if done:
                # Fill remaining with final values
                pursuer_trajectory[i:] = info['agent_position']
                target_trajectory[i:] = info['target_position']
                pursuer_velocities[i:] = env.agent_velocity
                distances[i:] = distances[i]
                break
        
        # Compute observability metrics
        from guidance_law import GuidanceLaw
        trajectory_positions = [pursuer_trajectory[i] for i in range(n_steps)]
        obs_metrics = GuidanceLaw.evaluate_trajectory(
            uvw, trajectory_positions, R, objective_type='trace'
        )
        
        return {
            'time': time,
            'pursuer_trajectory': pursuer_trajectory,
            'target_trajectory': target_trajectory,
            'pursuer_velocities': pursuer_velocities,
            'distances': distances,
            'observability_metrics': obs_metrics
        }
