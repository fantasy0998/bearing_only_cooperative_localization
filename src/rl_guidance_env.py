"""
Reinforcement Learning Environment for Guidance Law Learning

This module implements a Gym-compatible environment for learning optimal
guidance policies using reinforcement learning instead of hand-crafted heuristics.

The environment allows an RL agent to learn guidance strategies that optimize
observability in bearing-only cooperative localization scenarios.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from fisher_information_matrix import FisherInformationAnalyzer

logger = logging.getLogger(__name__)


class GuidanceEnvironment:
    """
    Gym-compatible environment for learning guidance policies.
    
    The agent learns to generate velocity commands that optimize observability
    while achieving mission objectives (e.g., target pursuit).
    
    State Space:
        - Relative position to target (3D)
        - Current velocity (3D)
        - FIM metrics (condition number, determinant, eigenvalues)
        - Bearing geometry features (angular diversity, etc.)
    
    Action Space:
        - Continuous: 3D direction vector (normalized to unit vector)
        - The agent outputs a desired direction of motion
    
    Reward Function:
        - Observability improvement (FIM determinant increase)
        - Progress toward target (distance reduction)
        - Penalty for excessive maneuvers
    """
    
    def __init__(
        self,
        landmarks: np.ndarray,
        initial_rotation: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        agent_speed: float = 10.0,
        max_steps: int = 100,
        dt: float = 0.1,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the guidance environment.
        
        Args:
            landmarks: Landmark positions in global frame, shape (3, n)
            initial_rotation: Initial rotation estimate, shape (3, 3)
            target_position: Target initial position, shape (3,)
            target_velocity: Target velocity (constant), shape (3,)
            agent_speed: Agent speed (m/s)
            max_steps: Maximum episode length
            dt: Time step (seconds)
            reward_weights: Dictionary of reward component weights
        """
        self.landmarks = landmarks
        self.rotation = initial_rotation
        self.target_position_0 = target_position.copy()
        self.target_velocity = target_velocity
        self.agent_speed = agent_speed
        self.max_steps = max_steps
        self.dt = dt
        
        # Reward weights (default to balanced)
        self.reward_weights = reward_weights or {
            'observability': 1.0,
            'pursuit': 0.5,
            'efficiency': 0.1
        }
        
        # State variables
        self.agent_position = None
        self.agent_velocity = None
        self.current_step = 0
        self.episode_history = []
        
        # State/action space dimensions
        self.state_dim = 16  # See get_state() for breakdown (was 15, updated to 16)
        self.action_dim = 3  # 3D direction vector
        
    def reset(self, agent_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            agent_position: Optional initial agent position. If None, use random.
        
        Returns:
            Initial state observation
        """
        # Initialize agent position
        if agent_position is None:
            # Random position in a sphere around target
            self.agent_position = self.target_position_0 + np.random.randn(3) * 5
        else:
            self.agent_position = agent_position.copy()
        
        # Initialize velocity (toward target)
        to_target = self.target_position_0 - self.agent_position
        if np.linalg.norm(to_target) > 1e-6:
            self.agent_velocity = (to_target / np.linalg.norm(to_target)) * self.agent_speed
        else:
            self.agent_velocity = np.array([self.agent_speed, 0.0, 0.0])
        
        self.current_step = 0
        self.episode_history = []
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector of shape (state_dim,)
            
        State components:
            [0:3]   - Relative position to target (normalized by initial distance)
            [3:6]   - Current velocity (normalized by agent speed)
            [6:9]   - FIM top 3 eigenvalues (log scale)
            [9]     - FIM condition number (log scale)
            [10]    - FIM determinant (log scale)
            [11]    - Distance to target (normalized)
            [12]    - Angle to target (radians)
            [13:16] - Bearing geometry features (3D)
        """
        # Compute current target position
        target_position = self.target_position_0 + self.target_velocity * (self.current_step * self.dt)
        
        # Relative position
        rel_pos = target_position - self.agent_position
        distance = np.linalg.norm(rel_pos)
        initial_distance = np.linalg.norm(self.target_position_0 - self.agent_position)
        normalized_rel_pos = rel_pos / (initial_distance + 1e-6)
        
        # Normalized velocity
        normalized_vel = self.agent_velocity / (self.agent_speed + 1e-6)
        
        # FIM metrics
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.landmarks, self.rotation, self.agent_position
        )
        eigenvalues = np.linalg.eigvalsh(FIM)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid log(0)
        
        # Take top 3 eigenvalues (FIM is 6x6)
        top_eigenvalues = eigenvalues[-3:]
        log_eigenvalues = np.log10(top_eigenvalues)
        
        cond_number = eigenvalues[-1] / eigenvalues[0]
        log_cond = np.log10(cond_number + 1e-10)
        
        det = np.prod(eigenvalues)
        log_det = np.log10(det + 1e-10)
        
        # Distance and angle to target
        normalized_distance = distance / (initial_distance + 1e-6)
        
        if distance > 1e-6:
            angle_to_target = np.arccos(
                np.clip(np.dot(normalized_vel, rel_pos / distance), -1, 1)
            )
        else:
            angle_to_target = 0.0
        
        # Bearing geometry features
        bearing_features = self._compute_bearing_features()
        
        # Construct state vector
        state = np.concatenate([
            normalized_rel_pos,      # [0:3]
            normalized_vel,           # [3:6]
            log_eigenvalues,          # [6:9]
            [log_cond],              # [9]
            [log_det],               # [10]
            [normalized_distance],    # [11]
            [angle_to_target],       # [12]
            bearing_features         # [13:16] (3 elements)
        ])
        
        return state
    
    def _compute_bearing_features(self) -> np.ndarray:
        """
        Compute bearing geometry features for state representation.
        
        Returns:
            Feature vector of shape (3,)
        """
        # Bearings from agent to landmarks
        bearings = self.landmarks - self.agent_position.reshape(3, 1)
        bearings = bearings / (np.linalg.norm(bearings, axis=0, keepdims=True) + 1e-10)
        
        # Angular diversity (average pairwise angle)
        n = bearings.shape[1]
        if n > 1:
            angles = []
            for i in range(n):
                for j in range(i+1, n):
                    cos_angle = np.dot(bearings[:, i], bearings[:, j])
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
            angular_diversity = np.mean(angles)
        else:
            angular_diversity = 0.0
        
        # Bearing spread (variance in azimuth and elevation)
        azimuths = np.arctan2(bearings[1, :], bearings[0, :])
        elevations = np.arcsin(np.clip(bearings[2, :], -1, 1))
        
        azimuth_spread = np.std(azimuths)
        elevation_spread = np.std(elevations)
        
        return np.array([angular_diversity, azimuth_spread, elevation_spread])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Desired direction vector, shape (3,)
                    Will be normalized and scaled to agent_speed
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Normalize action to unit vector
        action_norm = np.linalg.norm(action)
        if action_norm > 1e-6:
            direction = action / action_norm
        else:
            # If action is zero, maintain current direction
            direction = self.agent_velocity / (np.linalg.norm(self.agent_velocity) + 1e-6)
        
        # Store previous state for reward computation
        prev_position = self.agent_position.copy()
        prev_FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.landmarks, self.rotation, prev_position
        )
        
        # Update agent velocity and position
        self.agent_velocity = direction * self.agent_speed
        self.agent_position = self.agent_position + self.agent_velocity * self.dt
        
        # Update step counter
        self.current_step += 1
        
        # Compute reward
        reward, reward_components = self._compute_reward(prev_position, prev_FIM)
        
        # Check termination
        done = self._check_done()
        
        # Get next state
        next_state = self.get_state()
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'reward_components': reward_components,
            'agent_position': self.agent_position.copy(),
            'target_position': self.target_position_0 + self.target_velocity * (self.current_step * self.dt)
        }
        
        # Store history
        self.episode_history.append({
            'state': next_state,
            'action': direction,
            'reward': reward,
            'position': self.agent_position.copy()
        })
        
        return next_state, reward, done, info
    
    def _compute_reward(
        self, 
        prev_position: np.ndarray, 
        prev_FIM: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for current transition.
        
        Args:
            prev_position: Agent position before action
            prev_FIM: FIM before action
        
        Returns:
            Tuple of (total_reward, reward_components_dict)
        """
        # Current target position
        target_position = self.target_position_0 + self.target_velocity * (self.current_step * self.dt)
        
        # Component 1: Observability improvement
        current_FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.landmarks, self.rotation, self.agent_position
        )
        
        prev_det = np.linalg.det(prev_FIM + np.eye(6) * 1e-10)
        current_det = np.linalg.det(current_FIM + np.eye(6) * 1e-10)
        
        # Log-scale determinant improvement
        if prev_det > 0 and current_det > 0:
            det_improvement = np.log10(current_det) - np.log10(prev_det)
        else:
            det_improvement = 0.0
        
        observability_reward = det_improvement * self.reward_weights['observability']
        
        # Component 2: Pursuit progress
        prev_distance = np.linalg.norm(target_position - prev_position)
        current_distance = np.linalg.norm(target_position - self.agent_position)
        distance_reduction = prev_distance - current_distance
        
        pursuit_reward = distance_reduction * self.reward_weights['pursuit']
        
        # Component 3: Efficiency (penalize excessive maneuvering)
        # Encourage smooth, direct motion
        velocity_change = np.linalg.norm(
            self.agent_velocity - 
            (self.agent_position - prev_position) / (self.dt + 1e-10)
        )
        efficiency_penalty = -velocity_change * self.reward_weights['efficiency']
        
        # Total reward
        total_reward = observability_reward + pursuit_reward + efficiency_penalty
        
        components = {
            'observability': observability_reward,
            'pursuit': pursuit_reward,
            'efficiency': efficiency_penalty,
            'total': total_reward
        }
        
        return total_reward, components
    
    def _check_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if episode is done
        """
        # Check max steps
        if self.current_step >= self.max_steps:
            return True
        
        # Check if reached target
        target_position = self.target_position_0 + self.target_velocity * (self.current_step * self.dt)
        distance = np.linalg.norm(target_position - self.agent_position)
        if distance < 1.0:  # Within 1 meter
            return True
        
        return False
    
    def render(self) -> None:
        """Print current state (for debugging)."""
        target_position = self.target_position_0 + self.target_velocity * (self.current_step * self.dt)
        distance = np.linalg.norm(target_position - self.agent_position)
        
        print(f"\nStep {self.current_step}:")
        print(f"  Agent position: {self.agent_position}")
        print(f"  Target position: {target_position}")
        print(f"  Distance: {distance:.2f}")
        print(f"  Velocity: {self.agent_velocity}")
