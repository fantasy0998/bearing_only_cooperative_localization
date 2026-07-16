"""
Guidance Law Derivation from Condition Matrix and Fisher Information Matrix

This module implements guidance laws for bearing-only cooperative localization
based on optimizing observability metrics derived from the Fisher Information 
Matrix and condition number analysis.

Key Concepts:
1. Universal Guidance Law: Derives motion commands to optimize observability
   along specific directions based on FIM eigenstructure
2. Two-Agent Pursuit: Specialized guidance for a pursuit scenario where one
   agent flies straight and another pursues while optimizing observability

References:
- JS Russell et al., "Cooperative Localisation of a GPS-Denied UAV using 
  Direction-of-Arrival Measurements," IEEE TAES, 2019.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from fisher_information_matrix import FisherInformationAnalyzer

logger = logging.getLogger(__name__)


class GuidanceLaw:
    """
    Universal guidance law based on Fisher Information Matrix optimization.
    
    This class implements guidance strategies that optimize observability
    by maximizing/minimizing various FIM-based metrics.
    """
    
    @staticmethod
    def compute_optimal_direction(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        objective_type: str = 'trace',
        step_size: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the optimal motion direction to improve observability.
        
        Uses gradient-based optimization to find the direction that maximizes
        the specified FIM objective.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            objective_type: Type of objective to optimize
                          ('trace', 'determinant', 'min_eigenvalue', 'inverse_condition')
            step_size: Step size for numerical gradient computation
            
        Returns:
            Tuple of (optimal_direction, objective_value)
            - optimal_direction: 3D unit vector indicating best motion direction
            - objective_value: Current objective value
        """
        # Compute current objective
        current_obj = FisherInformationAnalyzer.compute_condition_based_objective(
            uvw, R, t, objective_type
        )
        
        # Compute gradient via finite differences
        gradient = np.zeros(3)
        for i in range(3):
            # Positive perturbation
            t_plus = t.copy()
            t_plus[i] += step_size
            obj_plus = FisherInformationAnalyzer.compute_condition_based_objective(
                uvw, R, t_plus, objective_type
            )
            
            # Negative perturbation
            t_minus = t.copy()
            t_minus[i] -= step_size
            obj_minus = FisherInformationAnalyzer.compute_condition_based_objective(
                uvw, R, t_minus, objective_type
            )
            
            # Central difference
            gradient[i] = (obj_plus - obj_minus) / (2 * step_size)
        
        # Normalize gradient to get unit direction
        if np.linalg.norm(gradient) > 1e-10:
            optimal_direction = gradient / np.linalg.norm(gradient)
        else:
            # If gradient is zero, use FIM eigenvector analysis
            FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(uvw, R, t)
            eigenvalues, eigenvectors = np.linalg.eigh(FIM)
            # Move in direction of smallest eigenvalue (worst observable direction)
            # FIM structure: [rotation(0:3), translation(3:6)]
            optimal_direction = eigenvectors[3:6, 0]  # Translation part of smallest eigenvector
            optimal_direction = optimal_direction / (np.linalg.norm(optimal_direction) + 1e-10)
        
        return optimal_direction, current_obj
    
    @staticmethod
    def compute_guidance_command(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        current_velocity: np.ndarray,
        objective_type: str = 'trace',
        max_acceleration: float = 1.0,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Compute guidance acceleration command to improve observability.
        
        This implements a proportional guidance law that steers the agent
        toward the optimal direction for observability.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            current_velocity: Current velocity vector, shape (3,)
            objective_type: Type of objective to optimize
            max_acceleration: Maximum allowed acceleration magnitude
            dt: Time step for guidance update
            
        Returns:
            Acceleration command vector, shape (3,)
        """
        # Get optimal direction
        optimal_dir, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, t, objective_type
        )
        
        # Compute desired velocity in optimal direction
        speed = np.linalg.norm(current_velocity)
        if speed < 1e-6:
            speed = 1.0  # Default speed
        desired_velocity = optimal_dir * speed
        
        # Compute acceleration command (proportional control)
        acceleration = (desired_velocity - current_velocity) / dt
        
        # Limit acceleration
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > max_acceleration:
            acceleration = acceleration * (max_acceleration / accel_magnitude)
        
        return acceleration
    
    @staticmethod
    def evaluate_trajectory(
        uvw: np.ndarray,
        trajectory: List[np.ndarray],
        R: np.ndarray,
        objective_type: str = 'trace'
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate observability along a trajectory.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            trajectory: List of positions (translation vectors)
            R: Rotation estimate (assumed constant)
            objective_type: Type of objective to evaluate
            
        Returns:
            Dictionary containing:
            - objectives: Array of objective values along trajectory
            - condition_numbers: Array of condition numbers
            - determinants: Array of FIM determinants
        """
        n_steps = len(trajectory)
        objectives = np.zeros(n_steps)
        condition_numbers = np.zeros(n_steps)
        determinants = np.zeros(n_steps)
        
        for i, t in enumerate(trajectory):
            # Compute FIM metrics
            obj = FisherInformationAnalyzer.compute_condition_based_objective(
                uvw, R, t, objective_type
            )
            analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
            
            objectives[i] = obj
            condition_numbers[i] = analysis['FIM_metrics']['condition_number']
            determinants[i] = analysis['FIM_metrics']['determinant']
        
        return {
            'objectives': objectives,
            'condition_numbers': condition_numbers,
            'determinants': determinants
        }


class TwoAgentPursuitGuidance:
    """
    Guidance law for two-agent pursuit scenario.
    
    Scenario:
    - Agent 1 (target): Flies in a straight line at constant velocity
    - Agent 2 (pursuer): Pursues Agent 1 while optimizing observability
    
    The pursuer uses bearing-only measurements to landmarks to localize
    itself relative to the target, and follows a guidance law that balances
    pursuit with observability optimization.
    """
    
    def __init__(
        self,
        pursuer_speed: float = 10.0,
        pursuit_gain: float = 0.5,
        observability_gain: float = 0.5,
        objective_type: str = 'trace'
    ):
        """
        Initialize pursuit guidance parameters.
        
        Args:
            pursuer_speed: Constant speed of pursuer
            pursuit_gain: Weight for pure pursuit component (0 to 1)
            observability_gain: Weight for observability component (0 to 1)
            objective_type: FIM objective to optimize
        """
        self.pursuer_speed = pursuer_speed
        self.pursuit_gain = pursuit_gain
        self.observability_gain = observability_gain
        self.objective_type = objective_type
        
        # Normalize gains
        total_gain = pursuit_gain + observability_gain
        if total_gain > 0:
            self.pursuit_gain = pursuit_gain / total_gain
            self.observability_gain = observability_gain / total_gain
    
    def compute_target_state(
        self,
        target_position_0: np.ndarray,
        target_velocity: np.ndarray,
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute target state at given time (straight-line motion).
        
        Args:
            target_position_0: Initial position of target, shape (3,)
            target_velocity: Constant velocity of target, shape (3,)
            time: Current time
            
        Returns:
            Tuple of (position, velocity)
        """
        position = target_position_0 + target_velocity * time
        velocity = target_velocity.copy()
        return position, velocity
    
    def compute_pursuit_direction(
        self,
        pursuer_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute pure pursuit direction (proportional navigation).
        
        Uses classic proportional navigation: steer toward predicted
        intercept point.
        
        Args:
            pursuer_position: Current pursuer position, shape (3,)
            target_position: Current target position, shape (3,)
            target_velocity: Target velocity, shape (3,)
            
        Returns:
            Unit vector indicating pursuit direction
        """
        # Vector from pursuer to target
        los_vector = target_position - pursuer_position
        los_distance = np.linalg.norm(los_vector)
        
        if los_distance < 1e-6:
            # Already at target
            return np.zeros(3)
        
        # Compute time to intercept (assuming constant closing speed)
        target_speed = np.linalg.norm(target_velocity)
        los_rate = np.dot(target_velocity, los_vector) / los_distance
        
        # Predicted intercept time (simplified)
        if self.pursuer_speed > target_speed:
            closing_speed = self.pursuer_speed - los_rate
            if closing_speed > 0:
                time_to_intercept = los_distance / closing_speed
            else:
                # Default: 1 second lookahead for prediction stability
                time_to_intercept = 1.0
        else:
            # Default: 1 second lookahead when cannot catch up
            time_to_intercept = 1.0
        
        # Predicted target position
        predicted_position = target_position + target_velocity * time_to_intercept
        
        # Direction to predicted intercept
        intercept_vector = predicted_position - pursuer_position
        pursuit_direction = intercept_vector / (np.linalg.norm(intercept_vector) + 1e-10)
        
        return pursuit_direction
    
    def compute_observability_direction(
        self,
        uvw: np.ndarray,
        R: np.ndarray,
        pursuer_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute direction that improves observability.
        
        Args:
            uvw: Landmark positions in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            pursuer_position: Current pursuer position (translation), shape (3,)
            
        Returns:
            Unit vector indicating observability-optimal direction
        """
        optimal_dir, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, pursuer_position, self.objective_type
        )
        return optimal_dir
    
    def compute_guidance_velocity(
        self,
        uvw: np.ndarray,
        R: np.ndarray,
        pursuer_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute pursuer velocity command balancing pursuit and observability.
        
        Args:
            uvw: Landmark positions in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            pursuer_position: Current pursuer position, shape (3,)
            target_position: Current target position, shape (3,)
            target_velocity: Target velocity, shape (3,)
            
        Returns:
            Velocity command for pursuer, shape (3,)
        """
        # Compute pursuit direction
        pursuit_dir = self.compute_pursuit_direction(
            pursuer_position, target_position, target_velocity
        )
        
        # Compute observability direction
        obs_dir = self.compute_observability_direction(
            uvw, R, pursuer_position
        )
        
        # Blend directions
        blended_direction = (
            self.pursuit_gain * pursuit_dir + 
            self.observability_gain * obs_dir
        )
        
        # Normalize and scale to desired speed
        direction_norm = np.linalg.norm(blended_direction)
        if direction_norm > 1e-10:
            velocity = (blended_direction / direction_norm) * self.pursuer_speed
        else:
            velocity = pursuit_dir * self.pursuer_speed
        
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
        Simulate the two-agent pursuit scenario.
        
        Args:
            uvw: Landmark positions in global frame, shape (3, n)
            R: Rotation estimate (assumed constant), shape (3, 3)
            pursuer_position_0: Initial pursuer position, shape (3,)
            target_position_0: Initial target position, shape (3,)
            target_velocity: Target constant velocity, shape (3,)
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            
        Returns:
            Dictionary containing:
            - time: Time array
            - pursuer_trajectory: Array of pursuer positions, shape (n_steps, 3)
            - target_trajectory: Array of target positions, shape (n_steps, 3)
            - pursuer_velocities: Array of pursuer velocities, shape (n_steps, 3)
            - distances: Array of pursuer-target distances
            - observability_metrics: Dict with FIM metrics over time
        """
        n_steps = int(duration / dt)
        time = np.linspace(0, duration, n_steps)
        
        # Initialize trajectory arrays
        pursuer_trajectory = np.zeros((n_steps, 3))
        target_trajectory = np.zeros((n_steps, 3))
        pursuer_velocities = np.zeros((n_steps, 3))
        distances = np.zeros(n_steps)
        
        # Initialize positions
        pursuer_pos = pursuer_position_0.copy()
        pursuer_trajectory[0] = pursuer_pos
        target_trajectory[0] = target_position_0
        distances[0] = np.linalg.norm(pursuer_pos - target_position_0)
        
        # Simulate
        for i in range(1, n_steps):
            # Update target position (straight line)
            target_pos, _ = self.compute_target_state(
                target_position_0, target_velocity, time[i]
            )
            
            # Compute pursuer velocity command
            velocity = self.compute_guidance_velocity(
                uvw, R, pursuer_pos, target_pos, target_velocity
            )
            
            # Update pursuer position
            pursuer_pos = pursuer_pos + velocity * dt
            
            # Store
            pursuer_trajectory[i] = pursuer_pos
            target_trajectory[i] = target_pos
            pursuer_velocities[i] = velocity
            distances[i] = np.linalg.norm(pursuer_pos - target_pos)
        
        # Evaluate observability along pursuer trajectory
        trajectory_positions = [pursuer_trajectory[i] for i in range(n_steps)]
        obs_metrics = GuidanceLaw.evaluate_trajectory(
            uvw, trajectory_positions, R, self.objective_type
        )
        
        return {
            'time': time,
            'pursuer_trajectory': pursuer_trajectory,
            'target_trajectory': target_trajectory,
            'pursuer_velocities': pursuer_velocities,
            'distances': distances,
            'observability_metrics': obs_metrics
        }
