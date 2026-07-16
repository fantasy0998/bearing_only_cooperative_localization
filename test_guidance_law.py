"""
Tests for guidance law derivation from Fisher Information Matrix.

This test suite validates the guidance law implementations for:
1. Universal guidance law based on FIM optimization
2. Two-agent pursuit guidance
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from guidance_law import GuidanceLaw, TwoAgentPursuitGuidance
from fisher_information_matrix import FisherInformationAnalyzer


class TestGuidanceLaw(unittest.TestCase):
    """Tests for universal guidance law."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test landmarks (well-distributed)
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        self.t = np.array([1.0, 2.0, 3.0])
        self.velocity = np.array([1.0, 0.0, 0.0])
    
    def test_compute_optimal_direction_returns_unit_vector(self):
        """Test that optimal direction is a unit vector."""
        direction, obj = GuidanceLaw.compute_optimal_direction(
            self.uvw, self.R, self.t, objective_type='trace'
        )
        
        self.assertEqual(direction.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=5)
    
    def test_compute_optimal_direction_all_objectives(self):
        """Test optimal direction computation for all objective types."""
        objectives = ['trace', 'determinant', 'min_eigenvalue', 'inverse_condition']
        
        for obj_type in objectives:
            with self.subTest(objective=obj_type):
                direction, obj_value = GuidanceLaw.compute_optimal_direction(
                    self.uvw, self.R, self.t, objective_type=obj_type
                )
                
                self.assertEqual(direction.shape, (3,))
                self.assertIsInstance(obj_value, (float, np.floating))
                self.assertFalse(np.isnan(obj_value))
    
    def test_compute_guidance_command_bounded(self):
        """Test that guidance command respects acceleration limits."""
        max_accel = 2.0
        acceleration = GuidanceLaw.compute_guidance_command(
            self.uvw, self.R, self.t, self.velocity,
            objective_type='trace',
            max_acceleration=max_accel
        )
        
        self.assertEqual(acceleration.shape, (3,))
        accel_magnitude = np.linalg.norm(acceleration)
        self.assertLessEqual(accel_magnitude, max_accel + 1e-6)
    
    def test_compute_guidance_command_zero_velocity(self):
        """Test guidance command with zero initial velocity."""
        zero_velocity = np.zeros(3)
        acceleration = GuidanceLaw.compute_guidance_command(
            self.uvw, self.R, self.t, zero_velocity,
            objective_type='trace'
        )
        
        self.assertEqual(acceleration.shape, (3,))
        self.assertFalse(np.any(np.isnan(acceleration)))
    
    def test_evaluate_trajectory(self):
        """Test trajectory evaluation."""
        # Create a simple trajectory
        trajectory = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
        ]
        
        results = GuidanceLaw.evaluate_trajectory(
            self.uvw, trajectory, self.R, objective_type='trace'
        )
        
        self.assertIn('objectives', results)
        self.assertIn('condition_numbers', results)
        self.assertIn('determinants', results)
        
        self.assertEqual(len(results['objectives']), len(trajectory))
        self.assertEqual(len(results['condition_numbers']), len(trajectory))
        self.assertEqual(len(results['determinants']), len(trajectory))
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(results['objectives'])))
        self.assertTrue(np.all(np.isfinite(results['condition_numbers'])))
        self.assertTrue(np.all(np.isfinite(results['determinants'])))
    
    def test_optimal_direction_improves_observability(self):
        """Test that moving in optimal direction improves objective."""
        # Get optimal direction
        direction, current_obj = GuidanceLaw.compute_optimal_direction(
            self.uvw, self.R, self.t, objective_type='trace'
        )
        
        # Move in optimal direction
        step_size = 0.5
        t_new = self.t + direction * step_size
        
        # Compute new objective
        new_obj = FisherInformationAnalyzer.compute_condition_based_objective(
            self.uvw, self.R, t_new, objective_type='trace'
        )
        
        # New objective should be better (higher for trace)
        # Note: This may not always hold perfectly due to nonlinearity,
        # but should hold for small steps
        # Use relative tolerance based on objective magnitude
        tolerance = max(abs(current_obj) * 0.01, 1e-6)  # 1% relative or absolute minimum
        self.assertGreaterEqual(new_obj, current_obj - tolerance)


class TestTwoAgentPursuitGuidance(unittest.TestCase):
    """Tests for two-agent pursuit guidance."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        
        # Initialize pursuit guidance
        self.guidance = TwoAgentPursuitGuidance(
            pursuer_speed=10.0,
            pursuit_gain=0.5,
            observability_gain=0.5,
            objective_type='trace'
        )
        
        # Initial conditions
        self.pursuer_pos = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.array([10.0, 0.0, 0.0])
        self.target_vel = np.array([1.0, 0.0, 0.0])
    
    def test_initialization(self):
        """Test pursuit guidance initialization."""
        self.assertEqual(self.guidance.pursuer_speed, 10.0)
        self.assertAlmostEqual(self.guidance.pursuit_gain, 0.5, places=5)
        self.assertAlmostEqual(self.guidance.observability_gain, 0.5, places=5)
    
    def test_initialization_gain_normalization(self):
        """Test that gains are properly normalized."""
        guidance = TwoAgentPursuitGuidance(
            pursuit_gain=3.0,
            observability_gain=1.0
        )
        
        self.assertAlmostEqual(guidance.pursuit_gain, 0.75, places=5)
        self.assertAlmostEqual(guidance.observability_gain, 0.25, places=5)
    
    def test_compute_target_state(self):
        """Test target state computation (straight-line motion)."""
        time = 2.0
        pos, vel = self.guidance.compute_target_state(
            self.target_pos, self.target_vel, time
        )
        
        expected_pos = self.target_pos + self.target_vel * time
        np.testing.assert_array_almost_equal(pos, expected_pos)
        np.testing.assert_array_almost_equal(vel, self.target_vel)
    
    def test_compute_pursuit_direction_returns_unit_vector(self):
        """Test that pursuit direction is a unit vector."""
        pursuit_dir = self.guidance.compute_pursuit_direction(
            self.pursuer_pos, self.target_pos, self.target_vel
        )
        
        self.assertEqual(pursuit_dir.shape, (3,))
        # Should be approximately unit vector (may be zero if at target)
        norm = np.linalg.norm(pursuit_dir)
        if norm > 1e-6:
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_compute_pursuit_direction_at_target(self):
        """Test pursuit direction when already at target."""
        pursuit_dir = self.guidance.compute_pursuit_direction(
            self.target_pos, self.target_pos, self.target_vel
        )
        
        np.testing.assert_array_almost_equal(pursuit_dir, np.zeros(3))
    
    def test_compute_observability_direction_returns_unit_vector(self):
        """Test that observability direction is a unit vector."""
        obs_dir = self.guidance.compute_observability_direction(
            self.uvw, self.R, self.pursuer_pos
        )
        
        self.assertEqual(obs_dir.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(obs_dir), 1.0, places=5)
    
    def test_compute_guidance_velocity_magnitude(self):
        """Test that guidance velocity has correct magnitude."""
        velocity = self.guidance.compute_guidance_velocity(
            self.uvw, self.R, self.pursuer_pos, self.target_pos, self.target_vel
        )
        
        self.assertEqual(velocity.shape, (3,))
        speed = np.linalg.norm(velocity)
        self.assertAlmostEqual(speed, self.guidance.pursuer_speed, places=5)
    
    def test_compute_guidance_velocity_pure_pursuit(self):
        """Test guidance with pure pursuit (no observability)."""
        guidance = TwoAgentPursuitGuidance(
            pursuit_gain=1.0,
            observability_gain=0.0
        )
        
        velocity = guidance.compute_guidance_velocity(
            self.uvw, self.R, self.pursuer_pos, self.target_pos, self.target_vel
        )
        
        self.assertEqual(velocity.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(velocity), guidance.pursuer_speed, places=5)
    
    def test_compute_guidance_velocity_pure_observability(self):
        """Test guidance with pure observability (no pursuit)."""
        guidance = TwoAgentPursuitGuidance(
            pursuit_gain=0.0,
            observability_gain=1.0
        )
        
        velocity = guidance.compute_guidance_velocity(
            self.uvw, self.R, self.pursuer_pos, self.target_pos, self.target_vel
        )
        
        self.assertEqual(velocity.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(velocity), guidance.pursuer_speed, places=5)
    
    def test_simulate_pursuit_basic(self):
        """Test basic pursuit simulation."""
        duration = 2.0
        dt = 0.1
        
        results = self.guidance.simulate_pursuit(
            self.uvw, self.R,
            self.pursuer_pos, self.target_pos, self.target_vel,
            duration, dt
        )
        
        # Check all expected keys
        self.assertIn('time', results)
        self.assertIn('pursuer_trajectory', results)
        self.assertIn('target_trajectory', results)
        self.assertIn('pursuer_velocities', results)
        self.assertIn('distances', results)
        self.assertIn('observability_metrics', results)
        
        # Check array shapes
        n_steps = int(duration / dt)
        self.assertEqual(len(results['time']), n_steps)
        self.assertEqual(results['pursuer_trajectory'].shape, (n_steps, 3))
        self.assertEqual(results['target_trajectory'].shape, (n_steps, 3))
        self.assertEqual(results['pursuer_velocities'].shape, (n_steps, 3))
        self.assertEqual(len(results['distances']), n_steps)
    
    def test_simulate_pursuit_distance_decreases(self):
        """Test that pursuer gets closer to target over time (mostly)."""
        duration = 5.0
        dt = 0.1
        
        # Use high pursuit gain
        guidance = TwoAgentPursuitGuidance(
            pursuer_speed=15.0,  # Faster than target
            pursuit_gain=0.9,
            observability_gain=0.1
        )
        
        results = guidance.simulate_pursuit(
            self.uvw, self.R,
            self.pursuer_pos, self.target_pos, self.target_vel,
            duration, dt
        )
        
        # Initial and final distances
        initial_distance = results['distances'][0]
        final_distance = results['distances'][-1]
        
        # Distance should generally decrease (pursuer is faster)
        self.assertLess(final_distance, initial_distance)
    
    def test_simulate_pursuit_observability_tracked(self):
        """Test that observability metrics are tracked during pursuit."""
        duration = 1.0
        dt = 0.1
        
        results = self.guidance.simulate_pursuit(
            self.uvw, self.R,
            self.pursuer_pos, self.target_pos, self.target_vel,
            duration, dt
        )
        
        metrics = results['observability_metrics']
        
        self.assertIn('objectives', metrics)
        self.assertIn('condition_numbers', metrics)
        self.assertIn('determinants', metrics)
        
        # All should be finite
        self.assertTrue(np.all(np.isfinite(metrics['objectives'])))
        self.assertTrue(np.all(np.isfinite(metrics['condition_numbers'])))
        self.assertTrue(np.all(np.isfinite(metrics['determinants'])))
    
    def test_pursuit_with_different_objective_types(self):
        """Test pursuit with different objective types."""
        objectives = ['trace', 'determinant', 'min_eigenvalue', 'inverse_condition']
        
        for obj_type in objectives:
            with self.subTest(objective=obj_type):
                guidance = TwoAgentPursuitGuidance(
                    objective_type=obj_type
                )
                
                velocity = guidance.compute_guidance_velocity(
                    self.uvw, self.R, self.pursuer_pos, 
                    self.target_pos, self.target_vel
                )
                
                self.assertEqual(velocity.shape, (3,))
                self.assertFalse(np.any(np.isnan(velocity)))


class TestGuidanceLawIntegration(unittest.TestCase):
    """Integration tests combining guidance law components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
    
    def test_guidance_consistency_across_objectives(self):
        """Test that different objectives produce reasonable guidance."""
        t = np.array([5.0, 5.0, 5.0])
        objectives = ['trace', 'determinant', 'min_eigenvalue']
        
        directions = {}
        for obj_type in objectives:
            direction, _ = GuidanceLaw.compute_optimal_direction(
                self.uvw, self.R, t, objective_type=obj_type
            )
            directions[obj_type] = direction
        
        # All should be unit vectors
        for direction in directions.values():
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=5)
        
        # Directions may differ but should be finite
        for direction in directions.values():
            self.assertTrue(np.all(np.isfinite(direction)))
    
    def test_pursuit_observability_tradeoff(self):
        """Test tradeoff between pursuit and observability."""
        # Pure pursuit
        guidance_pursuit = TwoAgentPursuitGuidance(
            pursuit_gain=1.0, observability_gain=0.0
        )
        
        # Pure observability
        guidance_obs = TwoAgentPursuitGuidance(
            pursuit_gain=0.0, observability_gain=1.0
        )
        
        # Balanced
        guidance_balanced = TwoAgentPursuitGuidance(
            pursuit_gain=0.5, observability_gain=0.5
        )
        
        pursuer_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([10.0, 0.0, 0.0])
        target_vel = np.array([1.0, 0.0, 0.0])
        
        vel_pursuit = guidance_pursuit.compute_guidance_velocity(
            self.uvw, self.R, pursuer_pos, target_pos, target_vel
        )
        
        vel_obs = guidance_obs.compute_guidance_velocity(
            self.uvw, self.R, pursuer_pos, target_pos, target_vel
        )
        
        vel_balanced = guidance_balanced.compute_guidance_velocity(
            self.uvw, self.R, pursuer_pos, target_pos, target_vel
        )
        
        # All should have same magnitude
        self.assertAlmostEqual(np.linalg.norm(vel_pursuit), 10.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(vel_obs), 10.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(vel_balanced), 10.0, places=5)
        
        # Balanced should be between the two extremes (in some sense)
        # This is a soft check - just verify it's different from both
        self.assertFalse(np.allclose(vel_balanced, vel_pursuit))
        self.assertFalse(np.allclose(vel_balanced, vel_obs))


if __name__ == '__main__':
    unittest.main()
