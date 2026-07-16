"""
Tests for RL-Based Guidance Components

This test suite validates:
1. RL environment (GuidanceEnvironment)
2. RL agent (PPOAgent)
3. RL-based guidance integration
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rl_guidance_env import GuidanceEnvironment
from rl_guidance_agent import PPOAgent, MLPNetwork
from rl_based_guidance import RLGuidanceLaw, RLTwoAgentPursuitGuidance


class TestGuidanceEnvironment(unittest.TestCase):
    """Tests for RL guidance environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        self.target_pos = np.array([20.0, 0.0, 0.0])
        self.target_vel = np.array([2.0, 0.5, 0.0])
        
        self.env = GuidanceEnvironment(
            landmarks=self.uvw,
            initial_rotation=self.R,
            target_position=self.target_pos,
            target_velocity=self.target_vel,
            agent_speed=10.0,
            max_steps=50
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.state_dim, 16)
        self.assertEqual(self.env.action_dim, 3)
        self.assertEqual(self.env.max_steps, 50)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        self.assertEqual(state.shape, (16,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertEqual(self.env.current_step, 0)
    
    def test_reset_with_position(self):
        """Test reset with specific position."""
        init_pos = np.array([5.0, 5.0, 5.0])
        state = self.env.reset(agent_position=init_pos)
        
        np.testing.assert_array_almost_equal(self.env.agent_position, init_pos)
        self.assertEqual(state.shape, (16,))
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        
        action = np.array([1.0, 0.0, 0.0])
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(next_state.shape, (16,))
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, bool)
        self.assertIn('step', info)
        self.assertIn('reward_components', info)
    
    def test_step_normalizes_action(self):
        """Test that step normalizes action to unit vector."""
        self.env.reset()
        
        # Non-unit action
        action = np.array([2.0, 2.0, 2.0])
        prev_pos = self.env.agent_position.copy()
        
        self.env.step(action)
        
        # Check that movement is in direction of action
        displacement = self.env.agent_position - prev_pos
        expected_dir = action / np.linalg.norm(action)
        actual_dir = displacement / np.linalg.norm(displacement)
        
        np.testing.assert_array_almost_equal(actual_dir, expected_dir, decimal=3)
    
    def test_reward_components(self):
        """Test that reward has all expected components."""
        self.env.reset()
        action = np.array([1.0, 0.0, 0.0])
        _, reward, _, info = self.env.step(action)
        
        components = info['reward_components']
        self.assertIn('observability', components)
        self.assertIn('pursuit', components)
        self.assertIn('efficiency', components)
        self.assertIn('total', components)
        
        # Total should be sum of components
        expected_total = (
            components['observability'] +
            components['pursuit'] +
            components['efficiency']
        )
        self.assertAlmostEqual(components['total'], expected_total, places=5)
    
    def test_episode_terminates(self):
        """Test that episode terminates correctly."""
        self.env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = np.random.randn(3)
            _, _, done, _ = self.env.step(action)
            steps += 1
        
        # Should terminate within max_steps
        self.assertTrue(done or steps == self.env.max_steps)
    
    def test_get_state_components(self):
        """Test that state contains expected components."""
        state = self.env.reset()
        
        # Check state dimensions
        self.assertEqual(len(state), 16)
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(state)))


class TestMLPNetwork(unittest.TestCase):
    """Tests for MLP network."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = MLPNetwork(input_dim=10, hidden_dim=32, output_dim=3)
        
        self.assertEqual(net.input_dim, 10)
        self.assertEqual(net.hidden_dim, 32)
        self.assertEqual(net.output_dim, 3)
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        net = MLPNetwork(input_dim=10, hidden_dim=32, output_dim=3)
        x = np.random.randn(10)
        
        output = net.forward(x)
        
        self.assertEqual(output.shape, (3,))
        self.assertTrue(np.all(np.isfinite(output)))
    
    def test_forward_batch(self):
        """Test forward pass with batch."""
        net = MLPNetwork(input_dim=10, hidden_dim=32, output_dim=3)
        x = np.random.randn(5, 10)
        
        output = net.forward(x)
        
        self.assertEqual(output.shape, (5, 3))
        self.assertTrue(np.all(np.isfinite(output)))
    
    def test_policy_network_bounded(self):
        """Test that policy network output is bounded."""
        net = MLPNetwork(input_dim=10, hidden_dim=32, output_dim=3, is_policy=True)
        x = np.random.randn(10)
        
        output = net.forward(x)
        
        # Output should be in [-1, 1] due to tanh
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))
    
    def test_get_set_params(self):
        """Test getting and setting parameters."""
        net = MLPNetwork(input_dim=10, hidden_dim=32, output_dim=3)
        
        # Get params
        params = net.get_params()
        self.assertIn('W1', params)
        self.assertIn('b1', params)
        
        # Modify
        params['W1'] = params['W1'] * 2
        
        # Set params
        net.set_params(params)
        
        # Check modified
        np.testing.assert_array_equal(net.W1, params['W1'])


class TestPPOAgent(unittest.TestCase):
    """Tests for PPO agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PPOAgent(
            state_dim=16,
            action_dim=3,
            hidden_dim=32,
            learning_rate=1e-3
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, 16)
        self.assertEqual(self.agent.action_dim, 3)
        self.assertIsNotNone(self.agent.policy)
        self.assertIsNotNone(self.agent.value)
    
    def test_select_action_deterministic(self):
        """Test deterministic action selection."""
        state = np.random.randn(16)
        
        action, log_prob = self.agent.select_action(state, deterministic=True)
        
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
        self.assertEqual(log_prob, 0.0)
    
    def test_select_action_stochastic(self):
        """Test stochastic action selection."""
        state = np.random.randn(16)
        
        action, log_prob = self.agent.select_action(state, deterministic=False)
        
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
        self.assertIsInstance(log_prob, (float, np.floating))
    
    def test_store_transition(self):
        """Test storing transitions."""
        state = np.random.randn(16)
        action = np.random.randn(3)
        reward = 1.0
        value = 0.5
        log_prob = -1.5
        
        self.agent.store_transition(state, action, reward, value, log_prob)
        
        self.assertEqual(len(self.agent.buffer['states']), 1)
        self.assertEqual(len(self.agent.buffer['actions']), 1)
        self.assertEqual(len(self.agent.buffer['rewards']), 1)
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        # Store some transitions
        for _ in range(5):
            self.agent.store_transition(
                np.random.randn(16),
                np.random.randn(3),
                1.0, 0.5, -1.5
            )
        
        self.agent.clear_buffer()
        
        self.assertEqual(len(self.agent.buffer['states']), 0)
    
    def test_update_with_empty_buffer(self):
        """Test update with empty buffer."""
        stats = self.agent.update()
        
        self.assertEqual(stats, {})
    
    def test_update_with_data(self):
        """Test update with data."""
        # Collect some experience
        for _ in range(10):
            state = np.random.randn(16)
            action, log_prob = self.agent.select_action(state)
            value = self.agent.value.forward(state).item()
            reward = np.random.randn()
            
            self.agent.store_transition(state, action, reward, value, log_prob)
        
        # Update
        stats = self.agent.update()
        
        self.assertIn('policy_loss', stats)
        self.assertIn('value_loss', stats)
        self.assertIn('mean_return', stats)


class TestRLGuidanceLaw(unittest.TestCase):
    """Tests for RL-based guidance law."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        self.t = np.array([5.0, 5.0, 5.0])
    
    def test_compute_optimal_direction(self):
        """Test optimal direction computation."""
        direction, obj_value = RLGuidanceLaw.compute_optimal_direction(
            self.uvw, self.R, self.t
        )
        
        self.assertEqual(direction.shape, (3,))
        # Direction should be approximately unit vector
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=2)
    
    def test_compute_guidance_command(self):
        """Test guidance command computation."""
        velocity = np.array([1.0, 0.0, 0.0])
        
        accel = RLGuidanceLaw.compute_guidance_command(
            self.uvw, self.R, self.t, velocity,
            max_acceleration=2.0
        )
        
        self.assertEqual(accel.shape, (3,))
        # Should respect acceleration limit
        self.assertLessEqual(np.linalg.norm(accel), 2.0 + 1e-6)


class TestRLTwoAgentPursuitGuidance(unittest.TestCase):
    """Tests for RL-based two-agent pursuit."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        
        self.guidance = RLTwoAgentPursuitGuidance(pursuer_speed=10.0)
        
        self.pursuer_pos = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.array([10.0, 0.0, 0.0])
        self.target_vel = np.array([1.0, 0.0, 0.0])
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.guidance.pursuer_speed, 10.0)
        self.assertIsNotNone(self.guidance.agent)
    
    def test_compute_guidance_velocity(self):
        """Test velocity computation."""
        velocity = self.guidance.compute_guidance_velocity(
            self.uvw, self.R, self.pursuer_pos, self.target_pos, self.target_vel
        )
        
        self.assertEqual(velocity.shape, (3,))
        # Should have correct magnitude
        self.assertAlmostEqual(np.linalg.norm(velocity), self.guidance.pursuer_speed, places=2)
    
    def test_simulate_pursuit(self):
        """Test pursuit simulation."""
        results = self.guidance.simulate_pursuit(
            self.uvw, self.R,
            self.pursuer_pos, self.target_pos, self.target_vel,
            duration=1.0, dt=0.1
        )
        
        self.assertIn('time', results)
        self.assertIn('pursuer_trajectory', results)
        self.assertIn('target_trajectory', results)
        self.assertIn('distances', results)
        
        # Check shapes
        n_steps = int(1.0 / 0.1)
        self.assertEqual(len(results['time']), n_steps)
        self.assertEqual(results['pursuer_trajectory'].shape[0], n_steps)


if __name__ == '__main__':
    unittest.main()
