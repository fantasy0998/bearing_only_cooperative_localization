"""
Training Script for RL Guidance Agent

This script trains a PPO agent to learn optimal guidance policies
for bearing-only cooperative localization.

The agent learns to balance:
1. Observability improvement (better FIM metrics)
2. Mission objectives (e.g., target pursuit)
3. Efficiency (smooth trajectories)
"""

import numpy as np
import sys
import os
import argparse
import logging
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rl_guidance_env import GuidanceEnvironment
from rl_guidance_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_scenario() -> Dict:
    """
    Generate random training scenario.
    
    Returns:
        Dictionary with scenario parameters
    """
    # Random landmarks
    n_landmarks = np.random.randint(6, 12)
    landmarks = np.random.randn(3, n_landmarks) * 15
    
    # Random initial rotation (identity for simplicity)
    rotation = np.eye(3)
    
    # Random target position and velocity
    target_position = np.random.randn(3) * 20
    target_velocity = np.random.randn(3) * 2
    
    # Random agent speed
    agent_speed = np.random.uniform(8, 15)
    
    return {
        'landmarks': landmarks,
        'rotation': rotation,
        'target_position': target_position,
        'target_velocity': target_velocity,
        'agent_speed': agent_speed
    }


def train_episode(
    env: GuidanceEnvironment,
    agent: PPOAgent,
    max_steps: int = 100
) -> Dict[str, float]:
    """
    Train one episode.
    
    Args:
        env: Training environment
        agent: RL agent
        max_steps: Maximum episode steps
    
    Returns:
        Episode statistics
    """
    state = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    
    for step in range(max_steps):
        # Select action
        action, log_prob = agent.select_action(state, deterministic=False)
        
        # Get value estimate
        value = agent.value.forward(state).item()
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, value, log_prob)
        
        episode_reward += reward
        episode_steps += 1
        state = next_state
        
        if done:
            break
    
    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'mean_reward': episode_reward / episode_steps if episode_steps > 0 else 0
    }


def evaluate_agent(
    agent: PPOAgent,
    n_episodes: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Args:
        agent: RL agent to evaluate
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
    
    Returns:
        Evaluation statistics
    """
    np.random.seed(seed)
    
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        # Generate scenario
        scenario = generate_random_scenario()
        
        # Create environment
        env = GuidanceEnvironment(
            landmarks=scenario['landmarks'],
            initial_rotation=scenario['rotation'],
            target_position=scenario['target_position'],
            target_velocity=scenario['target_velocity'],
            agent_speed=scenario['agent_speed']
        )
        
        # Run episode
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        
        for _ in range(100):
            action, _ = agent.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def train(
    n_episodes: int = 1000,
    update_frequency: int = 10,
    eval_frequency: int = 50,
    save_frequency: int = 100,
    model_save_path: str = 'models/rl_guidance_agent.pkl',
    hidden_dim: int = 64,
    learning_rate: float = 3e-4
) -> None:
    """
    Main training loop.
    
    Args:
        n_episodes: Total number of training episodes
        update_frequency: Update policy every N episodes
        eval_frequency: Evaluate every N episodes
        save_frequency: Save model every N episodes
        model_save_path: Path to save trained model
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate
    """
    logger.info("Starting RL guidance agent training")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Hidden dim: {hidden_dim}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Create agent
    agent = PPOAgent(
        state_dim=16,
        action_dim=3,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate
    )
    
    # Training statistics
    episode_rewards = []
    training_losses = []
    
    # Training loop
    for episode in range(n_episodes):
        # Generate random scenario
        scenario = generate_random_scenario()
        
        # Create environment
        env = GuidanceEnvironment(
            landmarks=scenario['landmarks'],
            initial_rotation=scenario['rotation'],
            target_position=scenario['target_position'],
            target_velocity=scenario['target_velocity'],
            agent_speed=scenario['agent_speed']
        )
        
        # Train episode
        stats = train_episode(env, agent)
        episode_rewards.append(stats['reward'])
        
        # Update policy
        if (episode + 1) % update_frequency == 0:
            update_stats = agent.update()
            if update_stats:
                training_losses.append(update_stats['policy_loss'])
                logger.info(
                    f"Episode {episode+1}/{n_episodes} - "
                    f"Reward: {stats['reward']:.2f} - "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} - "
                    f"Value Loss: {update_stats['value_loss']:.4f}"
                )
        
        # Evaluate
        if (episode + 1) % eval_frequency == 0:
            eval_stats = evaluate_agent(agent, n_episodes=10)
            logger.info(
                f"\nEvaluation at episode {episode+1}:\n"
                f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n"
                f"  Mean Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}\n"
            )
        
        # Save model
        if (episode + 1) % save_frequency == 0:
            agent.save(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
    
    # Final save
    agent.save(model_save_path)
    logger.info(f"\nTraining complete! Final model saved to {model_save_path}")
    
    # Final evaluation
    logger.info("\nRunning final evaluation...")
    final_eval = evaluate_agent(agent, n_episodes=50)
    logger.info(
        f"Final Evaluation (50 episodes):\n"
        f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}\n"
        f"  Mean Length: {final_eval['mean_length']:.1f} ± {final_eval['std_length']:.1f}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train RL guidance agent')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--update-freq', type=int, default=10,
                        help='Update policy every N episodes')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluate every N episodes')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--model-path', type=str, default='models/rl_guidance_agent.pkl',
                        help='Path to save trained model')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Run training
    train(
        n_episodes=args.episodes,
        update_frequency=args.update_freq,
        eval_frequency=args.eval_freq,
        save_frequency=args.save_freq,
        model_save_path=args.model_path,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()
