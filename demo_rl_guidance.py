"""
Demonstration: RL Guidance vs Heuristic Guidance

This script compares reinforcement learning-based guidance with
traditional heuristic-based guidance (FIM optimization).

Shows that RL can learn effective guidance policies from data
without manual heuristic design.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from guidance_law import GuidanceLaw, TwoAgentPursuitGuidance
from rl_based_guidance import RLGuidanceLaw, RLTwoAgentPursuitGuidance
from rl_guidance_env import GuidanceEnvironment
from rl_guidance_agent import PPOAgent
from fisher_information_matrix import FisherInformationAnalyzer


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def demo_rl_vs_heuristic_guidance():
    """Compare RL guidance with heuristic guidance."""
    print_section("DEMO 1: RL Guidance vs Heuristic Guidance")
    
    # Create test scenario
    np.random.seed(42)
    uvw = np.random.randn(3, 8) * 10
    R = np.eye(3)
    t = np.array([5.0, 5.0, 5.0])
    
    print(f"\nTest scenario:")
    print(f"  Landmarks: {uvw.shape[1]} points")
    print(f"  Current position: {t}")
    
    # Analyze current observability
    print("\n--- Current Observability ---")
    analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
    print(f"FIM Condition Number: {analysis['FIM_metrics']['condition_number']:.2e}")
    print(f"FIM Determinant: {analysis['FIM_metrics']['determinant']:.2e}")
    
    # Compare different guidance approaches
    print("\n--- Comparing Guidance Approaches ---")
    
    # Heuristic guidance (trace optimization)
    print("\n1. Heuristic Guidance (Trace Optimization):")
    heuristic_dir, heuristic_obj = GuidanceLaw.compute_optimal_direction(
        uvw, R, t, objective_type='trace'
    )
    print(f"   Direction: [{heuristic_dir[0]:6.3f}, {heuristic_dir[1]:6.3f}, {heuristic_dir[2]:6.3f}]")
    print(f"   Objective value: {heuristic_obj:.4e}")
    
    # Simulate moving in heuristic direction
    t_new_heuristic = t + heuristic_dir * 1.0
    obj_new_heuristic = FisherInformationAnalyzer.compute_condition_based_objective(
        uvw, R, t_new_heuristic, objective_type='trace'
    )
    print(f"   After moving 1 unit: {obj_new_heuristic:.4e}")
    print(f"   Improvement: {obj_new_heuristic - heuristic_obj:.4e}")
    
    # RL guidance (using untrained model as baseline)
    print("\n2. RL Guidance (Untrained - Random Initialization):")
    try:
        rl_dir, _ = RLGuidanceLaw.compute_optimal_direction(uvw, R, t)
        print(f"   Direction: [{rl_dir[0]:6.3f}, {rl_dir[1]:6.3f}, {rl_dir[2]:6.3f}]")
        
        # Simulate moving in RL direction
        t_new_rl = t + rl_dir * 1.0
        obj_new_rl = FisherInformationAnalyzer.compute_condition_based_objective(
            uvw, R, t_new_rl, objective_type='trace'
        )
        print(f"   After moving 1 unit: {obj_new_rl:.4e}")
        print(f"   Improvement: {obj_new_rl - heuristic_obj:.4e}")
        print(f"   (Note: Untrained RL model - would improve with training)")
    except Exception as e:
        print(f"   Could not compute RL guidance: {e}")
    
    # Note about training
    print("\n--- Training RL Model ---")
    print("To train the RL model, run:")
    print("  python train_rl_guidance.py --episodes 1000")
    print("\nAfter training, the RL agent can:")
    print("  ✓ Learn from diverse scenarios")
    print("  ✓ Discover strategies beyond hand-crafted heuristics")
    print("  ✓ Adapt to different mission objectives")
    print("  ✓ Balance multiple competing goals automatically")


def demo_rl_environment():
    """Demonstrate the RL training environment."""
    print_section("DEMO 2: RL Training Environment")
    
    # Create environment
    np.random.seed(42)
    uvw = np.random.randn(3, 8) * 10
    R = np.eye(3)
    target_pos = np.array([20.0, 0.0, 0.0])
    target_vel = np.array([2.0, 0.5, 0.0])
    
    print(f"\nEnvironment setup:")
    print(f"  Landmarks: {uvw.shape[1]} points")
    print(f"  Target position: {target_pos}")
    print(f"  Target velocity: {target_vel}")
    
    env = GuidanceEnvironment(
        landmarks=uvw,
        initial_rotation=R,
        target_position=target_pos,
        target_velocity=target_vel,
        agent_speed=12.0,
        max_steps=50
    )
    
    # Reset environment
    state = env.reset()
    print(f"\nState space dimension: {len(state)}")
    print(f"Action space dimension: {env.action_dim}")
    
    # Run a few random steps
    print("\n--- Sample Episode (Random Actions) ---")
    total_reward = 0.0
    
    for step in range(5):
        # Random action
        action = np.random.randn(3)
        
        # Step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}]")
        print(f"  Reward: {reward:.4f}")
        print(f"  Distance to target: {info['reward_components']}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.4f}")
    
    # Explain reward components
    print("\n--- Reward Components ---")
    print("The reward function combines:")
    print("  1. Observability improvement (FIM determinant increase)")
    print("  2. Pursuit progress (distance reduction to target)")
    print("  3. Efficiency penalty (penalize excessive maneuvering)")
    print("\nThis encourages the agent to learn policies that balance")
    print("all objectives rather than optimizing just one heuristic.")


def demo_training_quick():
    """Demonstrate quick training run."""
    print_section("DEMO 3: Quick RL Training Demonstration")
    
    print("\nTraining RL agent for 50 episodes (quick demo)...")
    print("For full training, use: python train_rl_guidance.py --episodes 1000")
    
    # Create agent
    agent = PPOAgent(
        state_dim=16,
        action_dim=3,
        hidden_dim=32,  # Smaller for demo
        learning_rate=1e-3  # Higher for faster learning in demo
    )
    
    # Training loop
    episode_rewards = []
    
    for episode in range(50):
        # Random scenario
        uvw = np.random.randn(3, 8) * 10
        R = np.eye(3)
        target_pos = np.random.randn(3) * 20
        target_vel = np.random.randn(3) * 2
        
        env = GuidanceEnvironment(
            landmarks=uvw,
            initial_rotation=R,
            target_position=target_pos,
            target_velocity=target_vel,
            agent_speed=10.0,
            max_steps=50
        )
        
        # Run episode
        state = env.reset()
        episode_reward = 0.0
        
        for _ in range(50):
            action, log_prob = agent.select_action(state, deterministic=False)
            value = agent.value.forward(state).item()
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update every 10 episodes
        if (episode + 1) % 10 == 0:
            update_stats = agent.update()
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
    
    print(f"\nTraining complete!")
    print(f"Initial 10 episodes avg reward: {np.mean(episode_rewards[:10]):.2f}")
    print(f"Final 10 episodes avg reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Improvement: {np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10]):.2f}")


def demo_rl_advantages():
    """Explain advantages of RL-based guidance."""
    print_section("DEMO 4: Advantages of RL-Based Guidance")
    
    print("\n📊 Comparison: RL vs Heuristic Guidance")
    print("\n" + "-"*80)
    print(f"{'Aspect':<30} {'Heuristic':<25} {'RL-Based':<25}")
    print("-"*80)
    
    comparisons = [
        ("Design effort", "Manual heuristic design", "Automatic from data"),
        ("Adaptability", "Fixed to one objective", "Learns multiple objectives"),
        ("Generalization", "Case-by-case tuning", "Generalizes from training"),
        ("Discovery", "Limited to known strategies", "Can discover novel strategies"),
        ("Multi-objective", "Manual weight tuning", "Learns optimal balance"),
        ("Scalability", "Complex for many criteria", "Scales with data"),
        ("Transfer learning", "Not applicable", "Can transfer to new tasks"),
        ("Data-driven", "No", "Yes"),
    ]
    
    for aspect, heuristic, rl in comparisons:
        print(f"{aspect:<30} {heuristic:<25} {rl:<25}")
    
    print("-"*80)
    
    print("\n🎯 Key Advantages of RL Approach:")
    print("\n1. **Data-Driven**: Learns from experience, not hand-crafted rules")
    print("   - Aligns with modern ML paradigm")
    print("   - Can leverage large-scale simulation data")
    print("   - Improves with more training")
    
    print("\n2. **Automatic Multi-Objective Optimization**:")
    print("   - Learns to balance pursuit, observability, efficiency")
    print("   - No manual weight tuning needed")
    print("   - Discovers optimal trade-offs automatically")
    
    print("\n3. **Generalization**:")
    print("   - Single trained model works across scenarios")
    print("   - No case-by-case heuristic design")
    print("   - Handles diverse landmark configurations")
    
    print("\n4. **Potential for Superior Performance**:")
    print("   - Can discover strategies beyond human-designed heuristics")
    print("   - Learns from success and failure")
    print("   - Optimizes for actual mission objectives, not proxy metrics")
    
    print("\n5. **Extensibility**:")
    print("   - Easy to add new objectives (just modify reward)")
    print("   - Can incorporate constraints naturally")
    print("   - Transfer learning to related tasks")
    
    print("\n💡 When to Use RL Guidance:")
    print("   ✓ When you have diverse training scenarios")
    print("   ✓ When objectives are complex or multi-faceted")
    print("   ✓ When you want data-driven solutions")
    print("   ✓ When heuristics are hard to design")
    print("   ✓ When you need generalization across conditions")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*80)
    print("# RL-Based Guidance for Bearing-Only Cooperative Localization")
    print("#"*80)
    print("\nThis demonstration shows how reinforcement learning can replace")
    print("hand-crafted heuristics for guidance law design, enabling:")
    print("  • Data-driven policy learning")
    print("  • Automatic multi-objective optimization")
    print("  • Generalization across scenarios")
    print("  • Discovery of novel strategies")
    
    # Run demonstrations
    demo_rl_vs_heuristic_guidance()
    demo_rl_environment()
    demo_training_quick()
    demo_rl_advantages()
    
    # Summary
    print_section("SUMMARY")
    print("\n✅ We have demonstrated:")
    print("  • RL environment for guidance learning")
    print("  • RL agent architecture (PPO)")
    print("  • Comparison with heuristic guidance")
    print("  • Training process and improvements")
    print("  • Key advantages of RL approach")
    
    print("\n📚 Next Steps:")
    print("  1. Train full model: python train_rl_guidance.py --episodes 1000")
    print("  2. Evaluate trained model on test scenarios")
    print("  3. Compare performance with heuristic baselines")
    print("  4. Fine-tune for specific mission requirements")
    
    print("\n🎓 This represents a significant advancement:")
    print("  Moving from hand-crafted heuristics → data-driven RL")
    print("  Aligning with modern machine learning paradigm")
    print("  Enabling automatic discovery of optimal guidance strategies")
    
    print("\n" + "#"*80 + "\n")


if __name__ == '__main__':
    main()
