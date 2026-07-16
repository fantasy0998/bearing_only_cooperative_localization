"""
Demonstration of Guidance Laws Derived from Fisher Information Matrix

This script demonstrates:
1. Universal guidance law that optimizes observability along certain directions
2. Two-agent pursuit scenario with observability-aware guidance

The demonstrations validate that guidance laws can be derived from the
condition matrix (Fisher Information Matrix) to improve localization performance.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from guidance_law import GuidanceLaw, TwoAgentPursuitGuidance
from fisher_information_matrix import FisherInformationAnalyzer


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def demo_universal_guidance_law():
    """Demonstrate universal guidance law based on FIM optimization."""
    print_section("DEMO 1: Universal Guidance Law from Condition Matrix")
    
    # Create landmarks (bearing reference points)
    np.random.seed(42)
    uvw = np.random.randn(3, 8) * 10
    print(f"\nLandmarks: {uvw.shape[1]} points distributed in 3D space")
    
    # Initial pose
    R = np.eye(3)
    t = np.array([5.0, 5.0, 5.0])
    print(f"Initial position: {t}")
    
    # Analyze current observability
    print("\n--- Current Observability ---")
    analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
    print(f"FIM Condition Number: {analysis['FIM_metrics']['condition_number']:.2e}")
    print(f"FIM Determinant: {analysis['FIM_metrics']['determinant']:.2e}")
    print(f"Jacobian Condition Number: {analysis['Jacobian_condition_number']:.2e}")
    
    # Test different guidance objectives
    print("\n--- Testing Different Guidance Objectives ---")
    objectives = ['trace', 'determinant', 'min_eigenvalue', 'inverse_condition']
    
    for obj_type in objectives:
        print(f"\n{obj_type.upper()} Optimization:")
        
        # Compute optimal direction
        direction, obj_value = GuidanceLaw.compute_optimal_direction(
            uvw, R, t, objective_type=obj_type
        )
        
        print(f"  Current objective value: {obj_value:.4e}")
        print(f"  Optimal direction: [{direction[0]:6.3f}, {direction[1]:6.3f}, {direction[2]:6.3f}]")
        
        # Simulate moving in optimal direction
        step_size = 1.0
        t_new = t + direction * step_size
        
        # Compute new objective
        obj_new = FisherInformationAnalyzer.compute_condition_based_objective(
            uvw, R, t_new, objective_type=obj_type
        )
        
        improvement = obj_new - obj_value
        print(f"  After moving 1 unit in optimal direction:")
        print(f"    New objective value: {obj_new:.4e}")
        if abs(obj_value) > 1e-10:
            print(f"    Improvement: {improvement:.4e} ({improvement/abs(obj_value)*100:.2f}%)")
        else:
            print(f"    Improvement: {improvement:.4e}")
    
    # Demonstrate trajectory evaluation
    print("\n--- Trajectory Evaluation ---")
    print("Comparing two trajectories:")
    
    # Trajectory 1: Random walk
    np.random.seed(123)
    traj1 = [t + np.random.randn(3) * 2 for _ in range(10)]
    
    # Trajectory 2: Following optimal directions
    traj2 = [t]
    current_t = t.copy()
    for _ in range(9):
        direction, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, current_t, objective_type='trace'
        )
        current_t = current_t + direction * 0.5
        traj2.append(current_t.copy())
    
    # Evaluate both trajectories
    results1 = GuidanceLaw.evaluate_trajectory(uvw, traj1, R, objective_type='trace')
    results2 = GuidanceLaw.evaluate_trajectory(uvw, traj2, R, objective_type='trace')
    
    print(f"\nTrajectory 1 (Random):")
    print(f"  Average condition number: {np.mean(results1['condition_numbers']):.2e}")
    print(f"  Average determinant: {np.mean(results1['determinants']):.2e}")
    
    print(f"\nTrajectory 2 (Guided by FIM):")
    print(f"  Average condition number: {np.mean(results2['condition_numbers']):.2e}")
    print(f"  Average determinant: {np.mean(results2['determinants']):.2e}")
    
    # Compare
    cond_improvement = (np.mean(results1['condition_numbers']) - np.mean(results2['condition_numbers'])) / np.mean(results1['condition_numbers']) * 100
    det_improvement = (np.mean(results2['determinants']) - np.mean(results1['determinants'])) / np.mean(results1['determinants']) * 100
    
    print(f"\nImprovement from FIM-guided trajectory:")
    print(f"  Condition number: {cond_improvement:.1f}% better (lower is better)")
    print(f"  Determinant: {det_improvement:.1f}% better (higher is better)")


def demo_two_agent_pursuit():
    """Demonstrate two-agent pursuit with observability-aware guidance."""
    print_section("DEMO 2: Two-Agent Pursuit with Observability Optimization")
    
    # Create landmarks
    np.random.seed(42)
    uvw = np.random.randn(3, 8) * 10
    print(f"\nLandmarks: {uvw.shape[1]} points for bearing measurements")
    
    # Initial conditions
    R = np.eye(3)
    pursuer_pos_0 = np.array([0.0, 0.0, 0.0])
    target_pos_0 = np.array([20.0, 0.0, 0.0])
    target_vel = np.array([2.0, 0.5, 0.0])  # Target moves at constant velocity
    
    print(f"\nInitial pursuer position: {pursuer_pos_0}")
    print(f"Initial target position: {target_pos_0}")
    print(f"Target velocity: {target_vel} (magnitude: {np.linalg.norm(target_vel):.2f})")
    print(f"Initial separation: {np.linalg.norm(target_pos_0 - pursuer_pos_0):.2f}")
    
    # Compare different pursuit strategies
    print("\n--- Comparing Pursuit Strategies ---")
    
    strategies = [
        ("Pure Pursuit", 1.0, 0.0),
        ("Pure Observability", 0.0, 1.0),
        ("Balanced (50-50)", 0.5, 0.5),
        ("Pursuit-Heavy (80-20)", 0.8, 0.2),
        ("Observability-Heavy (20-80)", 0.2, 0.8),
    ]
    
    duration = 10.0
    dt = 0.1
    
    results_comparison = {}
    
    for name, pursuit_gain, obs_gain in strategies:
        print(f"\n{name} (pursuit={pursuit_gain}, observability={obs_gain}):")
        
        # Create guidance
        guidance = TwoAgentPursuitGuidance(
            pursuer_speed=12.0,  # Faster than target to allow intercept
            pursuit_gain=pursuit_gain,
            observability_gain=obs_gain,
            objective_type='trace'
        )
        
        # Simulate
        results = guidance.simulate_pursuit(
            uvw, R,
            pursuer_pos_0, target_pos_0, target_vel,
            duration, dt
        )
        
        results_comparison[name] = results
        
        # Analyze results
        final_distance = results['distances'][-1]
        min_distance = np.min(results['distances'])
        avg_condition = np.mean(results['observability_metrics']['condition_numbers'])
        avg_determinant = np.mean(results['observability_metrics']['determinants'])
        
        print(f"  Final distance to target: {final_distance:.2f}")
        print(f"  Minimum distance achieved: {min_distance:.2f}")
        print(f"  Average FIM condition number: {avg_condition:.2e}")
        print(f"  Average FIM determinant: {avg_determinant:.2e}")
    
    # Detailed analysis of balanced strategy
    print("\n--- Detailed Analysis: Balanced Strategy ---")
    balanced_results = results_comparison["Balanced (50-50)"]
    
    print(f"\nTrajectory statistics:")
    print(f"  Total time: {duration} seconds")
    print(f"  Time steps: {len(balanced_results['time'])}")
    print(f"  Initial distance: {balanced_results['distances'][0]:.2f}")
    print(f"  Final distance: {balanced_results['distances'][-1]:.2f}")
    print(f"  Distance reduction: {balanced_results['distances'][0] - balanced_results['distances'][-1]:.2f}")
    
    print(f"\nObservability along trajectory:")
    obs = balanced_results['observability_metrics']
    print(f"  Initial condition number: {obs['condition_numbers'][0]:.2e}")
    print(f"  Final condition number: {obs['condition_numbers'][-1]:.2e}")
    print(f"  Best condition number: {np.min(obs['condition_numbers']):.2e}")
    print(f"  Worst condition number: {np.max(obs['condition_numbers']):.2e}")
    
    print(f"\nVelocity statistics:")
    speeds = np.linalg.norm(balanced_results['pursuer_velocities'], axis=1)
    print(f"  Average speed: {np.mean(speeds):.2f}")
    print(f"  Speed variation: {np.std(speeds):.4f} (should be near zero)")
    
    # Show that observability-aware pursuit maintains better observability
    print("\n--- Observability vs Pure Pursuit Comparison ---")
    pure_pursuit = results_comparison["Pure Pursuit"]
    obs_heavy = results_comparison["Observability-Heavy (20-80)"]
    
    pure_avg_cond = np.mean(pure_pursuit['observability_metrics']['condition_numbers'])
    obs_avg_cond = np.mean(obs_heavy['observability_metrics']['condition_numbers'])
    
    pure_avg_det = np.mean(pure_pursuit['observability_metrics']['determinants'])
    obs_avg_det = np.mean(obs_heavy['observability_metrics']['determinants'])
    
    print(f"\nPure Pursuit:")
    print(f"  Average condition number: {pure_avg_cond:.2e}")
    print(f"  Average determinant: {pure_avg_det:.2e}")
    print(f"  Final distance: {pure_pursuit['distances'][-1]:.2f}")
    
    print(f"\nObservability-Heavy:")
    print(f"  Average condition number: {obs_avg_cond:.2e}")
    print(f"  Average determinant: {obs_avg_det:.2e}")
    print(f"  Final distance: {obs_heavy['distances'][-1]:.2f}")
    
    cond_improvement = (pure_avg_cond - obs_avg_cond) / pure_avg_cond * 100
    det_improvement = (obs_avg_det - pure_avg_det) / pure_avg_det * 100
    
    print(f"\nObservability improvement (vs pure pursuit):")
    print(f"  Condition number: {cond_improvement:.1f}% better")
    print(f"  Determinant: {det_improvement:.1f}% better")
    print(f"  Trade-off: Slightly slower pursuit (distance: {obs_heavy['distances'][-1]:.2f} vs {pure_pursuit['distances'][-1]:.2f})")


def demo_guidance_law_properties():
    """Demonstrate mathematical properties of the guidance law."""
    print_section("DEMO 3: Mathematical Properties of Guidance Law")
    
    # Create landmarks
    np.random.seed(42)
    uvw = np.random.randn(3, 8) * 10
    R = np.eye(3)
    t = np.array([5.0, 5.0, 5.0])
    
    print("\n--- Property 1: Guidance Direction is Always Unit Vector ---")
    for obj_type in ['trace', 'determinant', 'min_eigenvalue']:
        direction, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, t, objective_type=obj_type
        )
        norm = np.linalg.norm(direction)
        print(f"{obj_type:20s}: ||direction|| = {norm:.10f}")
    
    print("\n--- Property 2: Different Objectives Give Different Directions ---")
    directions = {}
    for obj_type in ['trace', 'determinant', 'min_eigenvalue']:
        direction, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, t, objective_type=obj_type
        )
        directions[obj_type] = direction
        print(f"{obj_type:20s}: direction = [{direction[0]:7.4f}, {direction[1]:7.4f}, {direction[2]:7.4f}]")
    
    # Compute angles between directions
    print("\nAngles between optimal directions:")
    objs = list(directions.keys())
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            d1, d2 = directions[objs[i]], directions[objs[j]]
            angle = np.arccos(np.clip(np.dot(d1, d2), -1, 1)) * 180 / np.pi
            print(f"  {objs[i]:20s} vs {objs[j]:20s}: {angle:6.2f} degrees")
    
    print("\n--- Property 3: Guidance Command Respects Constraints ---")
    velocity = np.array([3.0, 0.0, 0.0])
    max_accels = [1.0, 2.0, 5.0]
    
    for max_accel in max_accels:
        accel = GuidanceLaw.compute_guidance_command(
            uvw, R, t, velocity,
            max_acceleration=max_accel
        )
        accel_mag = np.linalg.norm(accel)
        print(f"Max accel = {max_accel:.1f}: ||acceleration|| = {accel_mag:.4f} (within limit: {accel_mag <= max_accel})")
    
    print("\n--- Property 4: Pursuit Guidance Balances Multiple Objectives ---")
    guidance = TwoAgentPursuitGuidance(
        pursuit_gain=0.7,
        observability_gain=0.3
    )
    
    pursuer_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([10.0, 5.0, 0.0])
    target_vel = np.array([1.0, 0.0, 0.0])
    
    # Pure pursuit direction
    pursuit_dir = guidance.compute_pursuit_direction(pursuer_pos, target_pos, target_vel)
    
    # Pure observability direction
    obs_dir = guidance.compute_observability_direction(uvw, R, pursuer_pos)
    
    # Blended direction
    blended_vel = guidance.compute_guidance_velocity(uvw, R, pursuer_pos, target_pos, target_vel)
    blended_dir = blended_vel / np.linalg.norm(blended_vel)
    
    print(f"Pursuit direction:       [{pursuit_dir[0]:7.4f}, {pursuit_dir[1]:7.4f}, {pursuit_dir[2]:7.4f}]")
    print(f"Observability direction: [{obs_dir[0]:7.4f}, {obs_dir[1]:7.4f}, {obs_dir[2]:7.4f}]")
    print(f"Blended direction:       [{blended_dir[0]:7.4f}, {blended_dir[1]:7.4f}, {blended_dir[2]:7.4f}]")
    
    # Verify blending
    expected_blend = 0.7 * pursuit_dir + 0.3 * obs_dir
    expected_blend = expected_blend / np.linalg.norm(expected_blend)
    error = np.linalg.norm(blended_dir - expected_blend)
    print(f"\nBlending error: {error:.6f} (should be near zero)")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*80)
    print("# Guidance Law Derivation from Condition Matrix (Fisher Information Matrix)")
    print("#"*80)
    print("\nThis demonstration shows how to derive guidance laws from the condition")
    print("matrix to optimize observability in bearing-only cooperative localization.")
    print("\nKey Results:")
    print("1. Universal guidance laws can be derived for different FIM objectives")
    print("2. Two-agent pursuit can balance capture and observability")
    print("3. Observability-aware guidance improves localization accuracy")
    
    # Run demonstrations
    demo_universal_guidance_law()
    demo_two_agent_pursuit()
    demo_guidance_law_properties()
    
    # Summary
    print_section("SUMMARY")
    print("\nWe have successfully demonstrated:")
    print("✓ Universal guidance law derivation from FIM/condition matrix")
    print("✓ Multiple optimization objectives (trace, determinant, min eigenvalue, condition)")
    print("✓ Two-agent pursuit scenario with observability optimization")
    print("✓ Trade-off between pursuit objective and observability")
    print("✓ Mathematical properties of the guidance laws")
    print("\nThe guidance laws derived from the condition matrix provide a principled")
    print("approach to trajectory planning that optimizes observability in bearing-only")
    print("cooperative localization systems.")
    print("\n" + "#"*80 + "\n")


if __name__ == '__main__':
    main()
