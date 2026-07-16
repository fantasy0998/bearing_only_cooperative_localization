"""
Quick Start Guide for Guidance Law Usage

This script provides minimal, copy-paste examples for using the guidance law
implementation. Perfect for getting started quickly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from guidance_law import GuidanceLaw, TwoAgentPursuitGuidance
from fisher_information_matrix import FisherInformationAnalyzer


def example_1_basic_optimal_direction():
    """Example 1: Find optimal direction to improve observability"""
    print("="*70)
    print("EXAMPLE 1: Find Optimal Direction for Observability")
    print("="*70)
    
    # Setup: landmarks and current position
    np.random.seed(42)
    landmarks = np.random.randn(3, 8) * 10  # 8 landmarks in 3D space
    current_rotation = np.eye(3)
    current_position = np.array([5.0, 5.0, 5.0])
    
    # Find optimal direction
    optimal_dir, current_obj = GuidanceLaw.compute_optimal_direction(
        landmarks, current_rotation, current_position,
        objective_type='trace'  # Maximize trace of FIM
    )
    
    print(f"\nCurrent position: {current_position}")
    print(f"Current observability (trace): {current_obj:.2e}")
    print(f"Optimal direction to move: {optimal_dir}")
    print(f"Direction magnitude: {np.linalg.norm(optimal_dir):.4f} (should be 1.0)")
    
    # Move in optimal direction
    new_position = current_position + optimal_dir * 1.0
    new_obj = FisherInformationAnalyzer.compute_condition_based_objective(
        landmarks, current_rotation, new_position, objective_type='trace'
    )
    
    print(f"\nAfter moving 1 unit in optimal direction:")
    print(f"New position: {new_position}")
    print(f"New observability: {new_obj:.2e}")
    print(f"Improvement: {(new_obj - current_obj)/current_obj * 100:.2f}%")


def example_2_real_time_guidance():
    """Example 2: Real-time guidance with dynamic constraints"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Real-Time Guidance Command with Acceleration Limits")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    landmarks = np.random.randn(3, 8) * 10
    current_rotation = np.eye(3)
    current_position = np.array([5.0, 5.0, 5.0])
    current_velocity = np.array([2.0, 0.0, 0.0])  # Moving in x direction
    
    print(f"\nCurrent position: {current_position}")
    print(f"Current velocity: {current_velocity} (speed: {np.linalg.norm(current_velocity):.2f})")
    
    # Compute guidance command
    acceleration = GuidanceLaw.compute_guidance_command(
        landmarks, current_rotation, current_position, current_velocity,
        objective_type='trace',
        max_acceleration=1.5,  # Maximum 1.5 m/s^2
        dt=0.1  # 10 Hz update rate
    )
    
    print(f"Acceleration command: {acceleration}")
    print(f"Acceleration magnitude: {np.linalg.norm(acceleration):.4f} m/s^2")
    
    # Simulate one time step
    dt = 0.1
    new_velocity = current_velocity + acceleration * dt
    new_position = current_position + new_velocity * dt
    
    print(f"\nAfter {dt} seconds:")
    print(f"New velocity: {new_velocity}")
    print(f"New position: {new_position}")


def example_3_simple_pursuit():
    """Example 3: Simple two-agent pursuit"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Two-Agent Pursuit (Pursuer Chases Target)")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    landmarks = np.random.randn(3, 8) * 10
    current_rotation = np.eye(3)
    
    # Initial positions
    pursuer_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([15.0, 0.0, 0.0])
    target_vel = np.array([1.5, 0.0, 0.0])  # Target moving away
    
    print(f"\nPursuer position: {pursuer_pos}")
    print(f"Target position: {target_pos}")
    print(f"Target velocity: {target_vel}")
    print(f"Initial separation: {np.linalg.norm(target_pos - pursuer_pos):.2f} meters")
    
    # Create balanced pursuit guidance (50% pursuit, 50% observability)
    guidance = TwoAgentPursuitGuidance(
        pursuer_speed=12.0,      # Pursuer is faster than target
        pursuit_gain=0.5,        # 50% weight on catching target
        observability_gain=0.5,  # 50% weight on good measurements
        objective_type='trace'
    )
    
    # Get velocity command
    velocity_cmd = guidance.compute_guidance_velocity(
        landmarks, current_rotation, pursuer_pos, target_pos, target_vel
    )
    
    print(f"\nGuidance velocity command: {velocity_cmd}")
    print(f"Command speed: {np.linalg.norm(velocity_cmd):.2f} m/s (should be 12.0)")
    
    # Compare with pure pursuit
    pure_pursuit_guidance = TwoAgentPursuitGuidance(
        pursuer_speed=12.0,
        pursuit_gain=1.0,        # 100% pursuit
        observability_gain=0.0   # 0% observability
    )
    
    pure_pursuit_vel = pure_pursuit_guidance.compute_guidance_velocity(
        landmarks, current_rotation, pursuer_pos, target_pos, target_vel
    )
    
    print(f"\nPure pursuit velocity: {pure_pursuit_vel}")
    
    # Angle between the two strategies
    angle = np.arccos(np.clip(
        np.dot(velocity_cmd, pure_pursuit_vel) / 
        (np.linalg.norm(velocity_cmd) * np.linalg.norm(pure_pursuit_vel)),
        -1, 1
    )) * 180 / np.pi
    
    print(f"\nAngle between balanced and pure pursuit: {angle:.2f} degrees")
    print("(Balanced strategy steers slightly away from direct pursuit to improve observability)")


def example_4_full_pursuit_simulation():
    """Example 4: Full pursuit simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Pursuit Simulation (10 seconds)")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    landmarks = np.random.randn(3, 8) * 10
    current_rotation = np.eye(3)
    
    pursuer_pos_0 = np.array([0.0, 0.0, 0.0])
    target_pos_0 = np.array([20.0, 0.0, 0.0])
    target_vel = np.array([2.0, 0.5, 0.0])
    
    # Create guidance
    guidance = TwoAgentPursuitGuidance(
        pursuer_speed=12.0,
        pursuit_gain=0.7,        # Prioritize pursuit slightly
        observability_gain=0.3,
        objective_type='trace'
    )
    
    # Simulate
    print("\nSimulating pursuit for 10 seconds...")
    results = guidance.simulate_pursuit(
        landmarks, current_rotation,
        pursuer_pos_0, target_pos_0, target_vel,
        duration=10.0,
        dt=0.1
    )
    
    # Report results
    print(f"\nResults:")
    print(f"  Initial distance: {results['distances'][0]:.2f} m")
    print(f"  Final distance: {results['distances'][-1]:.2f} m")
    print(f"  Minimum distance: {np.min(results['distances']):.2f} m")
    print(f"  Distance reduction: {results['distances'][0] - results['distances'][-1]:.2f} m")
    
    obs = results['observability_metrics']
    print(f"\nObservability metrics:")
    print(f"  Average condition number: {np.mean(obs['condition_numbers']):.2e}")
    print(f"  Best condition number: {np.min(obs['condition_numbers']):.2e}")
    print(f"  Average determinant: {np.mean(obs['determinants']):.2e}")
    
    # Sample trajectory points
    print(f"\nTrajectory samples (every 2 seconds):")
    for i in range(0, len(results['time']), 20):
        t = results['time'][i]
        p_pos = results['pursuer_trajectory'][i]
        t_pos = results['target_trajectory'][i]
        dist = results['distances'][i]
        print(f"  t={t:.1f}s: Pursuer at {p_pos}, Target at {t_pos}, Distance={dist:.2f}")


def example_5_choosing_objective():
    """Example 5: Comparing different objectives"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing Different FIM Objectives")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    landmarks = np.random.randn(3, 8) * 10
    current_rotation = np.eye(3)
    current_position = np.array([5.0, 5.0, 5.0])
    
    objectives = ['trace', 'determinant', 'min_eigenvalue', 'inverse_condition']
    
    print("\nComparing guidance directions for different objectives:")
    print("(All point to different directions to improve different aspects of observability)\n")
    
    for obj_type in objectives:
        direction, obj_value = GuidanceLaw.compute_optimal_direction(
            landmarks, current_rotation, current_position,
            objective_type=obj_type
        )
        
        print(f"{obj_type:20s}: direction = [{direction[0]:7.4f}, {direction[1]:7.4f}, {direction[2]:7.4f}]")
        print(f"{'':20s}  objective = {obj_value:.4e}")
    
    print("\nRecommendations:")
    print("  - 'trace': Best for general use (A-optimality)")
    print("  - 'determinant': Best for balanced observability (D-optimality)")
    print("  - 'min_eigenvalue': Best to avoid blind spots (E-optimality)")
    print("  - 'inverse_condition': Best for numerical stability")


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# GUIDANCE LAW QUICK START EXAMPLES")
    print("#"*70)
    print("\nThese examples show how to use the guidance law implementation.")
    print("Copy and adapt these examples for your own applications.\n")
    
    example_1_basic_optimal_direction()
    example_2_real_time_guidance()
    example_3_simple_pursuit()
    example_4_full_pursuit_simulation()
    example_5_choosing_objective()
    
    print("\n" + "#"*70)
    print("# All examples completed successfully!")
    print("#"*70)
    print("\nNext steps:")
    print("  1. Try modifying the parameters in these examples")
    print("  2. Run 'python demo_guidance_law.py' for more detailed analysis")
    print("  3. Read GUIDANCE_LAW.md for complete documentation")
    print("  4. Run 'python test_guidance_law.py' to see all tests")
    print()
