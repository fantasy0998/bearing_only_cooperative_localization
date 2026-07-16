"""
Practical Example: Using FIM Analysis for Trajectory Optimization

This example demonstrates how researchers can use Fisher Information Matrix
analysis to design guidance strategies that improve observability in 
bearing-only cooperative localization.

Scenario: An agent needs to move to maximize observability of its pose
relative to landmarks at known positions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fisher_information_matrix import FisherInformationAnalyzer


def example_1_basic_observability_check():
    """
    Example 1: Basic observability check before solving
    
    Use case: Verify that your bearing measurements provide good observability
    before attempting to solve for pose.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Observability Check")
    print("="*70)
    
    # Landmarks at known positions (3 x n)
    landmarks_global = np.array([
        [10, -10, 10, -10, 0, 0],
        [10, 10, -10, -10, 10, -10],
        [5, 5, 5, 5, 10, 10]
    ], dtype=float)
    
    # Initial pose guess
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    
    # Check observability
    analysis = FisherInformationAnalyzer.analyze_observability(
        landmarks_global, R, t, measurement_noise_std=0.01
    )
    
    print("\nObservability Report:")
    print(analysis['analysis'])
    
    # Decision logic
    cond_num = analysis['FIM_metrics']['condition_number']
    if cond_num < 1000:
        print("\n✓ PASS: Good observability - safe to proceed with solving")
    elif cond_num < 1e6:
        print("\n⚠ WARNING: Moderate observability - results may be less accurate")
    else:
        print("\n✗ FAIL: Poor observability - consider improving bearing diversity")
        
        # Get suggestions
        suggestions = FisherInformationAnalyzer.suggest_improvements(
            landmarks_global, R, t
        )
        print("\nSuggestions:")
        print(f"  {suggestions['translation_suggestion']}")


def example_2_trajectory_planning():
    """
    Example 2: Plan trajectory to maximize observability
    
    Use case: Given multiple potential positions, choose the one that
    provides best observability.
    """
    print("\n" + "="*70)
    print("Example 2: Trajectory Planning for Maximum Observability")
    print("="*70)
    
    # Fixed landmarks
    landmarks = np.random.randn(3, 8) * 10
    R = np.eye(3)
    
    # Candidate positions to evaluate
    candidate_positions = [
        np.array([0, 0, 0]),
        np.array([5, 0, 0]),
        np.array([0, 5, 0]),
        np.array([0, 0, 5]),
        np.array([5, 5, 5]),
    ]
    
    print("\nEvaluating candidate positions...")
    
    # Evaluate each position using different criteria
    results = []
    for i, pos in enumerate(candidate_positions):
        # Compute multiple objectives
        trace_obj = FisherInformationAnalyzer.compute_condition_based_objective(
            landmarks, R, pos, objective_type='trace'
        )
        det_obj = FisherInformationAnalyzer.compute_condition_based_objective(
            landmarks, R, pos, objective_type='determinant'
        )
        cond_obj = FisherInformationAnalyzer.compute_condition_based_objective(
            landmarks, R, pos, objective_type='inverse_condition'
        )
        
        results.append({
            'position': pos,
            'trace': trace_obj,
            'determinant': det_obj,
            'inverse_condition': cond_obj
        })
    
    # Print comparison
    print(f"\n{'Position':<20} {'Trace':<15} {'Log(Det)':<15} {'1/Cond':<15}")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"{str(r['position']):<20} {r['trace']:<15.2e} "
              f"{r['determinant']:<15.2e} {r['inverse_condition']:<15.2e}")
    
    # Choose best by trace (A-optimality - most common)
    best_idx = max(range(len(results)), key=lambda i: results[i]['trace'])
    print(f"\nRecommendation: Move to position {results[best_idx]['position']}")
    print(f"  (maximizes trace of FIM)")


def example_3_online_monitoring():
    """
    Example 3: Online observability monitoring during operation
    
    Use case: Monitor observability during operation and trigger warnings
    when it degrades.
    """
    print("\n" + "="*70)
    print("Example 3: Online Observability Monitoring")
    print("="*70)
    
    # Simulate agent trajectory
    landmarks = np.random.randn(3, 8) * 10
    R = np.eye(3)
    
    print("\nSimulating agent movement...")
    print(f"{'Time':<10} {'Position':<25} {'Cond(FIM)':<15} {'Status':<20}")
    print("-" * 75)
    
    for t in range(10):
        # Agent moves
        position = np.array([t, np.sin(t), np.cos(t)])
        
        # Check observability at this position
        analysis = FisherInformationAnalyzer.analyze_observability(
            landmarks, R, position, measurement_noise_std=0.01
        )
        
        cond_num = analysis['FIM_metrics']['condition_number']
        
        # Status based on condition number
        if cond_num < 1000:
            status = "✓ GOOD"
        elif cond_num < 1e6:
            status = "⚠ MODERATE"
        else:
            status = "✗ POOR"
        
        print(f"{t:<10} {str(position):<25} {cond_num:<15.2e} {status:<20}")
        
        # Trigger warning
        if cond_num > 1e5:
            print(f"  → WARNING at t={t}: Consider changing trajectory!")


def example_4_comparing_guidance_strategies():
    """
    Example 4: Compare different guidance strategies
    
    Use case: Understand which observability criterion works best for
    your application.
    """
    print("\n" + "="*70)
    print("Example 4: Comparing Guidance Strategies")
    print("="*70)
    
    # Fixed scenario
    landmarks = np.array([
        [10, -10, 5, -5, 0, 8],
        [0, 0, 10, -10, 8, -8],
        [5, 5, 5, 5, 10, 10]
    ], dtype=float)
    R = np.eye(3)
    
    # Test positions along a line
    positions = [np.array([x, 0, 0]) for x in range(-5, 6)]
    
    # Different strategies
    strategies = {
        'A-optimal (trace)': 'trace',
        'D-optimal (determinant)': 'determinant',
        'E-optimal (min eigenvalue)': 'min_eigenvalue',
        'Condition number': 'inverse_condition'
    }
    
    # Compute objectives
    strategy_results = {name: [] for name in strategies}
    
    for pos in positions:
        for name, obj_type in strategies.items():
            obj_val = FisherInformationAnalyzer.compute_condition_based_objective(
                landmarks, R, pos, objective_type=obj_type
            )
            strategy_results[name].append(obj_val)
    
    # Find best position for each strategy
    print("\nBest position for each strategy:")
    print(f"{'Strategy':<30} {'Best Position':<20} {'Objective Value':<20}")
    print("-" * 75)
    
    for name, values in strategy_results.items():
        best_idx = np.argmax(values)
        best_pos = positions[best_idx]
        best_val = values[best_idx]
        print(f"{name:<30} {str(best_pos):<20} {best_val:<20.2e}")
    
    print("\nInterpretation:")
    print("  - A-optimal: Maximizes average observability (most common)")
    print("  - D-optimal: Balances observability across all directions")
    print("  - E-optimal: Focuses on worst observable direction")
    print("  - Condition number: Emphasizes numerical stability")


def example_5_poor_to_good_observability():
    """
    Example 5: Transforming poor observability to good
    
    Use case: Given poor initial bearings, how to improve them.
    """
    print("\n" + "="*70)
    print("Example 5: Improving Poor Observability")
    print("="*70)
    
    # Start with poor bearing configuration (nearly parallel)
    print("\nInitial Configuration (nearly parallel bearings):")
    landmarks_poor = np.zeros((3, 6))
    for i in range(6):
        # All pointing roughly in same direction
        direction = np.array([1, 0, 0]) + np.random.randn(3) * 0.05
        landmarks_poor[:, i] = 10 * direction / np.linalg.norm(direction)
    
    R = np.eye(3)
    t = np.array([0, 0, 0])
    
    analysis_poor = FisherInformationAnalyzer.analyze_observability(
        landmarks_poor, R, t
    )
    
    print(f"Condition Number: {analysis_poor['FIM_metrics']['condition_number']:.2e}")
    print(f"Determinant: {analysis_poor['FIM_metrics']['determinant']:.2e}")
    
    # Get suggestions
    suggestions = FisherInformationAnalyzer.suggest_improvements(
        landmarks_poor, R, t
    )
    print(f"\nWorst observable direction: {suggestions['worst_observable_direction']}")
    
    # Improve by adding diverse bearings
    print("\nImproved Configuration (added diverse bearings):")
    landmarks_improved = np.hstack([
        landmarks_poor,
        np.array([
            [-10, 0, 10],
            [0, 10, -10],
            [10, 10, 10]
        ]).T
    ])
    
    analysis_improved = FisherInformationAnalyzer.analyze_observability(
        landmarks_improved, R, t
    )
    
    print(f"Condition Number: {analysis_improved['FIM_metrics']['condition_number']:.2e}")
    print(f"Determinant: {analysis_improved['FIM_metrics']['determinant']:.2e}")
    
    # Compute improvement
    cond_improvement = (analysis_poor['FIM_metrics']['condition_number'] / 
                       analysis_improved['FIM_metrics']['condition_number'])
    det_improvement = (analysis_improved['FIM_metrics']['determinant'] / 
                      analysis_poor['FIM_metrics']['determinant'])
    
    print(f"\nImprovement:")
    print(f"  Condition number reduced by: {cond_improvement:.1f}x")
    print(f"  Determinant increased by: {det_improvement:.2e}x")
    print("\n✓ Observability successfully improved!")


def main():
    """Run all practical examples"""
    print("\n" + "="*70)
    print(" Practical Examples: FIM Analysis for Trajectory Optimization")
    print("="*70)
    
    try:
        example_1_basic_observability_check()
        example_2_trajectory_planning()
        example_3_online_monitoring()
        example_4_comparing_guidance_strategies()
        example_5_poor_to_good_observability()
        
        print("\n" + "="*70)
        print(" All Examples Complete!")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. Always check observability before solving")
        print("2. Use FIM metrics to choose best trajectory")
        print("3. Monitor observability during operation")
        print("4. Different criteria suit different applications")
        print("5. Poor observability can be improved with diverse bearings")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
