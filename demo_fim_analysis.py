"""
Demonstration of Fisher Information Matrix (FIM) Analysis
for Bearing-Only Cooperative Localization

This script demonstrates:
1. The relationship between FIM and condition number
2. How to use FIM metrics to assess observability
3. How to derive heuristics from condition number for improving observability
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fisher_information_matrix import FisherInformationAnalyzer
from bearing_only_solver import bearing_linear_solver, load_simulation_data
import matplotlib.pyplot as plt
import seaborn as sns


def demo_fim_condition_relationship():
    """
    Demonstrate the mathematical relationship between FIM and condition number.
    
    Key Insight: Since FIM = J^T * J / sigma^2, we have:
    cond(FIM) ≈ [cond(J)]^2
    
    This means the condition number of FIM is approximately the square of
    the Jacobian's condition number.
    """
    print("\n" + "="*70)
    print("DEMO 1: Relationship between FIM and Condition Number")
    print("="*70)
    
    # Create a simple scenario with varying bearing diversity
    n_configs = 10
    results = []
    
    for i in range(n_configs):
        # Create points with varying angular spread
        angle_spread = (i + 1) * np.pi / 20  # From narrow to wide spread
        n_points = 8
        
        # Generate points uniformly distributed on a cone
        uvw = np.zeros((3, n_points))
        for j in range(n_points):
            theta = 2 * np.pi * j / n_points
            phi = angle_spread * (j % 2)  # Alternate between different elevations
            
            # Spherical to Cartesian
            uvw[0, j] = 10 * np.cos(theta) * np.cos(phi)
            uvw[1, j] = 10 * np.sin(theta) * np.cos(phi)
            uvw[2, j] = 10 * np.sin(phi)
        
        # Simple rotation and translation
        R = np.eye(3)
        t = np.array([0.0, 0.0, 0.0])
        
        # Analyze observability
        analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
        
        results.append({
            'angle_spread': angle_spread * 180 / np.pi,
            'jacobian_cond': analysis['Jacobian_condition_number'],
            'fim_cond': analysis['FIM_metrics']['condition_number'],
            'fim_det': analysis['FIM_metrics']['determinant'],
            'fim_trace': analysis['FIM_metrics']['trace']
        })
    
    # Print results
    print("\nConfiguration Analysis:")
    print(f"{'Spread (deg)':<15} {'J Cond':<15} {'FIM Cond':<15} {'[J Cond]^2':<15} {'Ratio':<10}")
    print("-" * 75)
    
    for r in results:
        j_cond = r['jacobian_cond']
        fim_cond = r['fim_cond']
        j_squared = j_cond ** 2
        ratio = fim_cond / j_squared if j_squared > 0 else 0
        
        print(f"{r['angle_spread']:<15.2f} {j_cond:<15.2e} {fim_cond:<15.2e} {j_squared:<15.2e} {ratio:<10.2f}")
    
    print("\nKEY FINDING:")
    print("  cond(FIM) ≈ [cond(J)]^2")
    print("  This relationship holds because FIM = J^T * J / sigma^2")
    print("  Therefore, improving J's condition number also improves FIM's condition number")
    
    return results


def demo_observability_scenarios():
    """
    Demonstrate different bearing configurations and their observability.
    """
    print("\n" + "="*70)
    print("DEMO 2: Observability Analysis for Different Scenarios")
    print("="*70)
    
    scenarios = []
    
    # Scenario 1: Well-distributed bearings (GOOD)
    print("\n--- Scenario 1: Well-Distributed Bearings ---")
    uvw1 = np.array([
        [10, 10, 10, -10, -10, -10, 0, 0],
        [10, -10, 0, 10, -10, 0, 10, -10],
        [5, 5, -5, -5, 5, -5, 10, 10]
    ], dtype=float)
    
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    
    analysis1 = FisherInformationAnalyzer.analyze_observability(uvw1, R, t)
    print(analysis1['analysis'])
    scenarios.append(('Well-Distributed', analysis1))
    
    # Scenario 2: Nearly parallel bearings (BAD)
    print("\n--- Scenario 2: Nearly Parallel Bearings ---")
    uvw2 = np.zeros((3, 8))
    base_direction = np.array([1, 0, 0])
    for i in range(8):
        perturbation = np.random.randn(3) * 0.1
        direction = base_direction + perturbation
        uvw2[:, i] = 10 * direction / np.linalg.norm(direction)
    
    analysis2 = FisherInformationAnalyzer.analyze_observability(uvw2, R, t)
    print(analysis2['analysis'])
    scenarios.append(('Nearly Parallel', analysis2))
    
    # Scenario 3: Planar bearings (POOR)
    print("\n--- Scenario 3: Planar Bearings (all in XY plane) ---")
    uvw3 = np.zeros((3, 8))
    for i in range(8):
        angle = 2 * np.pi * i / 8
        uvw3[0, i] = 10 * np.cos(angle)
        uvw3[1, i] = 10 * np.sin(angle)
        uvw3[2, i] = 0  # All in plane
    
    analysis3 = FisherInformationAnalyzer.analyze_observability(uvw3, R, t)
    print(analysis3['analysis'])
    scenarios.append(('Planar', analysis3))
    
    # Compare scenarios
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    print(f"{'Scenario':<20} {'FIM Cond':<15} {'J Cond':<15} {'FIM Det':<15}")
    print("-" * 70)
    
    for name, analysis in scenarios:
        fim_cond = analysis['FIM_metrics']['condition_number']
        j_cond = analysis['Jacobian_condition_number']
        fim_det = analysis['FIM_metrics']['determinant']
        print(f"{name:<20} {fim_cond:<15.2e} {j_cond:<15.2e} {fim_det:<15.2e}")
    
    return scenarios


def demo_improvement_heuristics():
    """
    Demonstrate heuristics for improving observability using condition number.
    """
    print("\n" + "="*70)
    print("DEMO 3: Condition Number-Based Improvement Heuristics")
    print("="*70)
    
    # Start with poor configuration
    print("\n--- Initial Configuration (Poor Observability) ---")
    uvw_poor = np.zeros((3, 6))
    for i in range(6):
        # Concentrated in one direction
        angle = np.pi / 8 * i
        uvw_poor[0, i] = 10 * np.cos(angle)
        uvw_poor[1, i] = 10 * np.sin(angle)
        uvw_poor[2, i] = 0.5
    
    R = np.eye(3)
    t = np.array([1.0, 1.0, 1.0])
    
    analysis_poor = FisherInformationAnalyzer.analyze_observability(uvw_poor, R, t)
    print(f"Initial FIM Condition Number: {analysis_poor['FIM_metrics']['condition_number']:.2e}")
    print(f"Initial Determinant: {analysis_poor['FIM_metrics']['determinant']:.2e}")
    
    # Get suggestions
    suggestions = FisherInformationAnalyzer.suggest_improvements(uvw_poor, R, t)
    print("\nSuggestions for Improvement:")
    print(f"  Worst observable direction (pose space): {suggestions['worst_observable_direction']}")
    print(f"  {suggestions['rotation_suggestion']}")
    print(f"  {suggestions['translation_suggestion']}")
    
    # Apply heuristic: Add points in different directions
    print("\n--- Improved Configuration (Added Diverse Bearings) ---")
    uvw_improved = np.hstack([
        uvw_poor,
        np.array([
            [-10, -10, 0],
            [0, 0, 10],
            [10, -10, 5]
        ]).T
    ])
    
    analysis_improved = FisherInformationAnalyzer.analyze_observability(uvw_improved, R, t)
    print(f"Improved FIM Condition Number: {analysis_improved['FIM_metrics']['condition_number']:.2e}")
    print(f"Improved Determinant: {analysis_improved['FIM_metrics']['determinant']:.2e}")
    
    # Calculate improvement
    cond_improvement = analysis_poor['FIM_metrics']['condition_number'] / analysis_improved['FIM_metrics']['condition_number']
    det_improvement = analysis_improved['FIM_metrics']['determinant'] / analysis_poor['FIM_metrics']['determinant']
    
    print(f"\nImprovement Factor:")
    print(f"  Condition Number improved by: {cond_improvement:.2f}x (lower is better)")
    print(f"  Determinant improved by: {det_improvement:.2e}x (higher is better)")
    
    return analysis_poor, analysis_improved


def demo_guidance_strategies():
    """
    Demonstrate different guidance strategies based on FIM optimization.
    """
    print("\n" + "="*70)
    print("DEMO 4: Guidance Strategies for Observability Improvement")
    print("="*70)
    
    # Fixed bearings
    uvw = np.random.randn(3, 8) * 10
    R = np.eye(3)
    
    # Test different translation positions
    positions = []
    objectives = {
        'trace': [],
        'determinant': [],
        'min_eigenvalue': [],
        'inverse_condition': []
    }
    
    print("\nTesting different agent positions...")
    for i in range(10):
        t = np.array([i, 0, 0])  # Move along x-axis
        positions.append(t[0])
        
        for obj_type in objectives.keys():
            obj_val = FisherInformationAnalyzer.compute_condition_based_objective(
                uvw, R, t, objective_type=obj_type
            )
            objectives[obj_type].append(obj_val)
    
    # Print results
    print("\nObjective Values at Different Positions:")
    print(f"{'Position':<12} {'Trace':<15} {'Log(Det)':<15} {'Min Eig':<15} {'1/Cond':<15}")
    print("-" * 75)
    
    for i in range(len(positions)):
        print(f"{positions[i]:<12.1f} {objectives['trace'][i]:<15.2e} "
              f"{objectives['determinant'][i]:<15.2e} {objectives['min_eigenvalue'][i]:<15.2e} "
              f"{objectives['inverse_condition'][i]:<15.2e}")
    
    # Find best positions for each objective
    print("\nBest Positions for Each Objective:")
    for obj_type, values in objectives.items():
        best_idx = np.argmax(values)
        print(f"  {obj_type}: Position {positions[best_idx]:.1f} (value: {values[best_idx]:.2e})")
    
    print("\nKEY INSIGHT:")
    print("  Different objectives (trace, determinant, min eigenvalue) can guide")
    print("  the agent to maximize observability. Trace maximization is commonly")
    print("  used (A-optimality), while determinant maximization (D-optimality)")
    print("  and condition number minimization provide alternative strategies.")
    
    return positions, objectives


def demo_real_data_analysis():
    """
    Analyze observability on real simulation data from the repository.
    """
    print("\n" + "="*70)
    print("DEMO 5: FIM Analysis on Real Simulation Data")
    print("="*70)
    
    # Try to load simulation data
    gpath = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(gpath, 'taes/')
    
    if not os.path.exists(folder):
        print("Simulation data folder not found. Skipping real data analysis.")
        return None
    
    files = [os.path.join(folder, f) for f in os.listdir(folder) if 'simu_' in f]
    
    if len(files) == 0:
        print("No simulation files found. Skipping real data analysis.")
        return None
    
    # Analyze first simulation file
    data = load_simulation_data(files[0])
    uvw = data["p1"]
    xyz = data["p2"]
    bearing = data["bearing"]
    Rgt = data["Rgt"]
    tgt = data["tgt"]
    
    print(f"\nLoaded simulation data from: {os.path.basename(files[0])}")
    print(f"Number of points: {uvw.shape[1]}")
    
    # Analyze with ground truth pose
    print("\n--- Analysis with Ground Truth Pose ---")
    analysis_gt = FisherInformationAnalyzer.analyze_observability(uvw, Rgt, tgt)
    print(analysis_gt['analysis'])
    
    # Analyze with identity pose (before solving)
    print("\n--- Analysis with Initial Guess (Identity) ---")
    R_init = np.eye(3)
    t_init = np.zeros(3)
    analysis_init = FisherInformationAnalyzer.analyze_observability(uvw, R_init, t_init)
    print(analysis_init['analysis'])
    
    # Solve and analyze with estimated pose
    print("\n--- Solving for Pose ---")
    (R_est, t_est), solve_time = bearing_linear_solver.solve(uvw, xyz, bearing)
    
    print(f"Solution time: {solve_time:.4f} seconds")
    print("\n--- Analysis with Estimated Pose ---")
    analysis_est = FisherInformationAnalyzer.analyze_observability(uvw, R_est, t_est)
    print(analysis_est['analysis'])
    
    # Compare Jacobian condition with A matrix condition
    bearing_angle = np.zeros((2, bearing.shape[1]))
    for i in range(bearing.shape[1]):
        vec = bearing[:, i]
        phi = np.arcsin(vec[2])
        theta = np.arctan2(vec[1], vec[0])
        bearing_angle[:, i] = np.array([theta, phi])
    
    A = bearing_linear_solver.compute_A_matrix(
        uvw[0,:], uvw[1,:], uvw[2,:],
        bearing_angle[1,:], bearing_angle[0,:],
        bearing_angle.shape[1]
    )
    A_cond = bearing_linear_solver.compute_condition_number(A)
    
    print(f"\nComparison with Linear Solver's A matrix:")
    print(f"  A matrix condition number: {A_cond:.2e}")
    print(f"  Jacobian condition number: {analysis_est['Jacobian_condition_number']:.2e}")
    print(f"  FIM condition number: {analysis_est['FIM_metrics']['condition_number']:.2e}")
    
    return analysis_gt, analysis_est


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "="*70)
    print(" Fisher Information Matrix (FIM) and Condition Number Analysis")
    print(" for Bearing-Only Cooperative Localization")
    print("="*70)
    
    # Run all demos
    results1 = demo_fim_condition_relationship()
    scenarios = demo_observability_scenarios()
    analysis_poor, analysis_improved = demo_improvement_heuristics()
    positions, objectives = demo_guidance_strategies()
    real_analysis = demo_real_data_analysis()
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY: Key Findings")
    print("="*70)
    
    print("\n1. FIM-Condition Number Relationship:")
    print("   - cond(FIM) ≈ [cond(J)]^2 because FIM = J^T * J / sigma^2")
    print("   - This means improving Jacobian condition also improves FIM")
    print("   - Both metrics indicate the same underlying observability issues")
    
    print("\n2. Observability Assessment:")
    print("   - FIM condition number < 100: GOOD observability")
    print("   - FIM condition number > 1e6: POOR observability")
    print("   - Determinant of FIM: higher values indicate better observability")
    print("   - Minimum eigenvalue: indicates worst observable direction")
    
    print("\n3. Improvement Heuristics:")
    print("   - Maximize trace of FIM (A-optimality)")
    print("   - Maximize determinant of FIM (D-optimality)")
    print("   - Maximize minimum eigenvalue (E-optimality)")
    print("   - Minimize condition number (inverse as objective)")
    
    print("\n4. Practical Recommendations:")
    print("   - Avoid nearly parallel bearing vectors")
    print("   - Ensure 3D diversity (not just planar)")
    print("   - Use FIM analysis to guide agent trajectory")
    print("   - Monitor condition number during operation")
    
    print("\n" + "="*70)
    print(" Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        pass
    
    main()
