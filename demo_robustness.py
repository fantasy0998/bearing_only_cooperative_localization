"""
Demonstration of the robustness improvement for ill-conditioned bearing distributions.

This script compares the performance of the linear solver before and after the
regularization improvements when dealing with poorly distributed bearing vectors.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bearing_only_solver import bearing_linear_solver


def generate_ill_conditioned_scenario(n_points=6, parallel_factor=0.01):
    """
    Generate a test scenario with nearly parallel bearing vectors (ill-conditioned).
    
    Args:
        n_points: Number of bearing measurements
        parallel_factor: Controls how parallel the bearings are (smaller = more parallel)
    
    Returns:
        uvw, xyz, bearing, true_R, true_t
    """
    np.random.seed(42)
    
    # Create ground truth rotation and translation
    # Random rotation matrix
    angles = np.random.rand(3) * 0.3  # Small angles for realistic scenario
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    true_R = Rz @ Ry @ Rx
    
    true_t = np.array([10.0, 5.0, 2.0])
    
    # Generate points in global frame that are far away in similar direction
    # This creates nearly parallel bearing vectors
    base_point = np.array([100, 0, 0])  # Far point in one direction
    uvw = np.zeros((3, n_points))
    for i in range(n_points):
        perturbation = np.random.randn(3) * parallel_factor * 100
        uvw[:, i] = base_point + perturbation
    
    # Transform to local frame
    xyz = true_R.T @ (uvw - true_t[:, None])
    
    # Compute true bearing vectors
    bearing = np.zeros((3, n_points))
    for i in range(n_points):
        vec = true_R @ uvw[:, i] + true_t - xyz[:, i]
        bearing[:, i] = vec / np.linalg.norm(vec)
    
    return uvw, xyz, bearing, true_R, true_t


def generate_well_conditioned_scenario(n_points=6):
    """
    Generate a test scenario with well-distributed bearing vectors (well-conditioned).
    """
    np.random.seed(42)
    
    # Create ground truth rotation and translation
    angles = np.random.rand(3) * 0.3  # Small angles
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    true_R = Rz @ Ry @ Rx
    
    true_t = np.array([10.0, 5.0, 2.0])
    
    # Generate points in global frame in diverse directions
    uvw = np.random.randn(3, n_points) * 50 + 50  # Points spread in space
    
    # Transform to local frame
    xyz = true_R.T @ (uvw - true_t[:, None])
    
    # Compute true bearing vectors
    bearing = np.zeros((3, n_points))
    for i in range(n_points):
        vec = true_R @ uvw[:, i] + true_t - xyz[:, i]
        bearing[:, i] = vec / np.linalg.norm(vec)
    
    return uvw, xyz, bearing, true_R, true_t


def compute_rotation_error(R_est, R_true):
    """Compute rotation error in degrees"""
    R_error = R_est @ R_true.T
    trace = np.trace(R_error)
    # Clamp to valid range for acos
    trace = np.clip((trace - 1) / 2, -1, 1)
    angle_error = np.arccos(trace) * 180 / np.pi
    return angle_error


def compute_translation_error(t_est, t_true):
    """Compute translation error"""
    return np.linalg.norm(t_est - t_true)


def main():
    print("=" * 80)
    print("Demonstrating Robustness Improvement for Ill-Conditioned Scenarios")
    print("=" * 80)
    print()
    
    # Test with actual simulation data
    print("Test 1: Simulation Data from Repository")
    print("-" * 80)
    
    gpath = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(gpath, 'taes/')
    
    if os.path.exists(folder):
        from bearing_only_solver import load_simulation_data
        
        files = [os.path.join(folder, f) for f in os.listdir(folder) if 'simu_' in f]
        
        if len(files) > 0:
            # Test on first simulation file
            data = load_simulation_data(files[0])
            uvw = data["p1"]
            xyz = data["p2"]
            bearing = data["bearing"]
            
            # Compute condition number
            from math import asin, atan2
            bearing_angle = np.zeros((2, bearing.shape[1]))
            for i in range(bearing.shape[1]):
                vec = bearing[:, i]
                phi = asin(vec[2])
                theta = atan2(vec[1], vec[0])
                bearing_angle[:, i] = np.array([theta, phi])
            
            A = bearing_linear_solver.compute_A_matrix(
                uvw[0,:], uvw[1,:], uvw[2,:], 
                bearing_angle[1,:], bearing_angle[0,:], 
                bearing.shape[1]
            )
            cond_num = bearing_linear_solver.compute_condition_number(A)
            
            print(f"Number of bearing measurements: {bearing.shape[1]}")
            print(f"Condition number: {cond_num:.2e}")
            
            if cond_num > 1e10:
                print("Status: Ill-conditioned (regularization will be applied)")
            else:
                print("Status: Well-conditioned")
            
            (R_est, t_est), solve_time = bearing_linear_solver.solve(uvw, xyz, bearing)
            
            # Compute errors
            true_R = data["Rgt"]
            true_t = data["tgt"]
            
            rot_error = compute_rotation_error(R_est, true_R)
            trans_error = compute_translation_error(t_est, true_t)
            
            print(f"Rotation error: {rot_error:.6f} degrees")
            print(f"Translation error: {trans_error:.6f}")
            print(f"Solve time: {solve_time:.6f} seconds")
            print()
    
    # Test 2: Artificially created ill-conditioned scenario
    print("Test 2: Artificially Ill-Conditioned Scenario")
    print("-" * 80)
    print("Creating a test matrix with very high condition number...")
    
    # Create a simple ill-conditioned linear system to demonstrate regularization
    n = 12  # Size of the system (same as A matrix in bearing solver)
    m = 16  # Number of measurements
    
    # Create a matrix with controlled condition number
    np.random.seed(123)
    U, _ = np.linalg.qr(np.random.randn(m, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create singular values with large spread (ill-conditioned)
    S = np.logspace(0, -15, n)  # Singular values from 1 to 1e-15
    
    A_ill = U @ np.diag(S) @ V.T
    cond_ill = bearing_linear_solver.compute_condition_number(A_ill)
    
    print(f"Condition number: {cond_ill:.2e}")
    print(f"Status: {'Ill-conditioned' if cond_ill > 1e10 else 'Well-conditioned'}")
    
    # Create a random right-hand side
    x_true = np.random.randn(n)
    b = A_ill @ x_true
    
    # Solve without explicit regularization (uses lstsq)
    from scipy.linalg import lstsq
    x_lstsq = lstsq(A_ill, b)[0]
    error_lstsq = np.linalg.norm(x_lstsq - x_true)
    
    # Solve with regularization
    x_reg = bearing_linear_solver.solve_with_regularization(A_ill, b)
    error_reg = np.linalg.norm(x_reg - x_true)
    
    print(f"\nResults:")
    print(f"  Standard lstsq solution error: {error_lstsq:.6e}")
    print(f"  Regularized solution error: {error_reg:.6e}")
    print(f"  Improvement: {error_lstsq / error_reg:.2f}x" if error_reg < error_lstsq else "  (lstsq was better)")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("-" * 80)
    print("The linear solver now automatically detects ill-conditioned matrices")
    print("(condition number > 1e10) and applies Tikhonov regularization to")
    print("maintain numerical stability. This improves robustness when bearing")
    print("vectors are poorly distributed (e.g., nearly parallel directions).")
    print("=" * 80)


if __name__ == '__main__':
    main()
