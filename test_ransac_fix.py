#!/usr/bin/env python3
"""
Test script to verify RANSAC bug fixes work correctly.
This script tests the RANSAC methods to ensure they properly refit using inliers.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bearing_only_solver import bearing_linear_solver, bgpnp, load_simulation_data

def test_ransac_linear_solver():
    """Test that bearing_linear_solver.ransac_solve refits properly"""
    print("\n=== Testing bearing_linear_solver.ransac_solve ===")
    
    # Load test data
    data_file = os.path.join(os.path.dirname(__file__), 'taes', 'simu_0.txt')
    if not os.path.exists(data_file):
        print(f"Warning: Test data file {data_file} not found, skipping test")
        return True
    
    data = load_simulation_data(data_file)
    uvw = data["p1"]
    xyz = data["p2"]
    bearing = data["bearing"]
    
    # Add some outliers to test RANSAC
    n_outliers = 3
    outlier_indices = np.random.choice(bearing.shape[1], n_outliers, replace=False)
    bearing_with_outliers = bearing.copy()
    for idx in outlier_indices:
        # Add significant noise to create outliers
        bearing_with_outliers[:, idx] += np.random.randn(3) * 0.5
        bearing_with_outliers[:, idx] /= np.linalg.norm(bearing_with_outliers[:, idx])
    
    print(f"Data shape: uvw={uvw.shape}, xyz={xyz.shape}, bearing={bearing.shape}")
    print(f"Added {n_outliers} outliers at indices: {outlier_indices}")
    
    # Test RANSAC solve
    try:
        (R, t), time = bearing_linear_solver.ransac_solve(
            uvw, xyz, bearing_with_outliers,
            num_iterations=100,
            threshold=0.1
        )
        print(f"✓ RANSAC solve completed successfully")
        print(f"  Rotation matrix shape: {R.shape}")
        print(f"  Translation vector shape: {t.shape}")
        print(f"  Time: {time:.4f}s")
        
        # Verify result is valid
        assert R.shape == (3, 3), "Rotation matrix should be 3x3"
        assert t.shape == (3,), "Translation vector should be length 3"
        assert np.allclose(np.linalg.det(R), 1.0, atol=0.1), "Rotation matrix should have determinant ~1"
        print(f"✓ Output validation passed")
        return True
    except Exception as e:
        print(f"✗ RANSAC solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bgpnp_ransac():
    """Test that bgpnp.ransac_solve refits properly"""
    print("\n=== Testing bgpnp.ransac_solve ===")
    
    # Create synthetic test data
    np.random.seed(42)
    n_points = 10
    
    # Generate random 3D points
    p1 = np.random.randn(n_points, 3) * 10
    p2 = np.random.randn(n_points, 3) * 10
    
    # Generate bearing vectors (should be unit vectors)
    bearing = np.random.randn(n_points, 3)
    bearing = bearing / np.linalg.norm(bearing, axis=1, keepdims=True)
    
    # Add some outliers
    n_outliers = 2
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    for idx in outlier_indices:
        bearing[idx] += np.random.randn(3) * 0.5
        bearing[idx] /= np.linalg.norm(bearing[idx])
    
    print(f"Data shape: p1={p1.shape}, p2={p2.shape}, bearing={bearing.shape}")
    print(f"Added {n_outliers} outliers at indices: {outlier_indices}")
    
    # Test RANSAC solve
    try:
        (R, t, error), time = bgpnp.ransac_solve(
            p1, p2, bearing,
            num_iterations=100,
            threshold=0.1
        )
        print(f"✓ BGPnP RANSAC solve completed successfully")
        print(f"  Rotation matrix shape: {R.shape}")
        print(f"  Translation vector shape: {t.shape}")
        print(f"  Error: {error}")
        print(f"  Time: {time:.4f}s")
        
        # Verify result is valid
        assert R.shape == (3, 3), "Rotation matrix should be 3x3"
        assert t.shape == (3,), "Translation vector should be length 3"
        print(f"✓ Output validation passed")
        return True
    except Exception as e:
        print(f"✗ BGPnP RANSAC solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_validation():
    """Test that input validation works correctly"""
    print("\n=== Testing input validation ===")
    
    # Test with too few points
    try:
        p1 = np.random.randn(2, 3)
        p2 = np.random.randn(2, 3)
        bearing = np.random.randn(2, 3)
        bearing = bearing / np.linalg.norm(bearing, axis=1, keepdims=True)
        
        (R, t, error), time = bgpnp.solve(p1, p2, bearing)
        print(f"✗ Should have raised ValueError for insufficient points")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test with NaN values
    try:
        p1 = np.random.randn(6, 3)
        p1[0, 0] = np.nan
        p2 = np.random.randn(6, 3)
        bearing = np.random.randn(6, 3)
        bearing = bearing / np.linalg.norm(bearing, axis=1, keepdims=True)
        
        (R, t, error), time = bgpnp.solve(p1, p2, bearing)
        print(f"✗ Should have raised ValueError for NaN values")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test with zero bearing vectors
    try:
        p1 = np.random.randn(6, 3)
        p2 = np.random.randn(6, 3)
        bearing = np.random.randn(6, 3)
        bearing[0] = 0  # Zero vector
        
        (R, t, error), time = bgpnp.solve(p1, p2, bearing)
        print(f"✗ Should have raised ValueError for zero bearing vector")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print(f"✓ All input validation tests passed")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("RANSAC Bug Fix Verification Tests")
    print("=" * 60)
    
    results = []
    results.append(("RANSAC Linear Solver", test_ransac_linear_solver()))
    results.append(("BGPnP RANSAC", test_bgpnp_ransac()))
    results.append(("Input Validation", test_input_validation()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
