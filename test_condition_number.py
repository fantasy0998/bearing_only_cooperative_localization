"""
Test suite for verifying the robustness improvements to handle ill-conditioned matrices
in the bearing-only linear solver.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bearing_only_solver import bearing_linear_solver, load_simulation_data


class TestConditionNumberRobustness(unittest.TestCase):
    """Test the condition number checking and regularization functionality"""
    
    def test_condition_number_computation(self):
        """Test that condition number is computed correctly"""
        # Create a well-conditioned matrix
        A_well = np.eye(5)
        cond_well = bearing_linear_solver.compute_condition_number(A_well)
        self.assertAlmostEqual(cond_well, 1.0, places=10)
        
        # Create an ill-conditioned matrix
        A_ill = np.array([[1, 1], [1, 1.0001]])
        cond_ill = bearing_linear_solver.compute_condition_number(A_ill)
        self.assertGreater(cond_ill, 1e4)  # Should be ill-conditioned
        
    def test_regularization_solver(self):
        """Test that regularization solver produces valid solutions"""
        # Create a simple overdetermined system
        np.random.seed(42)
        A = np.random.randn(10, 5)
        x_true = np.random.randn(5)
        b = A @ x_true
        
        # Solve with regularization
        x_reg = bearing_linear_solver.solve_with_regularization(A, b, regularization=0.01)
        
        # Solution should be close to true solution for well-conditioned A
        self.assertTrue(np.allclose(x_reg, x_true, atol=0.1))
        
    def test_regularization_with_ill_conditioned_matrix(self):
        """Test regularization improves solution for ill-conditioned matrices"""
        # Create an ill-conditioned matrix
        A = np.array([[1, 1, 0],
                      [1, 1.0001, 0],
                      [0, 0, 1],
                      [2, 2, 0]])
        b = np.array([2, 2.0001, 1, 4])
        
        # Check it's ill-conditioned
        cond = bearing_linear_solver.compute_condition_number(A)
        self.assertGreater(cond, 1e4)
        
        # Solve with regularization (should not raise an error)
        x_reg = bearing_linear_solver.solve_with_regularization(A, b)
        
        # Verify the solution exists and has correct dimension
        self.assertEqual(x_reg.shape, (3,))
        
        # Verify residual is reasonable
        residual = np.linalg.norm(A @ x_reg - b)
        self.assertLess(residual, 10.0)  # Relaxed tolerance for ill-conditioned system
        
    def test_solve_with_nearly_parallel_bearings(self):
        """Test solve method with nearly parallel bearing vectors (ill-conditioned)"""
        # Create a scenario with nearly parallel bearing vectors
        # This should create an ill-conditioned A matrix
        np.random.seed(123)
        
        # Create 6 points with nearly parallel bearings
        n_points = 6
        uvw = np.random.randn(3, n_points)
        xyz = np.random.randn(3, n_points)
        
        # Create nearly parallel bearing vectors (all pointing roughly in the same direction)
        base_direction = np.array([1, 0, 0])
        bearing = np.zeros((3, n_points))
        for i in range(n_points):
            # Add small random perturbation to create nearly parallel vectors
            perturbation = np.random.randn(3) * 0.01
            vec = base_direction + perturbation
            bearing[:, i] = vec / np.linalg.norm(vec)
        
        # This should not crash, even though the matrix is ill-conditioned
        try:
            (R, t), _ = bearing_linear_solver.solve(uvw, xyz, bearing)
            
            # Verify R is a valid rotation matrix shape
            self.assertEqual(R.shape, (3, 3))
            self.assertEqual(t.shape, (3,))
            
            # Check that R is approximately orthogonal (may not be perfect due to ill-conditioning)
            RtR = R.T @ R
            # Relaxed tolerance for ill-conditioned case
            self.assertTrue(np.allclose(RtR, np.eye(3), atol=0.1))
            
        except Exception as e:
            self.fail(f"Solver failed with ill-conditioned bearings: {e}")
    
    def test_solve_with_well_conditioned_data(self):
        """Test that regularization doesn't hurt well-conditioned cases"""
        # Load a well-conditioned test case
        gpath = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(gpath, 'taes/')
        
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if 'simu_' in f]
            
            if len(files) > 0:
                # Test on first simulation file
                data = load_simulation_data(files[0])
                uvw = data["p1"]
                xyz = data["p2"]
                bearing = data["bearing"]
                
                # Solve should work without issues
                (R, t), _ = bearing_linear_solver.solve(uvw, xyz, bearing)
                
                # Verify solution quality
                self.assertEqual(R.shape, (3, 3))
                self.assertEqual(t.shape, (3,))
                
                # Check R is orthogonal
                RtR = R.T @ R
                self.assertTrue(np.allclose(RtR, np.eye(3), atol=1e-6))
                
                # Check determinant is 1 (proper rotation, not reflection)
                det = np.linalg.det(R)
                self.assertAlmostEqual(det, 1.0, places=5)
    
    def test_adaptive_regularization(self):
        """Test that adaptive regularization parameter is computed correctly"""
        # Create a matrix with known singular values
        U, _ = np.linalg.qr(np.random.randn(10, 5))
        V, _ = np.linalg.qr(np.random.randn(5, 5))
        S = np.array([100, 10, 1, 0.1, 0.01])  # Known singular values
        A = U @ np.diag(S) @ V.T
        
        b = np.random.randn(10)
        
        # Solve with adaptive regularization (regularization=None)
        x = bearing_linear_solver.solve_with_regularization(A, b, regularization=None)
        
        # Should produce a valid solution
        self.assertEqual(x.shape, (5,))
        
        # Verify it doesn't produce NaN or Inf
        self.assertFalse(np.any(np.isnan(x)))
        self.assertFalse(np.any(np.isinf(x)))


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure that the changes maintain backward compatibility"""
    
    def test_solve_returns_correct_types(self):
        """Verify the solve method returns correct types"""
        # Create simple test data
        np.random.seed(42)
        n = 6
        uvw = np.random.randn(3, n)
        xyz = np.random.randn(3, n)
        
        # Create valid bearing vectors
        bearing = np.random.randn(3, n)
        for i in range(n):
            bearing[:, i] = bearing[:, i] / np.linalg.norm(bearing[:, i])
        
        # Call solve method
        result, elapsed_time = bearing_linear_solver.solve(uvw, xyz, bearing)
        
        # Should return (R, t) and elapsed_time due to @timeit decorator
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        R, t = result
        
        # Check types and shapes
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape, (3,))
        self.assertIsInstance(elapsed_time, float)


if __name__ == '__main__':
    unittest.main()
