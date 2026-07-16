"""
Test suite for Fisher Information Matrix (FIM) analysis module.

Tests cover:
1. FIM computation
2. Jacobian computation
3. Observability metrics
4. Relationship between FIM and condition number
5. Improvement suggestions
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fisher_information_matrix import FisherInformationAnalyzer


class TestFisherInformationMatrix(unittest.TestCase):
    """Test FIM computation and analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple test configuration
        np.random.seed(42)
        self.n_points = 6
        self.uvw = np.random.randn(3, self.n_points) * 10
        self.R = np.eye(3)
        self.t = np.array([1.0, 2.0, 3.0])
        
    def test_skew_symmetric_matrix(self):
        """Test skew-symmetric matrix creation"""
        v = np.array([1, 2, 3])
        skew = FisherInformationAnalyzer.skew_symmetric(v)
        
        # Check it's skew-symmetric
        self.assertTrue(np.allclose(skew, -skew.T))
        
        # Check specific values
        self.assertAlmostEqual(skew[0, 1], -3)
        self.assertAlmostEqual(skew[0, 2], 2)
        self.assertAlmostEqual(skew[1, 2], -1)
        
    def test_jacobian_shape(self):
        """Test Jacobian has correct shape"""
        J = FisherInformationAnalyzer.compute_bearing_jacobian(
            self.uvw, self.R, self.t
        )
        
        # Should be (3*n_points, 6)
        expected_shape = (3 * self.n_points, 6)
        self.assertEqual(J.shape, expected_shape)
        
    def test_jacobian_no_nan_inf(self):
        """Test Jacobian contains no NaN or Inf values"""
        J = FisherInformationAnalyzer.compute_bearing_jacobian(
            self.uvw, self.R, self.t
        )
        
        self.assertFalse(np.any(np.isnan(J)))
        self.assertFalse(np.any(np.isinf(J)))
        
    def test_fim_shape(self):
        """Test FIM has correct shape"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        # FIM should be 6x6 (3 rotation + 3 translation parameters)
        self.assertEqual(FIM.shape, (6, 6))
        
    def test_fim_symmetric(self):
        """Test FIM is symmetric"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        self.assertTrue(np.allclose(FIM, FIM.T))
        
    def test_fim_positive_semidefinite(self):
        """Test FIM is positive semi-definite"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(FIM)
        self.assertTrue(np.all(eigenvalues >= -1e-10))  # Allow small numerical errors
        
    def test_fim_scales_with_noise(self):
        """Test FIM scales inversely with measurement noise"""
        sigma1 = 0.01
        sigma2 = 0.02
        
        FIM1 = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t, measurement_noise_std=sigma1
        )
        
        FIM2 = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t, measurement_noise_std=sigma2
        )
        
        # FIM should scale as 1/sigma^2 (inversely)
        # So FIM2 = FIM1 * (sigma1/sigma2)^2
        ratio = (sigma1 / sigma2) ** 2
        self.assertTrue(np.allclose(FIM2, FIM1 * ratio, rtol=1e-5))
        

class TestObservabilityMetrics(unittest.TestCase):
    """Test computation of observability metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        self.t = np.array([1.0, 2.0, 3.0])
        
    def test_metrics_keys(self):
        """Test that all expected metrics are computed"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        
        expected_keys = [
            'trace', 'determinant', 'condition_number',
            'smallest_eigenvalue', 'largest_eigenvalue',
            'eigenvalue_ratio', 'observability_index'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            
    def test_trace_positive(self):
        """Test trace is positive for valid FIM"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        self.assertGreater(metrics['trace'], 0)
        
    def test_condition_number_positive(self):
        """Test condition number is positive"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        self.assertGreater(metrics['condition_number'], 0)
        
    def test_eigenvalue_ordering(self):
        """Test eigenvalues are properly ordered"""
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            self.uvw, self.R, self.t
        )
        
        metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        self.assertLessEqual(
            metrics['smallest_eigenvalue'],
            metrics['largest_eigenvalue']
        )


class TestFIMConditionRelationship(unittest.TestCase):
    """Test the relationship between FIM and Jacobian condition numbers"""
    
    def test_fim_condition_approximation(self):
        """Test that cond(FIM) ≈ [cond(J)]^2"""
        np.random.seed(123)
        
        # Test with multiple configurations
        for _ in range(5):
            uvw = np.random.randn(3, 8) * 10
            R = np.eye(3)
            t = np.random.randn(3)
            
            # Compute Jacobian and FIM
            J = FisherInformationAnalyzer.compute_bearing_jacobian(uvw, R, t)
            FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
                uvw, R, t, measurement_noise_std=1.0
            )
            
            # Compute condition numbers
            cond_J = np.linalg.cond(J)
            cond_FIM = np.linalg.cond(FIM)
            
            # Check relationship: cond(FIM) ≈ [cond(J)]^2
            # We expect them to be close, but not exact due to numerical errors
            expected_cond_FIM = cond_J ** 2
            
            # Allow for reasonable tolerance (factor of 2)
            ratio = cond_FIM / expected_cond_FIM
            self.assertGreater(ratio, 0.5)
            self.assertLess(ratio, 2.0)
            
    def test_parallel_bearings_condition(self):
        """Test that parallel bearings lead to high condition number"""
        # Create nearly parallel bearings
        base = np.array([1, 0, 0])
        uvw_parallel = np.zeros((3, 6))
        for i in range(6):
            perturbation = np.random.randn(3) * 0.01
            direction = base + perturbation
            uvw_parallel[:, i] = 10 * direction / np.linalg.norm(direction)
        
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        FIM_parallel = FisherInformationAnalyzer.compute_fisher_information_matrix(
            uvw_parallel, R, t
        )
        
        metrics_parallel = FisherInformationAnalyzer.compute_fim_metrics(FIM_parallel)
        
        # Parallel bearings should have high condition number
        self.assertGreater(metrics_parallel['condition_number'], 100)
        
    def test_diverse_bearings_condition(self):
        """Test that diverse bearings lead to lower condition number"""
        # Create well-distributed bearings
        uvw_diverse = np.array([
            [10, -10, 10, -10, 0, 0],
            [10, 10, -10, -10, 10, -10],
            [5, 5, 5, 5, 10, 10]
        ], dtype=float)
        
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        FIM_diverse = FisherInformationAnalyzer.compute_fisher_information_matrix(
            uvw_diverse, R, t
        )
        
        metrics_diverse = FisherInformationAnalyzer.compute_fim_metrics(FIM_diverse)
        
        # Diverse bearings should have lower condition number
        self.assertLess(metrics_diverse['condition_number'], 1000)


class TestImprovementSuggestions(unittest.TestCase):
    """Test improvement suggestion functionality"""
    
    def test_suggestions_structure(self):
        """Test that suggestions have expected structure"""
        np.random.seed(42)
        uvw = np.random.randn(3, 6) * 10
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        suggestions = FisherInformationAnalyzer.suggest_improvements(uvw, R, t)
        
        expected_keys = [
            'current_observability',
            'worst_observable_direction',
            'worst_eigenvalue',
            'suggested_motion',
            'rotation_suggestion',
            'translation_suggestion'
        ]
        
        for key in expected_keys:
            self.assertIn(key, suggestions)
            
    def test_worst_direction_shape(self):
        """Test worst observable direction has correct shape"""
        np.random.seed(42)
        uvw = np.random.randn(3, 6) * 10
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        suggestions = FisherInformationAnalyzer.suggest_improvements(uvw, R, t)
        
        # Should be 6D (3 rotation + 3 translation)
        self.assertEqual(len(suggestions['worst_observable_direction']), 6)


class TestObjectiveFunctions(unittest.TestCase):
    """Test objective functions for guidance"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.uvw = np.random.randn(3, 8) * 10
        self.R = np.eye(3)
        self.t = np.array([1.0, 2.0, 3.0])
        
    def test_trace_objective(self):
        """Test trace objective computation"""
        obj = FisherInformationAnalyzer.compute_condition_based_objective(
            self.uvw, self.R, self.t, objective_type='trace'
        )
        
        self.assertIsInstance(obj, float)
        self.assertGreater(obj, 0)
        
    def test_determinant_objective(self):
        """Test determinant objective computation"""
        obj = FisherInformationAnalyzer.compute_condition_based_objective(
            self.uvw, self.R, self.t, objective_type='determinant'
        )
        
        self.assertIsInstance(obj, float)
        # Log determinant can be negative
        self.assertFalse(np.isnan(obj))
        
    def test_min_eigenvalue_objective(self):
        """Test minimum eigenvalue objective"""
        obj = FisherInformationAnalyzer.compute_condition_based_objective(
            self.uvw, self.R, self.t, objective_type='min_eigenvalue'
        )
        
        self.assertIsInstance(obj, float)
        self.assertGreaterEqual(obj, 0)
        
    def test_inverse_condition_objective(self):
        """Test inverse condition number objective"""
        obj = FisherInformationAnalyzer.compute_condition_based_objective(
            self.uvw, self.R, self.t, objective_type='inverse_condition'
        )
        
        self.assertIsInstance(obj, float)
        self.assertGreater(obj, 0)
        self.assertLessEqual(obj, 1.0)  # Inverse of condition >= 1
        
    def test_invalid_objective_type(self):
        """Test that invalid objective type raises error"""
        with self.assertRaises(ValueError):
            FisherInformationAnalyzer.compute_condition_based_objective(
                self.uvw, self.R, self.t, objective_type='invalid'
            )


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for complete analysis workflow"""
    
    def test_complete_analysis(self):
        """Test complete observability analysis"""
        np.random.seed(42)
        uvw = np.random.randn(3, 8) * 10
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
        
        # Check all expected keys
        expected_keys = [
            'FIM', 'FIM_metrics', 'Jacobian',
            'Jacobian_condition_number', 'analysis'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
            
        # Check types
        self.assertIsInstance(analysis['FIM'], np.ndarray)
        self.assertIsInstance(analysis['FIM_metrics'], dict)
        self.assertIsInstance(analysis['Jacobian'], np.ndarray)
        self.assertIsInstance(analysis['Jacobian_condition_number'], float)
        self.assertIsInstance(analysis['analysis'], str)
        
    def test_interpretation_output(self):
        """Test that interpretation produces valid output"""
        np.random.seed(42)
        uvw = np.random.randn(3, 8) * 10
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        
        analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
        interpretation = analysis['analysis']
        
        # Should contain key phrases
        self.assertIn('Condition Number', interpretation)
        self.assertIn('FIM', interpretation)
        
        # Should not be empty
        self.assertGreater(len(interpretation), 100)


if __name__ == '__main__':
    unittest.main()
