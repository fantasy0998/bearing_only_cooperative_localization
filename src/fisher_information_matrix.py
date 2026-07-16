"""
Fisher Information Matrix (FIM) Analysis for Bearing-Only Cooperative Localization

This module implements Fisher Information Matrix computation and analysis tools
for bearing-only localization. It explores the relationship between FIM and 
condition number, and provides heuristics for improving observability.

Key Concepts:
1. Fisher Information Matrix (FIM): Quantifies the amount of information that 
   bearing measurements carry about the pose parameters (rotation and translation)
2. Condition Number: Measures how sensitive the solution is to perturbations
3. Observability: The degree to which the system state can be determined from 
   the measurements

References:
- JS Russell et al., "Cooperative Localisation of a GPS-Denied UAV using 
  Direction-of-Arrival Measurements," IEEE TAES, 2019.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FisherInformationAnalyzer:
    """
    Analyzes observability using Fisher Information Matrix and condition number.
    """
    
    @staticmethod
    def compute_bearing_jacobian(uvw: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of bearing measurements with respect to pose parameters.
        
        For bearing-only localization, the measurement model is:
        h(R, t) = (R*uvw + t - xyz) / ||R*uvw + t - xyz||
        
        The Jacobian is computed with respect to the pose parameters [rotation, translation].
        For rotation, we use the axis-angle representation (3 parameters).
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            
        Returns:
            Jacobian matrix, shape (3n, 6) where:
            - First 3 columns: derivatives w.r.t. rotation (axis-angle)
            - Last 3 columns: derivatives w.r.t. translation
        """
        n_points = uvw.shape[1]
        J = np.zeros((3 * n_points, 6))
        
        for i in range(n_points):
            p = uvw[:, i]  # Point in global frame
            
            # Predicted point in local frame
            p_local = R @ p + t
            norm_p = np.linalg.norm(p_local)
            
            # Unit bearing vector
            b = p_local / norm_p
            
            # Jacobian w.r.t. translation
            # d(b)/d(t) = (I - b*b^T) / ||p||
            I_bb = np.eye(3) - np.outer(b, b)
            J_t = I_bb / norm_p
            
            # Jacobian w.r.t. rotation (using axis-angle parameterization)
            # d(b)/d(omega) where R = exp([omega]_x)
            # For small perturbations: delta_R ≈ [omega]_x * R
            # So: d(R*p)/d(omega) = [R*p]_x
            Rp = R @ p
            skew_Rp = FisherInformationAnalyzer.skew_symmetric(Rp)
            J_R = (I_bb @ skew_Rp) / norm_p
            
            # Assemble Jacobian for this measurement
            J[3*i:3*i+3, 0:3] = J_R  # Rotation part
            J[3*i:3*i+3, 3:6] = J_t  # Translation part
        
        return J
    
    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from 3D vector.
        
        Args:
            v: 3D vector
            
        Returns:
            3x3 skew-symmetric matrix [v]_x
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    @staticmethod
    def compute_fisher_information_matrix(
        uvw: np.ndarray, 
        R: np.ndarray, 
        t: np.ndarray,
        measurement_noise_std: float = 0.01
    ) -> np.ndarray:
        """
        Compute the Fisher Information Matrix for bearing-only measurements.
        
        FIM = J^T * Σ^{-1} * J
        
        where:
        - J is the measurement Jacobian
        - Σ is the measurement noise covariance
        
        For bearing measurements with isotropic Gaussian noise of std σ:
        Σ = σ^2 * I
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Rotation matrix estimate, shape (3, 3)
            t: Translation vector estimate, shape (3,)
            measurement_noise_std: Standard deviation of bearing noise (default: 0.01)
            
        Returns:
            Fisher Information Matrix, shape (6, 6)
        """
        # Compute Jacobian
        J = FisherInformationAnalyzer.compute_bearing_jacobian(uvw, R, t)
        
        # Measurement noise covariance (assume isotropic Gaussian)
        sigma_sq = measurement_noise_std ** 2
        
        # Fisher Information Matrix: FIM = J^T * J / sigma^2
        FIM = (J.T @ J) / sigma_sq
        
        return FIM
    
    @staticmethod
    def compute_fim_metrics(FIM: np.ndarray) -> Dict[str, float]:
        """
        Compute various metrics from the Fisher Information Matrix.
        
        Args:
            FIM: Fisher Information Matrix, shape (6, 6)
            
        Returns:
            Dictionary containing:
            - trace: Trace of FIM (sum of diagonal elements)
            - determinant: Determinant of FIM
            - condition_number: Condition number of FIM
            - smallest_eigenvalue: Smallest eigenvalue (worst observability direction)
            - largest_eigenvalue: Largest eigenvalue (best observability direction)
            - observability_index: Determinant-based observability metric
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(FIM)
        
        # Sort eigenvalues (smallest to largest)
        eigenvalues = np.sort(eigenvalues)
        
        # Compute metrics
        metrics = {
            'trace': np.trace(FIM),
            'determinant': np.linalg.det(FIM),
            'condition_number': np.linalg.cond(FIM),
            'smallest_eigenvalue': eigenvalues[0],
            'largest_eigenvalue': eigenvalues[-1],
            'eigenvalue_ratio': eigenvalues[-1] / (eigenvalues[0] + 1e-10),
            'observability_index': np.prod(eigenvalues) ** (1.0 / len(eigenvalues))  # Geometric mean
        }
        
        return metrics
    
    @staticmethod
    def analyze_observability(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        measurement_noise_std: float = 0.01
    ) -> Dict[str, Any]:
        """
        Comprehensive observability analysis using FIM and condition number.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Rotation matrix estimate, shape (3, 3)
            t: Translation vector estimate, shape (3,)
            measurement_noise_std: Standard deviation of bearing noise
            
        Returns:
            Dictionary containing:
            - FIM: Fisher Information Matrix
            - FIM_metrics: Various FIM-based metrics
            - Jacobian: Measurement Jacobian
            - Jacobian_condition_number: Condition number of J
            - analysis: Text description of observability quality
        """
        # Compute FIM
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            uvw, R, t, measurement_noise_std
        )
        
        # Compute FIM metrics
        fim_metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        
        # Compute Jacobian
        J = FisherInformationAnalyzer.compute_bearing_jacobian(uvw, R, t)
        J_cond = np.linalg.cond(J)
        
        # Analysis
        analysis = FisherInformationAnalyzer.interpret_observability(fim_metrics, J_cond)
        
        return {
            'FIM': FIM,
            'FIM_metrics': fim_metrics,
            'Jacobian': J,
            'Jacobian_condition_number': J_cond,
            'analysis': analysis
        }
    
    @staticmethod
    def interpret_observability(fim_metrics: Dict[str, float], jacobian_cond: float) -> str:
        """
        Provide textual interpretation of observability quality.
        
        Args:
            fim_metrics: Metrics computed from FIM
            jacobian_cond: Condition number of Jacobian
            
        Returns:
            Text description of observability quality
        """
        lines = []
        lines.append("=== Observability Analysis ===")
        
        # Condition number analysis
        fim_cond = fim_metrics['condition_number']
        if fim_cond < 10:
            lines.append(f"FIM Condition Number: {fim_cond:.2e} - EXCELLENT (well-conditioned)")
        elif fim_cond < 100:
            lines.append(f"FIM Condition Number: {fim_cond:.2e} - GOOD")
        elif fim_cond < 1000:
            lines.append(f"FIM Condition Number: {fim_cond:.2e} - MODERATE")
        elif fim_cond < 1e6:
            lines.append(f"FIM Condition Number: {fim_cond:.2e} - POOR (ill-conditioned)")
        else:
            lines.append(f"FIM Condition Number: {fim_cond:.2e} - VERY POOR (severely ill-conditioned)")
        
        if jacobian_cond < 10:
            lines.append(f"Jacobian Condition Number: {jacobian_cond:.2e} - EXCELLENT")
        elif jacobian_cond < 100:
            lines.append(f"Jacobian Condition Number: {jacobian_cond:.2e} - GOOD")
        else:
            lines.append(f"Jacobian Condition Number: {jacobian_cond:.2e} - NEEDS IMPROVEMENT")
        
        # Determinant analysis
        det = fim_metrics['determinant']
        if det > 1e6:
            lines.append(f"FIM Determinant: {det:.2e} - STRONG observability")
        elif det > 1e3:
            lines.append(f"FIM Determinant: {det:.2e} - GOOD observability")
        elif det > 1.0:
            lines.append(f"FIM Determinant: {det:.2e} - MODERATE observability")
        else:
            lines.append(f"FIM Determinant: {det:.2e} - WEAK observability")
        
        # Eigenvalue analysis
        min_eig = fim_metrics['smallest_eigenvalue']
        max_eig = fim_metrics['largest_eigenvalue']
        lines.append(f"Eigenvalue Range: [{min_eig:.2e}, {max_eig:.2e}]")
        
        if min_eig < 1e-3:
            lines.append("WARNING: Small minimum eigenvalue indicates poor observability in some directions")
        
        # Relationship between FIM and Jacobian condition numbers
        lines.append("\n=== FIM-Condition Number Relationship ===")
        lines.append(f"FIM is related to J^T*J, so:")
        lines.append(f"  cond(FIM) ≈ [cond(J)]^2")
        lines.append(f"  Expected: {jacobian_cond**2:.2e}")
        lines.append(f"  Actual:   {fim_cond:.2e}")
        ratio = fim_cond / (jacobian_cond**2 + 1e-10)
        lines.append(f"  Ratio:    {ratio:.2f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def suggest_improvements(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        measurement_noise_std: float = 0.01
    ) -> Dict[str, Any]:
        """
        Suggest trajectory modifications to improve observability.
        
        Based on the FIM analysis, this method suggests directions in which
        the agent should move to improve observability.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Current rotation estimate, shape (3, 3)
            t: Current translation estimate, shape (3,)
            measurement_noise_std: Standard deviation of bearing noise
            
        Returns:
            Dictionary containing:
            - current_observability: Current FIM metrics
            - worst_observable_direction: Direction with poorest observability
            - suggested_motion: Suggested motion direction
            - potential_improvement: Estimated improvement if suggestion followed
        """
        # Compute current FIM
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(
            uvw, R, t, measurement_noise_std
        )
        
        # Eigendecomposition of FIM
        eigenvalues, eigenvectors = np.linalg.eigh(FIM)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Worst observable direction (smallest eigenvalue)
        worst_direction = eigenvectors[:, 0]
        worst_eigenvalue = eigenvalues[0]
        
        # Suggest motion perpendicular to current bearings to improve diversity
        # This is a heuristic: move in the direction that maximizes trace(FIM)
        
        # For translation: move perpendicular to average bearing direction
        # For rotation: rotate to spread bearings more evenly
        
        # Compute current metrics
        current_metrics = FisherInformationAnalyzer.compute_fim_metrics(FIM)
        
        # Heuristic suggestions
        suggestions = {
            'current_observability': current_metrics,
            'worst_observable_direction': worst_direction,
            'worst_eigenvalue': worst_eigenvalue,
            'suggested_motion': f"Move/rotate in direction: {worst_direction[:3]}",
            'rotation_suggestion': f"Rotate around axis: {worst_direction[0:3] / (np.linalg.norm(worst_direction[0:3]) + 1e-10)}",
            'translation_suggestion': f"Translate in direction: {worst_direction[3:6] / (np.linalg.norm(worst_direction[3:6]) + 1e-10)}"
        }
        
        return suggestions
    
    @staticmethod
    def compute_condition_based_objective(
        uvw: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        objective_type: str = 'trace'
    ) -> float:
        """
        Compute an objective function for guidance based on FIM/condition number.
        
        This can be used in trajectory optimization to maximize observability.
        
        Args:
            uvw: Points in global frame, shape (3, n)
            R: Rotation matrix, shape (3, 3)
            t: Translation vector, shape (3,)
            objective_type: Type of objective ('trace', 'determinant', 'min_eigenvalue', 'inverse_condition')
            
        Returns:
            Objective value (higher is better for observability)
        """
        FIM = FisherInformationAnalyzer.compute_fisher_information_matrix(uvw, R, t)
        
        if objective_type == 'trace':
            # Maximize trace of FIM (commonly used in optimal experiment design)
            return np.trace(FIM)
        
        elif objective_type == 'determinant':
            # Maximize determinant of FIM (D-optimality criterion)
            det = np.linalg.det(FIM)
            return np.log(det + 1e-10)  # Use log to avoid numerical issues
        
        elif objective_type == 'min_eigenvalue':
            # Maximize minimum eigenvalue (E-optimality criterion)
            eigenvalues = np.linalg.eigvalsh(FIM)
            return eigenvalues[0]
        
        elif objective_type == 'inverse_condition':
            # Minimize condition number = maximize inverse condition number
            cond = np.linalg.cond(FIM)
            return 1.0 / (cond + 1e-10)
        
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    @staticmethod
    def compare_configurations(
        uvw_configs: list,
        R: np.ndarray,
        t: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Compare multiple bearing configurations for observability.
        
        Args:
            uvw_configs: List of point configurations to compare
            R: Rotation matrix
            t: Translation vector
            
        Returns:
            Dictionary mapping configuration index to its metrics
        """
        results = {}
        
        for i, uvw in enumerate(uvw_configs):
            analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
            results[i] = {
                'metrics': analysis['FIM_metrics'],
                'jacobian_cond': analysis['Jacobian_condition_number']
            }
        
        # Rank configurations
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['determinant'],
            reverse=True
        )
        
        print("\n=== Configuration Ranking (by determinant) ===")
        for rank, (idx, data) in enumerate(ranked):
            print(f"Rank {rank+1}: Config {idx}")
            print(f"  Determinant: {data['metrics']['determinant']:.2e}")
            print(f"  Condition Number: {data['metrics']['condition_number']:.2e}")
            print(f"  Jacobian Condition: {data['jacobian_cond']:.2e}")
        
        return results
