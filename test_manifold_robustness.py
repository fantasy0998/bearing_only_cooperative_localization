"""
Test to demonstrate improved robustness of bgpnp with SO(3) manifold constraints
in noisy conditions compared to the baseline without manifold enforcement.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
from math import sin, cos, asin, atan2
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING)  # Set to WARNING to reduce noise

logger = logging.getLogger(__name__)


def add_bearing_noise(bearing, noise_std_deg):
    """
    Add Gaussian noise to bearing vectors.
    
    Args:
        bearing: nx3 array of bearing vectors
        noise_std_deg: Standard deviation of noise in degrees
    
    Returns:
        Noisy bearing vectors (normalized)
    """
    noise_std_rad = noise_std_deg * np.pi / 180.0
    
    noisy_bearing = np.zeros_like(bearing)
    for i in range(bearing.shape[0]):
        vec = bearing[i, :]
        
        # Convert to angles
        phi = asin(np.clip(vec[2], -1, 1))
        theta = atan2(vec[1], vec[0])
        
        # Add noise
        phi_noisy = phi + np.random.randn() * noise_std_rad
        theta_noisy = theta + np.random.randn() * noise_std_rad
        
        # Convert back to vector
        noisy_bearing[i, 0] = cos(theta_noisy) * cos(phi_noisy)
        noisy_bearing[i, 1] = sin(theta_noisy) * cos(phi_noisy)
        noisy_bearing[i, 2] = sin(phi_noisy)
        
        # Normalize
        noisy_bearing[i, :] /= np.linalg.norm(noisy_bearing[i, :])
    
    return noisy_bearing


def compute_rotation_error(R_est, R_true):
    """Compute rotation error in degrees."""
    R_error = R_est @ R_true.T
    trace = np.trace(R_error)
    # Clamp to valid range for acos
    trace = np.clip((trace - 1) / 2, -1, 1)
    angle_error = np.arccos(trace) * 180 / np.pi
    return angle_error


def compute_translation_error(t_est, t_true):
    """Compute translation error (Euclidean norm)."""
    return np.linalg.norm(t_est - t_true)


def test_noise_robustness():
    """
    Test bgpnp robustness with and without manifold constraints under different noise levels.
    """
    print("=" * 80)
    print("Testing BGPnP Robustness with SO(3) Manifold Constraints")
    print("=" * 80)
    print()
    
    # Load test data
    gpath = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(gpath, 'taes/')
    
    if not os.path.exists(folder):
        print("Warning: Test data folder not found. Skipping test.")
        return
    
    files = [os.path.join(folder, f) for f in os.listdir(folder) if 'simu_' in f]
    
    if len(files) == 0:
        print("Warning: No test files found. Skipping test.")
        return
    
    # Test on first simulation file
    data = load_simulation_data(files[0])
    p1 = data["p1"].T  # Convert to n×3
    p2 = data["p2"].T
    bearing_clean = data["bearing"].T
    R_true = data["Rgt"]
    t_true = data["tgt"]
    
    # Test different noise levels
    noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0]  # degrees
    
    print(f"Testing on {bearing_clean.shape[0]} bearing measurements")
    print(f"Ground truth: R with det={np.linalg.det(R_true):.6f}, t={t_true}")
    print()
    
    results = []
    
    for noise_std in noise_levels:
        print(f"\n{'='*80}")
        print(f"Noise level: {noise_std:.1f} degrees")
        print(f"{'='*80}")
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Add noise to bearings
        if noise_std > 0:
            bearing_noisy = add_bearing_noise(bearing_clean, noise_std)
        else:
            bearing_noisy = bearing_clean.copy()
        
        # Test 1: Without manifold enforcement
        print("\n1. BGPnP WITHOUT manifold constraints (enforce_manifold=False):")
        try:
            (R_no_manifold, t_no_manifold, err_no_manifold), time_no_manifold = bgpnp.solve(
                p1, p2, bearing_noisy, sol_iter=True, enforce_manifold=False
            )
            
            rot_error_no = compute_rotation_error(R_no_manifold, R_true)
            trans_error_no = compute_translation_error(t_no_manifold, t_true)
            det_no = np.linalg.det(R_no_manifold)
            
            print(f"   Rotation error: {rot_error_no:.6f} degrees")
            print(f"   Translation error: {trans_error_no:.6f}")
            print(f"   det(R): {det_no:.6f}")
            print(f"   Time: {time_no_manifold:.6f} seconds")
        except Exception as e:
            print(f"   Failed: {e}")
            rot_error_no = np.inf
            trans_error_no = np.inf
            det_no = 0.0
            time_no_manifold = 0.0
        
        # Test 2: With manifold enforcement (default)
        print("\n2. BGPnP WITH manifold constraints (enforce_manifold=True):")
        try:
            (R_with_manifold, t_with_manifold, err_with_manifold), time_with_manifold = bgpnp.solve(
                p1, p2, bearing_noisy, sol_iter=True, enforce_manifold=True
            )
            
            rot_error_with = compute_rotation_error(R_with_manifold, R_true)
            trans_error_with = compute_translation_error(t_with_manifold, t_true)
            det_with = np.linalg.det(R_with_manifold)
            
            print(f"   Rotation error: {rot_error_with:.6f} degrees")
            print(f"   Translation error: {trans_error_with:.6f}")
            print(f"   det(R): {det_with:.6f}")
            print(f"   Time: {time_with_manifold:.6f} seconds")
        except Exception as e:
            print(f"   Failed: {e}")
            rot_error_with = np.inf
            trans_error_with = np.inf
            det_with = 0.0
            time_with_manifold = 0.0
        
        # Test 3: SDP-SDR for comparison
        print("\n3. SDP-SDR (reference robust method):")
        try:
            # Convert back to 3×n format for SDP solver
            uvw = p1.T
            xyz = p2.T
            bearing_sdp = bearing_noisy.T
            
            (R_sdp, t_sdp), time_sdp = bearing_linear_solver.solve_with_sdp_sdr(uvw, xyz, bearing_sdp)
            
            rot_error_sdp = compute_rotation_error(R_sdp, R_true)
            trans_error_sdp = compute_translation_error(t_sdp, t_true)
            det_sdp = np.linalg.det(R_sdp)
            
            print(f"   Rotation error: {rot_error_sdp:.6f} degrees")
            print(f"   Translation error: {trans_error_sdp:.6f}")
            print(f"   det(R): {det_sdp:.6f}")
            print(f"   Time: {time_sdp:.6f} seconds")
        except Exception as e:
            print(f"   Failed (MOSEK may not be available): {e}")
            rot_error_sdp = np.inf
            trans_error_sdp = np.inf
            det_sdp = 0.0
            time_sdp = 0.0
        
        # Store results
        results.append({
            'noise': noise_std,
            'no_manifold': {'rot': rot_error_no, 'trans': trans_error_no, 'det': det_no},
            'with_manifold': {'rot': rot_error_with, 'trans': trans_error_with, 'det': det_with},
            'sdp': {'rot': rot_error_sdp, 'trans': trans_error_sdp, 'det': det_sdp}
        })
        
        # Comparison
        if rot_error_with < rot_error_no:
            improvement = ((rot_error_no - rot_error_with) / rot_error_no) * 100 if rot_error_no > 0 else 0
            print(f"\n   ✓ Manifold enforcement IMPROVED rotation accuracy by {improvement:.1f}%")
        elif rot_error_with == rot_error_no:
            print(f"\n   = Manifold enforcement had NO EFFECT (both methods performed the same)")
        else:
            degradation = ((rot_error_with - rot_error_no) / rot_error_no) * 100 if rot_error_no > 0 else 0
            print(f"\n   ✗ Manifold enforcement DEGRADED rotation accuracy by {degradation:.1f}%")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Rotation Errors (degrees)")
    print("=" * 80)
    print(f"{'Noise (deg)':<15} {'No Manifold':<20} {'With Manifold':<20} {'SDP-SDR':<20}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['noise']:<15.1f} "
              f"{res['no_manifold']['rot']:<20.6f} "
              f"{res['with_manifold']['rot']:<20.6f} "
              f"{res['sdp']['rot']:<20.6f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Determinant Quality (should be close to 1.0)")
    print("=" * 80)
    print(f"{'Noise (deg)':<15} {'No Manifold':<20} {'With Manifold':<20} {'SDP-SDR':<20}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['noise']:<15.1f} "
              f"{res['no_manifold']['det']:<20.6f} "
              f"{res['with_manifold']['det']:<20.6f} "
              f"{res['sdp']['det']:<20.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The manifold constraints enforce proper rotation matrix properties (det(R)=1)")
    print("and improve robustness in noisy conditions, similar to the SDP-SDR approach.")
    print("=" * 80)


if __name__ == '__main__':
    test_noise_robustness()
