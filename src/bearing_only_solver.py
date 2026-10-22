import numpy as np
import os
import logging
from math import sin, cos, asin, acos, atan2
from typing import Optional, Tuple
import cvxpy as cp
import time
from scipy.optimize import least_squares
from scipy.linalg import lstsq

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        logger.debug(f"Elapsed time: {elapsed_time} seconds")
        return result, elapsed_time
    return wrapper

def to_bearing_angle(bearing: np.ndarray) -> np.ndarray:
    """
    Convert a 3D vector to bearing angles.

    Args:
        vec (np.ndarray): The 3D vector to convert.

    Returns:
        np.ndarray: The bearing angles.
    """
    if bearing.shape[0] > bearing.shape[1]:
        bearing = bearing.T

    bearing_angle = np.zeros((2, bearing.shape[1]))
    for j in range(bearing.shape[1]):
        vec = bearing[:, j]
        phi = asin(vec[2])
        theta = atan2(vec[1], vec[0])
        bearing_angle[:, j] = np.array([theta, phi])
    return bearing_angle if bearing.shape[0] > bearing.shape[1] else bearing_angle.T


def load_simulation_data(filename: str) -> dict:
    """
    Load simulation data from a file.

    Args:
        filename (str): The path to the file containing the simulation data.

    Returns:
        dict: A dictionary containing the loaded simulation data. The dictionary
              has the following keys:
              - "p1": A 2D numpy array of shape (3, n) representing the p1 data.
              - "p2": A 2D numpy array of shape (3, n) representing the p2 data.
              - "bearing": A 2D numpy array of shape (3, n) representing the bearing data.
              - "Rgt": A 2D numpy array of shape (3, 3) representing the Rgt data.
              - "tgt": A 1D numpy array of shape (n,) representing the tgt data.
    """
    with open(filename, 'r') as f:
        data = {}
        lines = f.readlines()

        # Read p1
        p1_lines = lines[0].strip().split(' ')
        p1 = np.array([float(x) for x in p1_lines]).reshape(-1,3)

        # Read p2
        p2_lines = lines[1].strip().split(' ')
        p2 = np.array([float(x) for x in p2_lines]).reshape(-1,3)

        # Read bearing
        bearing_lines = lines[2].strip().split(' ')
        bearing = np.array([float(x) for x in bearing_lines]).reshape(-1,3)

        # Read Rgt
        Rgt_lines = lines[3].strip().split(' ')
        Rgt = np.array([float(x) for x in Rgt_lines]).reshape(3, 3)

        # Read tgt
        tgt_lines = lines[4].strip().split(' ')
        tgt = np.array([float(x) for x in tgt_lines])

        data["p1"] = p1.T
        data["p2"] = p2.T
        data["bearing"] = bearing.T
        data["Rgt"] = Rgt
        data["tgt"] = tgt

    return data
class bearing_linear_solver():
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_reduced_Ab_matrix(uA: np.ndarray, vA: np.ndarray, wA: np.ndarray, phi: np.ndarray, theta: np.ndarray,
                                k: int, xB: np.ndarray, yB: np.ndarray, zB: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reduced A and b matrices for the bearing-only solver.

        Parameters:
        - uA (np.ndarray): Array of uA values. Shape: (k,)
        - vA (np.ndarray): Array of vA values. Shape: (k,)
        - wA (np.ndarray): Array of wA values. Shape: (k,)
        - phi (np.ndarray): Array of phi values. Shape: (k,)
        - theta (np.ndarray): Array of theta values. Shape: (k,)
        - k (int): Number of elements in the arrays.
        - xB (np.ndarray): Array of xB values. Shape: (k,)
        - yB (np.ndarray): Array of yB values. Shape: (k,)
        - zB (np.ndarray): Array of zB values. Shape: (k,)
        - R (np.ndarray): Array of R values. Shape: (3, 3)

        Returns:
        - A (np.ndarray): The reduced A matrix. Shape: (2 * k, 3)
        - b (np.ndarray): The reduced b matrix. Shape: (2 * k,)
        """
        r = R.flatten()
        A = np.zeros((2 * k, 3))
        b = np.zeros(2 * k)

        for i in range(k):
            A[2 * i, 0] = np.sin(phi[i])
            A[2 * i, 1] = 0
            A[2 * i, 2] = -np.cos(theta[i]) * np.cos(phi[i])

            A[2 * i + 1, 0] = 0
            A[2 * i + 1, 1] = np.sin(phi[i])
            A[2 * i + 1, 2] = -np.sin(theta[i]) * np.cos(phi[i])

            AA = np.zeros((2, 9))
            AA[0, 0] = uA[i] * np.sin(phi[i])
            AA[0, 1] = vA[i] * np.sin(phi[i])
            AA[0, 2] = wA[i] * np.sin(phi[i])
            AA[0, 3] = 0
            AA[0, 4] = 0
            AA[0, 5] = 0
            AA[0, 6] = -uA[i] * np.cos(theta[i]) * np.cos(phi[i])
            AA[0, 7] = -vA[i] * np.cos(theta[i]) * np.cos(phi[i])
            AA[0, 8] = -wA[i] * np.cos(theta[i]) * np.cos(phi[i])

            AA[0 + 1, 0] = 0
            AA[0 + 1, 1] = 0
            AA[0 + 1, 2] = 0
            AA[0 + 1, 3] = uA[i] * np.sin(phi[i])
            AA[0 + 1, 4] = vA[i] * np.sin(phi[i])
            AA[0 + 1, 5] = wA[i] * np.sin(phi[i])
            AA[0 + 1, 6] = -uA[i] * np.sin(theta[i]) * np.cos(phi[i])
            AA[0 + 1, 7] = -vA[i] * np.sin(theta[i]) * np.cos(phi[i])
            AA[0 + 1, 8] = -wA[i] * np.sin(theta[i]) * np.cos(phi[i])

            residual = AA.dot(r)
            logger.debug(f'Residual: {residual}')

            b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i] - residual[0]
            b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i] -  residual[1]

        return A, b

    @staticmethod
    def compute_A_matrix(uA: np.ndarray, vA: np.ndarray, wA: np.ndarray, phi: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the A matrix for bearing-only solver.

        Parameters:
        uA (np.ndarray): Array of uA values. Shape: (k,)
        vA (np.ndarray): Array of vA values. Shape: (k,)
        wA (np.ndarray): Array of wA values. Shape: (k,)
        phi (np.ndarray): Array of phi values. Shape: (k,)
        theta (np.ndarray): Array of theta values. Shape: (k,)
        k (int): Number of elements in the arrays.

        Returns:
        A (np.ndarray): A matrix of shape (2 * k, 12).
        """
        A = np.zeros((2 * k, 12))

        for i in range(k):
            A[2 * i, 0] = uA[i] * np.sin(phi[i])
            A[2 * i, 1] = vA[i] * np.sin(phi[i])
            A[2 * i, 2] = wA[i] * np.sin(phi[i])
            A[2 * i, 3] = 0
            A[2 * i, 4] = 0
            A[2 * i, 5] = 0
            A[2 * i, 6] = -uA[i] * np.cos(theta[i]) * np.cos(phi[i])
            A[2 * i, 7] = -vA[i] * np.cos(theta[i]) * np.cos(phi[i])
            A[2 * i, 8] = -wA[i] * np.cos(theta[i]) * np.cos(phi[i])
            A[2 * i, 9] = np.sin(phi[i])
            A[2 * i, 10] = 0
            A[2 * i, 11] = -np.cos(theta[i]) * np.cos(phi[i])

            A[2 * i + 1, 0] = 0
            A[2 * i + 1, 1] = 0
            A[2 * i + 1, 2] = 0
            A[2 * i + 1, 3] = uA[i] * np.sin(phi[i])
            A[2 * i + 1, 4] = vA[i] * np.sin(phi[i])
            A[2 * i + 1, 5] = wA[i] * np.sin(phi[i])
            A[2 * i + 1, 6] = -uA[i] * np.sin(theta[i]) * np.cos(phi[i])
            A[2 * i + 1, 7] = -vA[i] * np.sin(theta[i]) * np.cos(phi[i])
            A[2 * i + 1, 8] = -wA[i] * np.sin(theta[i]) * np.cos(phi[i])
            A[2 * i + 1, 9] = 0
            A[2 * i + 1, 10] = np.sin(phi[i])
            A[2 * i + 1, 11] = -np.sin(theta[i]) * np.cos(phi[i])

        return A

    @staticmethod
    def compute_b_vector(xB: np.ndarray, yB: np.ndarray, zB: np.ndarray, phi: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the b vector for the bearing-only solver.

        Parameters:
        - xB (np.ndarray): x-coordinates of the bearings. Shape: (k,)
        - yB (np.ndarray): y-coordinates of the bearings. Shape: (k,)
        - zB (np.ndarray): z-coordinates of the bearings. Shape: (k,)
        - phi (np.ndarray): azimuth angles of the bearings. Shape: (k,)
        - theta (np.ndarray): elevation angles of the bearings. Shape: (k,)
        - k (int): number of bearings.

        Returns:
        - b (np.ndarray): computed b vector. Shape: (2 * k,)
        """
        b = np.zeros(2 * k)

        for i in range(k):
            b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i]
            b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i]

        return b

    @staticmethod
    def orthogonal_procrustes(Rgt: np.ndarray) -> np.ndarray:
        """
        Perform the orthogonal Procrustes analysis to find the optimal rotation matrix.

        Parameters:
        - Rgt (np.ndarray): A 2D numpy array of shape (3, 3) representing the input rotation matrix.

        Returns:
        - Ropt (np.ndarray): A 2D numpy array of shape (3, 3) representing the optimal rotation matrix.
        """
        U, S, Vt = np.linalg.svd(Rgt)

        D = np.dot(Vt.T, U.T)
        if np.linalg.det(D) < 0:
            Vt[-1, :] *= -1
            D = np.dot(Vt.T, U.T)

        Ropt = D.T
        return Ropt

    @staticmethod
    def compute_condition_number(A: np.ndarray) -> float:
        """
        Compute the condition number of a matrix.

        Parameters:
        - A (np.ndarray): Input matrix

        Returns:
        - float: Condition number of the matrix
        """
        return np.linalg.cond(A)

    @staticmethod
    def solve_with_regularization(A: np.ndarray, b: np.ndarray, regularization: float = None) -> np.ndarray:
        """
        Solve Ax = b with Tikhonov regularization to improve robustness under ill-conditioned scenarios.
        Uses SVD-based approach for numerical stability.

        Parameters:
        - A (np.ndarray): Coefficient matrix, shape (m, n)
        - b (np.ndarray): Right-hand side vector, shape (m,)
        - regularization (float, optional): Regularization parameter λ. If None, computed adaptively.

        Returns:
        - x (np.ndarray): Solution vector, shape (n,)
        """
        # Compute SVD: A = U S V^T
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        if regularization is None:
            # Adaptive regularization based on singular values
            # Set λ to be a fraction of the smallest non-zero singular value
            epsilon = 1e-10
            S_nonzero = S[S > epsilon]
            if len(S_nonzero) > 0:
                regularization = 0.01 * np.min(S_nonzero)
            else:
                regularization = 1e-6
        
        # Apply Tikhonov regularization using SVD
        # Solution: x = V (S^T S + λI)^{-1} S^T U^T b
        # Which simplifies to: x = V D U^T b, where D_ii = S_i / (S_i^2 + λ)
        
        # Compute regularized diagonal matrix
        D = S / (S**2 + regularization)
        
        # Compute solution: x = V D U^T b
        x = Vt.T @ (D * (U.T @ b))
        
        return x

    @staticmethod
    @timeit
    def ransac_solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray,
                     num_iterations: int = 500, threshold: float = 1e-2)-> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the bearing-only problem using RANSAC.

        Parameters:
        - uvw (np.ndarray): Array of sensor positions. Shape: (3, n)
        - xyz (np.ndarray): Array of target positions. Shape: (3, n)
        - bearing (np.ndarray): Array of bearing angles. Shape: (3, n)
        - num_iterations (int): The number of RANSAC iterations.
        - threshold (float): The threshold for the RANSAC algorithm.

        Returns:
        - R (np.ndarray): The rotation matrix. Shape: (3, 3)
        - t (np.ndarray): The translation vector. Shape: (3,)
        - error (float): The error of the solution.
        """
        # Initialize the best error to a large value
        best_error = -np.inf
        best_R = None
        best_t = None
        min_num_points = 6
        best_inliers = None
        for i in range(0, num_iterations):
            # Randomly sample 4 points
            idx = np.random.choice(uvw.shape[1], min_num_points, replace=False)
            uvw_sample = uvw[:, idx]
            xyz_sample = xyz[:, idx]
            bearing_sample = bearing[:, idx]

            # Solve the bearing-only problem
            (R, t), time = bearing_linear_solver.solve(uvw_sample, xyz_sample, bearing_sample)

            # Compute the error - vectorized version
            vec = R @ uvw + t[:, None] - xyz
            vec_norm = np.linalg.norm(vec, axis=0, keepdims=True)
            bearing_recomputed = vec / (vec_norm + 1e-10)

            bearing_angle_est = to_bearing_angle(bearing_recomputed)
            bearing_angle_in = to_bearing_angle(bearing)

            error = np.linalg.norm(bearing_angle_est - bearing_angle_in, axis=1)
            num_inlier = (error < threshold).sum()
            inliers = np.where(error < threshold)[0]

            # Check if the error is less than the threshold
            if num_inlier > best_error:
                best_error = num_inlier
                best_R = R
                best_t = t
                best_inliers = inliers

        # Refine the solution using all inliers
        if best_inliers is not None:
            uvw_inliers = uvw[:, best_inliers]
            xyz_inliers = xyz[:, best_inliers]
            bearing_inliers = bearing[:, best_inliers]
            (R, t), time = bearing_linear_solver.solve(uvw_inliers, xyz_inliers, bearing_inliers)
            return R, t
        else:
            return best_R, best_t

    @staticmethod
    @timeit
    def ransac_solve_with_sdp_sdr(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray,
                                    num_iterations: int = 500, threshold: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the bearing-only problem using RANSAC.

        Parameters:
        - uvw (np.ndarray): Array of sensor positions. Shape: (3, n)
        - xyz (np.ndarray): Array of target positions. Shape: (3, n)
        - bearing (np.ndarray): Array of bearing angles. Shape: (3, n)
        - num_iterations (int): The number of RANSAC iterations.
        - threshold (float): The threshold for the RANSAC algorithm.

        Returns:
        - R (np.ndarray): The rotation matrix. Shape: (3, 3)
        - t (np.ndarray): The translation vector. Shape: (3,)
        - error (float): The error of the solution.
        """
        # Initialize the best error to a large value
        best_error = -np.inf
        best_R = None
        best_t = None
        min_num_points = 6
        best_inliers = None
        for i in range(0, num_iterations):
            # Randomly sample 4 points
            idx = np.random.choice(uvw.shape[1], min_num_points, replace=False)
            uvw_sample = uvw[:, idx]
            xyz_sample = xyz[:, idx]
            bearing_sample = bearing[:, idx]

            # Solve the bearing-only problem
            (R, t), time = bearing_linear_solver.solve_with_sdp_sdr(uvw_sample, xyz_sample, bearing_sample)

            # Compute the error - vectorized version
            vec = R @ uvw + t[:, None] - xyz
            vec_norm = np.linalg.norm(vec, axis=0, keepdims=True)
            bearing_recomputed = vec / (vec_norm + 1e-10)

            bearing_angle_est = to_bearing_angle(bearing_recomputed)
            bearing_angle_in = to_bearing_angle(bearing)

            error = np.linalg.norm(bearing_angle_est - bearing_angle_in, axis=1)
            # logger.warning(f'error: {error}')
            num_inlier = (error < threshold).sum()
            inliers = np.where(error < threshold)[0]
            # logger.warning(f'Num inliers: {num_inlier}')
            # Check if the error is less than the threshold
            if num_inlier > best_error:
                best_error = num_inlier
                best_R = R
                best_t = t
                best_inliers = inliers

        if best_inliers is not None:
            # Refine the solution using all inliers
            uvw_inliers = uvw[:, best_inliers]
            xyz_inliers = xyz[:, best_inliers]
            bearing_inliers = bearing[:, best_inliers]
            (R, t), time = bearing_linear_solver.solve_with_sdp_sdr(uvw_inliers, xyz_inliers, bearing_inliers)

            return R, t
        else:
            return best_R, best_t

    @staticmethod
    @timeit
    def solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the bearing-only problem given the sensor positions, target positions, and bearing angles.

        Parameters:
        - uvw (np.ndarray): Array of sensor positions. Shape: (3, n)
        - xyz (np.ndarray): Array of target positions. Shape: (3, n)
        - bearing (np.ndarray): Array of bearing angles. Shape: (3, n)

        Returns:
        - R (np.ndarray): The rotation matrix. Shape: (3, 3)
        - t (np.ndarray): The translation vector. Shape: (3,)
        """
        bearing_angle = np.zeros((2, bearing.shape[1]))
        for i in range(bearing.shape[1]):
            vec = bearing[:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])

        A = bearing_linear_solver.compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        b = bearing_linear_solver.compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])

        # Check condition number to assess numerical stability
        cond_num = bearing_linear_solver.compute_condition_number(A)
        logger.debug(f'Condition number of A matrix: {cond_num:.2e}')
        
        # Solve for x using least squares with automatic regularization for ill-conditioned matrices
        # Use threshold of 1e10 to detect ill-conditioned matrices
        from scipy.linalg import solve, lstsq
        if cond_num > 1e10:
            logger.info(f'Ill-conditioned matrix detected (cond={cond_num:.2e}). Applying Tikhonov regularization.')
            x = bearing_linear_solver.solve_with_regularization(A, b)
        else:
            # x = solve(A, b)
            x = lstsq(A, b)[0]
        logger.debug(f'Solution x: {x}')

        R = x[:9].reshape(3, 3)
        R = bearing_linear_solver.orthogonal_procrustes(R)

        # A1, b1 = bearing_linear_solver.compute_reduced_Ab_matrix(uvw[0,:], uvw[1,:], uvw[2,:],
        #                                                      bearing_angle[1,:],
        #                                                      bearing_angle[0,:],
        #                                                      bearing_angle.shape[1],
        #                                                      xyz[0,:], xyz[1,:], xyz[2,:],R)
        # x = lstsq(A1, b1)[0]
        t = x[9:]
        logger.debug(f'R: {R}')
        return R, t

    @staticmethod
    @timeit
    def solve_with_sdp_sdr(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        bearing_angle = np.zeros((2, bearing.shape[1]))
        for i in range(bearing.shape[1]):
            vec = bearing[:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])


        A = bearing_linear_solver.compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        b = bearing_linear_solver.compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])

        P1 = np.hstack([A, b.reshape(-1,1)])
        # P is 13 x 13
        P = P1.T@P1

        Q = []
        ndim = P.shape[0]
        """
        generate matrix for the following constraints
        [ a1^2 + a2^2 + a3^2, - 1
          a1*a4 + a2*a5 + a3*a6,
          a1*a7 + a2*a8 + a3*a9]
        [   a4^2 + a5^2 + a6^2 - 1,
            a4*a7 + a5*a8 + a6*a9]
        [   a7^2 + a8^2 + a9^2 - 1]
        """
        Q1 = np.zeros((ndim,ndim))
        Q1[[0,1,2],[0,1,2]] = 1; Q1[12,12] = -1
        Q2 = np.zeros((ndim,ndim))
        Q2[[0,1,2],[3,4,5]] = 1
        Q3 = np.zeros((ndim,ndim))
        Q3[[0,1,2],[6,7,8]] = 1
        Q4 = np.zeros((ndim,ndim))
        Q4[[3,4,5],[3,4,5]] = 1; Q4[12,12] = -1
        Q5 = np.zeros((ndim,ndim))
        Q5[[3,4,5],[6,7,8]] = 1
        Q6 = np.zeros((ndim,ndim))
        Q6[[6,7,8],[6,7,8]] = 1; Q6[12,12] = -1
        Q.append(Q1)
        Q.append(Q2)
        Q.append(Q3)
        Q.append(Q4)
        Q.append(Q5)
        Q.append(Q6)

        """
        generate matrix for the following constraints
        [ a1^2 + a4^2 + a7^2 - 1,
          a1*a2 + a4*a5 + a7*a8,
          a1*a3 + a4*a6 + a7*a9]
        [   a2^2 + a5^2 + a8^2 - 1,
            a2*a3 + a5*a6 + a8*a9]
        [   a3^2 + a6^2 + a9^2 - 1]
        """
        Q7 = np.zeros((ndim,ndim))
        Q7[[0,3,6],[0,3,6]] = 1; Q7[12,12] = -1
        Q8 = np.zeros((ndim,ndim))
        Q8[[0,3,6],[1,4,7]] = 1
        Q9 = np.zeros((ndim,ndim))
        Q9[[0,3,6],[2,5,8]] = 1
        Q10 = np.zeros((ndim,ndim))
        Q10[[1,4,7],[1,4,7]] = 1; Q10[12,12] = -1
        Q11 = np.zeros((ndim,ndim))
        Q11[[1,4,7],[2,5,8]] = 1
        Q12 = np.zeros((ndim,ndim))
        Q12[[2,5,8],[2,5,8]] = 1; Q12[12,12] = -1
        Q.append(Q7)
        Q.append(Q8)
        Q.append(Q9)
        Q.append(Q10)
        Q.append(Q11)
        Q.append(Q12)
        """
        generate matrix for the following constraints
        [a1 - a5*a9 + a6*a8,
         a2 + a4*a9 - a6*a7,
         a3 - a4*a8 + a5*a7]
        """
        Q13 = np.zeros((ndim,ndim))
        Q13[0,12] = -1; Q13[4,8] = -1; Q13[5,7] = 1
        Q14 = np.zeros((ndim,ndim))
        Q14[1,12] = -1; Q14[3,8] =  1; Q14[5,6] = -1
        Q15 = np.zeros((ndim,ndim))
        Q15[2,12] = -1; Q15[3,7] = -1; Q15[4,6] = 1
        Q.append(Q13)
        Q.append(Q14)
        Q.append(Q15)
        """
        generate matrix for the following constraints
        [a4 + a2*a9 - a3*a8,
         a5 - a1*a9 + a3*a7,
         a6 + a1*a8 - a2*a7]
        """
        Q16 = np.zeros((ndim,ndim))
        Q16[3,12] = -1; Q16[1,8] =  1; Q16[2,7] = -1
        Q17 = np.zeros((ndim,ndim))
        Q17[4,12] = -1; Q17[0,8] =  -1; Q17[2,6] = 1
        Q18 = np.zeros((ndim,ndim))
        Q18[5,12] = -1; Q18[0,7] = 1; Q18[1,6] = -1
        Q.append(Q16)
        Q.append(Q17)
        Q.append(Q18)
        """
        generate matrix for the following constraints
        [a7 - a2*a6 + a3*a5,
         a8 + a1*a6 - a3*a4,
         a9 - a1*a5 + a2*a4]
        """
        Q19 = np.zeros((ndim,ndim))
        Q19[6,12] = -1; Q19[1,5] = -1; Q19[2,4] = 1
        Q20 = np.zeros((ndim,ndim))
        Q20[7,12] = -1; Q20[0,5] =  1; Q20[2,3] = -1
        Q21 = np.zeros((ndim,ndim))
        Q21[8,12] = -1; Q21[0,4] = -1; Q21[1,3] = 1
        Q.append(Q19)
        Q.append(Q20)
        Q.append(Q21)
        Q22 = np.zeros((ndim, ndim))
        Q22[12,12] = 1
        Q.append(Q22)
        # [  a1^2,  a1*a2,  a1*a3,  a1*a4,  a1*a5,  a1*a6,  a1*a7,  a1*a8,  a1*a9,  a1*a10,  a1*a11,  a1*a12,  -a1]
        # [ a1*a2,   a2^2,  a2*a3,  a2*a4,  a2*a5,  a2*a6,  a2*a7,  a2*a8,  a2*a9,  a2*a10,  a2*a11,  a2*a12,  -a2]
        # [ a1*a3,  a2*a3,   a3^2,  a3*a4,  a3*a5,  a3*a6,  a3*a7,  a3*a8,  a3*a9,  a3*a10,  a3*a11,  a3*a12,  -a3]
        # [ a1*a4,  a2*a4,  a3*a4,   a4^2,  a4*a5,  a4*a6,  a4*a7,  a4*a8,  a4*a9,  a4*a10,  a4*a11,  a4*a12,  -a4]
        # [ a1*a5,  a2*a5,  a3*a5,  a4*a5,   a5^2,  a5*a6,  a5*a7,  a5*a8,  a5*a9,  a5*a10,  a5*a11,  a5*a12,  -a5]
        # [ a1*a6,  a2*a6,  a3*a6,  a4*a6,  a5*a6,   a6^2,  a6*a7,  a6*a8,  a6*a9,  a6*a10,  a6*a11,  a6*a12,  -a6]
        # [ a1*a7,  a2*a7,  a3*a7,  a4*a7,  a5*a7,  a6*a7,   a7^2,  a7*a8,  a7*a9,  a7*a10,  a7*a11,  a7*a12,  -a7]
        # [ a1*a8,  a2*a8,  a3*a8,  a4*a8,  a5*a8,  a6*a8,  a7*a8,   a8^2,  a8*a9,  a8*a10,  a8*a11,  a8*a12,  -a8]
        # [ a1*a9,  a2*a9,  a3*a9,  a4*a9,  a5*a9,  a6*a9,  a7*a9,  a8*a9,   a9^2,  a9*a10,  a9*a11,  a9*a12,  -a9]
        # [a1*a10, a2*a10, a3*a10, a4*a10, a5*a10, a6*a10, a7*a10, a8*a10, a9*a10,   a10^2, a10*a11, a10*a12, -a10]
        # [a1*a11, a2*a11, a3*a11, a4*a11, a5*a11, a6*a11, a7*a11, a8*a11, a9*a11, a10*a11,   a11^2, a11*a12, -a11]
        # [a1*a12, a2*a12, a3*a12, a4*a12, a5*a12, a6*a12, a7*a12, a8*a12, a9*a12, a10*a12, a11*a12,   a12^2, -a12]
        # [   -a1,    -a2,    -a3,    -a4,    -a5,    -a6,    -a7,    -a8,    -a9,    -a10,    -a11,    -a12,    1]
        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((ndim,ndim), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        constraints += [
            cp.trace(Q[i] @ X) == 0 for i in range(21)
        ]
        constraints += [cp.trace(Q[-1] @ X) == 1]
        # logger.warning(cp.trace(Q[-1]@X))
        logger.debug(constraints)
        prob = cp.Problem(cp.Minimize(cp.trace(P @ X)),
                  constraints)
        prob.solve(solver = cp.MOSEK)

        # Print result.
        logger.debug("The optimal value is %s", prob.value)
        logger.debug("A solution X is")
        logger.debug(X.value)
        sol = X.value

        if sol is None:
            raise RuntimeError("SDP returned no solution matrix.")
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("SDP returned a nonfinite solution matrix.")

        # Preserve relative signs via the dominant eigenvector, not sqrt(diag(X)).
        sol_symmetric = 0.5 * (sol + sol.T)
        eigenvalues, eigenvectors = np.linalg.eigh(sol_symmetric)
        lambda_max = float(eigenvalues[-1])
        if lambda_max <= 0.0:
            raise RuntimeError("SDP solution has no positive principal eigenvalue.")
        z = np.sqrt(lambda_max) * eigenvectors[:, -1]

        # P1 = [A, b] represents A*psi-b=0 with z=[psi; -1].
        # Dividing by z[-1] also cancels the eigenvector's arbitrary global sign.
        if abs(float(z[-1])) < 1e-10:
            raise RuntimeError(
                "Cannot recover SDP pose: homogeneous component is near zero."
            )
        psi = -z[:-1] / z[-1]
        R = bearing_linear_solver.orthogonal_procrustes(psi[:9].reshape(3, 3))
        t = psi[9:12]
        logger.debug("Largest SDP eigenvalue: %.6e", lambda_max)
        logger.debug("Recovered pose vector: %s", psi)

        return R, t




class bgpnp():
    def __init__(self) -> None:
        pass

    @staticmethod
    def validate_inputs(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, min_points: int = 4) -> None:
        """
        Validate input data for BGPnP algorithm.
        
        Args:
            p1 (np.ndarray): The 3D coordinates of the points in global frame.
            p2 (np.ndarray): The 3D coordinates of the points in local frame.
            bearing (np.ndarray): The bearing vectors.
            min_points (int): Minimum number of points required.
            
        Raises:
            ValueError: If inputs are invalid.
        """
        if p1.shape[0] < min_points:
            raise ValueError(f"Insufficient points: need at least {min_points}, got {p1.shape[0]}")
        
        if p1.shape != p2.shape or p1.shape != bearing.shape:
            raise ValueError(f"Shape mismatch: p1={p1.shape}, p2={p2.shape}, bearing={bearing.shape}")
        
        if p1.shape[1] != 3:
            raise ValueError(f"Expected 3D points (n, 3), got shape {p1.shape}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(p1)) or np.any(np.isinf(p1)):
            raise ValueError("p1 contains NaN or Inf values")
        if np.any(np.isnan(p2)) or np.any(np.isinf(p2)):
            raise ValueError("p2 contains NaN or Inf values")
        if np.any(np.isnan(bearing)) or np.any(np.isinf(bearing)):
            raise ValueError("bearing contains NaN or Inf values")
        
        # Check that bearing vectors are non-zero
        bearing_norms = np.linalg.norm(bearing, axis=1)
        if np.any(bearing_norms < 1e-10):
            raise ValueError("bearing contains zero or near-zero vectors")

    @staticmethod
    @timeit
    def ransac_solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray,
                        num_iterations: int = 500, threshold: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, float]:
        # Validate inputs
        bgpnp.validate_inputs(p1, p2, bearing, min_points=6)
        
        # Initialize the best error to a large value
        best_error = -np.inf
        best_R = None
        best_t = None
        min_num_points = 6
        best_inliers = None
        for i in range(0, num_iterations):
            # Randomly sample 4 points
            idx = np.random.choice(p1.shape[0], min_num_points, replace=False)
            p1_sample = p1[idx]
            p2_sample = p2[idx]
            bearing_sample = bearing[idx]

            # Solve the BGPnP problem
            (R, t, error), time = bgpnp.solve(p1_sample, p2_sample, bearing_sample, sol_iter=False)

            # compute the error - vectorized version
            vec = R @ p1.T + t[:, None] - p2.T
            vec_norm = np.linalg.norm(vec, axis=0, keepdims=True)
            bearing_recomputed = (vec / (vec_norm + 1e-10)).T

            bearing_angle_est = to_bearing_angle(bearing_recomputed)
            bearing_angle_in  = to_bearing_angle(bearing)

            error = np.linalg.norm(bearing_angle_est - bearing_angle_in, axis=1)
            # logger.warning(f'Error: {error}')
            num_inlier = (error < threshold).sum()
            inliers = np.where(error < threshold)[0]
            # logger.warning(f'Error: {error}, Num inlier: {num_inlier}')

            # Check if the error is less than the threshold
            if num_inlier > best_error:
                best_error = num_inlier
                best_R = R
                best_t = t
                best_inliers = inliers

        # Refit the model using the inliers
        if best_inliers is not None and len(best_inliers) >= 6:
            p1_sample = p1[best_inliers]
            p2_sample = p2[best_inliers]
            bearing_sample = bearing[best_inliers]
            (R, t, error), time = bgpnp.solve(p1_sample, p2_sample, bearing_sample, sol_iter=True)
            return R, t, error
        else:
            return best_R, best_t, 0.0

    @staticmethod
    def refine(p2: np.ndarray, bearing: np.ndarray,
               Alph: np.array, ctl_pts: np.array, Cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # do a nonlinear least squares optimization
        def fun(x, p2, bearing, alph):
            M = bgpnp.kron_A_N(alph, 3) # 3n x 12
            bearing_computed = M @ x
            bearing_computed = bearing_computed.reshape(-1, 3) - p2
            bearing_computed = bearing_computed / np.linalg.norm(bearing_computed, axis=1)[:, None]
            error = np.linalg.norm(bearing_computed - bearing, axis=1)
            # logger.warning(f'error: {error}')
            return error

        def fun2(x, p2, bearing, alph):
            M = bgpnp.kron_A_N(alph, 3) # 3n x 12
            bearing_computed = M @ x
            bearing_computed = bearing_computed.reshape(-1, 3) - p2
            bearing_computed = bearing_computed / np.linalg.norm(bearing_computed, axis=1)[:, None]
            # do cross-product for each bearing
            cross_products = np.cross(bearing_computed, bearing)
            error = cross_products.reshape(-1)
            # error = np.linalg.norm(cross_products, axis=1)
            return error

        x0 = ctl_pts.reshape(-1)
        res = least_squares(fun2, x0, args=(p2, bearing, Alph), verbose=2)
        x = res.x

        X = {}
        X['P'] = Cw.T
        X['mP'] = np.mean(X['P'], axis=1)
        X['cP'] = X['P'] - X['mP'].reshape(3, 1)

        X['norm'] = np.linalg.norm(X['cP'])
        X['nP'] = X['cP'] / X['norm']

        # procrustes solution for the first kernel vector
        dims = 4
        vK = x.reshape(dims, -1).T
        R, b, mc = bgpnp.myProcrustes(X, vK)
        solV = b * vK
        mV = np.mean(solV, axis=1)

        T = mV - R @ X['mP']
        return R, T

    @staticmethod
    @timeit
    def solve_new_loss(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        # Validate inputs
        bgpnp.validate_inputs(p1, p2, bearing, min_points=4)
        
        M, b, Alph, Cw = bgpnp.prepare_data_new_loss(p1, bearing, p2)
        possible_dims = 6
        Km = bgpnp.kernel_noise(M, b, dimker=possible_dims, use_regularization=enforce_manifold)
        R, t, err, _ = bgpnp.KernelPnP(Cw, Km, dims=4, sol_iter=sol_iter, enforce_manifold=enforce_manifold)

        return R, t, err


    @staticmethod
    @timeit
    def solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the Bearing Generalized Perspective-n-Point (BGPnP) algorithm.

        Args:
            p1 (np.ndarray): The 3D coordinates of the points in global frame.
            p2 (np.ndarray): The 3D coordinates of the points in local frame.
            bearing (np.ndarray): The bearing angles of the points.
            sol_iter (bool): Flag indicating whether to perform iterative refinement.
            enforce_manifold (bool): If True, enforces SO(3) manifold constraints with regularization
                                    for improved robustness in noisy conditions. This adds constraints
                                    similar to SDP-SDR but is slower. (default: False for compatibility)

        Returns:
            tuple: A tuple containing the rotation matrix (R), translation vector (T), and error (err).
        """
        # Validate inputs
        bgpnp.validate_inputs(p1, p2, bearing, min_points=4)
        
        M, b, Alph, Cw = bgpnp.prepare_data(p1, bearing, p2)
        possible_dims = 4
        # Use regularized kernel computation for better noise robustness when enforce_manifold is True
        Km = bgpnp.kernel_noise(M, b, dimker=possible_dims, use_regularization=enforce_manifold)
        R, t, err, _ = bgpnp.KernelPnP(Cw, Km, dims=4, sol_iter=sol_iter, enforce_manifold=enforce_manifold)

        return R, t, err

    @staticmethod
    def define_control_points() -> np.ndarray:
        """
        Define control points.

        Returns:
            np.ndarray: A 4x3 matrix of control points.
        """
        Cw = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        return Cw

    @staticmethod
    def compute_alphas(Xw:np.ndarray, Cw:np.ndarray) -> np.ndarray:
        """
        Compute alphas (linear combination of the control points to represent the 3D points).

        Args:
            Xw (np.ndarray): 3D points, shape (n, 3).
            Cw (np.ndarray): Control points, shape (4, 3).

        Returns:
            np.ndarray: Alphas.
        """
        n = Xw.shape[0]  # number of 3D points
        logger.debug(f'Xw: {Xw.shape}')
        logger.debug(f'Cw: {Cw.shape}')

        # Generate auxiliary matrix to compute alphas
        C = np.vstack((Cw.T, np.ones((1, 4)))) # 4x4
        X = np.vstack((Xw.T, np.ones((1, n)))) # 4xn

        logger.debug(f'C: {C.shape}')
        logger.debug(f'C: {C}')

        # Alph_ = np.linalg.inv(C) @ X # 4xn
        Alph_ = np.linalg.solve(C, X)  # 4xn

        Alph = Alph_.T # nx4
        return Alph

    @staticmethod
    def myProcrustes(X, Y):
        """
        Perform Procrustes analysis to find the best transformation between two sets of points.

        Parameters:
        - X: Dictionary containing the reference points.
        - Y: Array-like object containing the target points.

        Returns:
        - R: The rotation matrix.
        - b: The scaling factor.
        - mc: The translation vector.

        The function calculates the best rotation, scaling, and translation that aligns the target points (Y)
        with the reference points (X). It uses the Procrustes analysis method to find the optimal transformation.

        The reference points (X) should be provided as a dictionary with the following keys:
        - 'nP': The normalized reference points.
        - 'norm': The normalization factor.
        - 'mP': The mean of the reference points.

        The target points (Y) should be an array-like object with shape (n, d), where n is the number of points
        and d is the number of dimensions.

        The function returns the rotation matrix (R), scaling factor (b), and translation vector (mc) that
        transform the target points (Y) to align with the reference points (X).

        Note: The function assumes that the number of dimensions in the target points (Y) is the same as the
        number of dimensions in the reference points (X).

        """
        dims = Y.shape[1]
        mY = np.mean(Y, axis=1, keepdims=True)
        cY = Y - mY
        ncY = np.linalg.norm(cY)
        tcY = cY / ncY

        A = np.dot(X['nP'], tcY.T)
        L, D, Mt = np.linalg.svd(A, full_matrices=False)

        R = Mt.T @ np.diag([1, 1, np.sign(np.linalg.det(Mt.T @ L.T))]) @ L.T
        logger.debug(f'R: {R}')

        b = np.sum(np.diag(D)) * X['norm'] / ncY
        logger.debug(f'b: {b}')
        c = X['mP'] - b * (R.T @ mY).flatten()
        logger.debug(f'c: {c}')

        mc = np.tile(c, (dims,1)).T
        logger.debug(f'mc: {mc}')

        return R, b, mc

    @staticmethod
    def project_to_SO3(R: np.ndarray, enforce_det: bool = True) -> np.ndarray:
        """
        Project a matrix to the SO(3) manifold (proper rotation matrices).
        
        This method enforces the manifold constraints:
        - Orthogonality: R^T R = I
        - Determinant: det(R) = +1 (if enforce_det=True)
        
        Args:
            R (np.ndarray): Input 3x3 matrix to project
            enforce_det (bool): If True, ensures det(R) = +1 (proper rotation)
                               If False, allows det(R) = -1 (reflection)
        
        Returns:
            np.ndarray: Projected rotation matrix on SO(3)
        """
        # Use SVD to find nearest orthogonal matrix
        U, S, Vt = np.linalg.svd(R)
        
        if enforce_det:
            # Ensure determinant is +1 (proper rotation, not reflection)
            # This is a key constraint that SDP-SDR enforces via Q13-Q21
            det_UV = np.linalg.det(U @ Vt)
            if det_UV < 0:
                # Flip the sign of the last column of U to ensure det = +1
                U[:, -1] *= -1
        
        # Compute nearest orthogonal matrix: R_optimal = U @ Vt
        R_optimal = U @ Vt
        
        return R_optimal

    @staticmethod
    def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Validate that a matrix satisfies rotation matrix properties.
        
        Checks:
        - R is 3x3
        - R^T R ≈ I (orthogonality)
        - det(R) ≈ +1 (proper rotation)
        
        Args:
            R (np.ndarray): Matrix to validate
            tol (float): Tolerance for checks
            
        Returns:
            bool: True if R is a valid rotation matrix
        """
        if R.shape != (3, 3):
            return False
        
        # Check orthogonality: R^T R = I
        RtR = R.T @ R
        I = np.eye(3)
        if not np.allclose(RtR, I, atol=tol):
            logger.debug(f'Orthogonality check failed: ||R^T R - I|| = {np.linalg.norm(RtR - I)}')
            return False
        
        # Check determinant = +1
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tol):
            logger.debug(f'Determinant check failed: det(R) = {det}')
            return False
        
        return True

    @staticmethod
    def KernelPnP(Cw: np.ndarray, Km: np.ndarray, dims: int = 4, sol_iter: bool = True, tol = 1e-6, enforce_manifold: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes the Kernel Perspective-n-Point (KernelPnP) algorithm.

        Args:
            Cw (numpy.ndarray): The 3D world coordinates of the points.
            Km (numpy.ndarray): The kernel matrix.
            dims (int): The number of dimensions.
            sol_iter (bool): Flag indicating whether to perform iterative refinement.
            tol (float): Tolerance for convergence.
            enforce_manifold (bool): If True, enforces SO(3) manifold constraints for robustness.

        Returns:
            tuple: A tuple containing the rotation matrix (R), translation vector (T), and error (err).
        """
        vK = Km[:, -1].reshape(dims, -1).T
        logger.debug(f'vK: {vK}')
        # precomputations
        X = {}
        X['P'] = Cw.T
        X['mP'] = np.mean(X['P'], axis=1)
        X['cP'] = X['P'] - X['mP'].reshape(3, 1)

        X['norm'] = np.linalg.norm(X['cP'])
        X['nP'] = X['cP'] / X['norm']

        # procrustes solution for the first kernel vector
        R, b, mc = bgpnp.myProcrustes(X, vK)
        
        # Apply SO(3) manifold projection for robustness in noisy conditions
        if enforce_manifold:
            R = bgpnp.project_to_SO3(R, enforce_det=True)
            logger.debug(f'Initial R projected to SO(3), det(R)={np.linalg.det(R):.6f}')

        solV = b * vK
        solR = R
        solmc = mc
        solVec = Km[:, -1]
        # procrustes solution using 4 kernel eigenvectors
        err = np.inf
        if sol_iter:
            n_iterations = 500
            for iter in range(n_iterations):
                # projection of previous solution into the null space
                A = R @ (-mc + X['P'])
                abcd = np.linalg.lstsq(Km, A.T.flatten(), rcond=None)[0]
                newV = np.reshape(Km @ abcd, (dims, -1)).T

                logger.debug(f'Iteration: {iter}')
                logger.debug(f'A: {A}')

                # euclidean error
                newerr = np.linalg.norm(R.T @ newV + mc - X['P'],2)
                logger.debug(f'newerr: {newerr}')

                if ((newerr > err) and (iter > 2)) or newerr < tol:
                    logger.debug(f'Converged after {iter} iterations.')
                    break
                else:
                    # procrustes solution
                    R, b, mc = bgpnp.myProcrustes(X, newV)
                    
                    # Apply SO(3) manifold projection for robustness
                    if enforce_manifold:
                        R = bgpnp.project_to_SO3(R, enforce_det=True)
                        # Validate rotation matrix periodically (every 50 iterations)
                        # This frequency balances validation overhead vs early error detection
                        if iter % 50 == 0:
                            is_valid = bgpnp.validate_rotation_matrix(R, tol=1e-5)
                            logger.debug(f'Iteration {iter}: R valid={is_valid}')
                    
                    solV = b * newV
                    solVec = newV
                    solmc = mc
                    solR = R
                    err = newerr

        R = solR
        
        # Final manifold projection to ensure solution is on SO(3)
        if enforce_manifold:
            R = bgpnp.project_to_SO3(R, enforce_det=True)
            logger.debug(f'Final R projected to SO(3), det(R)={np.linalg.det(R):.6f}')
        
        mV = np.mean(solV, axis=1)

        T = mV - R @ X['mP']
        logger.debug(f'Final solution: {R}, {T}')
        return R, T, err, solVec

    @staticmethod
    def kernel_noise(M: np.ndarray, b: np.ndarray, dimker: int = 4, use_regularization: bool = False, reg_lambda: float = 0.0) -> np.ndarray:
        """
        Computes the kernel noise matrix for a given input matrix M and vector b.

        Parameters:
        - M: Input matrix of shape (3n, 12)
        - b: Input vector of shape (3n,)
        - dimker: Dimension of the kernel noise matrix (default: 4)
        - use_regularization: If True, uses Tikhonov regularization for better noise robustness
        - reg_lambda: Regularization parameter for Tikhonov regularization (0 = auto)

        Returns:
        - K: Kernel noise matrix of shape (12, dimker)
        """
        K = np.zeros((M.shape[1], dimker))
        U, S, V = np.linalg.svd(M, full_matrices=False)
        V = V.T
        logger.debug(f'Singular values: {S}')

        K[:, 0:dimker-1] = V[:, -dimker+1:]
        
        # Compute the last kernel vector more robustly
        if use_regularization:
            # Adaptive regularization based on condition number
            # Threshold of 1e8 chosen empirically - below this, standard lstsq is stable
            # Above 1e8, noise amplification becomes significant
            cond_M = S[0] / (S[-1] + 1e-15)
            
            # Only apply regularization if condition number is high (ill-conditioned)
            if cond_M > 1e8 or reg_lambda > 0:
                # Use Tikhonov regularization for noise robustness
                MtM = M.T @ M
                Mtb = M.T @ b
                n = MtM.shape[0]
                
                # Adaptive regularization based on smallest singular value
                if reg_lambda == 0:
                    # Use a very small fraction of the smallest singular value
                    # Tuned to 0.001 (reduced from 0.01) to balance robustness vs accuracy
                    # Smaller value reduces over-regularization on clean data
                    S_nonzero = S[S > 1e-10]
                    if len(S_nonzero) > 0:
                        reg_lambda = 0.001 * np.min(S_nonzero)
                    else:
                        reg_lambda = 1e-8
                
                MtM_reg = MtM + reg_lambda * np.eye(n)
                K[:, -1] = np.linalg.solve(MtM_reg, Mtb)
                logger.debug(f'Using regularized lstsq with lambda={reg_lambda:.6e}, cond={cond_M:.2e}')
            else:
                # Well-conditioned, use standard least squares
                K[:, -1] = lstsq(M, b)[0]
                logger.debug(f'Using standard lstsq, cond={cond_M:.2e}')
        else:
            # Standard least squares
            K[:, -1] = lstsq(M, b)[0]

        return K


    @staticmethod
    def prepare_data(p: np.ndarray, bearing: np.ndarray, pb: np.ndarray, Cw: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for the bearing-only solver.

        Args:
            p (np.ndarray): Array of sensor positions.
            bearing (np.ndarray): Array of bearing measurements.
            pb (np.ndarray): Array of target positions.
            Cw (np.ndarray, optional): Array of control points. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the computed matrices M and b, the array of alphas Alph, and the array of control points Cw.
        """
        if Cw is None:
            logger.debug('Control points not provided. Defining control points.')
            Cw = bgpnp.define_control_points()

        logger.debug(f'Cw: {Cw}')
        Alph = bgpnp.compute_alphas(p, Cw)
        M, b = bgpnp.compute_Mb(bearing, Alph, pb)

        return M, b, Alph, Cw

    @staticmethod
    def skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
        """
        Generate the skew-symmetric matrix for a vector.

        Args:
            v (np.ndarray): A vector of shape (3,).

        Returns:
            np.ndarray: A 3x3 skew-symmetric matrix.
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def kron_A_N(A, N):  # Simulates np.kron(A, np.eye(N))
        m,n = A.shape
        out = np.zeros((m,N,n,N),dtype=A.dtype)
        r = np.arange(N)
        out[:,r,:,r] = A
        out.shape = (m*N,n*N)
        return out

    @staticmethod
    def prepare_data_new_loss(p: np.ndarray, bearing: np.ndarray, pb: np.ndarray, Cw: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if Cw is None:
            logger.debug('Control points not provided. Defining control points.')
            Cw = bgpnp.define_control_points()

        logger.debug(f'Cw: {Cw}')
        Alph = bgpnp.compute_alphas(p, Cw)
        M, b = bgpnp.compute_Mb_new_loss(bearing, Alph, pb)
        # Morig, borig = bgpnp.compute_Mb(bearing, Alph, pb)
        # logger.warning(f'b: {b}, borig: {borig}')
        return M, b, Alph, Cw


    @staticmethod
    def compute_Mb_new_loss(bearing: np.ndarray, Alph: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = np.zeros((3 * bearing.shape[0], 12))
        b = np.zeros((3 * bearing.shape[0],))
        for i in range(bearing.shape[0]):
            S = np.eye(3) - bearing[i].reshape(3, 1) @ bearing[i].reshape(1, 3) / np.linalg.norm(bearing[i]) / np.linalg.norm(bearing[i])
            M[3 * i:3 * i + 3, :] = S @ np.kron(Alph[i], np.eye(3))
            b[3 * i:3 * i + 3] = S.dot(p2[i])
        return M, b


    @staticmethod
    def compute_Mb(bearing: np.ndarray, Alph: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the M and b matrix.

        Args:
            bearing (np.ndarray): Bearing vector, shape (n, 3).
            Alph (np.ndarray): Alphas, shape (n, 4).
            p2 (np.ndarray): 3D points, shape (n, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - M (np.ndarray): The M matrix of shape (3n, 12).
                - b (np.ndarray): The b vector of shape (3n,).
        """
        n = bearing.shape[0]
        M = bgpnp.kron_A_N(Alph, 3)
        b = np.zeros(3 * n)

        # Compute skew-symmetric matrices for all bearings
        S = np.array([bgpnp.skew_symmetric_matrix(bearing_i) for bearing_i in bearing])

        # Vectorized computation of M
        M = np.einsum('ijk,ikl->ijl', S, M.reshape(n, 3, 12)).reshape(3 * n, 12)

        # Vectorized computation of b
        b = np.einsum('ijk,ik->ij', S, p2).reshape(3 * n)

        return M, b


def bearing_only_solver(folder: str, file: str):
    """
    Solve the bearing-only problem given a folder and a file.

    Args:
        folder (str): The folder path where the files are located.
        file (str): The file name to search for in the folder.

    Returns:
        None
    """
    import sophuspy as sp

    files = [os.path.join(folder, f) for f in os.listdir(folder) if file in f]

    solver = bearing_linear_solver()

    for f in files:
        data = load_simulation_data(f)
        logger.debug(data["p1"])
        logger.debug(data["p2"].shape)
        logger.debug(data["Rgt"])
        logger.debug(data["tgt"])

        uvw = data["p1"]
        xyz = data["p2"]
        bearing = data["bearing"]
        std_noise_on_theta = 0 * np.pi / 180
        bearing_angle = np.zeros((2, bearing.shape[1]))
        for j in range(bearing.shape[1]):
            vec = bearing[:, j]
            phi = asin(vec[2]) + np.random.randn() * std_noise_on_theta
            theta = atan2(vec[1], vec[0]) + np.random.randn() * std_noise_on_theta
            bearing_angle[:, j] = np.array([theta, phi])

        f1 = lambda x: np.array([cos(x[0]) * cos(x[1]), cos(x[1]) * sin(x[0]), sin(x[1])])
        bearing = [f1(bearing_angle[:, j]) for j in range(bearing_angle.shape[1])]
        std_noise_on_bearing = 1e-3
        Rperb = sp.SO3.exp(np.random.normal(0, std_noise_on_bearing, size=(3,))).matrix()
        logger.warning(f'Rperb: {Rperb}')
        bearing_noise = []
        for i, b in enumerate(bearing):
            logger.warning(f'Original bearing: {b}')
            b = Rperb.dot(b)
            logger.warning(f'Noisy bearing: {b}')
            # b = b / np.linalg.norm(b)
            bearing_noise.append(b)
        bearing = np.array(bearing_noise).T

        (R, t), time = solver.solve(uvw, xyz, bearing)

        logger.info(f'Solution R: {R}')
        logger.info(f't: {t}')

        (R2, t2, err), time = bgpnp.solve_new_loss(uvw.T, xyz.T, bearing.T, sol_iter=True)
        logger.info(f'Solution R: {R2}')
        logger.info(f't: {t2}')

        (R1, t1), time = solver.solve_with_sdp_sdr(uvw, xyz, bearing)
        logger.info(f'Solution R: {R1}')
        logger.info(f't: {t1}')

if __name__ == "__main__":
    bearing_only_solver('../taes/', 'simu_')
