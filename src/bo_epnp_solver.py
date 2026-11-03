"""
BO-EPnP solver for bearing-only cooperative localization.

This module follows QUICK_START_GUIDE.md and provides:
1) BO-EPnP core solver
2) Sphere-aware loss (orthogonal projection operator)
3) WTLS initializer
4) ML-style nonlinear refinement
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    from bearing_only_solver import bgpnp
except ModuleNotFoundError:  # pragma: no cover - import fallback for repo-root test runs
    from src.bearing_only_solver import bgpnp

logger = logging.getLogger(__name__)


class BO_EPnP:
    """
    Bearing-only EPnP-style solver with optional WTLS and ML refinement.
    """

    @staticmethod
    def solve(
        uvw: np.ndarray,
        xyz: np.ndarray,
        bearing: np.ndarray,
        use_wtls: bool = False,
        use_ml_refine: Optional[bool] = None,
        control_point_method: str = "fixed",
        sigma_azimuth: float = np.deg2rad(0.3),
        sigma_elevation: float = np.deg2rad(0.5),
        R_ins_to_body: Optional[np.ndarray] = None,
        diagnostics: Optional[dict] = None,
        control_points_world: Optional[np.ndarray] = None,
        linear_solver: Optional[str] = None,
        structured_geometry: Optional[object] = None,
        structured_tls_options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve pose (R, t) from bearing-only correspondences.

        Args:
            uvw: 3D points in global frame, shape (3, n) or (n, 3)
            xyz: 3D points in local frame, shape (3, n) or (n, 3)
            bearing: bearing vectors, shape (3, n) or (n, 3)
            use_wtls: if True, use weighted TLS for linear stage
            use_ml_refine: None preserves legacy ML on, but defaults to off for
                           linear_solver="structured_tls". True explicitly enables ML.
            control_point_method: "fixed" (BGPnP controls, default) or explicit "pca"
            sigma_azimuth: azimuth noise std (rad) for ML likelihood weighting
            sigma_elevation: elevation noise std (rad) for ML likelihood weighting
            R_ins_to_body: known rotation from INS-frame vector to body-fixed vector.
                           If None, identity is used (i.e., bearing is already body-fixed).
            diagnostics: optional output dictionary for copied solver intermediates
                         and ML function evaluations; None disables recording.
            control_points_world: optional (4,3) world-frame controls computed by
                                  the caller; reused without recomputing PCA.
            linear_solver: None uses use_wtls; or "ls", "tls", "structured_tls".
            structured_geometry: fixed sensor derivatives/covariance/weights for
                                 the exact supplied controls and observations.
            structured_tls_options: StructuredTLSConfig field overrides.

        Returns:
            (R, t): rotation matrix (3,3) and translation vector (3,)
        """
        if use_ml_refine is None:
            use_ml_refine = linear_solver != "structured_tls"
        uvw_n3 = BO_EPnP._to_row_points(uvw)
        xyz_n3 = BO_EPnP._to_row_points(xyz)
        bearing_n3 = BO_EPnP._normalize_bearings(BO_EPnP._to_row_points(bearing))
        BO_EPnP._validate_inputs(uvw_n3, xyz_n3, bearing_n3)
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(schema_version=1, status="started",
                               use_wtls=use_wtls, use_ml_refine=use_ml_refine,
                               control_point_method=control_point_method,
                               world_points=uvw_n3.tolist(), local_points=xyz_n3.tolist(),
                               bearings=bearing_n3.tolist())

        if control_points_world is None:
            control_points = BO_EPnP._select_control_points(uvw_n3, method=control_point_method)
        else:
            control_points = np.array(control_points_world, dtype=float, copy=True)
            if control_points.shape != (4, 3) or not np.all(np.isfinite(control_points)):
                raise ValueError("control_points_world must be a finite (4,3) array")
            if np.linalg.matrix_rank(control_points[1:] - control_points[0]) < 3:
                raise ValueError("control_points_world must be affinely independent")
        if diagnostics is not None:
            diagnostics["control_point_source"] = (
                "provided" if control_points_world is not None else "computed")
        weights = bgpnp.compute_alphas(uvw_n3, control_points)

        linear_record = {} if diagnostics is not None else None
        linear_options = {"diagnostics": linear_record} if diagnostics is not None else {}
        A, b = BO_EPnP._build_sphere_aware_linear_system(xyz_n3, bearing_n3, weights, **linear_options)
        if diagnostics is not None:
            diagnostics.update(control_points=control_points.tolist(),
                               barycentric_weights=weights.tolist(), linear=linear_record)
            centered = uvw_n3 - np.mean(uvw_n3, axis=0)
            diagnostics["world_point_covariance_eigenvalues_descending"] = np.linalg.eigvalsh(
                centered.T @ centered / max(len(uvw_n3) - 1, 1))[::-1].tolist()

        if linear_solver not in (None, "ls", "tls", "structured_tls"):
            raise ValueError("Unknown BO linear_solver")
        if linear_solver in ("tls", "structured_tls"):
            use_wtls = True
        elif linear_solver == "ls":
            use_wtls = False
        if diagnostics is not None and linear_solver is not None:
            diagnostics.update(linear_solver=linear_solver, use_wtls=use_wtls)
        if use_wtls:
            tls_record = {} if diagnostics is not None or linear_solver == "structured_tls" else None
            tls_options = {"diagnostics": tls_record} if tls_record is not None else {}
            if diagnostics is not None:
                diagnostics["tls"] = tls_record
            c_prime = BO_EPnP._solve_wtls(A, b, **tls_options)
        else:
            c_prime = BO_EPnP._solve_ls(A, b)

        if linear_solver == "structured_tls":
            from src.structured_tls import StructuredTLSConfig, solve_structured_tls

            if structured_geometry is None:
                raise ValueError("Structured TLS requires sensor-frame geometry and covariance")
            expected_m = np.array([np.kron(w, np.eye(3)) for w in weights])
            expected_p = np.eye(3) - bearing_n3[:, :, None] * bearing_n3[:, None, :]
            if (structured_geometry.m.shape != expected_m.shape
                    or not np.allclose(structured_geometry.m, expected_m, rtol=1e-12, atol=1e-12)
                    or not np.allclose(structured_geometry.local, xyz_n3, rtol=1e-12, atol=1e-12)
                    or not np.allclose(structured_geometry.p, expected_p, rtol=1e-12, atol=1e-12)):
                raise ValueError("Structured geometry does not match controls, local points or bearings")
            stls_record = {}
            if diagnostics is not None:
                diagnostics["structured_tls"] = stls_record
            try:
                # Ask the existing WTLS implementation which initializer it used.
                stls_record["initialization_method"] = (
                    "ls_wtls_scale_fallback" if tls_record["fallback_weighted_ls"] else "wtls")
                result = solve_structured_tls(
                    structured_geometry, c_prime,
                    config=StructuredTLSConfig(**(structured_tls_options or {})),
                    diagnostics=stls_record)
                c_prime = result.controls
                if not result.diagnostics["converged"] and diagnostics is None:
                    raise ValueError("Structured TLS did not converge: " + result.diagnostics["status"])
            except (ValueError, np.linalg.LinAlgError) as error:
                stls_record.update(status="failed", error=str(error), converged=False)
                raise

        C_prime = c_prime.reshape(4, 3)
        recovery_record = {} if diagnostics is not None else None
        recovery_options = {"diagnostics": recovery_record} if diagnostics is not None else {}
        R, t = BO_EPnP._recover_pose(control_points, C_prime, **recovery_options)
        if diagnostics is not None:
            diagnostics.update(estimated_control_points=C_prime.tolist(), recovery=recovery_record,
                               pre_ml_pose={"rotation": R.tolist(), "translation": t.tolist()})

        if use_ml_refine:
            if R_ins_to_body is None:
                R_ins_to_body = np.eye(3)
            ml_record = {} if diagnostics is not None else None
            ml_options = {"diagnostics": ml_record} if diagnostics is not None else {}
            if diagnostics is not None:
                diagnostics["ml"] = ml_record
            R, t = BO_EPnP._ml_refinement_russell(
                uvw_n3=uvw_n3,
                xyz_n3=xyz_n3,
                bearing_n3=bearing_n3,
                R_init=R,
                t_init=t,
                sigma_azimuth=sigma_azimuth,
                sigma_elevation=sigma_elevation,
                R_ins_to_body=R_ins_to_body,
                **ml_options,
            )

        if diagnostics is not None:
            diagnostics.update(status="completed", final_pose={"rotation": R.tolist(), "translation": t.tolist()})
        return R, t

    # NEW FUNCTION (not in original library): robust input orientation unifier.
    @staticmethod
    def _to_row_points(points: np.ndarray) -> np.ndarray:
        """
        Convert points to shape (n, 3). Accepts (n,3) or (3,n).
        """
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        raise ValueError(f"Expected (n,3) or (3,n), got {arr.shape}")

    # NEW FUNCTION (not in original library): explicit bearing normalization helper.
    @staticmethod
    def _normalize_bearings(bearing_n3: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(bearing_n3, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("bearing contains zero or near-zero vectors")
        return bearing_n3 / norms

    # NEW FUNCTION (not in original library): BO-EPnP input checks at API boundary.
    @staticmethod
    def _validate_inputs(uvw_n3: np.ndarray, xyz_n3: np.ndarray, bearing_n3: np.ndarray) -> None:
        if uvw_n3.shape != xyz_n3.shape or uvw_n3.shape != bearing_n3.shape:
            raise ValueError(
                f"Shape mismatch: uvw={uvw_n3.shape}, xyz={xyz_n3.shape}, bearing={bearing_n3.shape}"
            )
        if uvw_n3.shape[0] < 4:
            raise ValueError(f"At least 4 points required, got {uvw_n3.shape[0]}")
        if np.any(~np.isfinite(uvw_n3)) or np.any(~np.isfinite(xyz_n3)) or np.any(~np.isfinite(bearing_n3)):
            raise ValueError("Inputs contain NaN or Inf")

    @staticmethod
    def _select_control_points(points_n3: np.ndarray, method: str = "fixed") -> np.ndarray:
        """
        Select 4 control points.

        method:
        - "fixed": reuse BGPnP world-frame control points, independent of input subset
        - "pca": EPnP-like adaptive control points
        """
        if method == "fixed":
            return bgpnp.define_control_points()
        if method == "pca":
            return BO_EPnP._select_control_points_pca(points_n3)
        raise ValueError(f"Unknown control_point_method: {method}")

    # NEW FUNCTION (not in original library): PCA/centroid-based control point selection.
    @staticmethod
    def _select_control_points_pca(points_n3: np.ndarray) -> np.ndarray:
        """
        EPnP-style control points from centroid + principal axes.
        """
        centroid = np.mean(points_n3, axis=0)
        centered = points_n3 - centroid
        cov = centered.T @ centered / max(points_n3.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        scales = np.sqrt(np.maximum(eigvals, 1e-12))
        c0 = centroid
        c1 = centroid + scales[0] * eigvecs[:, 0]
        c2 = centroid + scales[1] * eigvecs[:, 1]
        c3 = centroid + scales[2] * eigvecs[:, 2]
        return np.vstack([c0, c1, c2, c3])

    # NEW FUNCTION (not in original library): sphere-aware linear system assembly.
    @staticmethod
    def _build_sphere_aware_linear_system(
        xyz_n3: np.ndarray, bearing_n3: np.ndarray, weights_n4: np.ndarray,
        diagnostics: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linear system with orthogonal projection:
            P_lambda = I - q q^T
            P_lambda (M_i c' - xyz_i) = 0

        Uses 2 strongest rows per correspondence (rank-2 constraints).
        """
        n = xyz_n3.shape[0]
        A = np.zeros((2 * n, 12))
        b = np.zeros((2 * n,))
        eye3 = np.eye(3)
        if diagnostics is not None:
            diagnostics["selected_projection_rows"] = []

        for i in range(n):
            q = bearing_n3[i]
            P_lambda = eye3 - np.outer(q, q)
            M_i = np.kron(weights_n4[i], eye3)  # (3,12)
            A_i = P_lambda @ M_i
            b_i = P_lambda @ xyz_n3[i]

            row_norms = np.linalg.norm(A_i, axis=1)
            keep = np.argsort(row_norms)[-2:]
            if diagnostics is not None:
                diagnostics["selected_projection_rows"].append(keep.tolist())
            A[2 * i : 2 * i + 2, :] = A_i[keep, :]
            b[2 * i : 2 * i + 2] = b_i[keep]

        if diagnostics is not None:
            diagnostics.update(A=A.tolist(), b=b.tolist())
        return A, b

    @staticmethod
    def _solve_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Standard least squares with mild Tikhonov regularization fallback.
        """
        cond = np.linalg.cond(A)
        if cond > 1e10:
            reg = 1e-6
            AtA = A.T @ A + reg * np.eye(A.shape[1])
            Atb = A.T @ b
            return np.linalg.solve(AtA, Atb)
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # NEW FUNCTION (not in original library): weighted TLS solver for EIV.
    @staticmethod
    def _solve_wtls(A: np.ndarray, b: np.ndarray, row_weights: Optional[np.ndarray] = None,
                    diagnostics: Optional[dict] = None) -> np.ndarray:
        """
        Weighted TLS by row-whitening + SVD on augmented system [A|b].

        Note:
        - This is a practical WTLS initializer for EIV settings.
        - If the homogeneous scale is ill-posed, fallback to weighted LS.
        """
        if row_weights is None:
            row_weights = np.ones(A.shape[0], dtype=float)
        w = np.asarray(row_weights, dtype=float).reshape(-1)
        if w.shape[0] != A.shape[0]:
            raise ValueError(f"row_weights length mismatch: {w.shape[0]} vs {A.shape[0]}")
        if np.any(w <= 0):
            raise ValueError("row_weights must be positive")

        W_sqrt = np.sqrt(w)[:, None]
        Aw = A * W_sqrt
        bw = b * W_sqrt[:, 0]

        Z = np.hstack([Aw, bw[:, None]])
        # Six correspondences give A=(12,12), Z=(12,13). Reduced SVD omits
        # Z's right null vector in this wide case and returns a nonzero singular
        # direction instead, producing a spurious WTLS pose at the minimum budget.
        _, singular, vt = np.linalg.svd(Z, full_matrices=Z.shape[0] < Z.shape[1])
        v = vt[-1, :]
        denom = v[-1]
        if diagnostics is not None:
            diagnostics.update(row_weights=w.tolist(), A_w=Aw.tolist(), b_w=bw.tolist(),
                               augmented_matrix=Z.tolist(), singular_values=singular.tolist(),
                               singular_value_count=len(singular),
                               wide_matrix_nullspace=bool(Z.shape[0] < Z.shape[1]),
                               right_solution_vector=v.tolist(), denominator=float(denom),
                               fallback_weighted_ls=bool(np.abs(denom) < 1e-12))

        if np.abs(denom) < 1e-12:
            logger.warning("WTLS degenerate scale detected, fallback to weighted LS.")
            solution = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        else:
            solution = -v[:-1] / denom
        if diagnostics is not None:
            diagnostics["solution"] = solution.tolist()
        return solution

    @staticmethod
    def _recover_pose(C: np.ndarray, C_prime: np.ndarray,
                      diagnostics: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover pose from 3D-3D alignment:
            C_prime ~= R * C + t
        """
        mu_c = np.mean(C, axis=0)
        mu_cp = np.mean(C_prime, axis=0)
        X = C - mu_c
        Y = C_prime - mu_cp
        H = X.T @ Y
        U, singular, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if diagnostics is not None:
            diagnostics.update(cross_matrix=H.tolist(), singular_values=singular.tolist(),
                               reflection_corrected=bool(np.linalg.det(R) < 0))
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_cp - R @ mu_c
        if diagnostics is not None:
            diagnostics["alignment_residuals"] = (C_prime - ((R @ C.T).T + t)).tolist()
        return R, t

    # NEW FUNCTION (not in original library): explicit ML-style SE(3) refinement.
    @staticmethod
    def _ml_refinement(
        uvw_n3: np.ndarray,
        xyz_n3: np.ndarray,
        bearing_n3: np.ndarray,
        R_init: np.ndarray,
        t_init: np.ndarray,
        max_nfev: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ML-style refinement (Gaussian-noise assumption) by nonlinear least squares.

        Residual model:
            r_i = hat(b_pred_i) x b_meas_i
            b_pred_i = normalize(R*uvw_i + t - xyz_i)
        """

        def residual(delta: np.ndarray) -> np.ndarray:
            omega = delta[:3]
            upsilon = delta[3:]
            dR = BO_EPnP._se3_so3_exp(omega)
            R = dR @ R_init
            t = t_init + upsilon

            p_pred = (R @ uvw_n3.T).T + t.reshape(1, 3) - xyz_n3
            norms = np.linalg.norm(p_pred, axis=1, keepdims=True)
            b_pred = p_pred / (norms + 1e-12)
            return np.cross(b_pred, bearing_n3).reshape(-1)

        x0 = np.zeros(6, dtype=float)
        res = least_squares(residual, x0, method="lm", max_nfev=max_nfev)
        dR = BO_EPnP._se3_so3_exp(res.x[:3])
        R_ref = dR @ R_init
        t_ref = t_init + res.x[3:]
        return R_ref, t_ref

    # NEW FUNCTION (not in original library): Russell-style anisotropic ML refinement.
    @staticmethod
    def _ml_refinement_russell(
        uvw_n3: np.ndarray,
        xyz_n3: np.ndarray,
        bearing_n3: np.ndarray,
        R_init: np.ndarray,
        t_init: np.ndarray,
        sigma_azimuth: float,
        sigma_elevation: float,
        R_ins_to_body: np.ndarray,
        max_nfev: int = 120,
        diagnostics: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ML refinement aligned with Russell et al. (TAES 2020, Section V):
        - objective built from azimuth/elevation errors in body-fixed frame
        - anisotropic Gaussian noise (different sigma for az and el)
        - iterative nonlinear optimization initialized by algebraic solution

        We solve weighted least squares on residuals:
            r = [delta_az / sigma_az, delta_el / sigma_el]
        where wrapped angular error is used for azimuth.
        """
        if sigma_azimuth <= 0 or sigma_elevation <= 0:
            raise ValueError("sigma_azimuth and sigma_elevation must be positive")
        if R_ins_to_body.shape != (3, 3):
            raise ValueError(f"R_ins_to_body must be (3,3), got {R_ins_to_body.shape}")

        # Convert measured INS-frame bearings to body-fixed frame for likelihood model.
        b_meas_body = (R_ins_to_body @ bearing_n3.T).T
        b_meas_body = BO_EPnP._normalize_bearings(b_meas_body)
        az_meas, el_meas = BO_EPnP._bearing_to_az_el(b_meas_body)
        if diagnostics is not None:
            diagnostics.update(sigma_azimuth_rad=float(sigma_azimuth),
                               sigma_elevation_rad=float(sigma_elevation),
                               R_ins_to_body=R_ins_to_body.tolist(), method="lm", max_nfev=max_nfev,
                               initial_pose={"rotation": R_init.tolist(), "translation": t_init.tolist()},
                               trace_semantics="function_evaluations_not_accepted_iterations",
                               residual_evaluations=[], jacobian_evaluations=[])

        def residual(delta: np.ndarray) -> np.ndarray:
            omega = delta[:3]
            upsilon = delta[3:]
            dR = BO_EPnP._se3_so3_exp(omega)
            R = dR @ R_init
            t = t_init + upsilon

            # Predicted line-of-sight in INS frame.
            p_pred_ins = (R @ uvw_n3.T).T + t.reshape(1, 3) - xyz_n3
            p_pred_ins = p_pred_ins / (np.linalg.norm(p_pred_ins, axis=1, keepdims=True) + 1e-12)

            # Convert predicted bearing to body-fixed frame and then to az/el.
            p_pred_body = (R_ins_to_body @ p_pred_ins.T).T
            p_pred_body = BO_EPnP._normalize_bearings(p_pred_body)
            az_pred, el_pred = BO_EPnP._bearing_to_az_el(p_pred_body)

            d_az = BO_EPnP._wrap_to_pi(az_pred - az_meas)
            d_el = el_pred - el_meas

            # Negative log-likelihood under independent Gaussian noise -> weighted LS.
            r_az = d_az / sigma_azimuth
            r_el = d_el / sigma_elevation
            value = np.column_stack([r_az, r_el]).reshape(-1)
            if diagnostics is not None:
                diagnostics["residual_evaluations"].append(
                    {"delta": delta.tolist(), "residual": value.tolist(),
                     "cost": float(0.5 * np.dot(value, value))})
            return value

        # NEW FUNCTION (not in original library): analytic Jacobian for Russell-style residual.
        def jacobian(delta: np.ndarray) -> np.ndarray:
            omega = delta[:3]
            upsilon = delta[3:]
            dR = BO_EPnP._se3_so3_exp(omega)
            R = dR @ R_init
            t = t_init + upsilon

            n = uvw_n3.shape[0]
            J = np.zeros((2 * n, 6), dtype=float)
            eye3 = np.eye(3)
            eps = 1e-12

            for i in range(n):
                p = uvw_n3[i]
                q = xyz_n3[i]

                # u = R p + t - q (INS frame LOS before normalization)
                u = R @ p + t - q
                norm_u = np.linalg.norm(u)
                norm_u = max(norm_u, eps)
                u_hat = u / norm_u

                # v = R_ins_to_body * u_hat
                v = R_ins_to_body @ u_hat
                x, y, z = v
                rho2 = x * x + y * y
                rho2 = max(rho2, eps)
                rho = np.sqrt(rho2)
                den = rho2 + z * z
                den = max(den, eps)

                # d[az,el]/dv
                J_ang = np.array(
                    [
                        [-y / rho2, x / rho2, 0.0],
                        [
                            -x * z / (rho * den),
                            -y * z / (rho * den),
                            rho / den,
                        ],
                    ],
                    dtype=float,
                )

                # d(u_hat)/du = (I - u_hat u_hat^T) / ||u||
                J_norm = (eye3 - np.outer(u_hat, u_hat)) / norm_u

                # du/d[delta_theta, delta_t] ~= [-[R p]_x, I]
                Rp = R @ p
                J_u = np.hstack([-BO_EPnP._skew(Rp), eye3])

                # Angle weighting: [daz/sigma_az, del/sigma_el]
                J_w = np.array(
                    [[1.0 / sigma_azimuth, 0.0], [0.0, 1.0 / sigma_elevation]],
                    dtype=float,
                )

                Ji = J_w @ J_ang @ R_ins_to_body @ J_norm @ J_u
                J[2 * i : 2 * i + 2, :] = Ji

            if diagnostics is not None:
                diagnostics["jacobian_evaluations"].append({"delta": delta.tolist(), "jacobian": J.tolist()})
            return J

        x0 = np.zeros(6, dtype=float)
        # LM with analytic Jacobian for faster and more stable convergence.
        res = least_squares(residual, x0, jac=jacobian, method="lm", max_nfev=max_nfev)
        if diagnostics is not None:
            diagnostics.update(success=bool(res.success), status=int(res.status), message=str(res.message),
                               nfev=int(res.nfev), njev=None if res.njev is None else int(res.njev),
                               optimality=float(res.optimality), final_cost=float(res.cost),
                               final_delta=res.x.tolist(), final_residual=res.fun.tolist(),
                               final_jacobian=res.jac.tolist(), final_gradient=res.grad.tolist(),
                               initial_cost=diagnostics["residual_evaluations"][0]["cost"])
        dR = BO_EPnP._se3_so3_exp(res.x[:3])
        R_ref = dR @ R_init
        t_ref = t_init + res.x[3:]
        return R_ref, t_ref

    # NEW FUNCTION (not in original library): convert unit-bearing vectors to az/el.
    @staticmethod
    def _bearing_to_az_el(bearing_n3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Azimuth: atan2(y, x)
        Elevation: atan2(z, sqrt(x^2 + y^2))
        """
        x = bearing_n3[:, 0]
        y = bearing_n3[:, 1]
        z = bearing_n3[:, 2]
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x * x + y * y))
        return az, el

    # NEW FUNCTION (not in original library): wrap angle to [-pi, pi].
    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # NEW FUNCTION (not in original library): SO(3) exponential map.
    @staticmethod
    def _se3_so3_exp(omega: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(omega)
        if theta < 1e-12:
            return np.eye(3) + BO_EPnP._skew(omega)
        k = omega / theta
        K = BO_EPnP._skew(k)
        return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    # NEW FUNCTION (not in original library): skew helper for Lie update.
    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
