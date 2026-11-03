"""Local reweighted structured TLS in the 12 affine control coordinates.

Angle derivatives originate in the sensor frame and are rotated into the local
constraint frame. No covariance inverse, jitter, pose prior or ML is used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredTLSConfig:
    """Numerical tolerances; angular covariance is supplied by the caller."""

    max_iterations: int = 50
    step_tolerance: float = 1e-9
    cost_tolerance: float = 1e-10
    max_backtracks: int = 24
    covariance_min_eigenvalue: float = 1e-20
    covariance_max_condition: float = 1e12
    rank_rtol: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_backtracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("step_tolerance", "cost_tolerance", "covariance_min_eigenvalue",
                     "covariance_max_condition", "rank_rtol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rank_rtol >= 1 or self.covariance_max_condition < 1:
            raise ValueError("Invalid rank or covariance condition threshold")


@dataclass(frozen=True)
class StructuredTLSResult:
    """Always expose termination diagnostics alongside the last valid controls."""

    controls: np.ndarray
    diagnostics: dict


def bearing_derivatives(bearing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized q and dq/d(azimuth,elevation), using radians."""
    q = np.asarray(bearing, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError("bearing must be a finite nonzero 3-vector")
    q = q / np.linalg.norm(q)
    az = np.arctan2(q[1], q[0])
    el = np.arctan2(q[2], np.hypot(q[0], q[1]))
    ca, sa, cb, sb = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    return q, np.array([[-cb * sa, -sb * ca],
                        [cb * ca, -sb * sa], [0., cb]])


def projection_derivatives(bearing: np.ndarray) -> np.ndarray:
    """Return d(I-qq.T)/d(azimuth,elevation), shape (2,3,3)."""
    q, derivative = bearing_derivatives(bearing)
    return np.array([-np.outer(q, v) - np.outer(v, q) for v in derivative.T])


def tangent_basis(bearing: np.ndarray) -> np.ndarray:
    """Construct an orthonormal tangent basis with a nonparallel reference."""
    q, _ = bearing_derivatives(bearing)
    reference = np.eye(3)[np.argmin(np.abs(q))]
    u = np.cross(q, reference)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(q, u)))
    if (not np.allclose(basis.T @ q, 0., atol=1e-12)
            or not np.allclose(basis.T @ basis, np.eye(2), atol=1e-12)):
        raise ValueError("Tangent basis orthogonality failed")
    return basis


def validate_angle_covariances(covariance: np.ndarray, count: int) -> np.ndarray:
    """Validate shared/per-observation correlated angle covariance in rad^2."""
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape == (2, 2):
        sigma = np.broadcast_to(sigma, (count, 2, 2)).copy()
    if sigma.shape != (count, 2, 2) or not np.all(np.isfinite(sigma)):
        raise ValueError("Angle covariance must be (2,2) or (N,2,2), in rad^2")
    if not np.allclose(sigma, sigma.swapaxes(-1, -2), rtol=1e-12, atol=1e-24):
        raise ValueError("Angle covariance is not symmetric")
    if np.any(np.linalg.eigvalsh(sigma) <= 0):
        raise ValueError("Angle covariance is not positive definite")
    return sigma


class StructuredTLSGeometry:
    """Fixed weights, tangent rows and sensor derivative transport for a pool."""

    def __init__(self, local: np.ndarray, sensor_bearings: np.ndarray,
                 weights: np.ndarray, angle_covariances: np.ndarray,
                 rotations: np.ndarray | None = None,
                 observation_ids: list | None = None) -> None:
        self.local = np.asarray(local, dtype=float)
        bearings = np.asarray(sensor_bearings, dtype=float)
        weights = np.asarray(weights, dtype=float)
        n = len(self.local)
        if (self.local.shape != (n, 3) or bearings.shape != (n, 3)
                or weights.shape != (n, 4) or n == 0
                or not np.all(np.isfinite(self.local))
                or not np.all(np.isfinite(weights))):
            raise ValueError("Expected finite local/bearings (N,3), weights (N,4)")
        self.sigma = validate_angle_covariances(angle_covariances, n)
        rotations = (np.broadcast_to(np.eye(3), (n, 3, 3)) if rotations is None
                     else np.asarray(rotations, dtype=float))
        if (rotations.shape != (n, 3, 3) or not np.all(np.isfinite(rotations))
                or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-7)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-7)):
            raise ValueError("rotations must be proper local<-sensor rotations")
        self.ids = list(range(n)) if observation_ids is None else list(observation_ids)
        if len(self.ids) != n:
            raise ValueError("Observation IDs must align with rows")
        self.m = np.array([np.kron(w, np.eye(3)) for w in weights])
        unit, derivatives = zip(*(bearing_derivatives(q) for q in bearings))
        q = np.einsum("nij,nj->ni", rotations, np.asarray(unit))
        dq = np.einsum("nij,njk->nik", rotations, np.asarray(derivatives))
        basis = np.array([tangent_basis(v) for v in q])
        self.ut = basis.swapaxes(-1, -2)
        # P and both D are expressed in the same local frame as M c - p.
        self.p = np.eye(3) - q[:, :, None] * q[:, None, :]
        self.derivatives = np.array([
            [-np.outer(v, w) - np.outer(w, v) for w in d.T]
            for v, d in zip(q, dq)])
        self.a = self.ut @ self.p @ self.m

    def linearize(self, controls: np.ndarray, config: StructuredTLSConfig,
                  *, allow_invalid: bool = False) -> dict:
        """Recompute residual, angle Jacobian, covariance and Cholesky rows."""
        c = np.asarray(controls, dtype=float)
        if c.shape != (12,) or not np.all(np.isfinite(c)):
            raise ValueError("controls must be a finite 12-vector")
        d = np.einsum("nij,j->ni", self.m, c) - self.local
        residual = np.einsum("nij,nj->ni", self.ut @ self.p, d)
        jtheta = np.einsum("naij,nj->nia", self.derivatives, d)
        b = self.ut @ jtheta
        covariance = b @ self.sigma @ b.swapaxes(-1, -2)
        valid = np.ones(len(d), dtype=bool)
        reasons, eigenvalues, conditions = [], [], []
        for i, cov in enumerate(covariance):
            reason = None
            eigen = np.array([np.nan, np.nan])
            condition = np.inf
            if not np.all(np.isfinite(cov)):
                reason = "nonfinite_covariance"
            elif np.linalg.norm(cov - cov.T) > 1e-12 * max(np.linalg.norm(cov), np.finfo(float).tiny):
                reason = "asymmetric_covariance"
            else:
                # Average only roundoff-sized asymmetry, without changing eigenvalue scale.
                cov = (cov + cov.T) / 2
                covariance[i] = cov
                eigen = np.linalg.eigvalsh(cov)
                condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else np.inf
                if eigen[0] <= config.covariance_min_eigenvalue:
                    reason = "covariance_min_eigenvalue"
                elif condition > config.covariance_max_condition:
                    reason = "covariance_condition"
            eigenvalues.append(eigen)
            conditions.append(condition)
            if reason:
                valid[i] = False
                reasons.append({"id": self.ids[i], "reason": reason})
        if reasons and not allow_invalid:
            raise ValueError(f"Invalid structured covariance: {reasons}")
        white_a = np.zeros_like(self.a)
        white_r = np.zeros_like(residual)
        if np.any(valid):
            chol = np.linalg.cholesky(covariance[valid])
            white_a[valid] = np.linalg.solve(chol, self.a[valid])
            white_r[valid] = np.linalg.solve(chol, residual[valid, :, None])[..., 0]
        information = white_a.swapaxes(-1, -2) @ white_a
        if not np.all(np.isfinite(information)):
            raise ValueError("Nonfinite structured information")
        return dict(residual=residual, a=self.a, b=b, covariance=covariance,
                    covariance_eigenvalues=np.asarray(eigenvalues),
                    covariance_condition=np.asarray(conditions), valid=valid,
                    invalid=reasons, white_a=white_a, white_r=white_r,
                    information=information, cost=float(np.sum(white_r**2)))


def compute_structured_tls_information(
    local: np.ndarray, bearing: np.ndarray, control_weights: np.ndarray,
    c_reference: np.ndarray, angle_covariance: np.ndarray,
    rotation: np.ndarray | None = None,
    config: StructuredTLSConfig | None = None,
) -> dict:
    """Compute one observation's 12x12 H and intermediate formula terms."""
    geometry = StructuredTLSGeometry(
        np.asarray(local)[None], np.asarray(bearing)[None],
        np.asarray(control_weights)[None], angle_covariance,
        None if rotation is None else np.asarray(rotation)[None])
    return geometry.linearize(c_reference, config or StructuredTLSConfig())


def information_metrics(information: np.ndarray, rank_rtol: float,
                        regularization: float, rank_scale: float | None = None) -> dict:
    """Rank uses unregularized eigenvalues; logdet uses a fixed lambda."""
    h = np.asarray(information, dtype=float)
    eigen = np.linalg.eigvalsh((h + h.T) / 2)
    scale = max(float(eigen[-1]), np.finfo(float).tiny) if rank_scale is None else rank_scale
    rank = int(np.count_nonzero(eigen > rank_rtol * scale))
    sign, logdet = np.linalg.slogdet(h + regularization * np.eye(len(h)))
    if sign <= 0:
        raise ValueError("Regularized information is not positive definite")
    return dict(rank=rank, logdet=float(logdet), minimum_eigenvalue=float(eigen[0]),
                maximum_eigenvalue=float(eigen[-1]),
                condition_number=float(eigen[-1] / eigen[0]) if rank == len(h) else np.inf)


def greedy_structured_information(information: np.ndarray, valid: np.ndarray,
                                  count: int, rank_rtol: float,
                                  regularization: float) -> tuple[list[int], list[dict]]:
    """Deterministic rank-then-logdet selection, fixed pool scale and lambda."""
    remaining = np.flatnonzero(valid).tolist()
    scale = max(float(np.linalg.eigvalsh(information[valid].sum(axis=0))[-1]),
                np.finfo(float).tiny) if remaining else 1.
    total = np.zeros((12, 12))
    selected, history = [], []
    while remaining and len(selected) < count:
        best, best_metrics = None, None
        for index in remaining:
            metrics = information_metrics(total + information[index], rank_rtol,
                                          regularization, scale)
            score = (metrics["rank"], metrics["logdet"])
            if best_metrics is None or score > (best_metrics["rank"], best_metrics["logdet"]):
                best, best_metrics = index, metrics
        selected.append(best)
        remaining.remove(best)
        total += information[best]
        history.append({"index": best, **best_metrics})
    return selected, history


def solve_structured_tls(geometry: StructuredTLSGeometry, c_initial: np.ndarray,
                         *, config: StructuredTLSConfig | None = None,
                         diagnostics: dict | None = None) -> StructuredTLSResult:
    """Reweighted local GN with recomputed-covariance backtracking.

    A rejected direction returns the last valid estimate with explicit
    nonconvergence. Invalid covariance/rank raises; no invalid pose is returned.
    This update omits covariance derivatives and is not the exact gradient.
    """
    cfg = config or StructuredTLSConfig()
    record = diagnostics if diagnostics is not None else {}
    record.update(status="started", converged=False, iterations=0, history=[],
                  initial_control_points=np.asarray(c_initial).tolist())
    c = np.array(c_initial, dtype=float, copy=True)
    current = geometry.linearize(c, cfg)
    record["initial_cost"] = current["cost"]
    step_norm = 0.
    status = "max_iterations"
    for iteration in range(cfg.max_iterations):
        a = current["white_a"].reshape(-1, 12)
        r = current["white_r"].reshape(-1)
        singular = np.linalg.svd(a, compute_uv=False)
        if (len(singular) < 12 or singular[-1]**2 <= cfg.rank_rtol * singular[0]**2):
            record.update(status="rank_deficient", converged=False)
            raise ValueError("Structured TLS information rank is less than 12")
        delta = np.linalg.lstsq(a, -r, rcond=None)[0]
        step_norm = float(np.linalg.norm(delta))
        record["iterations"] = iteration + 1
        if step_norm <= cfg.step_tolerance * (1 + np.linalg.norm(c)):
            status = "step_tolerance"
            break
        accepted = False
        for backtrack in range(cfg.max_backtracks):
            step = 0.5**backtrack
            try:
                trial = geometry.linearize(c + step * delta, cfg)
            except ValueError:
                continue
            if trial["cost"] <= current["cost"]:
                accepted = True
                break
        if not accepted:
            status = "line_search_failed"
            break
        decrease = current["cost"] - trial["cost"]
        old_cost = current["cost"]
        c += step * delta
        step_norm *= step
        current = trial
        record["history"].append(dict(iteration=iteration + 1, cost=current["cost"],
                                      step_norm=step_norm, step_fraction=step))
        # A tiny backtracking fraction is a stall, not evidence of convergence.
        if step == 1. and decrease <= cfg.cost_tolerance * (1 + old_cost):
            status = "cost_tolerance"
            break
    correction = np.einsum(
        "nij,njk,nk->ni", geometry.sigma, current["b"].swapaxes(-1, -2),
        np.linalg.solve(current["covariance"], current["residual"][..., None])[..., 0])
    metrics = information_metrics(current["information"].sum(axis=0), cfg.rank_rtol, 1e-8)
    if metrics["rank"] < 12:
        record.update(status="rank_deficient", converged=False)
        raise ValueError("Final structured TLS information rank is less than 12")
    record.update(status=status, converged=status in ("step_tolerance", "cost_tolerance"),
                  final_cost=current["cost"], final_step_norm=step_norm,
                  final_control_points=c.tolist(), final_information=metrics,
                  angular_corrections_rad=correction.tolist(),
                  maximum_angular_correction_rad=float(np.max(np.linalg.norm(correction, axis=1))),
                  average_angular_correction_rad=float(np.mean(np.linalg.norm(correction, axis=1))),
                  covariance_max_condition=float(current["covariance_condition"].max()))
    return StructuredTLSResult(c, record)
