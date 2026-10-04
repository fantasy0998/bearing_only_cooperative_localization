"""
GaBP refinement for multi-agent bearing-only localization.

The module is intended to run after BO-EPnP has produced initial poses.
State convention follows docs/gn_bp_multi_agent_bearing_solver_design.md:
    p_i2 = R_i * p_world + t_i

GaBP is applied to the Gauss-Newton linearized tangent variables:
    delta_xi = [delta_theta, delta_t].
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


DEFAULT_SIGMA_AZIMUTH = np.deg2rad(0.2)
DEFAULT_SIGMA_ELEVATION = np.deg2rad(0.3)
STATE_SIZE = 6


@dataclass
class PoseState:
    """Pose mapping world coordinates into one agent's local i2 frame."""

    rotation: np.ndarray
    translation: np.ndarray

    def copy(self) -> "PoseState":
        """Return a deep copy of the pose arrays."""
        return PoseState(self.rotation.copy(), self.translation.copy())


@dataclass(frozen=True)
class AnchorBearingFactor:
    """Bearing observation made by one estimated agent toward a known anchor."""

    observer_id: str
    anchor_position_world: np.ndarray
    measured_angles: np.ndarray
    observer_point_local: np.ndarray | None = None
    sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH
    sigma_elevation: float = DEFAULT_SIGMA_ELEVATION
    observer_extrinsic_rotation: np.ndarray | None = None
    observer_extrinsic_translation: np.ndarray | None = None


@dataclass(frozen=True)
class PairwiseBearingFactor:
    """Bearing observation made by one estimated agent toward another agent."""

    observer_id: str
    target_id: str
    target_point_local: np.ndarray
    measured_angles: np.ndarray
    observer_point_local: np.ndarray | None = None
    sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH
    sigma_elevation: float = DEFAULT_SIGMA_ELEVATION
    observer_extrinsic_rotation: np.ndarray | None = None
    observer_extrinsic_translation: np.ndarray | None = None


@dataclass(frozen=True)
class PosePriorFactor:
    """Tangent-space prior for one pose state."""

    variable_id: str
    mean_delta: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class UnaryInfoFactor:
    """Information-form factor connected to one variable."""

    variable_id: str
    information: np.ndarray
    information_vector: np.ndarray
    residual_norm: float = 0.0


@dataclass(frozen=True)
class PairwiseInfoFactor:
    """Information-form factor connected to two variables."""

    first_id: str
    second_id: str
    omega_ii: np.ndarray
    omega_ij: np.ndarray
    omega_ji: np.ndarray
    omega_jj: np.ndarray
    eta_i: np.ndarray
    eta_j: np.ndarray
    residual_norm: float = 0.0


@dataclass
class GaBPSettings:
    """Solver settings for GN outer iterations and GaBP inner iterations."""

    max_gn_iterations: int = 10
    max_bp_iterations: int = 30
    gn_tolerance: float = 1e-7
    bp_tolerance: float = 1e-8
    message_damping: float = 0.5
    step_damping: float = 1.0
    diagonal_regularization: float = 1e-9


@dataclass
class GaBPHistoryEntry:
    """One GN iteration log entry."""

    iteration: int
    residual_norm: float
    max_increment_norm: float
    bp_iterations: int


@dataclass
class GaBPRefinementResult:
    """Output of the multi-agent GaBP refinement."""

    states: dict[str, PoseState]
    history: list[GaBPHistoryEntry] = field(default_factory=list)
    selection_metadata: dict[str, object] | None = None
    stop_reason: str = "unknown"
    initial_residual_norm: float | None = None
    final_residual_norm: float | None = None
    factor_counts: dict[str, object] = field(default_factory=dict)


def refine_with_gabp(
    initial_states: dict[str, PoseState],
    anchor_factors: Iterable[AnchorBearingFactor],
    pairwise_factors: Iterable[PairwiseBearingFactor],
    prior_states: dict[str, PoseState] | None = None,
    prior_covariance: np.ndarray | None = None,
    settings: GaBPSettings | None = None,
) -> GaBPRefinementResult:
    """
    Refine BO-EPnP initial poses using manifold GN and Gaussian BP.

    Args:
        initial_states: BO-EPnP pose initial values keyed by agent id.
        anchor_factors: Bearing factors from an estimated agent to a known anchor.
        pairwise_factors: Bearing factors between two estimated agents.
        prior_states: Optional BO-EPnP prior poses. Defaults to initial_states.
        prior_covariance: Optional 6x6 tangent-space prior covariance.
        settings: Optional GaBP/GN settings.

    Returns:
        Refined states and a compact iteration history.
    """
    cfg = settings or GaBPSettings()
    #initial_states 是初始位姿
    states = {key: _validate_pose_state(value).copy() for key, value in initial_states.items()}
    anchor_factor_list = list(anchor_factors)
    pairwise_factor_list = list(pairwise_factors)
    priors = prior_states if prior_states is not None else initial_states
    history: list[GaBPHistoryEntry] = []
    stop_reason = "max_gn_iterations"

    for iteration in range(cfg.max_gn_iterations):
        unary_info = _linearize_unary_factors(anchor_factor_list, states)
        if prior_covariance is not None:
            unary_info.extend(_build_prior_info_factors(priors, states, prior_covariance))
        pairwise_info = _linearize_pairwise_factors(pairwise_factor_list, states)
        # 计算残差范数，作为迭代日志的一部分
        residual_norm = float(
            np.sqrt(
                sum(f.residual_norm**2 for f in unary_info)
                + sum(f.residual_norm**2 for f in pairwise_info)
            )
        )

        deltas, bp_iterations = solve_linearized_gabp(
            variable_ids=states.keys(),
            unary_factors=unary_info,
            pairwise_factors=pairwise_info,
            settings=cfg,
        )

        max_increment_norm = 0.0
        for variable_id, delta in deltas.items():
            step = cfg.step_damping * delta
            _apply_tangent_update(states[variable_id], step)
            max_increment_norm = max(max_increment_norm, float(np.linalg.norm(step)))

        history.append(
            GaBPHistoryEntry(
                iteration=iteration,
                residual_norm=residual_norm,
                max_increment_norm=max_increment_norm,
                bp_iterations=bp_iterations,
            )
        )
        LOGGER.debug(
            "GN-GaBP iteration %d: residual=%.6e, max_delta=%.6e, bp_iters=%d",
            iteration,
            residual_norm,
            max_increment_norm,
            bp_iterations,
        )

        if max_increment_norm < cfg.gn_tolerance:
            stop_reason = "gn_increment_tolerance"
            break

    final_unary = _linearize_unary_factors(anchor_factor_list, states)
    if prior_covariance is not None:
        final_unary.extend(_build_prior_info_factors(priors, states, prior_covariance))
    final_pairwise = _linearize_pairwise_factors(pairwise_factor_list, states)
    final_residual = float(np.sqrt(sum(f.residual_norm**2 for f in final_unary)
                                   + sum(f.residual_norm**2 for f in final_pairwise)))
    return GaBPRefinementResult(
        states=states, history=history, stop_reason=stop_reason,
        initial_residual_norm=history[0].residual_norm if history else final_residual,
        final_residual_norm=final_residual,
        factor_counts={
            "anchor": len(anchor_factor_list), "pairwise": len(pairwise_factor_list),
            "observation_total": len(anchor_factor_list) + len(pairwise_factor_list),
            "by_observer": {
                node: {
                    "anchor": sum(f.observer_id == node for f in anchor_factor_list),
                    "pairwise_outgoing": sum(f.observer_id == node for f in pairwise_factor_list),
                } for node in states
            },
        },
    )


def solve_linearized_gabp(
    variable_ids: Iterable[str],
    unary_factors: Iterable[UnaryInfoFactor],
    pairwise_factors: Iterable[PairwiseInfoFactor],
    settings: GaBPSettings | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Solve one linearized Gaussian system using information-form BP.

    Factor information uses the canonical form:
        exp(-0.5 * x.T @ Lambda @ x + eta.T @ x)

    For residuals r + J dx, the canonical vector is eta = -J.T @ r.
    """
    cfg = settings or GaBPSettings()
    ids = list(variable_ids)
    # 将所有属于同一变量的单变量因子信息矩阵相加，信息向量相加，得到每个变量的总信息矩阵和总信息向量。
    unary_by_var = _sum_unary_information(ids, unary_factors)
    pairwise_list = list(pairwise_factors)

    factor_to_var: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for factor_idx, factor in enumerate(pairwise_list):
        factor_to_var[(factor_idx, factor.first_id)] = _zero_message()
        factor_to_var[(factor_idx, factor.second_id)] = _zero_message()

    bp_iteration_count = 0
    for bp_iteration in range(cfg.max_bp_iterations):
        bp_iteration_count = bp_iteration + 1
        variable_to_factor: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

        for factor_idx, factor in enumerate(pairwise_list):
            for variable_id in (factor.first_id, factor.second_id):
                info, eta = _copy_message(unary_by_var[variable_id])
                for other_idx, other_factor in enumerate(pairwise_list):
                    if other_idx == factor_idx or not _factor_contains(other_factor, variable_id):
                        continue
                    add_info, add_eta = factor_to_var[(other_idx, variable_id)]
                    info += add_info
                    eta += add_eta
                variable_to_factor[(variable_id, factor_idx)] = (info, eta)

        next_factor_to_var: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        max_message_change = 0.0
        for factor_idx, factor in enumerate(pairwise_list):
            msg_to_first = _factor_to_first_message(
                factor,
                variable_to_factor[(factor.second_id, factor_idx)],
                cfg.diagonal_regularization,
            )
            msg_to_second = _factor_to_second_message(
                factor,
                variable_to_factor[(factor.first_id, factor_idx)],
                cfg.diagonal_regularization,
            )
            for variable_id, new_message in (
                (factor.first_id, msg_to_first),
                (factor.second_id, msg_to_second),
            ):
                key = (factor_idx, variable_id)
                damped = _damp_message(factor_to_var[key], new_message, cfg.message_damping)
                max_message_change = max(
                    max_message_change,
                    float(
                        np.linalg.norm(damped[0] - factor_to_var[key][0])
                        + np.linalg.norm(damped[1] - factor_to_var[key][1])
                    ),
                )
                next_factor_to_var[key] = damped

        factor_to_var = next_factor_to_var
        if max_message_change < cfg.bp_tolerance:
            break

    deltas: dict[str, np.ndarray] = {}
    reg_eye = cfg.diagonal_regularization * np.eye(STATE_SIZE)
    for variable_id in ids:
        posterior_info, posterior_eta = _copy_message(unary_by_var[variable_id])
        for factor_idx, factor in enumerate(pairwise_list):
            if not _factor_contains(factor, variable_id):
                continue
            info, eta = factor_to_var[(factor_idx, variable_id)]
            posterior_info += info
            posterior_eta += eta
        deltas[variable_id] = np.linalg.solve(posterior_info + reg_eye, posterior_eta)

    return deltas, bp_iteration_count


def predict_anchor_angles(
    state: PoseState,
    anchor_position_world: np.ndarray,
    observer_point_local: np.ndarray | None = None,
    extrinsic_rotation: np.ndarray | None = None,
    extrinsic_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Predict azimuth/elevation for a known anchor in the observer measurement frame."""
    R, t = _pose_arrays(state)
    E = _extrinsic_rotation(extrinsic_rotation)
    e = _extrinsic_translation(extrinsic_translation)
    anchor = _as_vector3(anchor_position_world, "anchor_position_world")
    observer_point = _optional_vector3(observer_point_local, "observer_point_local")
    s = E @ (R @ anchor + t - observer_point) + e
    return vector_to_az_el(s)

# 将目标轨迹先旋转到世界系，再旋转到观测者系，最后转换成方位角和仰角
def predict_pairwise_angles(
    observer_state: PoseState,
    target_state: PoseState,
    observer_point_local: np.ndarray | None,
    target_point_local: np.ndarray,
    extrinsic_rotation: np.ndarray | None = None,
    extrinsic_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Predict azimuth/elevation from observer to target in the observer measurement frame."""
    Ri, ti = _pose_arrays(observer_state)
    Rj, tj = _pose_arrays(target_state)
    E = _extrinsic_rotation(extrinsic_rotation)
    e = _extrinsic_translation(extrinsic_translation)
    observer_point = _optional_vector3(observer_point_local, "observer_point_local")
    qj = _as_vector3(target_point_local, "target_point_local") - tj
    target_world = Rj.T @ qj
    s = E @ (Ri @ target_world + ti - observer_point) + e
    return vector_to_az_el(s)


def vector_to_az_el(vector: np.ndarray) -> np.ndarray:
    """Convert a 3D direction vector to [azimuth, elevation]."""
    s = _as_vector3(vector, "vector")
    norm = np.linalg.norm(s)
    if norm < 1e-12:
        raise ValueError("Cannot convert near-zero vector to azimuth/elevation.")
    n = s / norm
    azimuth = np.arctan2(n[1], n[0])
    elevation = np.arctan2(n[2], np.sqrt(n[0] * n[0] + n[1] * n[1]))
    return np.array([azimuth, elevation], dtype=float)


def az_el_to_vector(angles: np.ndarray) -> np.ndarray:
    """Convert [azimuth, elevation] angles to a unit direction vector."""
    azimuth, elevation = _as_angles(angles)
    cos_el = np.cos(elevation)
    return np.array(
        [np.cos(azimuth) * cos_el, np.sin(azimuth) * cos_el, np.sin(elevation)],
        dtype=float,
    )


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """SO(3) exponential map."""
    w = _as_vector3(omega, "omega")
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    axis = w / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map."""
    R = _as_rotation(rotation, "rotation")
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3)
    return (
        theta
        / (2.0 * np.sin(theta))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    )


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix [v]_x."""
    v = _as_vector3(vector, "vector")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# 输入是观测因子factors和初始位姿状态staes
def linearize_anchor_factor(
    factor: AnchorBearingFactor, states: dict[str, PoseState]
) -> UnaryInfoFactor:
    """Return whitened anchor information at the supplied fixed pose states."""
    return _linearize_anchor_factor(factor, states)


def linearize_pairwise_factor(
    factor: PairwiseBearingFactor, states: dict[str, PoseState]
) -> PairwiseInfoFactor:
    """Return whitened pairwise information at the supplied fixed pose states."""
    return _linearize_pairwise_factor(factor, states)


def _linearize_unary_factors(
    factors: list[AnchorBearingFactor], states: dict[str, PoseState]
) -> list[UnaryInfoFactor]:
    return [_linearize_anchor_factor(factor, states) for factor in factors]


def _linearize_pairwise_factors(
    factors: list[PairwiseBearingFactor], states: dict[str, PoseState]
) -> list[PairwiseInfoFactor]:
    return [_linearize_pairwise_factor(factor, states) for factor in factors]

## factor 观测因子 states 先验位姿状态 来自于BO-EPnP
def _linearize_anchor_factor(
    factor: AnchorBearingFactor, states: dict[str, PoseState]
) -> UnaryInfoFactor:
    if factor.observer_id not in states:
        raise KeyError(f"Unknown observer state: {factor.observer_id}")
    state = states[factor.observer_id]
    R, t = _pose_arrays(state)
    E = _extrinsic_rotation(factor.observer_extrinsic_rotation)
    e = _extrinsic_translation(factor.observer_extrinsic_translation)
    anchor = _as_vector3(factor.anchor_position_world, "anchor_position_world")
    observer_point = _optional_vector3(factor.observer_point_local, "observer_point_local")

    a = R @ anchor
    s = E @ (a + t - observer_point) + e
    residual, J_angle = _angle_residual_and_jacobian(
        s,
        factor.measured_angles,
        factor.sigma_azimuth,
        factor.sigma_elevation,
    )
    J_s = E @ np.hstack((-skew(a), np.eye(3)))
    J = J_angle @ J_s
    omega = J.T @ J
    eta = -J.T @ residual
    return UnaryInfoFactor(factor.observer_id, omega, eta, float(np.linalg.norm(residual)))


def _linearize_pairwise_factor(
    factor: PairwiseBearingFactor, states: dict[str, PoseState]
) -> PairwiseInfoFactor:
    if factor.observer_id not in states:
        raise KeyError(f"Unknown observer state: {factor.observer_id}")
    if factor.target_id not in states:
        raise KeyError(f"Unknown target state: {factor.target_id}")

    observer_state = states[factor.observer_id]
    target_state = states[factor.target_id]
    Ri, ti = _pose_arrays(observer_state)
    Rj, tj = _pose_arrays(target_state)
    E = _extrinsic_rotation(factor.observer_extrinsic_rotation)
    e = _extrinsic_translation(factor.observer_extrinsic_translation)
    observer_point = _optional_vector3(factor.observer_point_local, "observer_point_local")

    qj = _as_vector3(factor.target_point_local, "target_point_local") - tj
    gj = Rj.T @ qj
    aij = Ri @ gj
    s = E @ (aij + ti - observer_point) + e
    residual, J_angle = _angle_residual_and_jacobian(
        s,
        factor.measured_angles,
        factor.sigma_azimuth,
        factor.sigma_elevation,
    )

    J_s_observer = E @ np.hstack((-skew(aij), np.eye(3)))
    J_s_target = E @ Ri @ np.hstack((Rj.T @ skew(qj), -Rj.T))
    Ji = J_angle @ J_s_observer
    Jj = J_angle @ J_s_target

    omega_ii = Ji.T @ Ji
    omega_ij = Ji.T @ Jj
    omega_ji = Jj.T @ Ji
    omega_jj = Jj.T @ Jj
    eta_i = -Ji.T @ residual
    eta_j = -Jj.T @ residual

    return PairwiseInfoFactor(
        first_id=factor.observer_id,
        second_id=factor.target_id,
        omega_ii=omega_ii,
        omega_ij=omega_ij,
        omega_ji=omega_ji,
        omega_jj=omega_jj,
        eta_i=eta_i,
        eta_j=eta_j,
        residual_norm=float(np.linalg.norm(residual)),
    )


def _build_prior_info_factors(
    prior_states: dict[str, PoseState],
    current_states: dict[str, PoseState],
    covariance: np.ndarray,
) -> list[UnaryInfoFactor]:
    cov = _as_covariance(covariance)
    information = np.linalg.inv(cov)
    factors: list[UnaryInfoFactor] = []
    for variable_id, prior_state in prior_states.items():
        if variable_id not in current_states:
            continue
        current = current_states[variable_id]
        mean_delta = np.hstack(
            (
                so3_log(prior_state.rotation @ current.rotation.T),
                _as_vector3(prior_state.translation, "prior.translation")
                - _as_vector3(current.translation, "current.translation"),
            )
        )
        eta = information @ mean_delta
        factors.append(
            UnaryInfoFactor(
                variable_id=variable_id,
                information=information.copy(),
                information_vector=eta,
                residual_norm=float(np.linalg.norm(mean_delta)),
            )
        )
    return factors


def _angle_residual_and_jacobian(
    s: np.ndarray,
    measured_angles: np.ndarray,
    sigma_azimuth: float,
    sigma_elevation: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sigma_azimuth <= 0.0 or sigma_elevation <= 0.0:
        raise ValueError("sigma_azimuth and sigma_elevation must be positive.")

    s_vec = _as_vector3(s, "s")
    measured_azimuth, measured_elevation = _as_angles(measured_angles)
    norm_s = max(float(np.linalg.norm(s_vec)), 1e-12)
    n = s_vec / norm_s
    nx, ny, nz = n

    predicted = vector_to_az_el(s_vec)
    residual = np.array(
        [
            wrap_to_pi(measured_azimuth - predicted[0]) / sigma_azimuth,
            (measured_elevation - predicted[1]) / sigma_elevation,
        ],
        dtype=float,
    )

    rho2 = max(nx * nx + ny * ny, 1e-12)
    rho = np.sqrt(rho2)
    den = max(rho2 + nz * nz, 1e-12)
    J_ang_n = np.array(
        [
            [-ny / rho2, nx / rho2, 0.0],
            [-nx * nz / (rho * den), -ny * nz / (rho * den), rho / den],
        ],
        dtype=float,
    )
    J_n_s = (np.eye(3) - np.outer(n, n)) / norm_s
    J_res_ang = np.array(
        [[-1.0 / sigma_azimuth, 0.0], [0.0, -1.0 / sigma_elevation]],
        dtype=float,
    )
    return residual, J_res_ang @ J_ang_n @ J_n_s


def _factor_to_first_message(
    factor: PairwiseInfoFactor,
    incoming_second: tuple[np.ndarray, np.ndarray],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_second, eta_second = incoming_second
    A = factor.omega_jj + lambda_second + regularization * np.eye(STATE_SIZE)
    solved_omega = np.linalg.solve(A, factor.omega_ji)
    solved_eta = np.linalg.solve(A, eta_second + factor.eta_j)
    info = factor.omega_ii - factor.omega_ij @ solved_omega
    eta = factor.eta_i - factor.omega_ij @ solved_eta
    return _symmetrize(info), eta


def _factor_to_second_message(
    factor: PairwiseInfoFactor,
    incoming_first: tuple[np.ndarray, np.ndarray],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_first, eta_first = incoming_first
    A = factor.omega_ii + lambda_first + regularization * np.eye(STATE_SIZE)
    solved_omega = np.linalg.solve(A, factor.omega_ij)
    solved_eta = np.linalg.solve(A, eta_first + factor.eta_i)
    info = factor.omega_jj - factor.omega_ji @ solved_omega
    eta = factor.eta_j - factor.omega_ji @ solved_eta
    return _symmetrize(info), eta

# 返回一个字典，键值为变量id，值为一个元组，包含该变量的总信息矩阵和总信息向量。
# 按照键值将所有属于同一变量的单变量因子信息矩阵相加，信息向量相加，得到每个变量的总信息矩阵和总信息向量。
def _sum_unary_information(
    variable_ids: list[str], unary_factors: Iterable[UnaryInfoFactor]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by_var = {variable_id: _zero_message() for variable_id in variable_ids}
    for factor in unary_factors:
        if factor.variable_id not in by_var:
            raise KeyError(f"Unary factor references unknown variable {factor.variable_id}")
        info, eta = by_var[factor.variable_id]
        by_var[factor.variable_id] = (
            info + _as_information(factor.information, "factor.information"),
            eta + _as_state_vector(factor.information_vector, "factor.information_vector"),
        )
    return by_var


def _apply_tangent_update(state: PoseState, delta: np.ndarray) -> None:
    d = _as_state_vector(delta, "delta")
    state.rotation = so3_exp(d[:3]) @ state.rotation
    state.translation = state.translation + d[3:]


def _factor_contains(factor: PairwiseInfoFactor, variable_id: str) -> bool:
    return variable_id == factor.first_id or variable_id == factor.second_id


def _zero_message() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((STATE_SIZE, STATE_SIZE), dtype=float), np.zeros(STATE_SIZE, dtype=float)


def _copy_message(message: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return message[0].copy(), message[1].copy()


def _damp_message(
    old: tuple[np.ndarray, np.ndarray],
    new: tuple[np.ndarray, np.ndarray],
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = float(np.clip(damping, 0.0, 1.0))
    return (
        _symmetrize((1.0 - gamma) * old[0] + gamma * new[0]),
        (1.0 - gamma) * old[1] + gamma * new[1],
    )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _validate_pose_state(state: PoseState) -> PoseState:
    _as_rotation(state.rotation, "state.rotation")
    state.translation = _as_vector3(state.translation, "state.translation")
    return state


def _pose_arrays(state: PoseState) -> tuple[np.ndarray, np.ndarray]:
    return _as_rotation(state.rotation, "state.rotation"), _as_vector3(
        state.translation, "state.translation"
    )


def _extrinsic_rotation(rotation: np.ndarray | None) -> np.ndarray:
    return np.eye(3) if rotation is None else _as_rotation(rotation, "extrinsic_rotation")


def _extrinsic_translation(translation: np.ndarray | None) -> np.ndarray:
    return np.zeros(3) if translation is None else _as_vector3(translation, "extrinsic_translation")


def _optional_vector3(vector: np.ndarray | None, name: str) -> np.ndarray:
    return np.zeros(3) if vector is None else _as_vector3(vector, name)


def _as_rotation(rotation: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(rotation, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _as_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _as_angles(angles: np.ndarray) -> np.ndarray:
    arr = np.asarray(angles, dtype=float).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"angles must have shape (2,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("angles contain NaN or Inf")
    return arr


def _as_state_vector(vector: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.shape != (STATE_SIZE,):
        raise ValueError(f"{name} must have shape ({STATE_SIZE},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _as_covariance(covariance: np.ndarray) -> np.ndarray:
    arr = np.asarray(covariance, dtype=float)
    if arr.shape != (STATE_SIZE, STATE_SIZE):
        raise ValueError(f"covariance must have shape (6,6), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("covariance contains NaN or Inf")
    return _symmetrize(arr)


def _as_information(information: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(information, dtype=float)
    if arr.shape != (STATE_SIZE, STATE_SIZE):
        raise ValueError(f"{name} must have shape (6,6), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return _symmetrize(arr)
