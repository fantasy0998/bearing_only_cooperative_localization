"""Two-stage, truth-free distributed BO-EPnP + GNBP window pipeline.

Graph rows are observers and columns targets. Sparse edge indices always refer
to absolute trajectory columns. Evaluation truth is deliberately absent from
these types. Use exp_node_gnbp for existing NPZ loading and evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.bo_epnp_solver import BO_EPnP
from src.bearing_only_solver import bgpnp, bearing_linear_solver
from src.bgPnp_GABP import _build_prior_covariance
from src.bearing_segmentation import (
    BearingSegmentationConfig, segment_bearings_buchin, supplement_segment_candidates,
)
from src.gnbp_keypoint_selection import GNBPKeypointSelectionConfig, EdgeTimeKey, _Atom, _select_local
from src import multi_agent_bearing_gabp as kernel
from src.node_gnbp_backend import (
    UAVNode, ObservationFactor, NodeExecutor, freeze_states, readonly, refine_nodes,
)
from src.selection_trace import capture


@dataclass(frozen=True)
class NodeGNBPConfig:
    """Independent frontend, selection, segmentation and GN/BP settings."""
    initialization_method: str = "bo_epnp"
    anchor_selection_strategy: str = "rank_logdet"
    pairwise_selection_strategy: str = "schur_greedy"
    anchor_budget_per_node: int = 12
    pairwise_budget_per_node: int = 6
    anchor_budget_values: tuple[int, ...] = (6, 12, 18)
    pairwise_budget_values: tuple[int, ...] = (4, 6, 8)
    segmentation_enabled: bool = True
    representatives_per_segment: int = 1
    max_angle_deg: float = 3.0
    candidate_multiplier: float = 2.0
    max_time_gap_s: float | None = None
    candidate_scope: str = "full_window"
    max_gn_iterations: int = 8
    max_bp_iterations: int = 30
    gn_tolerance: float = 1e-7
    bp_tolerance: float = 1e-8
    message_damping: float = 0.6
    step_damping: float = 1.0
    diagonal_regularization: float = 1e-9
    prior_enabled: bool = True
    sigma_prior_rotation_deg: float = 5.0
    sigma_prior_translation: float = 0.5
    random_seed: int = 0
    trial_id: int = 0
    executor: str = "serial"
    workers: int = 2
    anchor_rank_rtol: float = 1e-10
    anchor_logdet_regularization: float = 1e-8
    schur: GNBPKeypointSelectionConfig = field(default_factory=lambda:
        GNBPKeypointSelectionConfig(candidate_policy="pairwise_only", candidate_scope="full_window"))

    def __post_init__(self) -> None:
        if self.initialization_method not in ("bo_epnp", "bgpnp", "sdp", "bls"):
            raise ValueError("Unknown initialization_method.")
        if self.anchor_selection_strategy not in ("random", "uniform_time", "rank_logdet"):
            raise ValueError("Unknown anchor_selection_strategy.")
        if self.pairwise_selection_strategy not in ("random", "uniform_time", "schur_greedy"):
            raise ValueError("Unknown pairwise_selection_strategy.")
        for name in ("anchor_budget_per_node", "max_gn_iterations", "max_bp_iterations", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("pairwise_budget_per_node", "random_seed", "trial_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("anchor_budget_values", "pairwise_budget_values"):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers.")
        if self.candidate_scope != "full_window":
            raise ValueError("Node pipeline candidates must use full_window.")
        if self.executor not in ("serial", "thread"):
            raise ValueError("executor must be serial or thread.")
        for name in ("gn_tolerance", "bp_tolerance", "diagonal_regularization", "step_damping",
                     "sigma_prior_rotation_deg", "sigma_prior_translation", "anchor_rank_rtol",
                     "anchor_logdet_regularization"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.message_damping) or not 0 < self.message_damping <= 1:
            raise ValueError("message_damping must lie in (0,1].")
        self.segmentation_config()
        if (self.schur.candidate_policy != "pairwise_only" or self.schur.max_total_factors_per_node is not None
                or self.schur.bearing_segmentation is not None or self.schur.candidate_scope != "full_window"):
            raise ValueError("Node pipeline controls scope, segmentation and extra budgets itself.")

    def segmentation_config(self) -> BearingSegmentationConfig:
        """Translate the common configuration to the reusable segmenter."""
        return BearingSegmentationConfig(self.max_angle_deg, self.representatives_per_segment,
                                         self.candidate_multiplier, self.max_time_gap_s)

    def backend_settings(self) -> kernel.GaBPSettings:
        """Pass independent iteration limits to the actual numerical scheduler."""
        names = kernel.GaBPSettings.__dataclass_fields__
        return kernel.GaBPSettings(**{name: getattr(self, name) for name in names})


@dataclass(frozen=True)
class DirectedObservations:
    """Sparse raw sensor bearings and known B3<-sensor extrinsics for one edge.

    bearings has shape (3,M); rotations (M,3,3); translation_sensor (M,3)
    follows the legacy s_sensor = Q.T @ s_B3 + e convention.
    """
    observer_id: str
    target_id: str
    absolute_indices: np.ndarray
    bearings: np.ndarray
    sigma_azimuth: float
    sigma_elevation: float
    rotation_b3_sensor: np.ndarray | None = None
    translation_sensor: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        idx = np.asarray(self.absolute_indices)
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(np.diff(idx) <= 0):
            raise ValueError("Edge absolute indices must be increasing nonnegative integers.")
        if np.shape(self.bearings) != (3, len(idx)):
            raise ValueError("Edge bearings must have shape (3,M).")
        for value in (self.sigma_azimuth, self.sigma_elevation):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Angular noise must be positive radians.")
        q = np.broadcast_to(np.eye(3), (len(idx), 3, 3)) if self.rotation_b3_sensor is None else self.rotation_b3_sensor
        e = np.zeros((len(idx), 3)) if self.translation_sensor is None else self.translation_sensor
        valid = np.ones(len(idx), dtype=bool) if self.valid_mask is None else np.asarray(self.valid_mask)
        if np.shape(q) != (len(idx), 3, 3) or np.shape(e) != (len(idx), 3) or valid.shape != (len(idx),):
            raise ValueError("Extrinsics or valid_mask are not aligned with edge indices.")
        if valid.dtype.kind != "b":
            raise ValueError("valid_mask must be boolean.")
        idx = idx.astype(int, copy=True); idx.setflags(write=False)
        mask = valid.copy(); mask.setflags(write=False)
        object.__setattr__(self, "absolute_indices", idx)
        object.__setattr__(self, "valid_mask", mask)
        for name, value in (("bearings", self.bearings), ("rotation_b3_sensor", q), ("translation_sensor", e)):
            object.__setattr__(self, name, readonly(value))

    @property
    def edge_key(self) -> str:
        """Use existing ID spelling without splitting IDs at underscores."""
        return f"{self.observer_id}_{self.target_id}"


@dataclass(frozen=True)
class WindowInput:
    """Validated graph, trajectories and measurements; intentionally no truth."""
    window_id: str
    node_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    estimated_ids: tuple[str, ...]
    observation_adjacency: np.ndarray
    absolute_indices: np.ndarray
    timestamps: np.ndarray | None
    anchor_position_world: Mapping[str, np.ndarray]
    estimated_position_local: Mapping[str, np.ndarray]
    edges: tuple[DirectedObservations, ...]

    def __post_init__(self) -> None:
        nodes, anchors, estimated = set(self.node_ids), set(self.anchor_ids), set(self.estimated_ids)
        if (len(nodes) != len(self.node_ids) or len(anchors) != len(self.anchor_ids)
                or len(estimated) != len(self.estimated_ids) or anchors & estimated
                or anchors | estimated != nodes or not estimated or not anchors):
            raise ValueError("Node roles must be unique, disjoint and cover node_ids.")
        if any(not isinstance(node, str) or not node for node in self.node_ids):
            raise ValueError("Node IDs must be nonempty strings.")
        matrix = np.asarray(self.observation_adjacency)
        if (matrix.shape != (len(nodes), len(nodes)) or not np.all(np.isin(matrix, (0, 1)))
                or np.any(np.diag(matrix))):
            raise ValueError("Invalid directed observation_adjacency.")
        for anchor in anchors:
            if np.any(matrix[self.node_ids.index(anchor)]):
                raise ValueError(f"Unsupported anchor-as-observer direction: {anchor}")
        points = {**self.anchor_position_world, **self.estimated_position_local}
        if set(self.anchor_position_world) != anchors or set(self.estimated_position_local) != estimated:
            raise ValueError("Trajectory keys must match graph roles.")
        length = next(iter(points.values())).shape[-1]
        if any(np.shape(p) != (3, length) for p in points.values()):
            raise ValueError("All trajectories must have shape (3,N).")
        idx = np.asarray(self.absolute_indices)
        if (idx.ndim != 1 or idx.dtype.kind not in "iu" or not len(idx) or np.any(idx < 0)
                or np.any(idx >= length) or np.any(np.diff(idx) <= 0)):
            raise ValueError("Window absolute indices must be increasing and in bounds.")
        if self.timestamps is not None:
            ts = np.asarray(self.timestamps)
            if ts.shape != (length,) or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
                raise ValueError("Timestamps must be finite, increasing and aligned to trajectories.")
            object.__setattr__(self, "timestamps", readonly(ts))
        seen, names = set(), set()
        for edge in self.edges:
            pair = (edge.observer_id, edge.target_id)
            if pair in seen or edge.edge_key in names or not set(pair) <= nodes:
                raise ValueError("Unknown or ambiguous/duplicate directed edge.")
            seen.add(pair); names.add(edge.edge_key)
            if not matrix[self.node_ids.index(pair[0]), self.node_ids.index(pair[1])]:
                raise ValueError(f"Data for undeclared edge: {pair}")
            if len(edge.absolute_indices) and edge.absolute_indices[-1] >= length:
                raise ValueError("Edge absolute index outside trajectory.")
        idx = idx.astype(int, copy=True); idx.setflags(write=False)
        object.__setattr__(self, "absolute_indices", idx)
        object.__setattr__(self, "observation_adjacency", readonly(matrix))
        object.__setattr__(self, "anchor_position_world", MappingProxyType({k: readonly(v) for k,v in self.anchor_position_world.items()}))
        object.__setattr__(self, "estimated_position_local", MappingProxyType({k: readonly(v) for k,v in self.estimated_position_local.items()}))


def stage_rng(config: NodeGNBPConfig, window: str, node: str, stage: str) -> np.random.Generator:
    """Derive a stable independent random stream unaffected by worker completion."""
    digest = hashlib.sha256(json.dumps([config.random_seed, config.trial_id, window, node, stage]).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))


def sample_candidates(keys: list[EdgeTimeKey], count: int, strategy: str,
                      timestamps: np.ndarray | None, rng: np.random.Generator) -> list[EdgeTimeKey]:
    """Random without replacement or true-time uniform nearest unused sampling."""
    keys = sorted(keys)
    count = min(count, len(keys))
    if not count:
        return []
    if strategy == "random":
        return [keys[i] for i in rng.choice(len(keys), count, replace=False)]
    if strategy != "uniform_time":
        raise ValueError("Expected random or uniform_time.")
    def stamp(key: EdgeTimeKey) -> float:
        return float(timestamps[key.time_index]) if timestamps is not None else float(key.time_index)
    remaining = sorted(keys, key=lambda k: (stamp(k), k))
    targets = np.linspace(stamp(remaining[0]), stamp(remaining[-1]), count)
    selected = []
    for target in targets:
        best = min(remaining, key=lambda k: (abs(stamp(k) - target), stamp(k), k))
        selected.append(best); remaining.remove(best)
    return selected


def _edge_lookup(window: WindowInput) -> dict[tuple[str, str], DirectedObservations]:
    return {(e.observer_id, e.target_id): e for e in window.edges}


def _edge_point(window: WindowInput, key: EdgeTimeKey) -> tuple:
    edge = _edge_lookup(window)[(key.observer_id, key.target_id)]
    pos = int(np.searchsorted(edge.absolute_indices, key.time_index))
    if pos >= len(edge.absolute_indices) or edge.absolute_indices[pos] != key.time_index:
        raise ValueError("Unknown absolute measurement index.")
    return edge, pos


def segment_node(node: UAVNode, payload: tuple) -> None:
    """Segment every owned edge before either selection stage or linearization."""
    window, cfg = payload
    started = time.perf_counter()
    edge_map = _edge_lookup(window)
    times = window.timestamps
    if times is None:
        times = np.arange(node.local_trajectory.shape[1], dtype=float)
    by_phase = {"anchor": {}, "pairwise": {}}
    diagnostics = {}
    for target in window.node_ids:
        if not window.observation_adjacency[window.node_ids.index(node.node_id), window.node_ids.index(target)]:
            continue
        edge = edge_map.get((node.node_id, target))
        name = f"{node.node_id}_{target}"
        phase = "anchor" if target in window.anchor_ids else "pairwise"
        info = {"segments": [], "invalid_inputs": [], "missing_data": edge is None,
                "time_rule": "timestamp_midpoint" if window.timestamps is not None else "lower_middle_index_fallback"}
        diagnostics[name] = info
        by_phase[phase][name] = ()
        if edge is None:
            continue
        positions = np.flatnonzero(np.isin(edge.absolute_indices, window.absolute_indices))
        idx = edge.absolute_indices[positions]
        b, q = edge.bearings[:, positions], edge.rotation_b3_sensor[positions]
        norms = np.linalg.norm(b, axis=0)
        valid = edge.valid_mask[positions].copy() & np.all(np.isfinite(b), axis=0) & (norms > 1e-12)
        valid &= np.all(np.isfinite(q), axis=(1, 2)) & np.all(np.isfinite(edge.translation_sensor[positions]), axis=1)
        if len(q):
            valid &= np.all(np.isclose(q @ q.transpose(0, 2, 1), np.eye(3), atol=1e-7), axis=(1, 2))
            valid &= np.isclose(np.linalg.det(q), 1., atol=1e-7)
        target_points = (window.anchor_position_world if phase == "anchor" else window.estimated_position_local)[target]
        valid &= np.all(np.isfinite(node.local_trajectory[:, idx]), axis=0) & np.all(np.isfinite(target_points[:, idx]), axis=0)
        info["invalid_inputs"] = idx[~valid].tolist()
        # Export the already-computed mask; plotting must not invent validity rules.
        info["valid_absolute_indices"] = idx[valid].tolist()
        info["raw_count"], info["valid_count"] = len(idx), int(valid.sum())
        directions = np.einsum("nij,jn->in", q, b)
        # Explicit invalid observations and sparse missing columns both break runs.
        runs, current = [], []
        for pos in range(len(idx)):
            if not valid[pos] or (current and idx[pos] != idx[current[-1]] + 1):
                if current:
                    runs.append(current); current = []
            if valid[pos]:
                current.append(pos)
        if current:
            runs.append(current)
        segments = []
        for run in runs:
            if cfg.segmentation_enabled:
                result = segment_bearings_buchin(directions[:, run], idx[run], times[idx[run]],
                    max_angle_rad=np.deg2rad(cfg.max_angle_deg), max_time_gap_s=cfg.max_time_gap_s)
                segments.extend(result.segments)
                info["segments"].extend({"indices": s.tolist(), "start_reason": reason if n else "valid_run_start"}
                    for n, (s, reason) in enumerate(zip(result.segments, result.split_reasons)))
            else:
                segments.extend(np.array([int(idx[p])]) for p in run)
                info["segments"].extend({"indices": [int(idx[p])], "start_reason": "segmentation_disabled"} for p in run)
        by_phase[phase][name] = tuple(segments)
    for phase, segments in by_phase.items():
        budget = cfg.anchor_budget_per_node if phase == "anchor" else cfg.pairwise_budget_per_node
        floor = int(np.ceil(cfg.candidate_multiplier * budget)) if cfg.segmentation_enabled else 0
        keep, reps, added = supplement_segment_candidates(segments,
            per_segment=cfg.representatives_per_segment, minimum_count=floor, timestamps=window.timestamps)
        candidates = []
        for name in segments:
            target = next(t for t in window.node_ids if f"{node.node_id}_{t}" == name)
            candidates.extend(EdgeTimeKey(node.node_id, target, int(i)) for i in keep[name])
            diagnostics[name].update({"representatives": {s: v.tolist() for s,v in reps[name].items()},
                "supplement_indices": added[name], "candidate_indices": keep[name].tolist(),
                "candidate_count": len(keep[name]), "candidate_floor": floor,
                "supplement_rule": "largest_uncovered_sample_position_gap"})
        node.candidates[phase] = sorted(candidates)
    node.segments = diagnostics
    node.diagnostics["segmentation_seconds"] = time.perf_counter() - started


def _anchor_arrays(window: WindowInput, keys: list[EdgeTimeKey]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world, local, bearing = [], [], []
    for key in keys:
        edge, pos = _edge_point(window, key)
        q = edge.rotation_b3_sensor[pos]
        world.append(window.anchor_position_world[key.target_id][:, key.time_index])
        local.append(window.estimated_position_local[key.observer_id][:, key.time_index] - q @ edge.translation_sensor[pos])
        direction = q @ edge.bearings[:, pos]
        bearing.append(direction / np.linalg.norm(direction))
    return np.asarray(world), np.asarray(local), np.asarray(bearing)


def anchor_matrix(window: WindowInput, keys: list[EdgeTimeKey]) -> tuple[np.ndarray, dict]:
    """Build fixed pool PCA parameterization of nonhomogeneous 2M-by-12 BO A."""
    world, local, bearings = _anchor_arrays(window, keys)
    controls = BO_EPnP._select_control_points(world, method="pca")
    weights = bgpnp.compute_alphas(world, controls)
    matrix, _ = BO_EPnP._build_sphere_aware_linear_system(local, bearings, weights)
    scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
    matrix = matrix / scale
    # c' contains the 3 coordinates of each of 4 transformed affine controls.
    # This is A c'=b, not a homogeneous camera matrix: no scale gauge removed.
    return matrix.reshape(len(keys), 2, matrix.shape[1]), {
        "system": "nonhomogeneous_A_cprime_equals_b", "columns": matrix.shape[1],
        "effective_target_rank": matrix.shape[1], "control_points": controls.tolist(),
        "column_norm_scale": scale.tolist(), "parameterization_scope": "fixed_candidate_pool"}


def _matrix_metrics(matrix: np.ndarray, rank_tolerance: float, regularization: float) -> tuple[int, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    padded = np.pad(singular**2, (0, matrix.shape[1] - len(singular)))
    return rank, float(np.log(padded + regularization).sum())


def select_anchor(node: UAVNode, window: WindowInput, cfg: NodeGNBPConfig) -> list[EdgeTimeKey]:
    """Share the anchor budget across all visible anchors; never repair random sets."""
    keys = node.candidates["anchor"]
    diagnostic = {"strategy": cfg.anchor_selection_strategy, "candidate_count": len(keys),
                  "requested_count": cfg.anchor_budget_per_node, "history": []}
    node.diagnostics["anchor_selection"] = diagnostic
    if len(keys) < 6:
        raise ValueError(f"Insufficient anchor candidates: {len(keys)} < 6 (12 nonhomogeneous unknowns).")
    blocks, description = anchor_matrix(window, keys)
    full = blocks.reshape(-1, blocks.shape[-1])
    tolerance = cfg.anchor_rank_rtol * max(1., np.linalg.norm(full, 2))
    diagnostic.update(description, rank_tolerance=tolerance,
                      logdet_regularization=cfg.anchor_logdet_regularization)
    diagnostic["pool_rank"] = _matrix_metrics(full, tolerance, cfg.anchor_logdet_regularization)[0]
    count = min(len(keys), cfg.anchor_budget_per_node)
    trace = node.diagnostics.get("selection_trace")
    if cfg.anchor_selection_strategy == "rank_logdet":
        remaining, selected = list(range(len(keys))), []
        current_score = (0, float(full.shape[1] * np.log(cfg.anchor_logdet_regularization)))
        capture(trace, "anchor", None, {},
                {"rank": current_score[0], "logdet": current_score[1]})
        while remaining and len(selected) < count:
            best, score = None, None
            for index in remaining:
                trial = blocks[selected + [index]].reshape(-1, full.shape[1])
                value = _matrix_metrics(trial, tolerance, cfg.anchor_logdet_regularization)
                if (score is None or value[0] > score[0]
                        or (value[0] == score[0] and value[1] > score[1] + 1e-12)):
                    best, score = index, value
            selected.append(best); remaining.remove(best)
            capture(trace, "anchor", keys[best],
                    {"rank": current_score[0], "logdet": current_score[1]},
                    {"rank": score[0], "logdet": score[1]}, winner=score,
                    mode="rank_then_regularized_logdet")
            diagnostic["history"].append({"id": _key_id(keys[best]), "rank": score[0],
                "logdet": score[1], "logdet_gain": score[1] - current_score[1]})
            current_score = score
        chosen = [keys[i] for i in selected]
    else:
        chosen = sample_candidates(keys, count, cfg.anchor_selection_strategy, window.timestamps,
                                   stage_rng(cfg, window.window_id, node.node_id, "anchor"))
        capture(trace, "anchor", None, {}, {}, source=cfg.anchor_selection_strategy)
        for key in chosen:
            capture(trace, "anchor", key, {}, {}, source=cfg.anchor_selection_strategy)
    selected_rank, logdet = _matrix_metrics(blocks[[keys.index(k) for k in chosen]].reshape(-1, full.shape[1]),
                                           tolerance, cfg.anchor_logdet_regularization)
    diagnostic.update({"selected_count": len(chosen), "selected_ids": [_key_id(k) for k in chosen],
        "selected_rank": selected_rank, "logdet": logdet,
        "budget_shortfall_reason": "insufficient_candidates_after_supplement" if len(chosen) < cfg.anchor_budget_per_node else None})
    node.selected["anchor"] = chosen
    if selected_rank < description["effective_target_rank"]:
        raise ValueError(f"Anchor rank insufficient: {selected_rank}/{description['effective_target_rank']}; no resampling/fallback.")
    return chosen


def solve_frontend(window: WindowInput, keys: list[EdgeTimeKey], method: str) -> kernel.PoseState:
    """Concatenate common-world anchor correspondences for the existing frontends."""
    world, local, bearings = _anchor_arrays(window, keys)
    if method == "bo_epnp":
        rotation, translation = BO_EPnP.solve(world, local, bearings, use_wtls=True,
                                             use_ml_refine=True, control_point_method="pca")
    elif method == "bgpnp":
        (rotation, translation, _), _ = bgpnp.solve(world, local, bearings, sol_iter=True, enforce_manifold=True)
    elif method in ("sdp", "bls"):
        solve = bearing_linear_solver.solve_with_sdp_sdr if method == "sdp" else bearing_linear_solver.solve
        (rotation, translation), _ = solve(world.T, local.T, bearings.T)
    else:
        raise ValueError(f"Unsupported frontend: {method}")
    state = kernel.PoseState(np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float).reshape(3))
    if (not np.all(np.isfinite(state.rotation)) or not np.all(np.isfinite(state.translation))
            or not np.allclose(state.rotation.T @ state.rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(state.rotation), 1., atol=1e-6)):
        raise ValueError("Frontend returned invalid/nonfinite pose.")
    return state


def initialize_node(node: UAVNode, payload: tuple) -> None:
    """Run exactly one configured solver; failure leaves the node uninitialized."""
    window, cfg = payload
    started = time.perf_counter()
    node.diagnostics["initialization"] = {"method": cfg.initialization_method, "success": False,
        "frontend_measurement_model": "existing_B3_frontend; BO_ML_uses_legacy_default_diagonal_noise"}
    try:
        selection_start = time.perf_counter()
        chosen = select_anchor(node, window, cfg)
        node.diagnostics["anchor_selection_seconds"] = time.perf_counter() - selection_start
        solve_start = time.perf_counter()
        node.state = solve_frontend(window, chosen, cfg.initialization_method)
        node.initial_state = node.state.copy()
        node.diagnostics["frontend_seconds"] = time.perf_counter() - solve_start
        node.diagnostics["initialization"]["success"] = True
        for key in chosen:
            factor = build_measurement(window, key)
            node.owned_factors[_key_id(key)] = ObservationFactor(_key_id(key), factor)
    except Exception as error:
        # Solver libraries raise their own exception classes (e.g. CVXPY SolverError).
        # Report them per node; never substitute another solver/state or swallow interrupts.
        node.state = node.initial_state = None
        node.owned_factors.clear()
        node.diagnostics["initialization"]["reason"] = f"{type(error).__name__}: {error}"
    node.diagnostics["initialization_seconds"] = time.perf_counter() - started


def build_measurement(window: WindowInput, key: EdgeTimeKey) -> kernel.AnchorBearingFactor | kernel.PairwiseBearingFactor:
    """Retain raw sensor angles, actual noise, local points and original extrinsics."""
    edge, pos = _edge_point(window, key)

    kwargs = dict(observer_id=key.observer_id, measured_angles=kernel.vector_to_az_el(edge.bearings[:, pos]),
        observer_point_local=window.estimated_position_local[key.observer_id][:, key.time_index],
        sigma_azimuth=edge.sigma_azimuth, sigma_elevation=edge.sigma_elevation,
        observer_extrinsic_rotation=edge.rotation_b3_sensor[pos].T,
        observer_extrinsic_translation=edge.translation_sensor[pos])
    
    if key.target_id in window.anchor_ids:
        return kernel.AnchorBearingFactor(anchor_position_world=window.anchor_position_world[key.target_id][:, key.time_index], **kwargs)
    return kernel.PairwiseBearingFactor(target_id=key.target_id,
        target_point_local=window.estimated_position_local[key.target_id][:, key.time_index], **kwargs)


def _geometry_gate(factor: kernel.PairwiseBearingFactor, states: Mapping, cfg: GNBPKeypointSelectionConfig) -> None:
    observer, target = states[factor.observer_id], states[factor.target_id]
    world = target.rotation.T @ (factor.target_point_local - target.translation)
    s = factor.observer_extrinsic_rotation @ (observer.rotation @ world + observer.translation - factor.observer_point_local)
    s = s + factor.observer_extrinsic_translation
    distance = np.linalg.norm(s)
    if not np.all(np.isfinite(s)) or distance < cfg.minimum_distance:
        raise ValueError("invalid_predicted_relative_distance")
    if (np.linalg.norm(s[:2]) / distance < cfg.minimum_horizontal_ratio
            or abs(np.cos(factor.measured_angles[1])) < cfg.minimum_horizontal_ratio):
        raise ValueError("angular_pole")
    if cfg.max_whitened_residual is not None:
        predicted = kernel.vector_to_az_el(s)
        residual = np.array([kernel.wrap_to_pi(factor.measured_angles[0] - predicted[0]) / factor.sigma_azimuth,
                             (factor.measured_angles[1] - predicted[1]) / factor.sigma_elevation])
        if np.linalg.norm(residual) > cfg.max_whitened_residual:
            raise ValueError("whitened_residual_gate")

## 计算现有观测得到的基础信息矩阵，用于判断哪个方向的观测可观性不好
def base_information(node: UAVNode, payload: tuple) -> np.ndarray:
    """Anchor-only selection base: deliberately distinct from backend pose prior."""
    states, scale = payload
    base = np.zeros((6, 6))
    for factor in node.owned_factors.values():
        info = kernel._linearize_anchor_factor(factor.measurement, states)
        base += scale.T @ info.information @ scale
    return readonly(base)


def select_pairwise(node: UAVNode, payload: tuple) -> None:
    """Select only owned extra measurements against immutable initial base blocks."""
    window, cfg, states, base = payload
    started = time.perf_counter()
    selection_cfg = replace(cfg.schur, extra_factors_per_node=cfg.pairwise_budget_per_node)
    scale = np.diag([1., 1., 1., *([selection_cfg.translation_scale] * 3)])
    candidates, rejected, atoms = {}, [], []
    for key in node.candidates["pairwise"]:
        try:
            ## 将观测打包变成一个因子，包含双方局部坐标系位置，位姿变换矩阵，以及bearing数值和噪声
            factor = build_measurement(window, key)
            ## 判断哪些bearing是病态的，主要是判断节点间距离是否过小，预测方向或实测方向是否接近角度极点，如果配置了残差门限，预测角度与实测角度的差异是否过大。
            _geometry_gate(factor, states, selection_cfg)
            candidates[key] = factor
        except (ValueError, np.linalg.LinAlgError) as error:
            rejected.append({"id": _key_id(key), "reason": str(error)})
    build_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    if cfg.pairwise_selection_strategy == "schur_greedy":
        for key, factor in candidates.items():
            ## 建立待估节点之间的信息矩阵
            info = kernel._linearize_pairwise_factor(factor, states)
            ## 用来按照Atom的格式打包bearing观测信息，便于后面执行多轮选点
            atoms.append(_Atom(key, factor, scale.T @ info.omega_ii @ scale,
                               scale.T @ info.omega_ij @ scale, scale.T @ info.omega_jj @ scale))
        chosen, diagnostic = _select_local(
            node.node_id, atoms, set(node.selected["anchor"]), base, selection_cfg,
            trace_sink=node.diagnostics.get("selection_trace"))
    else:
        proposed = sample_candidates(list(candidates), len(candidates), cfg.pairwise_selection_strategy,
                                     window.timestamps, stage_rng(cfg, window.window_id, node.node_id, "pairwise"))
        # Apply shared spacing/edge caps without scoring or geometry repair.
        # Uniform targets must use the requested count rather than the whole pool.
        if cfg.pairwise_selection_strategy == "uniform_time":
            proposed = sample_candidates(list(candidates), cfg.pairwise_budget_per_node, "uniform_time",
                                         window.timestamps, stage_rng(cfg, window.window_id, node.node_id, "pairwise"))
        chosen, counts = [], {}
        for key in proposed:
            if len(chosen) >= cfg.pairwise_budget_per_node:
                break
            if (selection_cfg.max_extra_factors_per_edge is not None
                    and counts.get(key.edge_key, 0) >= selection_cfg.max_extra_factors_per_edge):
                continue
            if any(old.edge_key == key.edge_key and abs(old.time_index-key.time_index) < selection_cfg.minimum_index_separation for old in chosen):
                continue
            chosen.append(key); counts[key.edge_key] = counts.get(key.edge_key, 0) + 1
        diagnostic = {"trial_score_count": 0, "stop_reason": "budget_exhausted" if len(chosen) == cfg.pairwise_budget_per_node else "no_eligible_candidates"}
        trace = node.diagnostics.get("selection_trace")
        capture(trace, "pairwise", None, {}, {}, source=cfg.pairwise_selection_strategy)
        for key in chosen:
            capture(trace, "pairwise", key, {}, {}, source=cfg.pairwise_selection_strategy)
    for key in chosen:
        node.owned_factors[_key_id(key)] = ObservationFactor(_key_id(key), candidates[key])
    node.selected["pairwise"] = chosen
    diagnostic.update({"strategy": cfg.pairwise_selection_strategy, "requested_count": cfg.pairwise_budget_per_node,
        "candidate_count": len(node.candidates["pairwise"]), "eligible_count": len(candidates),
        "selected_count": len(chosen), "selected_ids": [_key_id(k) for k in chosen],
        "rejected_candidates": rejected, "candidate_linearizations": len(atoms),
        "budget_shortfall_reason": diagnostic["stop_reason"] if len(chosen) < cfg.pairwise_budget_per_node else None,
        "construction_seconds": build_seconds, "scoring_seconds": time.perf_counter() - score_started,
        "base_information_source": "retained_anchors_only_no_pose_prior_no_BP_belief"})
    node.diagnostics["pairwise_selection"] = diagnostic
    node.diagnostics["pairwise_selection_seconds"] = time.perf_counter() - started


def _key_id(key: EdgeTimeKey) -> tuple[str, str, int]:
    return key.observer_id, key.target_id, int(key.time_index)


def pose_dict(state: kernel.PoseState | None) -> dict | None:
    """Serialize a pose without evaluation data."""
    return None if state is None else {"rotation": state.rotation.tolist(), "translation": state.translation.tolist()}


@dataclass
class NodeWindowResult:
    """Node objects for programmatic inspection plus JSON-compatible diagnostics."""
    nodes: dict[str, UAVNode]
    record: dict[str, Any]


def solve_window(window: WindowInput, config: NodeGNBPConfig | None = None,
                 *, trace_selection: bool = False) -> NodeWindowResult:
    """Execute all initialization barriers before selection/BP; no hidden fallback."""
    cfg = config or NodeGNBPConfig()
    started = time.perf_counter()
    nodes = {key: UAVNode(key, local_trajectory=window.estimated_position_local[key]) for key in window.estimated_ids}
    if trace_selection: ## 是否详细记录选点过程的开关，主要用于绘图时显示贪心打分函数的变化情况
        for node in nodes.values():
            node.diagnostics["selection_trace"] = []
    # 已提供的观测边
    present = {(edge.observer_id, edge.target_id) for edge in window.edges}
    # 拓扑中包含，但数据中未找到的观测边
    missing = [(i,j) for row,i in enumerate(window.node_ids) for col,j in enumerate(window.node_ids)
               if window.observation_adjacency[row,col] and (i,j) not in present]
    # 在后端是否加入位姿先验，开启先验会生成一个6*6协方差矩阵，增加先验是为了让最终结果靠近初始化的结果，即使不增加先验，后续节点也会从因子中获得协方差信息
    covariance = _build_prior_covariance(cfg.sigma_prior_rotation_deg, cfg.sigma_prior_translation) if cfg.prior_enabled else None

    record = {"schema_version": 1, "window_id": window.window_id, "absolute_indices": window.absolute_indices.tolist(),
        "config": asdict(cfg), "graph": {"node_ids": window.node_ids, "anchor_ids": window.anchor_ids,
        "estimated_ids": window.estimated_ids, "observation_adjacency": window.observation_adjacency.astype(int).tolist(),
        "convention": "row_observer_column_target", "missing_declared_edges": missing},
        "prior": {"enabled": cfg.prior_enabled, "mean_source": "configured_frontend_initial_pose",
            "covariance": covariance.tolist() if covariance is not None else None,
            "kernel": "multi_agent_bearing_gabp._build_prior_info_factors",
            "relinearized_each_gn": True, "initial_mean_delta": [0.] * 6,
            "observation_correlation": "initial pose uses retained anchors; existing prior behavior retained"},
        "time_range": None if window.timestamps is None else [float(window.timestamps[window.absolute_indices[0]]), float(window.timestamps[window.absolute_indices[-1]])],
        "status": "initialization_failed", "failed_node_ids": [], "timings": {}, "backend": None}
    with NodeExecutor(cfg.executor, cfg.workers) as executor:
        for name, operation in (("segmentation", segment_node), ("initialization", initialize_node)):
            ## 在这针对每个点执行轨迹分段并得到与锚节点观测的初始化位姿 
            stage_started = time.perf_counter()
            executor.map_nodes(nodes, operation, (window, cfg))
            if name == "initialization":
                initial_ready = time.perf_counter() - started
            record["timings"][name + "_wall_seconds"] = time.perf_counter() - stage_started
        record["failed_node_ids"] = sorted(key for key,node in nodes.items() if node.state is None)
        record["timings"]["time_to_initial_estimate_wall_seconds"] = initial_ready if not record["failed_node_ids"] else None
        record["timings"]["time_to_final_estimate_wall_seconds"] = None
        if not record["failed_node_ids"]:
            # 冻结的意思是保存一个位姿副本出来，原来node.state可以继续更新
            states = freeze_states({key: node.state for key,node in nodes.items()})
            scale = np.diag([1., 1., 1., *([cfg.schur.translation_scale] * 3)])
            stage_started = time.perf_counter()
            try:
                ## 把锚节点的bearing因子变成信息矩阵
                bases = executor.map_nodes(nodes, base_information, (states, readonly(scale)))
                base = MappingProxyType(dict(zip(sorted(nodes), bases)))
                ## 选点
                executor.map_nodes(nodes, select_pairwise, (window, cfg, states, base))
                record["timings"]["pairwise_selection_wall_seconds"] = time.perf_counter() - stage_started
                record["status"] = "backend_failed"
                record["backend"] = refine_nodes(nodes, executor=executor, settings=cfg.backend_settings(),
                                                   prior_covariance=covariance, window_id=window.window_id)
                record["timings"]["time_to_final_estimate_wall_seconds"] = time.perf_counter() - started
                record["status"] = "completed"  # Convergence flags are separate, never inferred from completion.
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                record["status"] = "backend_failed" if record["status"] == "backend_failed" else "selection_failed"
                record["failure_reason"] = f"{type(error).__name__}: {error}"
    record["nodes"] = {key: {"initial_pose": pose_dict(node.initial_state), "final_pose": pose_dict(node.state),
        "segments": node.segments, "candidates": {stage: [_key_id(k) for k in keys] for stage,keys in node.candidates.items()},
        "selected": {stage: [_key_id(k) for k in keys] for stage,keys in node.selected.items()},
        "owned_factor_ids": list(node.owned_factors), "incident_factor_ids": list(node.incident_factor_ids),
        "diagnostics": node.diagnostics} for key,node in nodes.items()}
    if record["backend"] is not None:
        record["timings"]["gnbp_wall_seconds"] = record["backend"]["gnbp_wall_seconds"]
    record["timings"]["pre_backend_node_compute_seconds_sum"] = sum(sum(float(value) for name,value in n.diagnostics.items()
        if name in ("segmentation_seconds", "initialization_seconds", "pairwise_selection_seconds")) for n in nodes.values())
    record["timings"]["end_to_end_wall_seconds"] = time.perf_counter() - started
    record["recording"] = {"selection_trace_enabled": trace_selection}
    return NodeWindowResult(nodes, record)
