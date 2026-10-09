"""One-snapshot, observer-owned Schur keypoint selection before GN/BP.

Public entry point: select_gnbp_keypoints. No pose updates, BP, network calls,
ground truth access, or initialization-derived priors are used during selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import TYPE_CHECKING

import numpy as np

from src.bearing_segmentation import (
    BearingSegmentationConfig, segment_bearings_buchin, supplement_segment_candidates,
)
from src.multi_agent_bearing_gabp import (
    AnchorBearingFactor,
    PairwiseBearingFactor,
    PoseState,
    _linearize_anchor_factor as linearize_anchor_factor,
    _linearize_pairwise_factor as linearize_pairwise_factor,
    vector_to_az_el,
)

if TYPE_CHECKING:
    from src.bgPnp_GABP import LoadedObservationBatch, MultiAgentWindowSample


@dataclass(frozen=True)
class GNBPKeypointSelectionConfig:
    """Extra-factor budgets per observer; angular units are radians.

    translation_scale is a fixed characteristic length in trajectory units.
    minimum_index_separation applies within one directed edge, including its
    retained initialization observations. Numerical tolerances apply after
    scaling delta_xi = diag(1,1,1,L,L,L) delta_z.
    """

    extra_factors_per_node: int = 8
    max_extra_factors_per_edge: int | None = None
    minimum_index_separation: int = 0
    translation_scale: float = 1.0
    stop_mode: str = "budget"
    information_threshold: float | None = None
    rank_rtol: float = 1e-10
    rank_atol: float = 1e-10
    logdet_regularization: float = 1e-8
    minimum_e_gain: float = 1e-10
    minimum_logdet_gain: float = 1e-10
    minimum_distance: float = 1e-8
    minimum_horizontal_ratio: float = 1e-6
    max_whitened_residual: float | None = None
    candidate_policy: str = "anchors_and_pairwise"
    candidate_scope: str = "full_window"
    max_total_factors_per_node: int | None = None

    bearing_segmentation: BearingSegmentationConfig | None = None

    def __post_init__(self) -> None:
        if self.bearing_segmentation is not None and self.candidate_policy != "pairwise_only":
            raise ValueError("Bearing segmentation supports only pairwise_only.")
        if self.candidate_policy not in ("anchors_and_pairwise", "pairwise_only"):
            raise ValueError("Unknown candidate_policy.")
        if self.candidate_scope not in ("full_window", "anchor_union"):
            raise ValueError("Unknown candidate_scope.")
        if self.max_total_factors_per_node is not None:
            _nonnegative_integer(self.max_total_factors_per_node, "max_total_factors_per_node")
        for name in ("extra_factors_per_node", "minimum_index_separation"):
            _nonnegative_integer(getattr(self, name), name)
        if self.max_extra_factors_per_edge is not None:
            _nonnegative_integer(
                self.max_extra_factors_per_edge, "max_extra_factors_per_edge"
            )
        for name in (
            "translation_scale", "rank_rtol", "rank_atol",
            "logdet_regularization", "minimum_distance",
            "minimum_horizontal_ratio",
        ):
            _positive_finite(getattr(self, name), name)
        for name in ("minimum_e_gain", "minimum_logdet_gain"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("information_threshold", "max_whitened_residual"):
            if getattr(self, name) is not None:
                _positive_finite(getattr(self, name), name)
        if self.minimum_horizontal_ratio >= 1:
            raise ValueError("minimum_horizontal_ratio must be less than one.")
        if self.stop_mode not in ("budget", "information_threshold"):
            raise ValueError("Unknown stop_mode.")
        if self.stop_mode == "information_threshold" and self.information_threshold is None:
            raise ValueError("information_threshold is required for threshold stopping.")


@dataclass(frozen=True, order=True)
class EdgeTimeKey:
    """One directed observation; time_index is an absolute batch index."""

    observer_id: str
    target_id: str
    time_index: int

    @property
    def edge_key(self) -> str:
        """Return the existing batch edge naming convention."""
        return f"{self.observer_id}_{self.target_id}"


@dataclass
class GNBPKeypointSelectionResult:
    """Selected factors and independent per-edge indices for one window.

    refinement_indices includes retained anchors exactly once. selection_order
    contains only additional observations, in greedy order for each observer.
    Factors are in the same measurement frame and use the same noise weights
    as the information used for selection; pass them directly to refine_with_gabp.
    """

    initialization_indices: dict[str, np.ndarray]
    refinement_indices: dict[str, np.ndarray]
    selection_order: dict[str, tuple[EdgeTimeKey, ...]]
    anchor_factors: list[AnchorBearingFactor]
    pairwise_factors: list[PairwiseBearingFactor]
    diagnostics: dict[str, object]

    def to_metadata(self) -> dict[str, object]:
        """Return serializable metadata without copying measurement factors."""
        return {
            "initialization_indices": {
                edge: indices.tolist() for edge, indices in self.initialization_indices.items()
            },
            "refinement_indices": {
                edge: indices.tolist() for edge, indices in self.refinement_indices.items()
            },
            "selection_order": {
                node: [(key.edge_key, key.time_index) for key in keys]
                for node, keys in self.selection_order.items()
            },
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class _Atom:
    key: EdgeTimeKey
    factor: AnchorBearingFactor | PairwiseBearingFactor
    ii: np.ndarray
    ij: np.ndarray | None = None
    jj: np.ndarray | None = None


def select_gnbp_keypoints(
    batch: LoadedObservationBatch,
    sample: MultiAgentWindowSample,
    initial_states: dict[str, PoseState],
    *,
    config: GNBPKeypointSelectionConfig | None = None,
    measurement_mode: str = "b4",
) -> GNBPKeypointSelectionResult:
    """Select second-stage bearing factors using fixed local Schur information.

    Args:
        batch: Raw trajectories, bearings and noise. Reference/GT fields are
            never read. B4 mode requires bearing_observation_b4 and per-observer
            rotation_b3_b4; b3_diagonal_approx explicitly opts into legacy noise.
        sample: Candidate window_indices and retained anchor_edge_keypoint_indices.
            All indices are absolute columns of batch arrays. Retained anchors
            cannot be removed and do not consume the additional-factor budget.
        initial_states: Initial world-to-local PoseState for each estimated node,
            from BO-EPnP, BGPnP or another initializer. Input states are not changed.
        config: Per-node/per-edge budgets, fixed state scale and stopping policy.
        measurement_mode: "b4" (default, original sensor noise frame), or
            "b3_diagonal_approx" (explicit diagonal B3 noise approximation).

    Returns:
        Per-edge final indices, greedy extra selection order, original nonlinear
        anchor/pairwise factors, and per-node rank/eigenvalue/stop diagnostics.
        This is an offline simulation of one neighbor snapshot: every local loop
        uses only its observer-owned candidates and fixed neighbor base blocks.

    Raises:
        ValueError: Invalid inputs, invalid retained observations, unavailable
            requested measurement frame, or numerically inconsistent Schur blocks.
        Invalid optional candidates are skipped and listed in diagnostics.
    """
    started = time.perf_counter()
    cfg = config or GNBPKeypointSelectionConfig()
    states, window, retained, atoms, rejected, preparation = _prepare_atoms(
        batch, sample, initial_states, cfg, measurement_mode
    )

    greedy_started = time.perf_counter()
    # The sole snapshot: compute every node's anchor-only base before any selection.
    base = {node: np.zeros((6, 6)) for node in states}
    for key in sorted(retained):
        base[key.observer_id] += atoms[key].ii
    selected = set(retained)
    orders: dict[str, tuple[EdgeTimeKey, ...]] = {}
    node_diagnostics: dict[str, object] = {}
    for node in sorted(states):
        owned = [atom for key, atom in sorted(atoms.items())
                 if key.observer_id == node and key not in retained]
        # No other observer's bearings or new selections enter this local loop.
        node_retained = {key for key in retained if key.observer_id == node}
        budget = cfg.extra_factors_per_node
        if cfg.max_total_factors_per_node is not None:
            budget = cfg.max_total_factors_per_node - len(node_retained)
            if budget < 0:
                raise ValueError(f"Total budget below retained anchor count for {node}.")
        order, diagnostics = _select_local(
            node, owned, node_retained, base, replace(cfg, extra_factors_per_node=budget)
        )
        diagnostics.update({
            "retained_anchor_count": len(node_retained),
            "eligible_candidate_count": len(owned),
            "requested_total_budget": cfg.max_total_factors_per_node,
            "effective_extra_budget": budget,
            "total_owned_factor_count": len(node_retained) + len(order),
        })
        selected.update(order)
        orders[node] = tuple(order)
        node_diagnostics[node] = diagnostics

    preparation["greedy_seconds"] = time.perf_counter() - greedy_started
    if preparation["bearing_segmentation"]["enabled"]:
        preparation["bearing_segmentation"]["selection_by_observer"] = node_diagnostics
        preparation["bearing_segmentation"]["greedy_trial_score_count"] = sum(
            d["trial_score_count"] for d in node_diagnostics.values())
    return _pack_result(sample, selected, atoms, orders, {
        **preparation,
        "method": "local_schur", "measurement_mode": measurement_mode,
        "prior_mode": "none", "translation_scale": cfg.translation_scale,
        "config": asdict(cfg), "budget_scope": "observer",
        "window_indices": window.tolist(), "nodes": node_diagnostics,
        "rejected_candidates": rejected,
        "selection_seconds": time.perf_counter() - started,
    })


def build_gnbp_candidate_factors(
    batch: LoadedObservationBatch,
    sample: MultiAgentWindowSample,
    initial_states: dict[str, PoseState],
    *,
    config: GNBPKeypointSelectionConfig | None = None,
    measurement_mode: str = "b4",
) -> GNBPKeypointSelectionResult:
    """Build retained anchors plus all valid candidates for a matched baseline.

    Uses the selector's identical frame, noise, candidate policy and quality gates.
    Budgets, spacing and information-gain stopping do not apply to this baseline.
    selection_order here is deterministic ID/time order, not a greedy order.
    """
    started = time.perf_counter()
    cfg = config or GNBPKeypointSelectionConfig(candidate_policy="pairwise_only")
    if cfg.bearing_segmentation is not None:
        raise ValueError("The all-candidate baseline must not enable bearing segmentation.")
    states, window, retained, atoms, rejected, preparation = _prepare_atoms(
        batch, sample, initial_states, cfg, measurement_mode
    )
    orders = {
        node: tuple(key for key in sorted(atoms)
                    if key.observer_id == node and key not in retained)
        for node in states
    }
    return _pack_result(sample, set(atoms), atoms, orders, {
        **preparation, "greedy_seconds": 0.0,
        "method": "all_candidates", "measurement_mode": measurement_mode,
        "prior_mode": "none", "config": asdict(cfg),
        "budget_scope": "observer", "budgets_applied": False,
        "window_indices": window.tolist(), "rejected_candidates": rejected,
        "selection_seconds": time.perf_counter() - started,
    })


def _prefilter_bearing_candidates(batch, states, retained, edges, candidate_times, cfg, mode):
    """Convert measured directions to frozen B3 and compress each directed edge."""
    scfg = cfg.bearing_segmentation
    max_angle_rad = np.deg2rad(scfg.max_angle_deg)
    diagnostics = {"enabled": True, "criterion": "maximum_pairwise_3d_angular_diameter",
                   "config": asdict(scfg), "frame": "b3_frozen_from_b4" if mode == "b4" else "b3_frozen",
                   "edges": {}, "observers": {}, "invalid_inputs": []}
    keep_by_edge = {}
    times = np.array(sorted(candidate_times), dtype=int)
    observations = batch.bearing_observation_b4 if mode == "b4" else batch.bearing_observation
    for observer in sorted(states):
        anchors = sum(key.observer_id == observer for key in retained)
        budget = cfg.extra_factors_per_node if cfg.max_total_factors_per_node is None else cfg.max_total_factors_per_node - anchors
        if budget < 0:
            raise ValueError(f"Total budget below retained anchor count for {observer}.")
        segments_by_edge = {}
        for i, target in edges:
            if i != observer or target not in states:
                continue
            edge = f"{i}_{target}"
            info = {"raw_scope_candidate_count": len(times), "valid_input_candidates": 0,
                    "segments": [], "invalid_inputs": [], "fast_check_count": 0,
                    "fast_accept_count": 0, "exact_pair_comparisons": 0}
            diagnostics["edges"][edge] = info
            if budget == 0:
                info["skipped_reason"] = "zero_extra_budget"
                segments_by_edge[edge] = ()
                continue
            b = np.asarray(observations[edge][:, times], dtype=float)
            norms = np.linalg.norm(b, axis=0)
            valid_b = np.all(np.isfinite(b), axis=0) & np.isfinite(norms) & (norms >= 1e-12)
            valid_q = np.ones(len(times), dtype=bool)
            if mode == "b4":
                q = np.asarray(batch.observer_rotation_b3_b4[observer][times], dtype=float)
                # Match _build_atom's E=Q.T orthogonality check and tolerances.
                finite_q = np.all(np.isfinite(q), axis=(1, 2))
                valid_q = finite_q.copy()
                if np.any(finite_q):
                    good_q = q[finite_q]
                    valid_q[finite_q] = (np.all(np.isclose(good_q @ good_q.transpose(0, 2, 1), np.eye(3), atol=1e-7), axis=(1, 2))
                                        & np.isclose(np.linalg.det(good_q), 1.0, atol=1e-7))
                b = np.einsum("nij,jn->in", q, b)
            ts = np.asarray(batch.timestamps)[times]
            valid_t = np.isfinite(ts)
            valid = valid_b & valid_q & valid_t
            for pos in np.flatnonzero(~valid):
                reason = 'invalid_bearing' if not valid_b[pos] else ('invalid_rotation' if not valid_q[pos] else 'invalid_timestamp')
                invalid = {"edge": edge, "index": int(times[pos]), "reason": reason}
                info["invalid_inputs"].append(invalid)
                diagnostics["invalid_inputs"].append(invalid)
            info["valid_input_candidates"] = int(valid.sum())
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            stops = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
            segments = []
            for start, stop in zip(starts, stops):
                result = segment_bearings_buchin(b[:, start:stop], times[start:stop], ts[start:stop],
                    max_angle_rad=max_angle_rad, max_time_gap_s=scfg.max_time_gap_s)
                for name in ('fast_check_count', 'fast_accept_count', 'exact_pair_comparisons'):
                    info[name] += getattr(result, name)
                for sid, (segment, reason) in enumerate(zip(result.segments, result.split_reasons)):
                    if sid == 0 and start > 0:
                        reason = 'invalid_input_break'
                    info["segments"].append({"indices": segment.tolist(), "start_index": int(segment[0]),
                        "end_index": int(segment[-1]), "length": len(segment), "start_reason": reason})
                    segments.append(segment)
            segments_by_edge[edge] = tuple(segments)
        valid_count = sum(diagnostics["edges"][edge]["valid_input_candidates"] for edge in segments_by_edge)
        floor = min(valid_count, int(np.ceil(scfg.candidate_multiplier * budget)))
        keep, representatives, added = supplement_segment_candidates(segments_by_edge,
            per_segment=scfg.representatives_per_segment, minimum_count=floor,
            timestamps=np.asarray(batch.timestamps))
        keep_by_edge.update({edge: set(values.tolist()) for edge, values in keep.items()})
        kept_count = sum(map(len, keep.values()))
        diagnostics["observers"][observer] = {"retained_anchor_count": anchors,
            "extra_budget": budget, "candidate_floor": floor, "valid_input_candidates": valid_count,
            "kept_prefilter": kept_count, "candidate_reduction": 1-kept_count/valid_count if valid_count else None}
        for edge in segments_by_edge:
            info = diagnostics["edges"][edge]
            info["initial_representatives"] = {sid: values.tolist() for sid, values in representatives[edge].items()}
            info["budget_supplement_indices"] = added[edge]
            info["keep_indices"] = keep[edge].tolist()
            info["kept_prefilter"] = len(keep[edge])
            count = info["valid_input_candidates"]
            info["candidate_reduction"] = 1-len(keep[edge])/count if count else None
    return keep_by_edge, diagnostics


def _prepare_atoms(batch, sample, initial_states, cfg, mode):
    states, window, retained, edges = _validate_inputs(batch, sample, initial_states, mode)
    candidate_times = (set(key.time_index for key in retained)
                       if cfg.candidate_scope == "anchor_union" else set(window.tolist()))
    started = time.perf_counter()
    keep, segmentation = None, {"enabled": False}
    if cfg.bearing_segmentation is not None:
        keep, segmentation = _prefilter_bearing_candidates(batch, states, retained, edges, candidate_times, cfg, mode)
    preparation = {"bearing_segmentation": segmentation,
                   "segmentation_and_representatives_seconds": time.perf_counter()-started if keep is not None else 0.0}
    started = time.perf_counter()
    scale = np.diag([1.0] * 3 + [cfg.translation_scale] * 3)
    atoms, rejected = {}, []
    attempts = valid_optional = 0
    for observer, target in edges:
        for index in window:
            key = EdgeTimeKey(observer, target, int(index))
            if key not in retained and (
                index not in candidate_times
                or (cfg.candidate_policy == "pairwise_only" and target not in states)
                or (keep is not None and index not in keep.get(key.edge_key, set()))
            ):
                continue
            attempts += int(key not in retained)
            try:
                atoms[key] = _build_atom(batch, key, states, scale, cfg, mode)
                valid_optional += int(key not in retained)
            except ValueError as exc:
                if key in retained:
                    raise ValueError(f"Invalid retained observation {key}: {exc}") from exc
                rejected.append({"edge": key.edge_key, "index": key.time_index,
                                 "reason": str(exc)})
    preparation.update(atom_construction_seconds=time.perf_counter()-started,
                       optional_atom_construction_count=attempts, valid_optional_atom_count=valid_optional)
    if keep is not None:
        segmentation.update(optional_atom_construction_count=attempts, valid_optional_atom_count=valid_optional)
    return states, window, retained, atoms, rejected, preparation


def _pack_result(sample, selected, atoms, orders, diagnostics):
    indices: dict[str, list[int]] = {}
    anchor_factors: list[AnchorBearingFactor] = []
    pairwise_factors: list[PairwiseBearingFactor] = []
    for key in sorted(selected):
        indices.setdefault(key.edge_key, []).append(key.time_index)
        factor = atoms[key].factor
        if isinstance(factor, AnchorBearingFactor):
            anchor_factors.append(factor)
        else:
            pairwise_factors.append(factor)
    return GNBPKeypointSelectionResult(
        initialization_indices={edge: np.asarray(value, dtype=int).copy()
                                for edge, value in sample.anchor_edge_keypoint_indices.items()},
        refinement_indices={edge: np.asarray(value, dtype=int) for edge, value in indices.items()},
        selection_order=orders,
        anchor_factors=anchor_factors,
        pairwise_factors=pairwise_factors,
        diagnostics={**diagnostics,
            "anchor_factor_count": len(anchor_factors),
            "pairwise_factor_count": len(pairwise_factors),
        },
    )


def _nonnegative_integer(value: int, name: str) -> None:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer.")


def _positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _indices(values: np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D integer array.")
    if np.any(array < 0) or np.any(array >= length) or np.unique(array).size != array.size:
        raise ValueError(f"{name} has duplicate or out-of-range indices.")
    return array.astype(int, copy=True)


def _validate_inputs(batch, sample, initial_states, mode):
    if mode not in ("b4", "b3_diagonal_approx"):
        raise ValueError("measurement_mode must be b4 or b3_diagonal_approx.")
    nodes = tuple(sample.estimated_nodes)
    if not nodes or len(set(nodes)) != len(nodes) or set(nodes) != set(initial_states):
        raise ValueError("initial_states must contain exactly the estimated nodes.")
    if set(nodes) & set(sample.anchor_nodes):
        raise ValueError("Anchor and estimated nodes must be disjoint.")
    states = {}
    for node in nodes:
        rotation = np.asarray(initial_states[node].rotation, dtype=float)
        translation = np.asarray(initial_states[node].translation, dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)):
            raise ValueError(f"Invalid initial pose for {node}.")
        states[node] = PoseState(rotation.copy(), translation.copy())
    length = len(batch.timestamps)
    window = _indices(sample.window_indices, "window_indices", length)
    if np.any(window < sample.start) or np.any(window >= sample.stop):
        raise ValueError("window_indices must lie in sample [start, stop).")
    for name in ("sigma_azimuth_deg", "sigma_elevation_deg"):
        _positive_finite(getattr(batch, name), name)
    observations = (getattr(batch, "bearing_observation_b4", None)
                    if mode == "b4" else batch.bearing_observation)
    rotations = getattr(batch, "observer_rotation_b3_b4", None)
    if observations is None or (mode == "b4" and rotations is None):
        raise ValueError(
            "B4 selection requires bearing_observation_b4 "
            "and observer_rotation_b3_b4."
        )
    edges = []
    for observer in nodes:
        for target in (*sample.anchor_nodes, *nodes):
            edge = f"{observer}_{target}"
            if target == observer or edge not in batch.bearing_observation:
                continue
            if edge not in observations or np.shape(observations[edge]) != (3, length):
                raise ValueError(f"Missing or invalid bearing array for {edge} in {mode}.")
            for node in (observer, target):
                points = (batch.anchor_position_w if node in sample.anchor_nodes
                          else batch.estimated_position_l)
                if node not in points or np.shape(points[node]) != (3, length):
                    raise ValueError(f"Invalid trajectory array for {node}.")
            if mode == "b4" and (
                observer not in rotations
                or np.shape(rotations[observer]) != (length, 3, 3)
            ):
                raise ValueError(f"Missing or invalid B3/B4 rotations for {observer}.")
            edges.append((observer, target))
    anchor_edges = {f"{i}_{j}": (i, j) for i, j in edges if j in sample.anchor_nodes}
    retained = set()
    window_set = set(window.tolist())
    for edge, values in sample.anchor_edge_keypoint_indices.items():
        if edge not in anchor_edges:
            raise ValueError(f"Unknown retained anchor edge: {edge}.")
        indices = _indices(values, edge, length)
        if not set(indices.tolist()) <= window_set:
            raise ValueError(f"Retained indices for {edge} lie outside the candidate window.")
        retained.update(EdgeTimeKey(*anchor_edges[edge], int(index)) for index in indices)
    if {key.observer_id for key in retained} != set(nodes):
        raise ValueError("Each estimated node needs retained initialization anchor observations.")
    return states, np.sort(window), retained, sorted(edges)


def _build_atom(batch, key, states, scale, cfg, mode):
    index = key.time_index
    observer = states[key.observer_id]
    point = np.asarray(batch.estimated_position_l[key.observer_id][:, index], dtype=float).copy()
    observations = batch.bearing_observation_b4 if mode == "b4" else batch.bearing_observation
    bearing = np.asarray(observations[key.edge_key][:, index], dtype=float)
    if not np.all(np.isfinite(bearing)) or np.linalg.norm(bearing) < 1e-12:
        raise ValueError("Nonfinite or zero bearing.")
    E = np.eye(3)
    if mode == "b4":
        E = np.asarray(batch.observer_rotation_b3_b4[key.observer_id][index], dtype=float).T.copy()
        if (not np.all(np.isfinite(E)) or not np.allclose(E.T @ E, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(E), 1.0, atol=1e-7)):
            raise ValueError("Invalid B3-to-B4 rotation.")
    kwargs = dict(
        observer_id=key.observer_id, measured_angles=vector_to_az_el(bearing),
        observer_point_local=point, observer_extrinsic_rotation=E,
        sigma_azimuth=np.deg2rad(batch.sigma_azimuth_deg),
        sigma_elevation=np.deg2rad(batch.sigma_elevation_deg),
    )
    if key.target_id in states:
        target = states[key.target_id]
        target_point = np.asarray(
            batch.estimated_position_l[key.target_id][:, index], dtype=float
        ).copy()
        world = target.rotation.T @ (target_point - target.translation)
        factor = PairwiseBearingFactor(
            target_id=key.target_id, target_point_local=target_point, **kwargs
        )
    else:
        world = np.asarray(batch.anchor_position_w[key.target_id][:, index], dtype=float).copy()
        factor = AnchorBearingFactor(anchor_position_world=world, **kwargs)
    s = E @ (observer.rotation @ world + observer.translation - point)
    distance = np.linalg.norm(s)
    if not np.all(np.isfinite(s)) or distance < cfg.minimum_distance:
        raise ValueError("Invalid predicted relative distance.")
    observed_horizontal_ratio = np.linalg.norm(bearing[:2]) / np.linalg.norm(bearing)
    if (np.linalg.norm(s[:2]) / distance < cfg.minimum_horizontal_ratio
            or observed_horizontal_ratio < cfg.minimum_horizontal_ratio):
        raise ValueError("Bearing is too close to an angular pole.")
    if isinstance(factor, AnchorBearingFactor):
        info = linearize_anchor_factor(factor, states)
        atom = _Atom(key, factor, scale.T @ info.information @ scale)
    else:
        info = linearize_pairwise_factor(factor, states)
        atom = _Atom(key, factor, scale.T @ info.omega_ii @ scale,
                     scale.T @ info.omega_ij @ scale, scale.T @ info.omega_jj @ scale)
    if cfg.max_whitened_residual is not None and info.residual_norm > cfg.max_whitened_residual:
        raise ValueError("Whitened residual exceeds the configured quality gate.")
    return atom


def _psd(matrix, cfg):
    matrix = (matrix + matrix.T) / 2
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Nonfinite information matrix.")
    values, vectors = np.linalg.eigh(matrix)
    tolerance = max(cfg.rank_atol, cfg.rank_rtol * max(1.0, float(np.max(np.abs(values)))))
    if values[0] < -tolerance:
        raise ValueError("Information matrix is not positive semidefinite within tolerance.")
    values = np.maximum(values, 0.0)
    return (vectors * values) @ vectors.T, values, vectors, tolerance


def _schur(self_block, cross, neighbor, cfg):
    marginal = self_block.copy()
    pseudoinverse_used = False
    for node, block in sorted(cross.items()):
        if not np.any(block):
            continue
        matrix, values, vectors, tolerance = _psd(neighbor[node], cfg)
        if values[0] > tolerance:
            chol = np.linalg.cholesky(matrix)
            solution = np.linalg.solve(chol.T, np.linalg.solve(chol, block.T))
        else:
            pseudoinverse_used = True
            keep = values > tolerance
            basis = vectors[:, keep]
            projection = basis @ (basis.T @ block.T)
            range_tolerance = max(
                cfg.rank_atol, cfg.rank_rtol * max(1.0, np.linalg.norm(block))
            )
            if np.linalg.norm(projection - block.T) > range_tolerance:
                raise ValueError(
                    "Schur cross block is outside the retained neighbor range; "
                    "review rank tolerance."
                )
            solution = (basis / values[keep]) @ (basis.T @ block.T)
        marginal -= block @ solution
    return _psd(marginal, cfg)[0], pseudoinverse_used


def _add_atom(self_block, cross, neighbor, atom):
    trial_cross, trial_neighbor = cross.copy(), neighbor.copy()
    if atom.ij is not None:
        target = atom.key.target_id
        trial_cross[target] = cross[target] + atom.ij
        trial_neighbor[target] = neighbor[target] + atom.jj
    return self_block + atom.ii, trial_cross, trial_neighbor


def _metrics(matrix, rank_tolerance, cfg):
    values = np.maximum(np.linalg.eigvalsh(matrix), 0.0)
    return {
        "rank": int(np.count_nonzero(values > rank_tolerance)),
        "lambda_min": float(values[0]),
        "logdet": float(np.log(values + cfg.logdet_regularization).sum()),
    }


def _select_local(node, candidates, retained, base, cfg):
    neighbors = sorted({atom.key.target_id for atom in candidates if atom.ij is not None})
    self_block = base[node].copy()
    cross = {target: np.zeros((6, 6)) for target in neighbors}
    neighbor = {target: base[target].copy() for target in neighbors}
    current, used_pinv = _schur(self_block, cross, neighbor, cfg)
    rank_tolerance = max(cfg.rank_atol, cfg.rank_rtol * max(1.0, np.linalg.norm(current, 2)))
    initial_metrics = _metrics(current, rank_tolerance, cfg)
    metrics = initial_metrics.copy()
    selected = []
    history = []
    edge_counts: dict[str, int] = {}
    active = set(retained)
    trial_score_count = 0
    stop = "budget_exhausted"
    for _ in range(cfg.extra_factors_per_node):
        reached = (metrics["rank"] == 6 and cfg.information_threshold is not None
                   and metrics["lambda_min"] >= cfg.information_threshold)
        if reached and cfg.stop_mode == "information_threshold":
            stop = "information_threshold"
            break
        mode = "rank" if metrics["rank"] < 6 else ("d" if reached else "e")
        best = None
        for atom in candidates:
            key = atom.key
            if key in active:
                continue
            if (cfg.max_extra_factors_per_edge is not None
                    and edge_counts.get(key.edge_key, 0) >= cfg.max_extra_factors_per_edge):
                continue
            if any(old.edge_key == key.edge_key
                   and abs(old.time_index - key.time_index) < cfg.minimum_index_separation
                   for old in active):
                continue
            trial_score_count += 1
            blocks = _add_atom(self_block, cross, neighbor, atom)
            trial, trial_pinv = _schur(*blocks, cfg)
            trial_metrics = _metrics(trial, rank_tolerance, cfg)
            dr = trial_metrics["rank"] - metrics["rank"]
            de = trial_metrics["lambda_min"] - metrics["lambda_min"]
            dd = trial_metrics["logdet"] - metrics["logdet"]
            score = (dr, dd) if mode == "rank" else ((dd,) if mode == "d" else (de, dd))
            # Candidates are sorted by directed ID/time; strict comparison retains ties.
            if best is None or score > best[0]:
                best = (score, atom, blocks, trial, trial_metrics, trial_pinv, dr, de, dd)
        if best is None:
            stop = "no_eligible_candidates"
            break
        score, atom, blocks, trial, updated, trial_pinv, dr, de, dd = best
        useful = (dr > 0 or dd > cfg.minimum_logdet_gain) if mode == "rank" else (
            dd > cfg.minimum_logdet_gain if mode == "d" else
            de > cfg.minimum_e_gain or dd > cfg.minimum_logdet_gain
        )
        if not useful:
            stop = "no_information_gain"
            break
        self_block, cross, neighbor = blocks
        current, metrics = trial, updated
        used_pinv |= trial_pinv
        active.add(atom.key)
        selected.append(atom.key)
        edge_counts[atom.key.edge_key] = edge_counts.get(atom.key.edge_key, 0) + 1
        history.append({"edge": atom.key.edge_key, "index": atom.key.time_index,
                        "mode": mode, "score": tuple(float(x) for x in score), **metrics})
    return selected, {
        "trial_score_count": trial_score_count,
        "initial": initial_metrics, "final": metrics, "history": history,
        "stop_reason": stop, "selected_extra_count": len(selected),
        "used_pseudoinverse": used_pinv, "rank_tolerance": rank_tolerance,
        "scaled_marginal_information": current.tolist(),
    }
