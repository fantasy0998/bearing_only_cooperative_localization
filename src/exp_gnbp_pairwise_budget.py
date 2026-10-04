"""Fixed-data BGPnP/GNBP comparison with observer-owned pairwise budgets.

Run from repository root: python -m src.exp_gnbp_pairwise_budget
This module never generates observations or changes the BGPnP frontend.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.bgPnp_GABP import (
    BgpnpGabpSolver, BoEpnpFrontEnd, LoadedObservationBatch,
    MiluvObservationReader, MultiAgentWindowSample, UniformKeypointSelector,
    _build_prior_covariance, _solve_anchor_only_algorithm,
    get_pairwise_observation_by_union, slice_window_sample,
)
from src.gnbp_keypoint_selection import (
    GNBPKeypointSelectionConfig,
    build_gnbp_candidate_factors, select_gnbp_keypoints,
)
from src.multi_agent_bearing_gabp import (
    AnchorBearingFactor, GaBPSettings, PairwiseBearingFactor, PoseState,
    refine_with_gabp, so3_log,
)

LOGGER = logging.getLogger(__name__)
DATA_ROOT = Path("data/miluv_bgpnp_gabp_test")
DEFAULT_OBSERVATIONS = (
    DATA_ROOT / "observations/miluv_bearing_az01_el02.npz",
    DATA_ROOT / "observations/miluv_bearing_az02_el03.npz",
)


@dataclass(frozen=True)
class PairwiseBudgetExperimentConfig:
    """Immutable experiment inputs; original NPZ files are always read-only."""

    observation_paths: tuple[Path, ...] = DEFAULT_OBSERVATIONS
    output_root: Path = DATA_ROOT / "pairwise_budget"
    figure_root: Path = Path("docs/figures/gnbp_pairwise_budget")
    keypoint_counts: tuple[int, ...] = (6, 12, 18)
    initialization_selection_count: int = 18
    total_budgets: tuple[int, ...] = (12, 18, 24)
    candidate_scope: str = "anchor_union"
    primary_anchor_id: str = "ifo001"
    include_existing_methods: bool = True
    settings: GaBPSettings = field(default_factory=lambda: GaBPSettings(
        max_gn_iterations=8, max_bp_iterations=30, gn_tolerance=1e-7,
        bp_tolerance=1e-8, message_damping=0.6, step_damping=1.0,
        diagonal_regularization=1e-9,
    ))

    def __post_init__(self) -> None:
        for values in (self.keypoint_counts, self.total_budgets):
            if not values or any(isinstance(x, bool) or not isinstance(x, int)
                                 or x <= 0 for x in values):
                raise ValueError("Counts and budgets must be nonempty positive integers.")
            if len(set(values)) != len(values):
                raise ValueError("Duplicate counts or budgets are not allowed.")
        if self.candidate_scope not in ("anchor_union", "full_window"):
            raise ValueError("Unknown candidate_scope.")
        if self.initialization_selection_count < max(self.keypoint_counts):
            raise ValueError("initialization_selection_count must cover all K values.")
        if not self.observation_paths or len(set(self.observation_paths)) != len(self.observation_paths):
            raise ValueError("Observation paths must be nonempty and unique.")


def file_sha256(path: Path) -> str:
    """Hash a source file without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_initialization_sample(
    batch: LoadedObservationBatch, count: int,
) -> MultiAgentWindowSample:
    """Reproduce WindowSampleBuilder selection without its hardcoded plot writes."""
    window = np.arange(batch.timestamps.size, dtype=int)
    selector = UniformKeypointSelector()
    indices = {
        f"{observer}_{anchor}": selector.select(
            window, count, batch, edge_tuple=(observer, anchor),
            method="linear_system_condition",
        )
        for anchor in batch.anchor_nodes for observer in batch.estimated_nodes
        if f"{observer}_{anchor}" in batch.bearing_observation
    }
    if not indices:
        raise ValueError("No initialization anchor edges.")
    return MultiAgentWindowSample(
        start=0, stop=batch.timestamps.size, window_indices=window,
        anchor_edge_keypoint_indices=indices, anchor_nodes=batch.anchor_nodes,
        estimated_nodes=batch.estimated_nodes, estimated_se3_gt=batch.estimated_se3,
    )


def pose_errors(state: PoseState, truth: dict[str, Any]) -> dict[str, float]:
    """Compute rotation and absolute/relative translation errors in stated units."""
    angle = float(np.linalg.norm(so3_log(state.rotation @ truth["rotation_local_world"].T)))
    distance = float(np.linalg.norm(state.translation - truth["translation_local"]))
    return {"rotation_rad": angle, "rotation_deg": float(np.rad2deg(angle)),
            "translation_abs_m": distance,
            "translation_relative": distance / max(float(np.linalg.norm(truth["translation_local"])), 1e-12)}


def _pose(state):
    return {"R_world_to_local": state.rotation.copy(),
            "t_world_to_local": state.translation.copy()}


def _valid_pose(state):
    return (np.all(np.isfinite(state.rotation)) and np.all(np.isfinite(state.translation))
            and np.allclose(state.rotation.T @ state.rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(state.rotation), 1.0, atol=1e-5))


def count_factors(
    anchors: list[AnchorBearingFactor], pairwise: list[PairwiseBearingFactor],
    nodes: tuple[str, ...], indices: dict[str, Any], retained: dict[str, Any],
) -> dict[str, Any]:
    """Count actual factor objects, assigning a pairwise observation to its observer."""
    by_observer = {}
    for node in nodes:
        a = sum(f.observer_id == node for f in anchors)
        p = sum(f.observer_id == node for f in pairwise)
        by_observer[node] = {"anchor": a, "pairwise_outgoing": p, "total_owned": a + p}
    by_edge = {}
    for edge, values in indices.items():
        retained_count = len(set(values) & set(retained.get(edge, [])))
        by_edge[edge] = {"retained": retained_count, "added": len(values) - retained_count,
                         "final": len(values)}
    total = len(anchors) + len(pairwise)
    if sum(v["final"] for v in by_edge.values()) != total:
        raise ValueError("Factor objects and edge indices disagree.")
    return {"graph": {"anchor": len(anchors), "pairwise": len(pairwise),
                      "observation_total": total, "residual_dimension": 2 * total},
            "by_observer": by_observer, "by_edge": by_edge}


def _record(batch, sample, dataset_hash, method, budget, stage1_seconds):
    return {
        "identity": {"dataset_hash": dataset_hash, "method": method,
                     "K": len(next(iter(sample.anchor_edge_keypoint_indices.values()))),
                     "T": budget, "budget_scope": "observer"},
        "noise_deg": [batch.sigma_azimuth_deg, batch.sigma_elevation_deg],
        "config": {}, "poses": {}, "counts": None,
        "indices": {"initialization_by_edge": sample.anchor_edge_keypoint_indices},
        "timing": {"stage1_selection": stage1_seconds, "initialization": 0.0,
                   "stage2_selection": 0.0, "refinement": 0.0, "total": 0.0},
        "status": "pending", "failure_reason": None,
    }


def _attach_poses(record, batch, initial, final, failures=None):
    failures = failures or {}
    for node in batch.estimated_nodes:
        truth = batch.estimated_se3[node]
        before, after = initial.get(node), final.get(node)
        reason = failures.get(node)
        if after is not None and not _valid_pose(after):
            reason = "nonfinite_or_invalid_rotation"
        record["poses"][node] = {
            "initial": _pose(before) if before is not None else None,
            "final": _pose(after) if after is not None else None,
            "truth": {"R_world_to_local": truth["rotation_local_world"].copy(),
                      "t_world_to_local": truth["translation_local"].copy()},
            "errors_initial": pose_errors(before, truth) if before is not None else None,
            "errors_final": pose_errors(after, truth) if after is not None and reason is None else None,
            "status": "success" if after is not None and reason is None else "failed",
            "failure_reason": reason or (None if after is not None else "no_final_pose"),
        }
    record["status"] = ("success" if all(p["status"] == "success" for p in record["poses"].values())
                        else "failed")


def _anchor_estimates(batch, sample, anchor, method):
    states, errors, node_times = {}, {}, {}
    for node in sample.estimated_nodes:
        started = time.perf_counter()
        try:
            estimate = (BoEpnpFrontEnd.solve_for_agent(batch, sample, node, anchor)
                        if method == "bo_epnp" else
                        _solve_anchor_only_algorithm(batch, sample, node, anchor, method))
            state = PoseState(estimate.rotation.copy(), estimate.translation.copy())
            if not _valid_pose(state):
                raise ValueError("nonfinite_or_invalid_rotation")
            states[node] = state
        except Exception as exc:
            errors[node] = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("%s failed for node %s", method, node)
        node_times[node] = time.perf_counter() - started
    return states, errors, node_times


def _legacy_factors(batch, sample, anchor):
    sliced = get_pairwise_observation_by_union(batch, sample)
    anchors = BgpnpGabpSolver._build_anchor_factors(sliced, anchor)
    pairwise = BgpnpGabpSolver._build_pairwise_factors(sliced)
    indices = {edge: sliced.keypoint_indices.copy() for edge in sliced.bearing_observation
               if edge.split("_", 1)[1] == anchor
               or edge.split("_", 1)[1] in sample.estimated_nodes}
    return anchors, pairwise, indices


def _refinement_record(batch, sample, initial, init_errors, init_time, record, cfg):
    method = record["identity"]["method"]
    selection_cfg = GNBPKeypointSelectionConfig(
        candidate_policy="pairwise_only", candidate_scope=cfg.candidate_scope,
        max_total_factors_per_node=record["identity"]["T"],
    )
    if method == "local_schur_mixed":
        selection_cfg = GNBPKeypointSelectionConfig(
            extra_factors_per_node=8, max_extra_factors_per_edge=6,
        )
    scope = "anchor_union" if method == "legacy_union" else selection_cfg.candidate_scope
    record["identity"]["candidate_scope"] = scope
    record["config"] = {
        "initializer": "bgpnp", "sol_iter": True, "enforce_manifold": True,
        "measurement_mode": "b3_legacy_default_noise" if method == "legacy_union" else "b4",
        "selection": asdict(selection_cfg) if method != "legacy_union" else None,
        "GN_BP": asdict(cfg.settings), "primary_anchor_id": cfg.primary_anchor_id,
        "prior": "initial_pose_5deg_0.5m" if method == "legacy_union" else "none",
    }
    record["timing"]["initialization"] = init_time
    record["timing"]["initialization_reused"] = True
    selection_started = time.perf_counter()
    stage = "initialization"
    try:
        if init_errors or set(initial) != set(sample.estimated_nodes):
            raise ValueError(f"Incomplete BGPnP initialization: {init_errors}")
        stage = "factor_construction_and_selection"
        prior = None
        if method == "legacy_union":
            anchors, pairwise, indices = _legacy_factors(batch, sample, cfg.primary_anchor_id)
            prior = _build_prior_covariance(5.0, 0.5)
        else:
            builder = build_gnbp_candidate_factors if method == "pairwise_all" else select_gnbp_keypoints
            selection = builder(batch, sample, initial, config=selection_cfg, measurement_mode="b4")
            anchors, pairwise = selection.anchor_factors, selection.pairwise_factors
            indices = selection.refinement_indices
            record["selection"] = selection.to_metadata()
        record["timing"]["stage2_selection"] = time.perf_counter() - selection_started
        record["indices"]["refinement_by_edge"] = indices
        record["counts"] = count_factors(anchors, pairwise, sample.estimated_nodes,
                                        indices, sample.anchor_edge_keypoint_indices)
        stage = "refinement"
        started = time.perf_counter()
        result = refine_with_gabp(initial, anchors, pairwise, prior_states=initial,
                                  prior_covariance=prior, settings=cfg.settings)
        record["timing"]["refinement"] = time.perf_counter() - started
        record["refinement"] = {
            "history": [asdict(entry) for entry in result.history],
            "history_residual_evaluation": "before_update",
            "stop_reason": result.stop_reason,
            "initial_residual_norm": result.initial_residual_norm,
            "final_residual_norm": result.final_residual_norm,
            "initial_objective": 0.5 * result.initial_residual_norm**2,
            "final_objective": 0.5 * result.final_residual_norm**2,
            "bp_convergence_certified": False,
        }
        _attach_poses(record, batch, initial, result.states)
    except Exception as exc:
        LOGGER.exception("%s failed", method)
        record["failure_reason"] = f"{stage}: {type(exc).__name__}: {exc}"
        _attach_poses(record, batch, initial, {}, {
            node: init_errors.get(node, record["failure_reason"]) for node in sample.estimated_nodes
        })
    record["timing"]["execution_after_cached_initialization"] = time.perf_counter() - selection_started
    record["timing"]["total"] = (record["timing"]["stage1_selection"] + init_time
                                   + record["timing"]["execution_after_cached_initialization"])
    return record


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8")


def save_records(directory: Path, records: list[dict[str, Any]]) -> None:
    """Persist full poses and parallel flat metric/count tables within one run."""
    np.savez_compressed(directory / "results.npz", schema_version=3,
                        results=np.array(records, dtype=object))
    metric_rows, factor_rows = [], []
    for record in records:
        identity = record["identity"]
        common = {**identity, "sigma_azimuth_deg": record["noise_deg"][0],
                  "sigma_elevation_deg": record["noise_deg"][1], "status": record["status"]}
        counts = record["counts"]
        factor_rows.append({**common, **(counts["graph"] if counts else {}),
                            "failure_reason": record["failure_reason"]})
        for node, pose in record["poses"].items():
            row = {**common, "node": node, "status": pose["status"],
                   "failure_reason": pose["failure_reason"]}
            for phase in ("initial", "final"):
                row.update({f"{phase}_{key}": value for key, value in (pose[f"errors_{phase}"] or {}).items()})
            if counts:
                row.update(counts["by_observer"][node])
                row["graph_observation_total"] = counts["graph"]["observation_total"]
            row.update({f"seconds_{key}": value for key, value in record["timing"].items()
                        if isinstance(value, float)})
            refinement = record.get("refinement", {})
            row["gn_iterations"] = len(refinement.get("history", []))
            row["bp_iterations_total"] = sum(x["bp_iterations"] for x in refinement.get("history", []))
            row["stop_reason"] = refinement.get("stop_reason")
            metric_rows.append(row)
    for name, rows in (("per_agent_metrics.csv", metric_rows), ("factor_counts.csv", factor_rows)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def validate_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Verify saved all/selected comparisons; failures remain explicitly recorded."""
    paired = 0
    for record in records:
        if record["status"] != "success":
            continue
        for pose in record["poses"].values():
            final = pose["final"]
            if not _valid_pose(PoseState(final["R_world_to_local"], final["t_world_to_local"])):
                raise ValueError("Invalid persisted pose.")
        counts = record["counts"]
        if counts["graph"]["observation_total"] != sum(x["total_owned"] for x in counts["by_observer"].values()):
            raise ValueError("Owned factor counts do not sum to graph total.")
        identity = record["identity"]
        if identity["method"] != "pairwise_selected":
            continue
        reference = next(r for r in records if r["identity"]["dataset_hash"] == identity["dataset_hash"]
                         and r["identity"]["K"] == identity["K"]
                         and r["identity"]["method"] == "pairwise_all")
        if reference["status"] != "success":
            raise ValueError("Selected record has no successful matched reference.")
        for node, pose in record["poses"].items():
            for key in ("R_world_to_local", "t_world_to_local"):
                np.testing.assert_array_equal(pose["initial"][key], reference["poses"][node]["initial"][key])
            if counts["by_observer"][node]["total_owned"] > identity["T"]:
                raise ValueError("Per-observer total budget exceeded.")
        indices = record["indices"]
        for edge, retained in indices["initialization_by_edge"].items():
            np.testing.assert_array_equal(sorted(retained), indices["refinement_by_edge"][edge])
        for edge, selected in indices["refinement_by_edge"].items():
            if len(set(selected)) != len(selected):
                raise ValueError("Duplicate observation factor.")
            if not set(selected) <= set(reference["indices"]["refinement_by_edge"].get(edge, [])):
                raise ValueError("Selected factors are not a matched-all subset.")
        paired += 1
    return {"records": len(records), "paired_selected_records": paired,
            "failed_records": sum(r["status"] != "success" for r in records)}


def run_experiment(config: PairwiseBudgetExperimentConfig | None = None) -> Path:
    """Run fixed observations and return a unique directory of results/manifest."""
    cfg = config or PairwiseBudgetExperimentConfig()
    hashes = {str(path.resolve()): file_sha256(path) for path in cfg.observation_paths}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    output = cfg.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 3, "run_id": run_id, "status": "running", "config": asdict(cfg),
        "source_hashes_before": hashes, "datasets": [], "skipped": [],
        "python": sys.version, "executable": sys.executable, "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "timing_policy": "single run, no warmup; reused stage1 and BGPnP costs charged to each method; excludes IO/plots",
        "pose_convention": "p_local = R_world_to_local @ p_world + t_world_to_local",
        "observations": "noisy bearings synthesized from MILUV mocap; fixed files",
        "source_code_sha256": {
            name: file_sha256(Path("src") / name) for name in (
                "exp_gnbp_pairwise_budget.py", "gnbp_keypoint_selection.py",
                "bgPnp_GABP.py", "multi_agent_bearing_gabp.py",
                "bearing_only_solver.py", "bo_epnp_solver.py",
            )
        },
    }
    records = []
    _write_json(output / "manifest.json", manifest)
    handler = logging.FileHandler(output / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    old_level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    try:
        with (output / "solver_stdout.txt").open("w", encoding="utf-8") as solver_output, redirect_stdout(solver_output):
            for path in cfg.observation_paths:
                batch = MiluvObservationReader.load(path)
                if batch.bearing_observation_b4 is None or batch.observer_rotation_b3_b4 is None:
                    raise ValueError(f"B4 arrays missing in {path}; refusing to regenerate data.")
                dataset_hash = hashes[str(path.resolve())]
                manifest["datasets"].append({
                    "path": str(path.resolve()), "sha256": dataset_hash,
                    "samples": len(batch.timestamps), "timestamp_start": float(batch.timestamps[0]),
                    "timestamp_end": float(batch.timestamps[-1]), "nodes": batch.robot_ids,
                    "edges": sorted(batch.bearing_observation),
                    "noise_deg": [batch.sigma_azimuth_deg, batch.sigma_elevation_deg],
                })
                LOGGER.info("Selecting initialization points: %s", path.name)
                started = time.perf_counter()
                full_sample = build_initialization_sample(batch, cfg.initialization_selection_count)
                stage1_seconds = time.perf_counter() - started
                for k in cfg.keypoint_counts:
                    sample = slice_window_sample(full_sample, np.arange(k, dtype=int))
                    initial, init_errors, init_node_times = _anchor_estimates(
                        batch, sample, cfg.primary_anchor_id, "bgpnp")
                    init_time = sum(init_node_times.values())
                    if cfg.include_existing_methods:
                        for method in ("bls", "bsdp", "bo_epnp", "bgpnp"):
                            record = _record(batch, sample, dataset_hash, method, None, stage1_seconds)
                            record["identity"]["candidate_scope"] = "anchor_initialization"
                            states, errors, node_times = ((initial, init_errors, init_node_times)
                                if method == "bgpnp" else _anchor_estimates(batch, sample, cfg.primary_anchor_id, method))
                            record["config"] = {"algorithm": method, "primary_anchor_id": cfg.primary_anchor_id,
                                                "measurement_mode": "b3_frontend",
                                                "frontend_options": ({"sol_iter": True, "enforce_manifold": True}
                                                    if method == "bgpnp" else {"use_wtls": True, "use_ml_refine": True,
                                                        "control_point_method": "pca"} if method == "bo_epnp" else {})}
                            record["timing"].update(initialization=sum(node_times.values()),
                                                    total=stage1_seconds + sum(node_times.values()))
                            indices = {f"{node}_{cfg.primary_anchor_id}": sample.anchor_edge_keypoint_indices[f"{node}_{cfg.primary_anchor_id}"]
                                       for node in sample.estimated_nodes}
                            record["indices"]["refinement_by_edge"] = indices
                            record["counts"] = {"graph": {"anchor": sum(map(len, indices.values())), "pairwise": 0,
                                "observation_total": sum(map(len, indices.values())), "residual_dimension": 2 * sum(map(len, indices.values()))},
                                "by_observer": {node: {"anchor": len(indices[f"{node}_{cfg.primary_anchor_id}"]), "pairwise_outgoing": 0,
                                    "total_owned": len(indices[f"{node}_{cfg.primary_anchor_id}"])} for node in sample.estimated_nodes},
                                "by_edge": {edge: {"retained": len(values), "added": 0, "final": len(values)} for edge, values in indices.items()}}
                            _attach_poses(record, batch, states if method == "bgpnp" else {}, states, errors)
                            record["failure_reason"] = str(errors) if errors else None
                            records.append(record)
                    variants = [("pairwise_all", None)]
                    if cfg.include_existing_methods:
                        variants = [("legacy_union", None), ("local_schur_mixed", None)] + variants
                    retained_counts = [sum(len(v) for edge, v in sample.anchor_edge_keypoint_indices.items()
                                           if edge.split("_", 1)[0] == node) for node in sample.estimated_nodes]
                    for budget in cfg.total_budgets:
                        if budget < max(retained_counts):
                            manifest["skipped"].append({"dataset_hash": dataset_hash, "K": k, "T": budget,
                                                        "reason": "total_budget_below_retained_anchors"})
                        else:
                            variants.append(("pairwise_selected", budget))
                    for method, budget in variants:
                        LOGGER.info("Running %s K=%d T=%s noise=%s", method, k, budget, path.name)
                        record = _record(batch, sample, dataset_hash, method, budget, stage1_seconds)
                        record = _refinement_record(batch, sample, initial, init_errors, init_time, record, cfg)
                        records.append(record)
                        save_records(output, records)
                        LOGGER.info("Finished %s status=%s factors=%s", method, record["status"],
                                    record["counts"]["graph"] if record["counts"] else None)
        manifest["status"] = "complete"
        manifest["record_count"] = len(records)
        manifest["failed_record_count"] = sum(r["status"] != "success" for r in records)
        manifest["validation"] = validate_records(records)
        figures = cfg.figure_root / run_id
        figures.mkdir(parents=True, exist_ok=False)
        manifest["figures"] = [str(p) for p in plot_results(records, figures)]
    except Exception:
        manifest["status"] = "aborted"
        LOGGER.exception("Experiment aborted; partial results retained at %s", output)
        raise
    finally:
        after = {path: file_sha256(Path(path)) for path in hashes}
        manifest["source_hashes_after"] = after
        manifest["data_unchanged"] = after == hashes
        save_records(output, records)
        _write_json(output / "manifest.json", manifest)
        LOGGER.removeHandler(handler)
        handler.close()
        LOGGER.setLevel(old_level)
        if after != hashes:
            raise RuntimeError("Source observation hashes changed during experiment.")
    return output


def plot_results(records: list[dict[str, Any]], directory: Path) -> list[Path]:
    """Plot measured tradeoffs, initial/final errors, runtime and selected indices."""
    paths = []
    dataset_ids = list(dict.fromkeys(r["identity"]["dataset_hash"] for r in records))
    for dataset_number, dataset in enumerate(dataset_ids):
        subset = [r for r in records if r["identity"]["dataset_hash"] == dataset]
        nodes = list(subset[0]["poses"])
        selected = [r for r in subset if r["identity"]["method"] == "pairwise_selected" and r["status"] == "success"]
        matched = [r for r in subset if r["identity"]["method"] == "pairwise_all" and r["status"] == "success"]
        fig, axes = plt.subplots(len(nodes), 2, figsize=(11, 3.5 * len(nodes)), squeeze=False)
        for row, node in enumerate(nodes):
            for col, metric in enumerate(("rotation_deg", "translation_abs_m")):
                ax = axes[row, col]
                for k in sorted(set(r["identity"]["K"] for r in selected)):
                    group = sorted((r for r in selected if r["identity"]["K"] == k), key=lambda r: r["identity"]["T"])
                    ax.plot([r["counts"]["graph"]["observation_total"] for r in group],
                            [r["poses"][node]["errors_final"][metric] for r in group], "o-", label=f"selected K={k}")
                ax.scatter([r["counts"]["graph"]["observation_total"] for r in matched],
                           [r["poses"][node]["errors_final"][metric] for r in matched], marker="x", color="black", label="matched all")
                ax.set(xlabel="Actual graph observation factors", ylabel=metric, title=node)
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
        fig.suptitle(f"Noise az/el (deg): {subset[0]['noise_deg']}")
        paths.append(_save_figure(fig, directory / f"noise_{dataset_number}_error_factors.png"))
        main = [r for r in selected if r["identity"]["T"] == r["identity"]["K"] + 6]
        fig, axes = plt.subplots(len(nodes), 2, figsize=(10, 3.5 * len(nodes)), squeeze=False)
        for row, node in enumerate(nodes):
            for col, metric in enumerate(("rotation_deg", "translation_abs_m")):
                for phase in ("initial", "final"):
                    axes[row, col].plot([r["identity"]["K"] for r in main],
                        [r["poses"][node][f"errors_{phase}"][metric] for r in main], "o-", label=phase)
                axes[row, col].set(xlabel="K (T=K+6)", ylabel=metric, title=node)
                axes[row, col].legend()
                axes[row, col].grid(alpha=.25)
        paths.append(_save_figure(fig, directory / f"noise_{dataset_number}_poses.png"))
        timing_rows = sorted(main + matched, key=lambda r: (r["identity"]["K"], r["identity"]["method"]))
        fig, ax = plt.subplots(figsize=(11, 4))
        bottom = np.zeros(len(timing_rows))
        for key in ("stage1_selection", "initialization", "stage2_selection", "refinement"):
            values = np.array([r["timing"][key] for r in timing_rows])
            ax.bar(np.arange(len(values)), values, bottom=bottom, label=key)
            bottom += values
        ax.set_xticks(np.arange(len(timing_rows)), [f"{r['identity']['method']}\nK={r['identity']['K']}" for r in timing_rows], fontsize=8)
        ax.set_ylabel("Seconds (shared setup cost charged per method)")
        ax.legend(fontsize=8)
        paths.append(_save_figure(fig, directory / f"noise_{dataset_number}_timing.png"))
        fig, axes = plt.subplots(max(1, len(main)), 1, figsize=(12, 3 * max(1, len(main))), squeeze=False)
        for row, record in enumerate(main):
            ax = axes[row, 0]
            indices = record["indices"]
            edges = sorted(indices["refinement_by_edge"])
            for y, edge in enumerate(edges):
                retained = set(indices["initialization_by_edge"].get(edge, []))
                final = set(indices["refinement_by_edge"][edge])
                ax.scatter(sorted(retained), [y] * len(retained), marker="s", color="tab:blue")
                ax.scatter(sorted(final - retained), [y] * len(final - retained), marker="x", color="tab:orange")
            ax.set_yticks(range(len(edges)), edges)
            ax.set(xlabel="Absolute sample index", title=f"K={record['identity']['K']} T={record['identity']['T']}; blue=retained anchors, orange=added pairwise")
        paths.append(_save_figure(fig, directory / f"noise_{dataset_number}_keypoints.png"))
        fig, axes = plt.subplots(len(nodes), 2, figsize=(12, 4 * len(nodes)), squeeze=False)
        methods = list(dict.fromkeys(r["identity"]["method"] for r in subset))
        for row, node in enumerate(nodes):
            for col, metric in enumerate(("rotation_deg", "translation_abs_m")):
                ax = axes[row, col]
                for method in methods:
                    group = [r for r in subset if r["identity"]["method"] == method
                             and r["status"] == "success"
                             and (method != "pairwise_selected" or r["identity"]["T"] == r["identity"]["K"] + 6)]
                    if group:
                        ax.plot([r["identity"]["K"] for r in group],
                                [r["poses"][node]["errors_final"][metric] for r in group],
                                "o-", label=method)
                ax.set(xlabel="Initialization K (selected T=K+6)", ylabel=metric,
                       title=node, yscale="log")
                ax.grid(alpha=.25)
                ax.legend(fontsize=7)
        paths.append(_save_figure(fig, directory / f"noise_{dataset_number}_all_methods.png"))
    return paths


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """CLI: use --smoke for a single fixed-data matched comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--candidate-scope", choices=("anchor_union", "full_window"), default="anchor_union")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = PairwiseBudgetExperimentConfig(candidate_scope=args.candidate_scope)
    if args.smoke:
        cfg = replace(cfg, observation_paths=cfg.observation_paths[:1], keypoint_counts=(6,),
                      total_budgets=(12,), include_existing_methods=False)
    path = run_experiment(cfg)
    LOGGER.info("Results: %s", path.resolve())


if __name__ == "__main__":
    main()
