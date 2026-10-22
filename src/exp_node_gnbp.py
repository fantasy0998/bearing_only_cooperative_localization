"""Read-only fixed-NPZ experiments for the node BO-EPnP/GNBP pipeline.

python -m src.exp_node_gnbp --observations /path/to/existing.npz --window-size 200
No data generation, frontend-result fallback or implicit budget/strategy sweep.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from src.boepnp_gnbp_nodes import DirectedObservations, NodeGNBPConfig, WindowInput, solve_window
from src.gnbp_keypoint_selection import GNBPKeypointSelectionConfig
from src.multi_agent_bearing_gabp import so3_log


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_npz_inputs(path: Path, *, measurement_mode: str = "b4",
                    input_adjacency_convention: str = "target_observer") -> tuple[dict, dict, dict]:
    """Separate observation-only constructor arguments from evaluator-only truth.

    Existing MILUV files use target/observer adjacency. New files can explicitly
    request observer_target; graph-less files infer only actually stored edges.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Existing observation NPZ required: {path}")
    if input_adjacency_convention not in ("target_observer", "observer_target"):
        raise ValueError("Unknown input adjacency convention.")
    if measurement_mode not in ("b4", "b3_diagonal_approx"):
        raise ValueError("Unknown measurement_mode.")
    with np.load(path, allow_pickle=True) as raw:
        observations = raw["observation_data"].item()
        # Only this evaluator adapter reads reference_data; never passed to WindowInput.
        truth = raw["reference_data"].item().get("estimated_se3", {}) if "reference_data" in raw.files else {}
        nodes = tuple(str(i) for i in raw["robot_ids"].tolist())
        anchors = tuple(str(i) for i in raw["anchor_nodes"].tolist())
        estimated = tuple(str(i) for i in raw["estimated_nodes"].tolist())
        length = next(iter(observations["anchor_position_w"].values())).shape[1]
        ts = np.asarray(raw["timestamps"], dtype=float) if "timestamps" in raw.files else None
        sigma_az, sigma_el = np.deg2rad(float(raw["sigma_azimuth_deg"])), np.deg2rad(float(raw["sigma_elevation_deg"]))
        bearings = observations.get("bearing_observation_b4") if measurement_mode == "b4" else observations.get("bearing_observation")
        rotations = observations.get("observer_rotation_b3_b4")
        if bearings is None or (measurement_mode == "b4" and rotations is None):
            raise ValueError("B4 input requires raw bearings and B3<-B4 rotations; use explicit b3_diagonal_approx only for legacy files.")
        mapping = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    key = f"{i}_{j}"
                    if key in mapping:
                        raise ValueError("Ambiguous serialized edge key; supply DirectedObservations programmatically.")
                    mapping[key] = (i,j)
        if not set(bearings) <= mapping.keys():
            raise ValueError(f"Unrecognized NPZ edge IDs: {set(bearings)-mapping.keys()}")
        if "adjacency_matrix" in raw.files:
            adjacency = np.asarray(raw["adjacency_matrix"])
            if input_adjacency_convention == "target_observer":
                adjacency = adjacency.T
            graph_source = f"explicit_npz_{input_adjacency_convention}"
        else:
            adjacency = np.zeros((len(nodes), len(nodes)), dtype=int)
            for key in bearings:
                i,j = mapping[key]
                adjacency[nodes.index(i), nodes.index(j)] = 1
            graph_source = "inferred_from_saved_directed_edge_keys"
        edges = []
        for key, data in sorted(bearings.items()):
            i,j = mapping[key]
            indices = observations.get("edge_absolute_indices", {}).get(key, np.arange(length))
            q = None if measurement_mode != "b4" else np.asarray(rotations[i])[indices]
            # Existing archives use dense arrays; optional sparse arrays align with indices.
            b = np.asarray(data)
            if b.shape[1] == length and len(indices) != length:
                b = b[:, indices]
            mask = observations.get("edge_valid_masks", {}).get(key)
            if mask is not None and len(mask) == length and len(indices) != length:
                mask = np.asarray(mask)[indices]
            edges.append(DirectedObservations(i,j, np.asarray(indices), b, sigma_az, sigma_el,
                                              q, valid_mask=mask))
        arguments = dict(node_ids=nodes, anchor_ids=anchors, estimated_ids=estimated,
            observation_adjacency=adjacency, timestamps=ts,
            anchor_position_world=observations["anchor_position_w"],
            estimated_position_local=observations["estimated_position_l"], edges=tuple(edges))
        metadata = {"observation_path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_count": length, "measurement_mode": measurement_mode, "graph_source": graph_source,
            "sigma_azimuth_deg": float(np.rad2deg(sigma_az)), "sigma_elevation_deg": float(np.rad2deg(sigma_el)),
            "saved_window_sizes": raw["window_sizes"].tolist() if "window_sizes" in raw.files else []}
    return arguments, truth, metadata


def evaluate_record(record: dict, truth: dict) -> dict:
    """Evaluate estimates only after solving; never influence choice or initialization."""
    errors = {}
    for key,node in record["nodes"].items():
        if key not in truth:
            continue
        errors[key] = {}
        for stage in ("initial_pose", "final_pose"):
            pose = node[stage]
            if pose is None:
                continue
            angle = np.linalg.norm(so3_log(np.asarray(pose["rotation"]) @ truth[key]["rotation_local_world"].T))
            translation = np.linalg.norm(np.asarray(pose["translation"]) - truth[key]["translation_local"])
            errors[key][stage] = {"rotation_deg": float(np.rad2deg(angle)), "translation_m": float(translation)}
    return errors


def validate_records(records: list[dict]) -> dict[str, int]:
    """Check saved schema, unique ownership, candidate membership and true budgets."""
    statuses = {}
    for record in records:
        if record["schema_version"] != 1:
            raise ValueError("Unknown record schema.")
        cfg = record["config"]
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        for node, info in record["nodes"].items():
            window = set(record["absolute_indices"])
            owned = [tuple(fid) for fid in info["owned_factor_ids"]]
            if len(set(owned)) != len(owned) or any(k[0] != node or k[2] not in window for k in owned):
                raise ValueError("Duplicate, wrong-owner or out-of-window factor.")
            for phase in ("anchor", "pairwise"):
                selected = [tuple(fid) for fid in info["selected"].get(phase, [])]
                candidates = {tuple(fid) for fid in info["candidates"].get(phase, [])}
                if (len(set(selected)) != len(selected) or not set(selected) <= candidates
                        or len(selected) > cfg[phase + "_budget_per_node"]):
                    raise ValueError("Incorrect selected set/budget.")
            if record["status"] == "completed":
                expected = {tuple(k) for values in info["selected"].values() for k in values}
                if set(owned) != expected:
                    raise ValueError("Final factors differ from the once-selected measurement set.")
        backend = record["backend"]
        if backend is not None:
            if not 1 <= backend["gn_iterations"] <= cfg["max_gn_iterations"]:
                raise ValueError("Invalid GN iteration count.")
            if any(not 1 <= h["bp_iterations"] <= cfg["max_bp_iterations"] for h in backend["history"]):
                raise ValueError("Invalid BP iteration count.")
    return statuses


def save_records(output: Path, records: list[dict], metadata: dict) -> None:
    """Save lossless JSON, compact CSV and run provenance after schema validation."""
    summary = validate_records(records)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({**metadata, "status_counts": summary}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    rows = []
    for r in records:
        for node, info in r["nodes"].items():
            errors = r.get("evaluation", {}).get(node, {})
            row = {"window": r["window_id"], "dataset": r["dataset"]["observation_path"], "node": node,
                "status": r["status"], "frontend": r["config"]["initialization_method"],
                "anchor_strategy": r["config"]["anchor_selection_strategy"], "pairwise_strategy": r["config"]["pairwise_selection_strategy"],
                "anchor_budget": r["config"]["anchor_budget_per_node"], "pairwise_budget": r["config"]["pairwise_budget_per_node"],
                "anchor_actual": len(info["selected"].get("anchor", [])), "pairwise_actual": len(info["selected"].get("pairwise", [])),
                "segmentation_enabled": r["config"]["segmentation_enabled"],
                "representatives_per_segment": r["config"]["representatives_per_segment"],
                "executor": r["config"]["executor"],
                "end_to_end_wall_seconds": r["timings"]["end_to_end_wall_seconds"],
                "gn_iterations": (r["backend"] or {}).get("gn_iterations"), "gn_converged": (r["backend"] or {}).get("gn_converged"),
                "bp_converged": (r["backend"] or {}).get("bp_converged"),
                "failure": info["diagnostics"].get("initialization", {}).get("reason", r.get("failure_reason", ""))}
            for stage in ("initial_pose", "final_pose"):
                for metric in ("rotation_deg", "translation_m"):
                    row[f"{stage}_{metric}"] = errors.get(stage, {}).get(metric)
            rows.append(row)
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def plot_records(records: list[dict], output: Path) -> None:
    """Plot frontend versus final errors without assuming method-name lists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("rotation_deg", "translation_m")):
        x,y = [],[]
        for record in records:
            if record["status"] != "completed":
                continue
            for error in record.get("evaluation", {}).values():
                if "initial_pose" in error and "final_pose" in error:
                    x.append(error["initial_pose"][metric]); y.append(error["final_pose"][metric])
        ax.scatter(x,y,s=18, alpha=.7)
        upper = max(x+y+[1e-3]); ax.plot([0,upper],[0,upper], "k--", linewidth=.7)
        ax.set(xlabel="Frontend", ylabel="After GNBP", title=metric)
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "errors.png", dpi=150); plt.close(fig)


def _provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "src").glob("*.py") if path.name in (
                "exp_node_gnbp.py", "boepnp_gnbp_nodes.py", "node_gnbp_backend.py", "bearing_segmentation.py",
                "gnbp_keypoint_selection.py", "multi_agent_bearing_gabp.py", "bo_epnp_solver.py", "bgPnp_GABP.py", "bearing_only_solver.py",
                "selection_trace.py")}}


def main() -> None:
    """Run explicit windows/budget scans on existing input files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods-config", type=Path, help="Explicit JSON list of config overrides; no automatic strategy Cartesian product")
    parser.add_argument("--graph-config", type=Path, help="JSON node_ids/anchor_ids/estimated_ids/observation_adjacency")
    parser.add_argument("--output-root", type=Path, default=Path("data/node_gnbp"))
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--window-starts", type=int, nargs="+", default=[0])
    parser.add_argument("--scan-budgets", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export-config", type=Path, help="Independent plot export options")
    parser.add_argument("--export-suite", default="manual")
    parser.add_argument("--export-context", type=Path, help="Input/noise/repeat provenance JSON")
    parser.add_argument("--measurement-mode", choices=["b4", "b3_diagonal_approx"], default="b4")
    parser.add_argument("--input-adjacency-convention", choices=["target_observer", "observer_target"], default="target_observer")
    for name in ("anchor-budget-per-node", "pairwise-budget-per-node", "representatives-per-segment",
                 "max-gn-iterations", "max-bp-iterations", "random-seed", "trial-id", "workers"):
        parser.add_argument("--"+name, type=int)
    for name in ("max-angle-deg", "candidate-multiplier", "max-time-gap-s"):
        parser.add_argument("--"+name, type=float)
    for name, choices in (("initialization-method", ["bo_epnp", "bgpnp", "sdp", "bls"]),
        ("anchor-selection-strategy", ["random", "uniform_time", "rank_logdet"]),
        ("pairwise-selection-strategy", ["random", "uniform_time", "schur_greedy"]),
        ("executor", ["serial", "thread"])):
        parser.add_argument("--"+name, choices=choices)
    parser.add_argument("--segmentation-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prior-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--anchor-budget-values", type=int, nargs="+")
    parser.add_argument("--pairwise-budget-values", type=int, nargs="+")
    args = parser.parse_args()
    from analysis.r5_config import export_options
    export_config = (json.loads(args.export_config.read_text(encoding="utf-8"))
                     if args.export_config else {})
    recording_options = export_options(export_config)
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if "schur" in values:
        values["schur"] = GNBPKeypointSelectionConfig(**values["schur"])
    for name in NodeGNBPConfig.__dataclass_fields__:
        if getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    cfg = NodeGNBPConfig(**values)
    method_configs = [cfg]
    if args.methods_config:
        specs = json.loads(args.methods_config.read_text(encoding="utf-8"))
        if not isinstance(specs, list) or not specs:
            parser.error("methods-config must be a nonempty list of config overrides.")
        method_configs = []
        for spec in specs:
            if "schur" in spec:
                spec["schur"] = GNBPKeypointSelectionConfig(**spec["schur"])
            method_configs.append(replace(cfg, **spec))
    if args.window_size < 1 or any(i < 0 for i in args.window_starts):
        parser.error("Windows require a positive size and nonnegative starts.")
    records, provenance = [], _provenance()
    export_windows = []
    run_started = time.perf_counter()
    for path in args.observations:
        inputs, truth, metadata = load_npz_inputs(path, measurement_mode=args.measurement_mode,
                                                 input_adjacency_convention=args.input_adjacency_convention)
        if args.graph_config:
            graph = json.loads(args.graph_config.read_text(encoding="utf-8"))
            for name in ("node_ids", "anchor_ids", "estimated_ids"):
                graph[name] = tuple(graph[name])
            graph["observation_adjacency"] = np.asarray(graph["observation_adjacency"])
            # Do not silently discard stored edges or invent trajectories for extra nodes.
            inputs.update(graph)
        for start in args.window_starts:
            stop = start + args.window_size
            if stop > metadata["sample_count"]:
                parser.error(f"Window [{start},{stop}) exceeds {path} length {metadata['sample_count']}.")
            window = WindowInput(window_id=f"{metadata['sha256'][:12]}:{start}:{stop}",
                                 absolute_indices=np.arange(start,stop), **inputs)
            runs = []
            for method_cfg in method_configs:
                budgets = list(itertools.product(method_cfg.anchor_budget_values, method_cfg.pairwise_budget_values)) if args.scan_budgets else [(method_cfg.anchor_budget_per_node, method_cfg.pairwise_budget_per_node)]
                runs.extend(replace(method_cfg, anchor_budget_per_node=a, pairwise_budget_per_node=p) for a,p in budgets)
            for run_cfg in runs:
                result = solve_window(
                    window, run_cfg,
                    trace_selection=recording_options.get("trace_selection", False))
                record = result.record
                record.update(dataset=metadata, provenance=provenance)
                record["evaluation"] = evaluate_record(record, truth)
                if recording_options.get("r5"):
                    record["evaluation_reference"] = json.loads(json.dumps(truth, default=_json_default))
                records.append(record)
                if args.export_config:
                    export_windows.append(window)
                label = f"{run_cfg.initialization_method}/{run_cfg.anchor_selection_strategy}/{run_cfg.pairwise_selection_strategy}/seg={run_cfg.segmentation_enabled}"
                record["method_label"] = label
                print(f"{path.name} [{start},{stop}) {label} K_A={run_cfg.anchor_budget_per_node} K_P={run_cfg.pairwise_budget_per_node}: {record['status']}", flush=True)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_root / run_name
    save_records(output, records, {**provenance, "config": asdict(cfg), "method_configs": [asdict(c) for c in method_configs], "record_count": len(records),
        "first_stage_cache": "disabled; each recorded run includes actual computation",
        "experiment_wall_seconds": time.perf_counter()-run_started})
    if args.export_config:
        from src.node_gnbp_result_export import export_plot_artifacts

        context = json.loads(args.export_context.read_text(encoding="utf-8")) if args.export_context else {}
        context["suite_id"] = args.export_suite
        export_plot_artifacts(output, export_windows, records, [r["dataset"] for r in records], export_config, context)
    if args.plot:
        plot_records(records, output)
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
