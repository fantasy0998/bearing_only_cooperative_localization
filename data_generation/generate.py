"""Generate solver-compatible bearing NPZ files from MILUV mocap CSVs only.

Direct: python data_generation/generate.py --config data_generation/config.json
Module: python -m data_generation.generate --help
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np

if __package__:
    from . import miluv_bearing_core as core
else:
    import miluv_bearing_core as core

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULTS = {
    "schema_version": 1,
    "experiment_dir": None,
    "output_root": "outputs",
    "anchor_nodes": ["ifo001"],
    "estimated_nodes": ["ifo002", "ifo003"],
    "directed_edges": None,
    "dt": 0.05,
    "max_samples": None,
    "start_point": None,
    "end_point": None,
    "reference_index": 0,
    "noise_levels": [{"noise_id": "noise_1", "sigma_azimuth_deg": 0.1, "sigma_elevation_deg": 0.2}],
    "random_seeds": [7],
    "window_sizes": [200],
    "statistics": {"enabled": True, "quantiles": [0.05, 0.5, 0.95]},
}


def nullable_int(value: str) -> int | None:
    """Accept 'none' for optional sample/CSV limits."""
    return None if value.lower() == "none" else int(value)


def optional_dt(value: str) -> float | None:
    """Accept 'auto' to estimate the common sampling interval."""
    return None if value.lower() == "auto" else float(value)


def build_argparser() -> argparse.ArgumentParser:
    """CLI overrides JSON; omitted options keep their configured values."""
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config; default: beside this script",
    )
    parser.add_argument(
        "--experiment-dir", type=Path, help="MILUV directory containing robot/mocap.csv"
    )
    parser.add_argument(
        "--output-root", type=Path, help="Parent of a new timestamped output folder"
    )
    parser.add_argument("--anchor-nodes", nargs="+", help="Anchor IDs, in the required node order")
    parser.add_argument("--estimated-nodes", nargs="+", help="Estimated-node IDs")
    parser.add_argument(
        "--edges", nargs="+", help="Explicit observer:target pairs; omit to keep JSON graph"
    )
    parser.add_argument("--dt", type=optional_dt, help="Sampling interval (seconds), or auto")
    parser.add_argument(
        "--max-samples", type=nullable_int, help="Limit synchronized samples, or none"
    )
    parser.add_argument(
        "--start-point", type=nullable_int, help="First robot CSV start row (zero based), or none"
    )
    parser.add_argument(
        "--end-point", type=nullable_int, help="First robot CSV end row (inclusive), or none"
    )
    parser.add_argument(
        "--reference-index", type=int, help="Frozen B3 reference in the synchronized grid"
    )
    parser.add_argument(
        "--sigma-azimuth-deg", type=float, nargs="+", help="Paired azimuth noise levels, degrees"
    )
    parser.add_argument(
        "--sigma-elevation-deg",
        type=float,
        nargs="+",
        help="Paired elevation noise levels, degrees",
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        dest="random_seeds",
        type=int,
        nargs="+",
        help="One or more noise seeds",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        help="Advertised solver window sizes; no extra noise draws",
    )
    parser.add_argument(
        "--statistics",
        action=argparse.BooleanOptionalAction,
        help="Write descriptive statistics JSON",
    )
    parser.add_argument(
        "--statistics-quantiles", type=float, nargs="+", help="Statistic quantiles in [0,1]"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/input headers/time range without writing NPZ",
    )
    return parser


def _merge(default: dict, supplied: dict, prefix: str = "") -> dict:
    if not isinstance(supplied, dict):
        raise ValueError(f"Expected JSON object at {prefix or 'root'}")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"Unknown config keys at {prefix}: {sorted(unknown)}")
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        result[key] = (
            _merge(default[key], value, prefix + key + ".")
            if isinstance(default[key], dict)
            else value
        )
    return result


def resolve_config(args: argparse.Namespace) -> dict:
    """Resolve JSON paths against the config file; CLI paths against the cwd."""
    overrides = vars(args).copy()
    config_path = overrides.pop("config").resolve()
    cfg = _merge(DEFAULTS, json.loads(config_path.read_text(encoding="utf-8-sig")))
    for key in ("experiment_dir", "output_root"):
        if key in overrides:
            cfg[key] = str(overrides.pop(key).resolve())
        elif cfg[key] is not None:
            cfg[key] = str((config_path.parent / cfg[key]).resolve())
    az = overrides.pop("sigma_azimuth_deg", None)
    el = overrides.pop("sigma_elevation_deg", None)
    if az is not None or el is not None:
        if az is None or el is None or len(az) != len(el):
            raise ValueError("Provide both sigma lists with equal lengths (paired, not Cartesian)")
        cfg["noise_levels"] = [
            {"noise_id": f"noise_{i}", "sigma_azimuth_deg": a, "sigma_elevation_deg": e}
            for i, (a, e) in enumerate(zip(az, el), 1)
        ]
    if "edges" in overrides:
        cfg["directed_edges"] = [pair.split(":") for pair in overrides.pop("edges")]
    if "statistics" in overrides:
        cfg["statistics"]["enabled"] = overrides.pop("statistics")
    if "statistics_quantiles" in overrides:
        cfg["statistics"]["quantiles"] = overrides.pop("statistics_quantiles")
    overrides.pop("dry_run", None)
    cfg.update(overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Reject invalid topology, malformed IDs, unknown keys and unsafe values."""
    if cfg["schema_version"] != 1:
        raise ValueError("Expected schema_version=1")
    if cfg["experiment_dir"] is None or not Path(cfg["experiment_dir"]).is_dir():
        raise ValueError("experiment_dir must be an existing MILUV directory")
    anchors, estimated = cfg["anchor_nodes"], cfg["estimated_nodes"]
    if (
        not isinstance(anchors, list)
        or not isinstance(estimated, list)
        or not anchors
        or not estimated
    ):
        raise ValueError("Provide nonempty anchor_nodes and estimated_nodes lists")
    nodes = anchors + estimated
    if any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in nodes):
        raise ValueError("Node IDs must use letters, digits, underscores or hyphens")
    if len(set(nodes)) != len(nodes):
        raise ValueError("Node IDs must be distinct across both roles")
    serialized = [f"{a}_{b}" for a in nodes for b in nodes if a != b]
    if len(set(serialized)) != len(serialized):
        raise ValueError("Node IDs produce ambiguous edge keys in the existing reader")
    edges = cfg["directed_edges"]
    if edges is None:
        # Match the original harness's row-major target/observer adjacency scan.
        edges = [[a, b] for b in nodes for a in estimated if a != b]
    if (
        not isinstance(edges, list)
        or not edges
        or any(not isinstance(e, (list, tuple)) or len(e) != 2 for e in edges)
    ):
        raise ValueError("directed_edges must be a nonempty list of observer/target pairs")
    if any(a not in estimated or b not in nodes or a == b for a, b in edges):
        raise ValueError("Edges must originate at estimated nodes and target another known node")
    if len(set(map(tuple, edges))) != len(edges):
        raise ValueError("Duplicate directed edge")
    cfg["directed_edges"] = edges
    for key in ("start_point", "end_point", "reference_index", "max_samples"):
        value = cfg[key]
        if value is None and key != "reference_index":
            continue
        minimum = 2 if key == "max_samples" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if cfg["dt"] is not None and (
        type(cfg["dt"]) not in (int, float) or not np.isfinite(cfg["dt"]) or cfg["dt"] <= 0
    ):
        raise ValueError("dt must be null or a positive finite number")
    for key, minimum in (("random_seeds", 0), ("window_sizes", 1)):
        values = cfg[key]
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < minimum for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must be a nonempty list of unique integers >= {minimum}")
    if not isinstance(cfg["noise_levels"], list) or not cfg["noise_levels"]:
        raise ValueError("noise_levels must be a nonempty list")
    names = []
    for level in cfg["noise_levels"]:
        if not isinstance(level, dict) or set(level) != {
            "noise_id",
            "sigma_azimuth_deg",
            "sigma_elevation_deg",
        }:
            raise ValueError("Each noise level needs exactly noise_id and two sigma fields")
        if not isinstance(level["noise_id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", level["noise_id"]
        ):
            raise ValueError("Invalid noise_id")
        names.append(level["noise_id"])
        for key in ("sigma_azimuth_deg", "sigma_elevation_deg"):
            value = level[key]
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError("Noise sigmas must be finite and positive for the existing solver")
    if len(set(names)) != len(names):
        raise ValueError("Noise IDs must be unique")
    stats = cfg["statistics"]
    if (
        type(stats["enabled"]) is not bool
        or not isinstance(stats["quantiles"], list)
        or not stats["quantiles"]
    ):
        raise ValueError("statistics requires enabled (boolean) and a nonempty quantiles list")
    if any(
        type(q) not in (int, float) or not np.isfinite(q) or not 0 <= q <= 1
        for q in stats["quantiles"]
    ):
        raise ValueError("Statistic quantiles must lie in [0,1]")


def validate_input(cfg: dict) -> dict:
    """Load CSVs and verify the synchronized grid without generating noise."""
    frames = core.load_three_robot_mocap(
        Path(cfg["experiment_dir"]), cfg["anchor_nodes"] + cfg["estimated_nodes"]
    )
    timestamps = core.build_common_timestamps(
        frames, cfg["dt"], cfg["max_samples"], cfg["start_point"], cfg["end_point"]
    )
    if cfg["reference_index"] >= len(timestamps):
        raise ValueError("reference_index must be inside the synchronized grid")
    if max(cfg["window_sizes"]) > len(timestamps):
        raise ValueError(f"Largest window exceeds {len(timestamps)} synchronized samples")
    return {
        "csv_rows": {n: len(frame) for n, frame in frames.items()},
        "sample_count": len(timestamps),
        "first_timestamp_s": float(timestamps[0]),
        "last_timestamp_s": float(timestamps[-1]),
        "dt_s": float(np.median(np.diff(timestamps))),
        "observation_files": len(cfg["noise_levels"]) * len(cfg["random_seeds"]),
    }


def build_solver_arrays(dataset: dict, cfg: dict, level: dict, seed: int) -> dict:
    """Use the existing nested NPZ contract, including raw B4 and frozen B3."""
    anchors, estimated = cfg["anchor_nodes"], cfg["estimated_nodes"]
    nodes = anchors + estimated
    states, ref = dataset["states"], cfg["reference_index"]
    local, poses, rotations = {}, {}, {}
    for node in estimated:
        state = states[node]
        r = state["rotation_wb"][ref].T
        t = -r @ state["position_w"][:, ref]
        local[node] = r @ state["position_w"] + t[:, None]
        poses[node] = {
            "rotation_local_world": r,
            "translation_local": t,
            "rotation_wb_start": state["rotation_wb"][ref],
            "position_w_start": state["position_w"][:, ref],
        }
        rotations[node] = core.compute_b3_from_b4_rotation_sequence(state["rotation_wb"], ref)
    observations = {
        f"{e['observer_id']}_{e['target_id']}": e for e in dataset["observations"].values()
    }
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=int)
    for observer, target in cfg["directed_edges"]:
        adjacency[nodes.index(target), nodes.index(observer)] = 1
    observation_data = {
        "anchor_position_w": {n: states[n]["position_w"] for n in anchors},
        "estimated_position_l": local,
        "bearing_observation": {key: value["bearing_b3"] for key, value in observations.items()},
        "bearing_observation_b4": {key: value["bearing_b4"] for key, value in observations.items()},
        "observer_rotation_b3_b4": rotations,
    }
    reference_data = {
        "estimated_position_w": {n: states[n]["position_w"] for n in estimated},
        "estimated_se3": poses,
        "bearing_noiseless_b4": {k: v["bearing_b4_noiseless"] for k, v in observations.items()},
        "bearing_noiseless_b3": {k: v["bearing_b3_noiseless"] for k, v in observations.items()},
    }
    return {
        "experiment_dir": dataset["experiment_dir"],
        "timestamps": dataset["timestamps"],
        "timeshift_s": dataset["timeshift_s"],
        "reference_index": ref,
        "reference_timestamp_s": dataset["timestamps"][ref],
        "random_seed": seed,
        "sigma_azimuth_deg": level["sigma_azimuth_deg"],
        "sigma_elevation_deg": level["sigma_elevation_deg"],
        "noise_level_id": level["noise_id"],
        "noise_realization_id": f"seed_{seed}",
        "robot_ids": np.asarray(nodes),
        "anchor_nodes": np.asarray(anchors),
        "estimated_nodes": np.asarray(estimated),
        "adjacency_matrix": adjacency,
        "adjacency_convention": "target_observer",
        "window_sizes": np.asarray(cfg["window_sizes"]),
        "window_size": cfg["window_sizes"][0],
        "observation_data": np.array(observation_data, dtype=object),
        "reference_data": np.array(reference_data, dtype=object),
        "data_origin": "bearing synthesized from MILUV mocap ground truth; Gaussian angular noise in B4",
    }


def summarize(values: np.ndarray, quantiles: list[float]) -> dict:
    """Population descriptive statistics; never estimate independent trial count."""
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std_ddof0": float(np.std(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": {str(q): float(np.quantile(values, q)) for q in quantiles},
    }


def dataset_statistics(dataset: dict, quantiles: list[float]) -> dict:
    """Describe trajectories and effective angular errors after wrap/clipping."""
    result = {
        "sample_count": len(dataset["timestamps"]),
        "nodes": {},
        "edges": {},
        "noise_statistics_frame": "B4",
        "angular_unit": "deg",
        "position_unit": "m",
        "effective_noise": "after azimuth wrapping and elevation clipping; std uses ddof=0",
    }
    for node, state in dataset["states"].items():
        p = state["position_w"]
        result["nodes"][node] = {
            "position_min_m": p.min(axis=1).tolist(),
            "position_max_m": p.max(axis=1).tolist(),
            "path_length_m": float(np.linalg.norm(np.diff(p, axis=1), axis=0).sum()),
        }
    for key, edge in dataset["observations"].items():
        delta = edge["angles_b4"] - edge["angles_b4_noiseless"]
        delta[0] = core.wrap_to_pi(delta[0])
        b, b0 = edge["bearing_b4"], edge["bearing_b4_noiseless"]
        angular = np.rad2deg(
            np.arctan2(np.linalg.norm(np.cross(b.T, b0.T), axis=1), np.sum(b * b0, axis=0))
        )
        result["edges"][key] = {
            "observer_id": edge["observer_id"],
            "target_id": edge["target_id"],
            "azimuth_noise_deg": summarize(np.rad2deg(delta[0]), quantiles),
            "elevation_noise_deg": summarize(np.rad2deg(delta[1]), quantiles),
            "bearing_angle_error_deg": summarize(angular, quantiles),
            "range_m": summarize(edge["range_w"], quantiles),
            "max_unit_norm_error": float(np.max(np.abs(np.linalg.norm(b, axis=0) - 1))),
        }
    return result


def write_json(path: Path, value: Any) -> None:
    """Save finite UTF-8 JSON."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def generate(cfg: dict) -> Path:
    """Create new observation batches and audit files, without importing solvers."""
    validate_config(cfg)
    preflight = validate_input(cfg)
    source_paths = [
        Path(cfg["experiment_dir"]) / n / "mocap.csv"
        for n in cfg["anchor_nodes"] + cfg["estimated_nodes"]
    ]
    timeshift = Path(cfg["experiment_dir"]) / "timeshift.yaml"
    if timeshift.is_file():
        source_paths.append(timeshift)
    provenance = {
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "code": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(core.__file__))
        },
        "noise_seed_policy": "reset default_rng(seed) for each level; same seed pairs standard draws across noise levels",
        "frame_convention": "R_WB maps body to world; B3 frozen at reference_index; B4 body at each sample",
        "trajectory_convention": "p_local = R_WB(ref).T @ (p_world - p_world(ref))",
        "adjacency_convention": "rows=target, columns=observer",
        "timeshift_usage": "metadata only; all mocap CSVs already share their time axis",
    }
    output = Path(cfg["output_root"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "observations").mkdir()
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "provenance.json", provenance)
    manifest = {
        "status": "running",
        "preflight": preflight,
        "files": [],
        "statistics_enabled": cfg["statistics"]["enabled"],
    }
    write_json(output / "manifest.json", manifest)
    try:
        for level in cfg["noise_levels"]:
            for seed in cfg["random_seeds"]:
                dataset = core.synthesize_three_robot_bearing_observations(
                    experiment_dir=Path(cfg["experiment_dir"]),
                    anchor_id=cfg["anchor_nodes"][0],
                    robot_ids=cfg["anchor_nodes"] + cfg["estimated_nodes"],
                    directed_edges=cfg["directed_edges"],
                    dt=cfg["dt"],
                    max_samples=cfg["max_samples"],
                    start_point=cfg["start_point"],
                    end_point=cfg["end_point"],
                    reference_index=cfg["reference_index"],
                    sigma_azimuth_deg=level["sigma_azimuth_deg"],
                    sigma_elevation_deg=level["sigma_elevation_deg"],
                    random_seed=seed,
                )
                arrays = build_solver_arrays(dataset, cfg, level, seed)
                relative = Path("observations") / f"{level['noise_id']}_seed{seed}.npz"
                np.savez_compressed(output / relative, **arrays)
                row = {
                    "path": relative.as_posix(),
                    "noise_id": level["noise_id"],
                    "random_seed": seed,
                    "sigma_azimuth_deg": level["sigma_azimuth_deg"],
                    "sigma_elevation_deg": level["sigma_elevation_deg"],
                    "sample_count": len(dataset["timestamps"]),
                    "edge_count": len(dataset["observations"]),
                    "sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                }
                if cfg["statistics"]["enabled"]:
                    stats_path = relative.with_suffix(".statistics.json")
                    write_json(
                        output / stats_path,
                        dataset_statistics(dataset, cfg["statistics"]["quantiles"]),
                    )
                    row["statistics_path"] = stats_path.as_posix()
                manifest["files"].append(row)
                write_json(output / "manifest.json", manifest)
                LOGGER.info(
                    "Generated %s (%s samples, %s edges)",
                    relative,
                    row["sample_count"],
                    row["edge_count"],
                )
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    LOGGER.info("Saved data generation batch: %s", output)
    return output


def main() -> None:
    """Run the configurable generator or a read-only input preflight."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_argparser()
    args = parser.parse_args()
    try:
        cfg = resolve_config(args)
        if getattr(args, "dry_run", False):
            LOGGER.info("Preflight: %s", json.dumps(validate_input(cfg), ensure_ascii=False))
            LOGGER.info("Resolved config: %s", json.dumps(cfg, ensure_ascii=False))
        else:
            generate(cfg)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Data generation failed: {error}\n")


if __name__ == "__main__":
    main()
