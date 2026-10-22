"""Node RMSE, phase success, budget coverage and timing-repeat reduction."""

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from analysis.plot_io import digest


def matches(run: dict, *, suite: str | None = None, **values: object) -> bool:
    """Match exact solver settings, not a display label."""
    return (suite is None or run["plot_export"]["context"]["suite_id"] == suite) and all(
        run["config"].get(k) == v for k, v in values.items()
    )


def invariant_key(run: dict, changed: set[str]) -> str:
    """Separate topologies and every algorithm setting except a figure's sweep."""
    # trial_id is the selection replication unit, not a different algorithm.
    cfg = {k: v for k, v in run["config"].items() if k not in changed | {"trial_id"}}
    identity = [run["plot_export"]["graph_id"], cfg]
    if "source_code_version" in run["plot_export"]:
        identity.append(run["plot_export"]["source_code_version"])
    mode = run.get("dataset", {}).get("measurement_mode")
    if mode != "b4":
        identity.append({"measurement_mode": mode})
    return digest(identity)


def instance_key(run: dict, *, across_noise: bool = False) -> str:
    """Pair the same sequence/window/realization/selection trial across methods."""
    c = run["plot_export"]["context"]
    parts = [
        c["base_sequence_id"],
        run["absolute_indices"],
        c["noise_realization_id"],
        run["config"]["random_seed"],
        run["config"]["trial_id"],
    ]
    if not across_noise:
        parts.extend([c["noise_level_id"], run["dataset"]["sha256"]])
    return digest(parts)


def phase_sample(run: dict, nodes: list[str], stage: str, require_budget: bool = True) -> dict:
    """Compute one network/node sample; no successful-node-only averaging."""
    cfg, meta = run["config"], run["plot_export"]
    initial = stage == "initial_pose"
    time_field = (
        "time_to_initial_estimate_wall_seconds"
        if initial
        else "time_to_final_estimate_wall_seconds"
    )
    success = all(run["nodes"][n].get(stage) is not None for n in nodes)
    success &= initial or run["status"] == "completed"
    stages = ["anchor"] if initial else ["anchor", "pairwise"]
    counts = {n: {s: len(run["nodes"][n]["selected"].get(s, [])) for s in stages} for n in nodes}
    budget_ok = all(counts[n][s] == cfg[s + "_budget_per_node"] for n in nodes for s in stages)
    available = all(stage in run.get("evaluation", {}).get(n, {}) for n in nodes)
    usable = bool(success and available and (budget_ok or not require_budget))
    row = {
        "run_id": meta["run_id"],
        "logical_run_id": meta["logical_run_id"],
        "instance_id": instance_key(run),
        "paired_noise_instance_id": instance_key(run, across_noise=True),
        "noise": meta["context"]["noise_level_id"],
        "stage": stage,
        "nodes": nodes,
        "status": run["status"],
        "phase_success": bool(success),
        "budget_ok": budget_ok,
        "actual_counts": counts,
        "usable_error": usable,
        "wall_seconds": run["timings"].get(time_field),
        "rotation_deg": None,
        "translation_m": None,
        "per_node_errors": {n: run.get("evaluation", {}).get(n, {}).get(stage) for n in nodes},
        "gn_converged": (run.get("backend") or {}).get("gn_converged") if not initial else None,
        "bp_converged": (run.get("backend") or {}).get("bp_converged") if not initial else None,
        "reason": (
            run.get("failure_reason", "")
            if not success
            else ("actual_budget_shortfall" if not budget_ok else "")
        ),
    }
    row["initialization_diagnostics"] = {
        n: run["nodes"][n].get("diagnostics", {}).get("initialization", {}) for n in nodes
    }
    row["pairwise_selection_diagnostics"] = {
        n: run["nodes"][n].get("diagnostics", {}).get("pairwise_selection", {}) for n in nodes
    }
    row["backend_diagnostics"] = run.get("backend")
    if not available:
        row["reason"] += ";missing_evaluation"
    if row["wall_seconds"] is None:
        row["reason"] += ";network_stage_time_unavailable"
    if usable:
        for metric in ("rotation_deg", "translation_m"):
            values = [run["evaluation"][n][stage][metric] for n in nodes]
            if not np.all(np.isfinite(values)) or np.any(np.array(values) < 0):
                raise ValueError("Pose error must be finite and nonnegative.")
            row[metric] = float(np.sqrt(np.mean(np.square(values))))
    return row


def reduce_repeats(samples: list[dict]) -> list[dict]:
    """One error per logical trial; median physical-repeat wall time, no pseudo-n."""
    groups = defaultdict(list)
    for row in samples:
        groups[row["logical_run_id"]].append(row)
    reduced = []
    for rows in groups.values():
        signatures = [
            json.dumps(
                {k: r[k] for k in ("phase_success", "budget_ok", "rotation_deg", "translation_m")},
                sort_keys=True,
            )
            for r in rows
        ]
        if len(set(signatures)) != 1:
            raise ValueError(
                "Timing repeats changed accuracy/status; cannot treat them as one deterministic trial."
            )
        row = dict(rows[0])
        times = [r["wall_seconds"] for r in rows if r["wall_seconds"] is not None]
        row["wall_seconds"] = float(np.median(times)) if times else None
        row["timing_repeats"] = len(rows)
        reduced.append(row)
    return reduced


def summarize(samples: list[dict], *, require_time: bool) -> dict:
    """RMSE across logical errors and median network time of the same usable runs."""
    logical = reduce_repeats(samples)
    good = [
        r
        for r in logical
        if r["usable_error"] and (not require_time or r["wall_seconds"] is not None)
    ]
    result = {
        "physical_attempts": len(samples),
        "logical_attempts": len(logical),
        "n": len(good),
        "phase_successes": sum(r["phase_success"] for r in logical),
        "actual_budget_successes": sum(r["budget_ok"] for r in logical),
        "gn_converged_count": sum(r.get("gn_converged") is True for r in logical),
        "bp_converged_count": sum(r.get("bp_converged") is True for r in logical),
        "rotation_deg": None,
        "translation_m": None,
        "wall_seconds": None,
    }
    if good:
        for metric in ("rotation_deg", "translation_m"):
            result[metric] = float(np.sqrt(np.mean([r[metric] ** 2 for r in good])))
        times = [r["wall_seconds"] for r in good if r["wall_seconds"] is not None]
        result["wall_seconds"] = float(np.median(times)) if times else None
    return result
