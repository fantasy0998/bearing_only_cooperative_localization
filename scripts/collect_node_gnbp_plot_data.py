"""Resolve an explicit experiment plan into existing CLI invocations (no solver changes)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(value: str, plan_path: Path) -> Path:
    """Resolve paths relative to repo, plan folder, or supplied spec package."""
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (ROOT, plan_path.parent, plan_path.parent.parent):
        if (base / path).exists():
            return (base / path).resolve()
    return ROOT / path


def sequence_hash(path: Path) -> str:
    """Check that named common sequences really share trajectories and timestamps."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=True) as raw:
        obs = raw["observation_data"].item()
        for name in ("anchor_position_w", "estimated_position_l", "observer_rotation_b3_b4"):
            for key, values in sorted(obs.get(name, {}).items()):
                a = np.ascontiguousarray(values)
                h.update(f"{name}:{key}:{a.shape}:{a.dtype}".encode())
                h.update(a.tobytes())
        if "timestamps" in raw:
            h.update(raw["timestamps"].tobytes())
    return h.hexdigest()


def main() -> None:
    """Execute each input/trial/repeat/suite serially and keep exact commands/logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    ## dry-run 只检查配置不执行算法
    parser.add_argument("--dry-run", action="store_true", help="Resolve an r5 plan without running experiments")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") == "plot_collection_plan_r5":
        from scripts.node_gnbp_r5_collection import collect
        collect(plan_path, dry_run=args.dry_run)
        return
    if args.dry_run:
        parser.error("--dry-run currently requires an r5 plan")
    if plan.get("plan_version") != 1 or plan.get("scan_budgets", False):
        raise ValueError("Expected plan_version=1 and explicit method lists (scan_budgets=false).")
    sequences = {}
    for item in plan["inputs"]:
        path = resolve_path(item["path"], plan_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resolve input placeholder first: {path}")
        item["path"] = str(path)
        digest = sequence_hash(path)
        name = item["base_sequence_id"]
        if name in sequences and sequences[name] != digest:
            raise ValueError(f"Base sequence {name} contains different trajectories/extrinsics.")
        sequences[name] = digest
        item["sequence_sha256"] = digest
    output = ROOT / plan["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_collection_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calls = []
    for item in plan["inputs"]:
        for trial in plan["selection_trial_ids"]:
            for repeat in range(plan["timing_repeats"]):
                for job in plan["jobs"]:
                    tag = f"{item['noise_level_id']}_{item['noise_realization_id']}_t{trial}_r{repeat}_{job['suite_id']}"
                    context = {
                        **item,
                        "timing_repeat": repeat,
                        "invocation_id": output.name + ":" + tag,
                    }
                    context.pop("path")
                    context_path = output / (tag + "_context.json")
                    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
                    command = [
                        sys.executable,
                        "-m",
                        "src.exp_node_gnbp",
                        "--observations",
                        item["path"],
                        "--config",
                        str(resolve_path(plan["base_config"], plan_path)),
                        "--methods-config",
                        str(resolve_path(job["methods_config"], plan_path)),
                        "--export-config",
                        str(resolve_path(plan["export_config"], plan_path)),
                        "--export-context",
                        str(context_path),
                        "--export-suite",
                        job["suite_id"],
                        "--output-root",
                        str(output / "runs"),
                        "--window-size",
                        str(plan["window_size"]),
                        "--trial-id",
                        str(trial),
                        "--measurement-mode",
                        plan["measurement_mode"],
                        "--input-adjacency-convention",
                        plan["input_adjacency_convention"],
                        "--window-starts",
                        *map(str, plan["window_starts"]),
                    ]
                    if plan.get("graph_config"):
                        command.extend(
                            ["--graph-config", str(resolve_path(plan["graph_config"], plan_path))]
                        )
                    print(f"Running {tag}", flush=True)
                    with (output / (tag + ".log")).open("w", encoding="utf-8") as stream:
                        completed = subprocess.run(
                            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                        )
                    calls.append(
                        {"tag": tag, "command": command, "returncode": completed.returncode}
                    )
                    (output / "commands.json").write_text(
                        json.dumps(calls, indent=2), encoding="utf-8"
                    )
                    if completed.returncode:
                        raise RuntimeError(
                            f"Collection command failed; inspect {output / (tag + '.log')}"
                        )
    print(f"Collected: {output}", flush=True)


if __name__ == "__main__":
    main()
