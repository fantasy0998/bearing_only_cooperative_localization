"""r5 plan adapter for the existing serial collector and experiment CLI."""

from __future__ import annotations

from dataclasses import asdict, replace
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

from analysis.plot_io import dump_json, write_csv, load_runs
from analysis.r5_config import CONTRACTS, checked_config, export_options, fixed_policy
from src.boepnp_gnbp_nodes import NodeGNBPConfig, WindowInput
from src.exp_node_gnbp import _provenance, load_npz_inputs
from src.gnbp_keypoint_selection import GNBPKeypointSelectionConfig

MAPPING = {"frontend": "initialization_method", "anchor_budget": "anchor_budget_per_node",
           "pairwise_budget": "pairwise_budget_per_node", "anchor_selection_strategy": "anchor_selection_strategy",
           "pairwise_selection_strategy": "pairwise_selection_strategy"}


def read(path: Path) -> dict:
    """Read UTF-8 config without relying on the Windows locale."""
    return json.loads(path.read_text(encoding="utf-8"))


def signature(value: object) -> str:
    """Hash normalized full solver identities, excluding display roles."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(value: str, parent: Path) -> Path:
    """Resolve r5 paths against the JSON containing them, never the cwd."""
    if "PATH_TO_" in value or "REPLACE_WITH" in value:
        raise ValueError(f"Unresolved placeholder: {value}")
    return (parent / value).resolve()


def resolve_experiments(plan: dict, parent: Path) -> tuple[list, list, dict]:
    """Bind five semantic fields and validate each complete config in the real class."""
    base_path = resolve(plan["base_algorithm_config"], parent)
    adapter_path = resolve(plan["adapter_contract"], parent)
    adapter = checked_config(read(adapter_path), "config_adapter_contract.json")
    mutable = {"implementation_status"} | {"semantic_fields." + k + ".existing_key_path" for k in MAPPING}
    mutable |= {"semantic_fields." + k + ".existing_value_mapping" for k in MAPPING}
    fixed_policy(adapter, CONTRACTS["config_adapter_contract.json"], mutable)
    for semantic, actual in MAPPING.items():
        rule = adapter["semantic_fields"][semantic]
        if rule["existing_key_path"] != actual:
            raise ValueError(f"Unresolved or unsupported adapter mapping for {semantic}")
        if "enum" in rule and rule["existing_value_mapping"] != {v: v for v in rule["enum"]}:
            raise ValueError(f"Unsupported enum mapping for {semantic}")
    values = read(base_path)
    if "schur" in values:
        values["schur"] = GNBPKeypointSelectionConfig(**values["schur"])
    base = NodeGNBPConfig(**values)
    unique, references = {}, []
    for filename in plan["suite_files"]:
        suite_path = resolve(filename, parent)
        raw = read(suite_path)
        contract = next((name for name, value in CONTRACTS.items()
                         if value.get("suite_id") == raw.get("suite_id") and name.startswith("experiments_")), None)
        if contract is None:
            raise ValueError(f"Unknown experiment suite: {raw.get('suite_id')}")
        suite = checked_config(raw, contract)
        fixed_policy(suite, CONTRACTS[contract], {"base_algorithm_config", "algorithm_adapter", "export_config", "experiments"})
        if resolve(suite["algorithm_adapter"], suite_path.parent) != adapter_path:
            raise ValueError("Suite adapter differs from collection adapter")
        if resolve(suite["export_config"], suite_path.parent) != resolve(plan["export_config"], parent):
            raise ValueError("Suite export config differs from collection export config")
        if resolve(suite["base_algorithm_config"], suite_path.parent) != base_path:
            raise ValueError("Suite base differs from collection base")
        if suite["schema_version"] != "plot_collection_r5":
            raise ValueError("Unsupported suite schema")
        for experiment in suite["experiments"]:
            if set(experiment) - {"config_id", "solver_semantics", "run_mode", "figure_roles", "requires_selection_trace"}:
                raise ValueError(f"Unknown experiment fields: {experiment}")
            semantics = experiment["solver_semantics"]
            if set(semantics) != set(MAPPING) or experiment["run_mode"] != "existing_full_pipeline":
                raise ValueError("Each experiment must specify the five validated semantics and full pipeline")
            override = {MAPPING[k]: v for k, v in semantics.items()}
            for name, value in semantics.items():
                rule = adapter["semantic_fields"][name]
                if "enum" in rule and value not in rule["enum"]:
                    raise ValueError(f"Unknown {name}: {value}")
                if rule.get("type") == "integer" and (type(value) is not int or value < rule["minimum"]):
                    raise ValueError(f"Invalid integer budget {name}: {value}")
            cfg = asdict(replace(base, **override))
            key = signature(cfg)
            if key not in unique:
                unique[key] = {"config_id": experiment["config_id"], "config": cfg,
                               "overrides": override, "roles": [], "suites": []}
            entry = unique[key]
            entry["roles"] = sorted(set(entry["roles"] + experiment["figure_roles"]))
            entry["suites"] = sorted(set(entry["suites"] + [suite["suite_id"]]))
            references.append({"suite": suite["suite_id"], "config_id": experiment["config_id"], "config_hash": key})
        if "budget_grid" in suite:
            grid = suite["budget_grid"]
            expected = {(a, p) for a in grid["anchor_budgets"] for p in grid["pairwise_budgets"]}
            actual = [(e["solver_semantics"]["anchor_budget"], e["solver_semantics"]["pairwise_budget"]) for e in suite["experiments"]]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("F5 explicit list does not cover the grid exactly once")
    return list(unique.values()), references, asdict(base)


def finalize_collection(output: Path, tasks: list[dict], refs: list[dict], reused: list[dict]) -> None:
    """Audit saved records, including failed stages, without invoking a solver."""
    roots = sorted({r["source"] for r in reused})
    if (output / "runs").exists():
        roots.append(str(output / "runs"))
    runs = load_runs(roots) if roots else []
    rows, links, capabilities = [], [], []
    config_hashes = {c["config_id"]: signature(c["config"]) for c in read(output / "resolved_algorithm_configs.json")}
    for task in tasks:
        run = next((r for r in runs if r["dataset"]["sha256"] == task["input_hash"]
                    and r["absolute_indices"][0] == task["window_start"]
                    and signature(r["config"]) == task["config_hash"]
                    and r["plot_export"]["context"]["timing_repeat"] == task["repeat"]), None)
        row = dict(task)
        if run is None:
            row.update(status="missing_record", run_id=None)
        else:
            meta = run["plot_export"]
            initial = all(n["initial_pose"] is not None for n in run["nodes"].values())
            final = run["status"] == "completed" and all(n["final_pose"] is not None for n in run["nodes"].values())
            budget = all(n[s]["actual"] == n[s]["requested"] for n in meta["budget_counts"]["nodes"].values()
                         for s in ("anchor", "pairwise"))
            row.update(run_id=meta["run_id"], source=run["_source"], status=run["status"],
                       initial_available=initial, final_available=final, actual_budget_met=budget,
                       failure_reason=run.get("failure_reason", ""), budget_counts=meta["budget_counts"],
                       timing_protocol=meta["timing_protocol"])
            capabilities.append({**row, **meta.get("capabilities", {}),
                                 "bearing_snapshot": True, "pose_evaluation": bool(run["evaluation"]),
                                 "trace_required_by_F2": task["trace_required_by_F2"]})
            for ref in refs:
                if ref["config_hash"] == config_hashes[task["config_id"]]:
                    links.append({"suite": ref["suite"], "configuration_reference": ref["config_id"],
                                  "run_id": meta["run_id"], "source": run["_source"], "noise": task["noise"],
                                  "trial": task["trial"], "repeat": task["repeat"]})
        rows.append(row)
    write_csv(output / "collection_coverage.csv", rows)
    write_csv(output / "figure_capabilities.csv", capabilities)
    write_csv(output / "budget_grid_coverage.csv", [r for r in rows if "F5.final" in r["roles"]])
    dump_json(output / "missing_runs.json", [r for r in rows if r["status"] == "missing_record"])
    dump_json(output / "physical_run_references.json", {"suite_references": refs, "physical_references": links, "reused": reused})
    dump_json(output / "collection_summary.json", {"planned": len(tasks), "saved": sum(r["run_id"] is not None for r in rows),
              "completed": sum(r["status"] == "completed" for r in rows), "suite_reference_count": len(links),
              "missing": sum(r["status"] == "missing_record" for r in rows)})


def collect(plan_path: Path, *, dry_run: bool = False) -> Path:
    """Inventory actual records, save a preflight, then call the original CLI."""
    plan = checked_config(read(plan_path), "collection_plan_r5.json")
    fixed_policy(plan, CONTRACTS["collection_plan_r5.json"],
                 {"base_algorithm_config", "suite_files", "export_config", "adapter_contract", "existing_result_roots",
                  "output_root", "datasets", "windows.window_size", "windows.window_starts", "trials.trial_ids",
                  "trials.timing_repeats", "reuse.enabled", "notes"})
    if (type(plan["trials"]["timing_repeats"]) is not int or plan["trials"]["timing_repeats"] < 1
            or not plan["trials"]["trial_ids"] or not plan["windows"]["window_starts"] or not plan["datasets"]):
        raise ValueError("Provide nonempty inputs/windows/trials and positive timing repeats")
    for values in (plan["trials"]["trial_ids"], plan["windows"]["window_starts"]):
        if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Trial IDs and window starts must be unique nonnegative integers")
    for data in plan["datasets"]:
        if set(data) != set(CONTRACTS["collection_plan_r5.json"]["datasets"][0]):
            raise ValueError("Unknown or missing dataset fields")
    if plan["execution"]["run_mode"] != "existing_full_pipeline" or plan["execution"]["budget_cartesian_scan"]:
        raise ValueError("r5 requires explicit full-pipeline configurations without another budget scan")
    configs, refs, base = resolve_experiments(plan, plan_path.parent)
    options_path = resolve(plan["export_config"], plan_path.parent)
    options = export_options(read(options_path))
    sources = _provenance()["source_sha256"]
    root = resolve(plan["output_root"], plan_path.parent)
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "collection_resolved_plan.json", plan)
    dump_json(output / "resolved_algorithm_configs.json", configs)
    dump_json(output / "adapter_mapping_resolved.json", {"mapping": MAPPING, "base": base, "validated": True,
                "source_code_version": sources, "base_file_sha256": hashlib.sha256(resolve(plan["base_algorithm_config"], plan_path.parent).read_bytes()).hexdigest()})
    existing, inventory = [], []
    roots = [resolve(p, plan_path.parent) for p in plan["existing_result_roots"]] + [root]
    seen = set()
    for source in roots:
        if not source.exists():
            inventory.append({"directory": str(source), "reason": "missing_existing_result_root"})
            continue
        for path in source.rglob("plot_runs.json"):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            for run in read(path):
                run["source_directory"] = str(path.parent)
                same_code = run.get("provenance", {}).get("source_sha256") == sources
                trace = run.get("recording", {}).get("selection_trace_enabled", False)
                manifest = read(path.parent / "plot_manifest.json")
                complete_export = (manifest.get("capabilities", {}).get("r5_additive_export", False)
                                   and all((path.parent / f).is_file() for f in manifest.get("additional_files", {}).values()))
                reusable = same_code and trace == options.get("trace_selection", False) and complete_export
                inventory.append({"directory": str(path.parent), "run_id": run["plot_export"]["run_id"],
                    "config_hash": signature(run["config"]), "status": run["status"], "trace": trace,
                    "same_code": same_code, "action": "reuse" if reusable else "version_or_timing_recollection",
                    "reason": "exact_source_and_trace_protocol" if reusable else "no_verified_cross_version_compatibility",
                    "source_code_version": run.get("provenance", {}).get("source_sha256", {})})
                if same_code and complete_export and plan["reuse"]["enabled"]:
                    existing.append(run)
    write_csv(output / "r3_inventory.csv", inventory)
    dump_json(output / "code_version_compatibility.json", {"cross_version_reuse": False,
        "reason": "No unverified cross-version reuse; historical SDP pre-fix is not equivalent to signed recovery.",
        "current_source_sha256": sources})
    tasks, batches, reuse_refs, sequences = [], [], [], {}
    for data in plan["datasets"]:
        path = resolve(data["observation_path"], plan_path.parent)
        args, truth, meta = load_npz_inputs(path)
        seq = signature([args["node_ids"], args["timestamps"].tolist() if args["timestamps"] is not None else None,
                         {k: v.tolist() for k, v in args["anchor_position_world"].items()},
                         {k: v.tolist() for k, v in args["estimated_position_local"].items()},
                         [e.rotation_b3_sensor.tolist() for e in args["edges"]]])
        sequence = data["base_sequence_id"]
        if "REPLACE_WITH" in sequence or (sequence in sequences and sequences[sequence] != seq):
            raise ValueError("Invalid base_sequence_id or nonidentical sequence/extrinsics")
        sequences[sequence] = seq
        for field in ("sigma_azimuth_deg", "sigma_elevation_deg"):
            if data[field] is not None and not np.isclose(data[field], meta[field]):
                raise ValueError(f"Dataset {field} disagrees with NPZ")
        for trial in plan["trials"]["trial_ids"]:
            for repeat in range(plan["trials"]["timing_repeats"]):
                for start in plan["windows"]["window_starts"]:
                    stop = start + plan["windows"]["window_size"]
                    if plan["windows"]["window_size"] < 1 or stop > meta["sample_count"]:
                        raise ValueError(f"Invalid window [{start},{stop}) for {path}")
                    window = WindowInput(window_id=f"{meta['sha256'][:12]}:{start}:{stop}", absolute_indices=np.arange(start, stop), **args)
                    pending = []
                    for config in configs:
                        cfg = dict(config["config"], trial_id=trial)
                        needs_trace = any(role.startswith("F2") for role in config["roles"])
                        needs_timing = any(role.startswith("F3a") for role in config["roles"])
                        old = next((r for r in existing if r["dataset"]["sha256"] == meta["sha256"]
                            and r["absolute_indices"] == window.absolute_indices.tolist()
                            and r["graph"]["node_ids"] == list(window.node_ids)
                            and r["graph"]["anchor_ids"] == list(window.anchor_ids)
                            and r["graph"]["estimated_ids"] == list(window.estimated_ids)
                            and r["graph"]["observation_adjacency"] == window.observation_adjacency.tolist()
                            and signature(r["config"]) == signature(cfg)
                            and r["plot_export"]["context"]["base_sequence_id"] == sequence
                            and r["plot_export"]["context"]["noise_level_id"] == data["noise_level_id"]
                            and r["plot_export"]["context"]["noise_realization_id"] == data["noise_realization_id"]
                            and (not needs_trace or r.get("recording", {}).get("selection_trace_enabled", False))
                            and (not needs_timing or r.get("recording", {}).get("selection_trace_enabled", False)
                                 == options.get("trace_selection", False))
                            and r["plot_export"]["context"]["timing_repeat"] == repeat), None)
                        task = {"input": str(path), "input_hash": meta["sha256"], "noise": data["noise_level_id"],
                            "window_start": start, "trial": trial, "repeat": repeat, "config_id": config["config_id"],
                            "config_hash": signature(cfg), "roles": config["roles"], "action": "reuse" if old else "collect",
                            "trace_required_by_F2": needs_trace, "timing_required_by_F3a": needs_timing,
                            "anchor_budget": cfg["anchor_budget_per_node"], "pairwise_budget": cfg["pairwise_budget_per_node"]}
                        tasks.append(task)
                        if old:
                            reuse_refs.append({**task, "source": old["source_directory"], "run_id": old["plot_export"]["run_id"]})
                        else:
                            pending.append(config["overrides"])
                    if not pending:
                        continue
                    tag = f"input_{meta['sha256'][:12]}_{signature(data)[:8]}_t{trial}_r{repeat}_w{start}"
                    methods = output / (tag + "_methods.json")
                    dump_json(methods, pending)
                    context = {**data, "timing_repeat": repeat, "invocation_id": output.name + ":" + tag}
                    context.pop("observation_path")
                    context_path = output / (tag + "_context.json")
                    dump_json(context_path, context)
                    command = [sys.executable, "-m", "src.exp_node_gnbp", "--observations", str(path),
                        "--config", str(resolve(plan["base_algorithm_config"], plan_path.parent)),
                        "--methods-config", str(methods), "--export-config", str(options_path),
                        "--export-context", str(context_path), "--export-suite", "r5_shared",
                        "--output-root", str(output / "runs"), "--window-size", str(plan["windows"]["window_size"]),
                        "--window-starts", str(start), "--trial-id", str(trial),
                        "--measurement-mode", "b4", "--input-adjacency-convention", "target_observer"]
                    batches.append({"tag": tag, "command": command})
    dump_json(output / "physical_run_references.json", {"suite_references": refs, "reused": reuse_refs})
    write_csv(output / "collection_dry_run.csv", tasks)
    write_csv(output / "figure_capabilities.csv", tasks)
    dump_json(output / "missing_runs.json", [t for t in tasks if t["action"] == "collect"])
    dump_json(output / "trace_collection_plan.json", [t for t in tasks if t["trace_required_by_F2"]])
    write_csv(output / "budget_grid_coverage.csv", [t for t in tasks if "F5.final" in t["roles"]])
    dump_json(output / "commands.json", batches)
    (output / "actual_commands.txt").write_text("\n".join(subprocess.list2cmdline(b["command"]) for b in batches), encoding="utf-8")
    help_check = subprocess.run([sys.executable, "-m", "src.exp_node_gnbp", "--help"], cwd=ROOT, capture_output=True, text=True)
    dump_json(output / "parser_validation.json", {"returncode": help_check.returncode,
        "config_class_validation": True, "references": len(refs), "distinct_configs": len(configs),
        "pending": sum(t["action"] == "collect" for t in tasks), "reused": len(reuse_refs)})
    if help_check.returncode:
        raise RuntimeError(help_check.stderr)
    print(f"r5 preflight: {len(refs)} references, {len(configs)} configurations, {len(tasks)} tasks; {len(reuse_refs)} reused. {output}", flush=True)
    if not dry_run:
        for batch in batches:
            print("Running " + batch["tag"], flush=True)
            with (output / (batch["tag"] + ".log")).open("w", encoding="utf-8") as stream:
                result = subprocess.run(batch["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            batch["returncode"] = result.returncode
            dump_json(output / "commands.json", batches)
            if result.returncode:
                finalize_collection(output, tasks, refs, reuse_refs)
                raise RuntimeError(f"Collection failed: {output / (batch['tag'] + '.log')}")
        plot = read(plan_path.parent / "plot_node_gnbp_r5.json")
        plot["results_roots"] = sorted(set([str(output / "runs")] if batches else []) | {r["source"] for r in reuse_refs})
        plot["output_dir"] = str(ROOT / "figures/node_gnbp/r5" / output.name)
        dump_json(output / "plot_config.json", plot)
        finalize_collection(output, tasks, refs, reuse_refs)
    return output
