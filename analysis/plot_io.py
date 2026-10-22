"""Validated readers and traceable figure outputs; never import algorithm modules."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def digest(value: Any) -> str:
    """Stable identity for grouping and filenames."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def slug(value: str) -> str:
    """A readable portable name, with a hash to avoid sanitized-ID collisions."""
    return re.sub(r"[^\w.-]+", "-", value)[:80] + "-" + digest(value)[:6]


def dump_json(path: Path, value: Any) -> None:
    """Save strict JSON."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write heterogeneous rows with explicit structured values."""
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def load_runs(roots: list[str]) -> list[dict]:
    """Read versioned exports and snapshots, rejecting corrupt/old/incomplete data."""
    runs, seen, seen_files = [], set(), set()
    for root in roots:
        path = Path(root)
        files = (
            [path / "plot_manifest.json"]
            if (path / "plot_manifest.json").exists()
            else sorted(path.rglob("plot_manifest.json"))
        )
        if not files:
            raise ValueError(f"No plot exports under {path}; collect with --export-config first.")
        for manifest_path in files:
            if manifest_path.resolve() in seen_files:
                continue
            seen_files.add(manifest_path.resolve())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("export_schema_version") != 1:
                raise ValueError(f"Unsupported export schema: {manifest_path}")
            for run in json.loads(
                (manifest_path.parent / "plot_runs.json").read_text(encoding="utf-8")
            ):
                for field in ("plot_export", "config", "nodes", "graph", "timings", "evaluation"):
                    if field not in run:
                        raise ValueError(
                            f"Missing {field}; recollect with full export: {manifest_path}"
                        )
                meta = run["plot_export"]
                if run.get("schema_version") != 1 or meta.get("export_schema_version") != 1:
                    raise ValueError("Unsupported run/export schema version.")
                for node_id, node in run["nodes"].items():
                    for field in (
                        "segments",
                        "candidates",
                        "selected",
                        "initial_pose",
                        "final_pose",
                        "diagnostics",
                    ):
                        if field not in node:
                            raise ValueError(
                                f"Missing node {node_id}.{field}; enable full export."
                            )
                for field in (
                    "time_to_initial_estimate_wall_seconds",
                    "time_to_final_estimate_wall_seconds",
                ):
                    if field not in run["timings"]:
                        raise ValueError(
                            f"Missing timing {field}; recollect with stage-ready instrumentation."
                        )
                for field in (
                    "noise_level_id",
                    "noise_realization_id",
                    "sigma_azimuth_deg",
                    "sigma_elevation_deg",
                    "base_sequence_id",
                ):
                    if field not in meta["context"]:
                        raise ValueError(f"Missing context {field}; enable noise metadata export.")
                if meta["run_id"] in seen:
                    raise ValueError(f"Duplicate physical run_id {meta['run_id']}")
                seen.add(meta["run_id"])
                snapshot = manifest["snapshots"].get(meta["snapshot_id"])
                if snapshot is None:
                    raise ValueError("Missing input snapshot; enable save_plot_inputs.")
                input_path = manifest_path.parent / snapshot["path"]
                if hashlib.sha256(input_path.read_bytes()).hexdigest() != snapshot["sha256"]:
                    raise ValueError(f"Snapshot checksum mismatch: {input_path}")
                with np.load(input_path, allow_pickle=False) as raw:
                    arrays = {key: raw[key] for key in raw.files}
                for edge in snapshot["edges"]:
                    for field in (
                        "absolute_indices",
                        "ordinal",
                        "bearing_b3",
                        "valid_mask",
                        "break_before",
                    ):
                        if edge["prefix"] + "_" + field not in arrays:
                            raise ValueError(
                                f"Missing snapshot field {field}; enable full plot export."
                            )
                    prefix = edge["prefix"] + "_"
                    idx = arrays[prefix + "absolute_indices"]
                    valid = arrays[prefix + "valid_mask"]
                    bearing = arrays[prefix + "bearing_b3"]
                    if (
                        not np.array_equal(idx, run["absolute_indices"])
                        or not np.array_equal(
                            arrays[prefix + "ordinal"], np.arange(1, len(idx) + 1)
                        )
                        or bearing.shape != (len(idx), 3)
                        or valid.shape != (len(idx),)
                        or valid.dtype.kind != "b"
                        or arrays[prefix + "break_before"].shape != (len(idx),)
                    ):
                        raise ValueError("Snapshot shapes or raw-slot alignment are invalid.")
                    if not np.all(np.isfinite(bearing[valid])):
                        raise ValueError("Solver-valid snapshot bearings must be finite.")
                run["_snapshot"] = snapshot
                run["_arrays"] = arrays
                run["_source"] = str(manifest_path.parent)
                runs.append(run)
    return runs


def select_nodes(run: dict, requested: str | list[str]) -> list[str]:
    """Only estimated nodes have pose errors and own the selection process."""
    available = run["graph"]["estimated_ids"]
    nodes = list(available) if requested == "all" else list(requested)
    if not nodes or len(set(nodes)) != len(nodes) or not set(nodes) <= set(available):
        raise ValueError(
            f"Invalid/duplicate plot_nodes {requested}; estimated IDs are {available}"
        )
    return nodes


def selected_edges(run: dict, options: dict, stage: str | None = None) -> list[dict]:
    """Filter explicit directed endpoints; never infer IDs by splitting strings."""
    nodes = select_nodes(run, options["plot_nodes"])
    edges = [e for e in run["_snapshot"]["edges"] if e["observer_id"] in nodes]
    wanted = options.get("plot_edges", "all")
    if wanted != "all":
        pairs = {
            (e["observer_id"], e["target_id"]) if isinstance(e, dict) else tuple(e) for e in wanted
        }
        available = {(e["observer_id"], e["target_id"]) for e in edges}
        if not pairs <= available:
            raise ValueError(f"Unknown or not-owned directed edges: {pairs - available}")
        edges = [e for e in edges if (e["observer_id"], e["target_id"]) in pairs]
    return [e for e in edges if stage is None or e["stage"] == stage]


def save_figure(
    fig: Any,
    name: str,
    rows: list[dict],
    coverage: list[dict],
    config: dict,
    sources: list[dict],
    details: dict | None = None,
) -> None:
    """Every figure includes raw plotted values, coverage and its resolved config."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for extension in config["formats"]:
        fig.savefig(output / f"{name}.{extension}", dpi=config["dpi"], bbox_inches="tight")
    write_csv(output / f"{name}.csv", rows)
    write_csv(output / f"{name}.coverage.csv", coverage)
    if "r5" in config:
        node_rows = []
        for run in sources:
            for node_id, node in run["nodes"].items():
                for stage in ("initial_pose", "final_pose"):
                    node_rows.append({"run_id": run["plot_export"]["run_id"], "node_id": node_id,
                        "stage": stage, "status": run["status"], "pose_available": node.get(stage) is not None,
                        "noise": run["plot_export"]["context"]["noise_level_id"],
                        "anchor_budget": run["config"]["anchor_budget_per_node"],
                        "pairwise_budget": run["config"]["pairwise_budget_per_node"],
                        **run.get("evaluation", {}).get(node_id, {}).get(stage, {})})
        write_csv(output / f"{name}.per_node.csv", node_rows)
    dump_json(
        output / f"{name}.figure_config.json",
        {
            "config": config,
            "details": details or {},
            "sources": [{"directory": r["_source"], **r["plot_export"]} for r in sources],
        },
    )


def write_gallery(output: Path, issues: list[dict]) -> None:
    """Create a local image index linking each figure's audit files."""
    cards = []
    resolved = json.loads((output / "resolved_plot_config.json").read_text(encoding="utf-8"))
    order = resolved.get("r5", {}).get("figure_order", ["F1", "F2", "F3", "F4", "F5"])
    def key(path):
        family = path.stem.split("_")[0]
        return (order.index(family) if family in order else len(order), path.stem)
    for path in sorted(output.glob("F*.png"), key=key):
        name = html.escape(path.stem)
        links = " · ".join(
            f'<a href="{name}{suffix}">{label}</a>'
            for suffix, label in (
                (".pdf", "PDF"),
                (".svg", "SVG"),
                (".csv", "数据"),
                (".per_node.csv", "逐节点误差"),
                (".coverage.csv", "覆盖记录"),
                (".figure_config.json", "配置与来源"),
            )
            if (output / (path.stem + suffix)).exists()
        )
        metadata = json.loads(path.with_suffix(".figure_config.json").read_text(encoding="utf-8"))
        source = metadata["sources"]
        contexts = [s["context"] for s in source]
        detail = metadata["details"]
        node_ids = detail.get("node_ids", metadata["config"]["metrics"]["plot_nodes"])
        if node_ids == "all" and source:
            node_ids = list(source[0]["budget_counts"]["nodes"])
        with path.with_suffix(".coverage.csv").open(encoding="utf-8-sig", newline="") as stream:
            cov = list(csv.DictReader(stream))
        attempts = [r for r in cov if "usable_error" in r]
        successful = sum(r.get("usable_error") == "True" for r in attempts)
        stages = sorted({r["stage"] for r in cov if r.get("stage")})
        budgets = sorted({(n["anchor"]["requested"], n["pairwise"]["requested"])
                           for s in source for n in s["budget_counts"]["nodes"].values()})
        description = {"节点": node_ids, "聚合": detail.get("aggregation", metadata["config"]["metrics"]["node_mode"]),
                       "位姿阶段": detail.get("pose_stage", stages or "选点/观测诊断"),
                       "KA/KP": budgets, "噪声": sorted({c["noise_level_id"] for c in contexts}),
                       "有效/尝试样本": f"{successful}/{len(attempts)}" if attempts else "诊断图，见覆盖表"}
        description_text = html.escape(json.dumps(description, ensure_ascii=False))
        cards.append(f'<article><h2>{name}</h2><p>{description_text}</p><p>{links}</p>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a></article>')
    issue_text = "".join(f"<li>{html.escape(str(i))}</li>" for i in issues)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GNBP 实验绘图</title>'
    page += '<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;background:#f4f6f8;color:#182331}article{background:white;padding:20px;margin:24px 0;border-radius:8px}h2{font-size:16px;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#0072b2}li{margin:8px 0}</style>'
    page += "<h1>BO-EPnP + GNBP：实验绘图</h1><p>图像由已保存的实验结果独立绘制。单次运行仅用于诊断；completed 不等于达到收敛阈值。F3 为当前公共流程的全网络阶段就绪耗时，导出不计入。</p>"
    page += '<p><a href="resolved_plot_config.json">解析后的绘图配置</a> · <a href="coverage_issues.csv">缺项清单</a></p>'
    page += f"<details open><summary>缺项/覆盖提示：{len(issues)} 条</summary><ul>{issue_text}</ul></details>"
    page += "".join(cards) + "</html>"
    (output / "index.html").write_text(page, encoding="utf-8")
