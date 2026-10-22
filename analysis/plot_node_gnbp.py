"""Replot exported node-GNBP experiments without importing/running any solver.

python analysis/plot_node_gnbp.py --config configs/node_gnbp_plot/plot_node_gnbp.resolved.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

# Enable both the documented direct script command and python -m analysis....
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plot_style import configure_style
from analysis.plot_figures import (
    plot_accuracy_runtime,
    plot_budget,
    plot_diagnostics,
    plot_fixed_budget,
)
from analysis.plot_io import dump_json, load_runs, select_nodes, write_csv, write_gallery


def merge_config(defaults: dict, override: dict) -> dict:
    """Deep merge options while rejecting unknown keys/likely spelling errors."""
    result = copy.deepcopy(defaults)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Unknown plotting option: {key}")
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result[key], dict)
            else value
        )
    return result


def validate_config(config: dict, runs: list[dict]) -> None:
    """Fail explicitly on unsupported semantics instead of silently ignoring them."""
    d, m, f = config["diagnostics"], config["metrics"], config["figures"]
    expected = [
        (config, "plot_schema_version", 1),
        (config, "strict", True),
        (d, "edge_scope", "owned"),
        (d, "bearing_frame", "b3"),
        (d, "x_axis", "raw_observation_ordinal"),
        (d, "angle_unit", "deg"),
        (d, "break_at_missing", True),
        (m, "aggregate", "equal_node_rmse"),
        (m, "rotation_metric", "geodesic_deg"),
        (m, "translation_metric", "transform_translation_m"),
        (m, "time_scope", "network"),
        (m, "time_repeat_summary", "median"),
        (m, "error_repeat_unit", "input_window_selection_trial"),
        (m, "report_failures", True),
        (f["segmentation"], "show_selected", False),
        (f["segmentation"], "show_representatives", False),
        (f["selection"], "show_candidates", False),
        (f["accuracy_runtime"], "noise_layout", "combined"),
        (f["accuracy_runtime"], "connect_by", "noise"),
        (f["accuracy_runtime"], "runtime_mode", "pipeline_to_estimate"),
        (f["accuracy_runtime"], "method_encoding", "color_and_line"),
        (f["accuracy_runtime"], "noise_encoding", "marker"),
        (f["accuracy_runtime"], "show_error_bars", False),
        (f["fixed_budget"], "noise_layout", "separate"),
        (f["fixed_budget"], "plot_type", "box_with_points"),
        (f["fixed_budget"], "pose_stage", "final_pose"),
    ]
    for section, key, value in expected:
        if section[key] != value:
            raise ValueError(f"Unsupported {key}={section[key]!r}; expected {value!r}.")
    if d["axes_mode"] not in ("stacked", "twin_y") or m["node_mode"] not in (
        "aggregate",
        "per_node",
    ):
        raise ValueError("axes_mode must be stacked/twin_y; node_mode aggregate/per_node.")
    if d["max_edges_per_page"] < 1 or m["minimum_boxplot_samples"] < 5 or config["dpi"] < 1:
        raise ValueError("Invalid pagination/DPI; boxplots require at least five logical samples.")
    if not set(config["formats"]) <= {"png", "svg", "pdf"} or not config["formats"]:
        raise ValueError("formats must be a nonempty subset of png,svg,pdf.")
    if not set(f["selection"]["stages"]) <= {"anchor", "pairwise"}:
        raise ValueError("Unknown selection stage.")
    for key in ("accuracy_runtime", "fixed_budget"):
        if f[key]["y_scale"] not in ("log", "linear"):
            raise ValueError("y_scale must be log or linear.")
    if f["accuracy_runtime"]["x_scale"] not in ("log", "linear"):
        raise ValueError("x_scale must be log or linear.")
    order = m["noise_order"]
    if len(set(order)) != len(order) or not order:
        raise ValueError("noise_order must contain unique explicit IDs.")
    contexts = {}
    for run in runs:
        select_nodes(run, d["plot_nodes"])
        select_nodes(run, m["plot_nodes"])
        context = run["plot_export"]["context"]
        noise = context["noise_level_id"]
        sigmas = (context["sigma_azimuth_deg"], context["sigma_elevation_deg"])
        if noise not in order:
            raise ValueError(
                f"noise_order omits actual noise {noise}; all means every actual noise."
            )
        if noise in contexts and contexts[noise] != sigmas:
            raise ValueError(f"Noise ID {noise} has inconsistent sigma values.")
        contexts[noise] = sigmas
    noise_levels = f["accuracy_runtime"]["noise_levels"]
    if noise_levels != "all" and (not noise_levels or not set(noise_levels) <= set(order)):
        raise ValueError("noise_levels must be all or known noise IDs.")

## 可以读外部指定config，也可以使用默认参数配置
def main() -> None:
    """Resolve CLI > JSON > defaults, render separately for every graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--plot-nodes", nargs="+")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=[      
            "segmentation",
            "selection",
            "accuracy_runtime",
            "fixed_budget",
            "budget_selection",
        ],
    )
    args = parser.parse_args()
    ## 读取外部配置
    defaults = json.loads(
        (ROOT / "configs/node_gnbp_plot/plot_node_gnbp.json").read_text(encoding="utf-8")
    )
    ## 按外部文件涉及的配置修改本地配置，不涉及的配置按照默认来
    config = (
        merge_config(defaults, json.loads(args.config.read_text(encoding="utf-8")))
        if args.config
        else defaults
    )
    ## 读取结果文件位置，输出路径
    if args.results:
        config["results_roots"] = args.results
    if args.output:
        config["output_dir"] = args.output
    ## 在结果中绘制哪个待估无人机
    if args.plot_nodes:
        nodes = "all" if args.plot_nodes == ["all"] else args.plot_nodes
        config["diagnostics"]["plot_nodes"] = nodes
        config["metrics"]["plot_nodes"] = nodes
    ## 输出图片位置
    if args.figures:
        for key, spec in config["figures"].items():
            spec["enabled"] = key in args.figures
    # All relative paths are repository-relative, independent of terminal cwd.
    ## 配置输出路径
    config["results_roots"] = [str((ROOT / p).resolve()) for p in config["results_roots"]]
    config["output_dir"] = str((ROOT / config["output_dir"]).resolve())
    ## 读取算法运行结果
    runs = load_runs(config["results_roots"])
    ## graphs是无人机之间不同的观测图，如ABC三机，只有A-B，A-C与A-B，A-C，B-C是两种图，对应两个graph_id
    graphs = list(dict.fromkeys(r["plot_export"]["graph_id"] for r in runs))
    ## 选择绘制哪一种观测图
    requested = config["graph_ids"]
    ## all_separate 表示所有观测图的结果都进行绘制，分别输出在不同的图中
    if requested != "all_separate":
        if not isinstance(requested, list) or not requested or not set(requested) <= set(graphs):
            raise ValueError(f"graph_ids must be all_separate or a subset of {graphs}")
        graphs = requested
    ## 删选用于绘图的结果
    runs = [r for r in runs if r["plot_export"]["graph_id"] in graphs]
    validate_config(config, runs)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "resolved_plot_config.json", config)
    configure_style()
    issues = []
    ## 按照观测图分组绘制不同的结果
    for graph in graphs:
        group = [r for r in runs if r["plot_export"]["graph_id"] == graph]
        for family in ("segmentation", "selection"):
            if config["figures"][family]["enabled"]:
                plot_diagnostics(group, config, family, issues)
        for name, function in (
            ("accuracy_runtime", plot_accuracy_runtime),
            ("fixed_budget", plot_fixed_budget),
            ("budget_selection", plot_budget),
        ):
            if config["figures"][name]["enabled"]:
                function(group, config, issues)
    write_csv(output / "coverage_issues.csv", issues)
    dump_json(
        output / "plot_summary.json",
        {
            "physical_runs": len(runs),
            "graphs": graphs,
            "issues": issues,
            "png_count": len(list(output.glob("F*.png"))),
        },
    )
    write_gallery(output, issues)
    print(f"Plots saved: {output}; coverage issues: {len(issues)}")


if __name__ == "__main__":
    main()
