"""Adapt the r5 external protocol to the existing r3 plotting interfaces."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((Path(__file__).with_name("r5_contracts.json")).read_text(encoding="utf-8"))


def checked_config(value: dict, filename: str) -> dict:
    """Merge protocol defaults; reject unknown keys at every dictionary level."""
    def merge(base, supplied, path):
        if not isinstance(base, dict):
            return copy.deepcopy(supplied)
        if not isinstance(supplied, dict):
            raise ValueError(f"Expected object at {path}")
        unknown = set(supplied) - set(base)
        if unknown:
            raise ValueError(f"Unknown r5 option at {path}: {sorted(unknown)}")
        result = copy.deepcopy(base)
        for key, item in supplied.items():
            result[key] = merge(base[key], item, path + "." + key)
        return result
    result = merge(CONTRACTS[filename], value, filename)
    if result["schema_version"] != CONTRACTS[filename]["schema_version"]:
        raise ValueError("Unsupported r5 schema_version")
    return result


def fixed_policy(actual: dict, reference: dict, mutable: set[str], prefix: str = "") -> None:
    """Fail on unsupported protocol changes rather than silently ignoring them."""
    for key, default in reference.items():
        path = prefix + key
        if path in mutable:
            continue
        value = actual[key]
        if isinstance(default, dict):
            fixed_policy(value, default, mutable, path + ".")
        elif value != default:
            raise ValueError(f"Unsupported r5 policy {path}={value!r}")


def export_options(raw: dict) -> dict:
    """Keep numerical settings separate from recording/export switches."""
    if raw.get("schema_version") != "plot_export_r5":
        return raw
    cfg = checked_config(raw, "export_plot_data_r5.json")
    fixed_policy(cfg, CONTRACTS["export_plot_data_r5.json"],
                 {"enabled", "output_subdir", "selection_trace.enabled"})
    if Path(cfg["output_subdir"]).is_absolute() or ".." in Path(cfg["output_subdir"]).parts:
        raise ValueError("output_subdir must remain inside the run directory")
    return {"enabled": cfg["enabled"], "r5": cfg,
            "trace_selection": cfg["enabled"] and cfg["selection_trace"]["enabled"]}


def adapt_plot(raw: dict, path: Path, defaults: dict) -> dict:
    """Translate r5 figure names/options without replacing the r3 renderer."""
    cfg = checked_config(raw, "plot_node_gnbp_r5.json")
    mutable = {"results_roots", "output_dir", "graph_ids", "figure_order",
               "diagnostics.plot_nodes", "diagnostics.plot_edges", "diagnostics.axes_mode",
               "diagnostics.unwrap_azimuth", "diagnostics.max_edges_per_page", "diagnostics.example",
               "metrics.plot_nodes", "metrics.node_mode", "metrics.noise_order", "metrics.noise_levels",
               "metrics.log_display_floor"}
    mutable |= {"style." + k for k in ("font_family", "font_size_pt", "line_width", "marker_size",
                                       "grid_alpha", "formats", "dpi")}
    for name in cfg["figures"]:
        mutable.add(f"figures.{name}.enabled")
    mutable |= {"figures.segmentation.source", "figures.stage_selection.source",
                "figures.stage_selection.show_selection_order", "figures.stage_selection.show_score_after",
                "figures.stage_selection.show_rank", "figures.stage_selection.show_baseline",
                "figures.stage_selection.annotate_score_values", "figures.stage_selection.score_marker",
                "figures.stage_selection.score_value_format", "figures.stage_selection.show_segment_boundaries",
                "figures.accuracy_runtime.anchor_budget", "figures.accuracy_runtime.gnbp_pairwise_budgets",
                "figures.accuracy_runtime.noise_levels", "figures.accuracy_runtime.error_y_scale",
                "figures.accuracy_budget_methods.error_y_scale", "figures.accuracy_budget_methods.marker",
                "figures.accuracy_budget_methods.noise_levels", "figures.fixed_budget_strategies.minimum_boxplot_samples",
                "figures.budget_sensitivity.error_y_scale", "figures.budget_sensitivity.series_styles",
                "figures.budget_sensitivity.noise_levels", "figures.budget_sensitivity.show_selection_small_multiples"}
    fixed_policy(cfg, CONTRACTS["plot_node_gnbp_r5.json"], mutable)
    if len(cfg["figure_order"]) != 6 or set(cfg["figure_order"]) != {"F1", "F2", "F3a", "F3b", "F4", "F5"}:
        raise ValueError("figure_order must list each r5 family exactly once")
    if any(value <= 0 for value in cfg["metrics"]["log_display_floor"].values()):
        raise ValueError("Log display floors must be positive")
    if cfg["diagnostics"]["example"]["selection"] != "explicit":
        raise ValueError("Configure an explicit diagnostic example")
    for key in ("accuracy_budget_methods", "budget_sensitivity", "accuracy_runtime"):
        levels = cfg["figures"][key]["noise_levels"]
        if levels != "all" and (not levels or not set(levels) <= set(cfg["metrics"]["noise_order"])):
            raise ValueError("Unknown or empty noise_levels")
    styles = cfg["figures"]["budget_sensitivity"]["series_styles"]
    expected_keys = {"pairwise_budget", "label", "color", "marker", "linestyle"}
    if any(set(s) != expected_keys for s in styles) or len(styles) != 4 or {s["pairwise_budget"] for s in styles} != {2, 4, 6, 8}:
        raise ValueError("F5 requires one complete style per KP=2,4,6,8")
    if cfg["metrics"]["node_mode"] not in ("aggregate", "per_node", "both"):
        raise ValueError("node_mode must be aggregate/per_node/both")
    for name in ("accuracy_budget_methods", "budget_sensitivity", "accuracy_runtime"):
        if cfg["figures"][name]["error_y_scale"] not in ("linear", "log"):
            raise ValueError("error_y_scale must be linear or log")
    result = copy.deepcopy(defaults)
    result["r5"] = cfg
    for name in ("results_roots", "output_dir"):
        value = cfg[name]
        result[name] = ([str((path.parent / p).resolve()) for p in value]
                        if isinstance(value, list) else str((path.parent / value).resolve()))
    result["graph_ids"] = cfg["graph_ids"]
    result["formats"], result["dpi"] = cfg["style"]["formats"], cfg["style"]["dpi"]
    d = cfg["diagnostics"]
    result["diagnostics"].update({k: v for k, v in d.items() if k != "example"})
    result["diagnostics"]["example"] = d["example"]
    result["diagnostics"]["show_selection_order"] = cfg["figures"]["stage_selection"]["show_selection_order"]
    result["metrics"].update({k: cfg["metrics"][k] for k in ("plot_nodes", "node_mode", "noise_order")})
    result["metrics"]["minimum_boxplot_samples"] = cfg["figures"]["fixed_budget_strategies"]["minimum_boxplot_samples"]
    for target, source in (("segmentation", "segmentation"), ("selection", "stage_selection")):
        item = cfg["figures"][source]
        result["figures"][target].update(item["source"], enabled=item["enabled"], suite=None)
        if target == "selection":
            result["figures"][target]["show_boundaries"] = item["show_segment_boundaries"]
    a = cfg["figures"]["accuracy_runtime"]
    result["figures"]["accuracy_runtime"].update(
        enabled=a["enabled"], suite=None, anchor_budget=a["anchor_budget"],
        frontend_methods=a["frontends"], gnbp_pairwise_budgets=a["gnbp_pairwise_budgets"],
        gnbp_initialization_method=a["gnbp_frontend"], noise_levels=a["noise_levels"],
        y_scale=a["error_y_scale"])
    result["figures"]["accuracy_runtime"]["frontend_reference_pairwise_budget"] = a["frontend_reference_pairwise_budget"]
    result["figures"]["fixed_budget"].update(
        enabled=cfg["figures"]["fixed_budget_strategies"]["enabled"], suite=None)
    result["figures"]["budget_selection"]["enabled"] = cfg["figures"]["budget_sensitivity"]["show_selection_small_multiples"]
    for key in ("accuracy_budget_methods", "budget_sensitivity"):
        result["figures"][key] = cfg["figures"][key]
    return result
