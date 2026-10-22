"""Five figure families from exported measurements and recorded estimates only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analysis.plot_io import digest, save_figure, select_nodes, selected_edges, slug
from analysis.plot_metrics import (
    invariant_key,
    instance_key,
    matches,
    phase_sample,
    reduce_repeats,
    summarize,
)
from analysis.plot_style import COLORS, MARKERS, ZERO_FLOOR


def _display_floor(config: dict, metric: str) -> float:
    return config.get("r5", {}).get("metrics", {}).get("log_display_floor", {}).get(metric, ZERO_FLOOR)


def angle_series(run: dict, edge: dict, unwrap: bool = True) -> tuple[np.ndarray, ...]:
    """Return raw-slot angles; unwrap each valid contiguous run independently."""
    prefix, data = edge["prefix"] + "_", run["_arrays"]
    idx, x = data[prefix + "absolute_indices"], data[prefix + "ordinal"]
    b = data[prefix + "bearing_b3"]
    valid, breaks = data[prefix + "valid_mask"], data[prefix + "break_before"]
    az = np.arctan2(b[:, 1], b[:, 0])
    el = np.arctan2(b[:, 2], np.hypot(b[:, 0], b[:, 1]))
    az[~valid], el[~valid] = np.nan, np.nan
    start = None
    for pos in range(len(x) + 1):
        boundary = pos == len(x) or not valid[pos] or breaks[pos]
        if boundary and start is not None:
            if unwrap:
                az[start:pos] = np.unwrap(az[start:pos])
            start = None
        if pos < len(x) and valid[pos] and start is None:
            start = pos
    return idx, x, np.rad2deg(az), np.rad2deg(el), valid, breaks


def boundary_positions(
    run: dict, edge: dict, idx: np.ndarray, x: np.ndarray
) -> list[tuple[float, str]]:
    """Boundaries belong between recorded segments, never on the selected sample."""
    info = (
        run["nodes"]
        .get(edge["observer_id"], {})
        .get("segments", {})
        .get(f"{edge['observer_id']}_{edge['target_id']}", {})
    )
    lookup = dict(zip(idx.tolist(), x.tolist()))
    result = []
    segments = info.get("segments", [])
    for previous, current in zip(segments, segments[1:]):
        a, b = previous["indices"][-1], current["indices"][0]
        if a in lookup and b in lookup:
            result.append(((lookup[a] + lookup[b]) / 2, current["start_reason"]))
    return result


def edge_panel(
    axes: list[Any],
    run: dict,
    edge: dict,
    options: dict,
    *,
    selected: bool,
    boundaries: bool,
    column_label: str = "",
) -> tuple[list[dict], dict]:
    """Plot one directed edge, including real selected IDs on both angle axes."""
    idx, x, az, el, valid, breaks = angle_series(run, edge, options["unwrap_azimuth"])
    # Insert NaN *between* sparse contiguous-window slots; preserve boundary samples.
    xp, values = [], [[], []]
    for pos in range(len(x)):
        if pos and breaks[pos]:
            xp.append(np.nan)
            for a in values:
                a.append(np.nan)
        xp.append(x[pos])
        values[0].append(az[pos])
        values[1].append(el[pos])
    observer, target, stage = edge["observer_id"], edge["target_id"], edge["stage"]
    node = run["nodes"].get(observer, {})
    ids = (
        [
            int(k[2])
            for k in node.get("selected", {}).get(stage, [])
            if k[0] == observer and k[1] == target
        ]
        if selected
        else []
    )
    lookup = {int(v): p for p, v in enumerate(idx)}
    if any(i not in lookup or not valid[lookup[i]] for i in ids):
        raise ValueError("Recorded selected ID is absent/invalid in snapshot.")
    slots = [lookup[i] for i in ids]
    order = {}
    if selected and options.get("show_selection_order"):
        for event in node.get("diagnostics", {}).get("selection_trace", []):
            if event["stage"] == stage and event["event_type"] == "add":
                for a, b, i in event["selected_observation_ids"]:
                    if a == observer and b == target:
                        order[int(i)] = event["event_index"]
    bounds = boundary_positions(run, edge, idx, x) if boundaries else []
    az_label = "Azimuth, unwrapped (deg)" if options["unwrap_azimuth"] else "Azimuth (deg)"
    for axis, curve, label, points in zip(axes, values, (az_label, "Elevation (deg)"), (az, el)):
        twin_elevation = options["axes_mode"] == "twin_y" and label == "Elevation (deg)"
        axis.plot(
            xp,
            curve,
            color="#009E73" if twin_elevation else "#525866",
            lw=0.95,
            ls="-." if twin_elevation else "-",
        )
        if selected:
            axis.scatter(
                x[slots],
                points[slots],
                marker="o" if stage == "anchor" else "D",
                s=28,
                facecolors="none",
                edgecolors=COLORS[0 if stage == "anchor" else 1],
                linewidths=1.2,
                zorder=5,
            )
            ordered_ids = sorted(ids, key=lambda i: lookup[i])
            last_x = None
            lane = 0
            midpoint = .5 * (np.nanmin(points) + np.nanmax(points)) if ids else 0
            for i in ordered_ids:
                if i in order:
                    pos = lookup[i]
                    lane = (lane + 1) % 3 if last_x is not None and x[pos] - last_x < 8 else 0
                    last_x = x[pos]
                    direction = -1 if points[pos] > midpoint else 1
                    axis.annotate(str(order[i]), (x[pos], points[pos]), xytext=(2, direction * (8 + 10 * lane)),
                                  textcoords="offset points", fontsize=7, annotation_clip=True,
                                  va="top" if direction < 0 else "bottom",
                                  bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": .4})
            if order:
                axis.margins(y=.28)
        for position, reason in bounds:
            angular = "angle" in reason or "diameter" in reason
            axis.axvline(
                position,
                color="#A5A9B0" if angular else "#9467BD",
                ls="--" if angular else ":",
                lw=0.65,
                alpha=0.65,
            )
        axis.set_ylabel(label)
        axis.set_xlim(0.5, float(x[-1]) + 0.5)
    axes[-1].set_xlabel("Raw observation ordinal (window slots)")
    cfg, context = run["config"], run["plot_export"]["context"]
    info = node.get("segments", {}).get(f"{observer}_{target}", {})
    title = f"{observer} → {target}  |  valid={int(valid.sum())}/{len(x)}, segments={len(info.get('segments', []))}"
    if selected:
        strategy = cfg[stage + "_selection_strategy"]
        total = len(node.get("selected", {}).get(stage, []))
        title += f"\n{stage}: edge {len(ids)}/{int(valid.sum())}; node {total}/{cfg[stage + '_budget_per_node']} · {strategy}"
        title += f" · (KA,KP)=({cfg['anchor_budget_per_node']},{cfg['pairwise_budget_per_node']})"
    if column_label:
        title = column_label + "\n" + title
    axes[0].set_title(title, fontsize=9)
    if not np.any(valid):
        axes[0].text(
            0.5, 0.5, "No valid recorded observations", transform=axes[0].transAxes, ha="center"
        )
    rows = [
        {
            "run_id": run["plot_export"]["run_id"],
            "observer": observer,
            "target": target,
            "stage": stage,
            "absolute_index": int(i),
            "ordinal": int(o),
            "valid": bool(v),
            "break_before": bool(g),
            "azimuth_deg": float(a) if v else None,
            "elevation_deg": float(e) if v else None,
            "selected": int(i) in ids,
            "selection_event_index": order.get(int(i)),
            "anchor_budget": cfg["anchor_budget_per_node"],
            "pairwise_budget": cfg["pairwise_budget_per_node"],
        }
        for i, o, a, e, v, g in zip(idx, x, az, el, valid, breaks)
    ]
    coverage = {
        "run_id": run["plot_export"]["run_id"],
        "observer": observer,
        "target": target,
        "noise": context["noise_level_id"],
        "status": run["status"],
        "raw_slots": len(x),
        "valid": int(valid.sum()),
        "selected_edge": len(ids) if selected else None,
        "segments": len(info.get("segments", [])),
        "missing_data": edge["missing_data"],
        "boundaries": bounds,
    }
    return rows, coverage


def diagnostic_instance(runs: list[dict], config: dict) -> list[dict]:
    """Deterministic first configured noise/input/window/realization, never best error."""
    options = config["diagnostics"]
    if "r5" in config:
        example = options["example"]
        if example["selection"] != "explicit":
            raise ValueError("r5 diagnostic example must be explicit")
        found = [r for r in runs if r["absolute_indices"][0] == example["window_start"]
                 and r["config"]["trial_id"] == example["trial_id"]
                 and all(r["plot_export"]["context"][k] == example[k] for k in
                         ("base_sequence_id", "noise_level_id", "noise_realization_id"))]
        return found
    available = [r for r in runs if r["config"]["trial_id"] == options["selection_trial_id"]]
    noise = options["noise_level"]
    if noise == "first_in_config_order":
        noise = next(
            (
                n
                for n in config["metrics"]["noise_order"]
                if any(r["plot_export"]["context"]["noise_level_id"] == n for r in available)
            ),
            None,
        )
    available = [r for r in available if r["plot_export"]["context"]["noise_level_id"] == noise]
    if options["window"] != "first_in_input_order":
        available = [r for r in available if r["window_id"] == options["window"]]
    if options["realization"] != "first_in_input_order":
        available = [
            r
            for r in available
            if r["plot_export"]["context"]["noise_realization_id"] == options["realization"]
        ]
    if not available:
        return []
    identity = instance_key(available[0])
    return [r for r in available if instance_key(r) == identity]


def _bearing_figure(count: int, columns: int, mode: str) -> tuple[Any, list[list[list[Any]]]]:
    if mode == "stacked":
        fig, axes = plt.subplots(
            count * 2, columns, figsize=(6.5 * columns, 3.5 * count), squeeze=False
        )
        panels = [
            [[axes[2 * row, col], axes[2 * row + 1, col]] for col in range(columns)]
            for row in range(count)
        ]
    else:
        fig, axes = plt.subplots(count, columns, figsize=(7 * columns, 3 * count), squeeze=False)
        panels = [
            [[axes[row, col], axes[row, col].twinx()] for col in range(columns)]
            for row in range(count)
        ]
        for row in panels:
            for pair in row:
                pair[1].spines["right"].set_visible(True)
    return fig, panels


def _context_title(run: dict) -> str:
    c = run["plot_export"]["context"]
    idx = run["absolute_indices"]
    return f"{c['noise_level_id']} (σaz={c['sigma_azimuth_deg']:g}°, σel={c['sigma_elevation_deg']:g}°) · window [{idx[0]},{idx[-1]+1}) · threshold {run['config']['max_angle_deg']:g}°"


def plot_diagnostics(runs: list[dict], config: dict, family: str, issues: list[dict]) -> None:
    """F1/F2 with explicit per-edge panels and actual selection markers."""
    spec, options = config["figures"][family], config["diagnostics"]
    candidates = [
        r
        for r in diagnostic_instance(runs, config)
        if matches(
            r,
            suite=spec["suite"],
            initialization_method=spec["frontend"],
            anchor_budget_per_node=spec["anchor_budget"],
            pairwise_budget_per_node=spec["pairwise_budget"],
            **({k: spec[k] for k in ("anchor_selection_strategy", "pairwise_selection_strategy")}
               if "r5" in config else {}),
        )
    ]
    if not candidates:
        issues.append({"figure": family, "reason": "missing_diagnostic_configuration"})
        return
    groups = defaultdict(list)
    for run in candidates:
        groups[invariant_key(run, set())].append(run)
    for group_id, group in groups.items():
        run = min(
            group,
            key=lambda r: (
                r["plot_export"]["context"]["timing_repeat"],
                r["plot_export"]["run_id"],
            ),
        )
        stages = [None] if family == "segmentation" else spec["stages"]
        if family == "selection" and "r5" in config:
            plot_selection_scores(run, config, issues)
        for stage in stages:
            edges = selected_edges(run, options, stage)
            if not edges:
                issues.append({"figure": family, "stage": stage, "reason": "no_owned_edges"})
            for page in range(0, len(edges), options["max_edges_per_page"]):
                batch = edges[page : page + options["max_edges_per_page"]]
                fig, panels = _bearing_figure(len(batch), 1, options["axes_mode"])
                rows, coverage = [], []
                for row, edge in enumerate(batch):
                    values, cov = edge_panel(
                        panels[row][0],
                        run,
                        edge,
                        options,
                        selected=family == "selection",
                        boundaries=spec["show_boundaries"],
                    )
                    rows.extend(values)
                    coverage.append(cov)
                fig.suptitle(
                    (
                        "F1 · Bearing segmentation"
                        if family == "segmentation"
                        else f"F2 · {stage.capitalize()} selected observations"
                    )
                    + "\n"
                    + _context_title(run),
                    y=1.005,
                )
                fig.tight_layout()
                name = f"F{'1' if family == 'segmentation' else '2'}_{family}_{stage or 'all'}_g{run['plot_export']['graph_id'][:8]}_{slug('+'.join(select_nodes(run, options['plot_nodes'])))}_KA{spec['anchor_budget']}_KP{spec['pairwise_budget']}_{group_id}_p{page // options['max_edges_per_page'] + 1}"
                save_figure(
                    fig,
                    name,
                    rows,
                    coverage,
                    config,
                    [run],
                    {
                        "boundary_legend": "gray dashed: angular; purple dotted: missing/invalid/time gap",
                        "example_policy": "first configured instance",
                    },
                )
                plt.close(fig)


def plot_budget(runs: list[dict], config: dict, issues: list[dict]) -> None:
    """F5 uses budget-specific actual sets; checks the shared frontend for KP sweeps."""
    spec, options = config["figures"]["budget_selection"], config["diagnostics"]
    candidates = [
        r
        for r in diagnostic_instance(runs, config)
        if ("r5" in config or r["plot_export"]["context"]["suite_id"] in spec["suites"])
        and matches(
            r,
            initialization_method=spec["frontend"],
            anchor_selection_strategy=spec["anchor_strategy"],
            pairwise_selection_strategy=spec["pairwise_strategy"],
        )
    ]
    groups = defaultdict(list)
    for run in candidates:
        groups[invariant_key(run, {"anchor_budget_per_node", "pairwise_budget_per_node"})].append(
            run
        )
    if not groups:
        issues.append({"figure": "budget_selection", "reason": "missing_diagnostic_configuration"})
    for group_id, group in groups.items():
        for stage in ("anchor", "pairwise"):
            budgets = spec[stage + "_budgets"]
            other = "pairwise" if stage == "anchor" else "anchor"
            fixed = spec[stage + "_view_fixed_" + other + "_budget"]
            chosen = []
            for budget in budgets:
                found = [
                    r
                    for r in group
                    if r["config"][stage + "_budget_per_node"] == budget
                    and r["config"][other + "_budget_per_node"] == fixed
                ]
                chosen.append(
                    min(
                        found,
                        key=lambda r: (
                            r["plot_export"]["context"]["timing_repeat"],
                            r["plot_export"]["run_id"],
                        ),
                    )
                    if found
                    else None
                )
            present = [r for r in chosen if r is not None]
            if not present:
                issues.append(
                    {"figure": "budget_selection", "stage": stage, "reason": "missing_all_budgets"}
                )
                continue
            frontend_equal = True
            if stage == "pairwise":
                for r in present[1:]:
                    for node in r["graph"]["estimated_ids"]:
                        a, b = present[0]["nodes"][node], r["nodes"][node]
                        frontend_equal &= (
                            a["selected"].get("anchor") == b["selected"].get("anchor")
                            and a["initial_pose"] == b["initial_pose"]
                        )
                if not frontend_equal:
                    issues.append(
                        {
                            "figure": "budget_selection",
                            "stage": stage,
                            "reason": "initial_pose_or_anchor_selection_differs_across_KP; figure_skipped",
                        }
                    )
                    continue
            edges = selected_edges(present[0], options, stage)
            if not edges:
                issues.append(
                    {"figure": "budget_selection", "stage": stage, "reason": "no_owned_edges"}
                )
            for page in range(0, len(edges), options["max_edges_per_page"]):
                batch = edges[page : page + options["max_edges_per_page"]]
                fig, panels = _bearing_figure(len(batch), len(budgets), options["axes_mode"])
                rows, coverage = [], []
                for col, (budget, run) in enumerate(zip(budgets, chosen)):
                    if run is None:
                        coverage.append({"budget": budget, "reason": "missing_configuration"})
                        for row in range(len(batch)):
                            panels[row][col][0].text(
                                0.5,
                                0.5,
                                f"Missing K={budget}",
                                ha="center",
                                transform=panels[row][col][0].transAxes,
                            )
                        continue
                    for row, edge in enumerate(batch):
                        values, cov = edge_panel(
                            panels[row][col],
                            run,
                            edge,
                            options,
                            selected=True,
                            boundaries=spec["show_boundaries"],
                            column_label=f"K_{'A' if stage == 'anchor' else 'P'}={budget}",
                        )
                        rows.extend(values)
                        coverage.append(cov)
                if spec["share_axes"]:
                    for row in range(len(batch)):
                        for angle in range(2):
                            limits = [
                                panels[row][col][angle].get_ylim()
                                for col, r in enumerate(chosen)
                                if r is not None
                            ]
                            for col in range(len(chosen)):
                                panels[row][col][angle].set_ylim(
                                    min(a for a, b in limits), max(b for a, b in limits)
                                )
                fig.suptitle(
                    f"F5 · {stage.capitalize()} budget selection · fixed {other} budget={fixed}\n{_context_title(present[0])}",
                    y=1.005,
                )
                fig.tight_layout()
                name = f"F5_{stage}_g{present[0]['plot_export']['graph_id'][:8]}_{slug('+'.join(select_nodes(present[0], options['plot_nodes'])))}_K{'-'.join(map(str,budgets))}_fixed{other}{fixed}_{group_id}_p{page // options['max_edges_per_page'] + 1}"
                save_figure(
                    fig,
                    name,
                    rows,
                    coverage,
                    config,
                    present,
                    {"frontend_equal_across_KP": frontend_equal, "not_assumed_nested": True},
                )
                plt.close(fig)


def _metric_nodes(run: dict, options: dict) -> list[list[str]]:
    nodes = select_nodes(run, options["plot_nodes"])
    if options["node_mode"] == "both":
        return [nodes] + ([[n] for n in nodes] if len(nodes) > 1 else [])
    return [nodes] if options["node_mode"] == "aggregate" else [[n] for n in nodes]


def plot_selection_scores(run: dict, config: dict, issues: list[dict]) -> None:
    """Draw recorded set scores in actual acceptance order; never rerun selection."""
    spec = config["r5"]["figures"]["stage_selection"]
    for node_id in select_nodes(run, config["diagnostics"]["plot_nodes"]):
        node = run["nodes"][node_id]
        for stage in spec["stages"]:
            events = [e for e in node["diagnostics"].get("selection_trace", []) if e["stage"] == stage]
            if not events:
                issues.append({"figure": "F2", "node": node_id, "stage": stage,
                               "reason": "missing_actual_order_and_score; bearing retained"})
                continue
            if [e["event_index"] for e in events] != list(range(len(events))):
                raise ValueError("Selection trace must contain baseline and contiguous event indices")
            fields = (["logdet"] + (["lambda_min"] if stage == "pairwise" else [])) if spec["show_score_after"] else []
            if spec["show_rank"]:
                fields.append("rank")
            if not fields:
                continue
            shown = [e for e in events if spec["show_baseline"] or e["event_type"] != "baseline"]
            fig, axes = plt.subplots(len(fields), 1, figsize=(9, 2.3 * len(fields)), squeeze=False, sharex=True)
            x = [e["event_index"] for e in shown]
            for axis, field in zip(axes[:, 0], fields):
                y = [e["score_after"].get(field) for e in shown]
                values = [v if isinstance(v, (float, int)) and np.isfinite(v) else np.nan for v in y]
                axis.plot(x, values, marker=spec["score_marker"], color=COLORS[0])
                axis.set_ylabel({"rank": "Recorded rank", "logdet": "Regularized set logdet",
                                 "lambda_min": "Minimum eigenvalue"}[field])
                if field == "rank":
                    from matplotlib.ticker import MaxNLocator
                    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                if not any(np.isfinite(values)):
                    axis.text(.5, .5, "Not scored by this selection strategy", transform=axis.transAxes, ha="center")
                if spec["annotate_score_values"]:
                    for xi, yi in zip(x, values):
                        if np.isfinite(yi):
                            axis.annotate(format(yi, spec["score_value_format"]), (xi, yi),
                                          xytext=(3, 5), textcoords="offset points", fontsize=8)
            axes[-1, 0].set(xlabel="Actual node-stage selection event (0 = baseline)", xticks=x)
            definition = ("Rank then regularized logdet of the fixed-pool normalized 12-column matrix."
                          if stage == "anchor" else
                          "Accumulated Schur information at the initial poses. Rank / E / D mode and winner deltas are recorded in CSV.")
            fig.suptitle(f"F2 · {node_id} · {stage} · {run['config'][stage + '_selection_strategy']}\n{_context_title(run)}")
            fig.text(.5, -.01, definition + "\nSet objectives: larger information is preferred; no monotonicity imposed.", ha="center", fontsize=8)
            rows = [{"run_id": run["plot_export"]["run_id"], "node_id": node_id, **e,
                     "observations": [{"observer_id": a, "target_id": b, "absolute_index": i,
                                       "ordinal": run["absolute_indices"].index(i) + 1,
                                       "final_member": [a, b, i] in [list(k) for k in node["selected"][stage]]}
                                      for a, b, i in e["selected_observation_ids"]]}
                    for e in events]
            name = f"F2_score_{stage}_{slug(node_id)}_{run['plot_export']['run_id']}"
            save_figure(fig, name, rows, [{"node": node_id, "stage": stage, "events": len(events)-1,
                        "actual_order_available": True}], config, [run],
                        {"definition": definition, "objective_direction": "lexicographic maximization",
                         "node_ids": [node_id], "stage": stage, "baseline_recorded": True,
                         "aggregation": "one node, one preselected diagnostic instance"})
            plt.close(fig)


def _controlled_initials(runs: list[dict], nodes: list[str], *, compare_pose: bool) -> bool:
    """Compare anchor sets and, for KP sweeps, already recorded initial poses."""
    paired = defaultdict(list)
    for run in runs:
        paired[instance_key(run)].append(run)
    for group in paired.values():
        for node_id in nodes:
            # A failed/unavailable initialization remains a missing sample.
            available = [r["nodes"][node_id] for r in group if r["nodes"][node_id]["initial_pose"] is not None]
            if not available:
                continue
            first = available[0]
            ids = sorted(map(tuple, first["selected"].get("anchor", [])))
            for node in available[1:]:
                if sorted(map(tuple, node["selected"].get("anchor", []))) != ids:
                    return False
                if compare_pose and any(not np.allclose(node["initial_pose"][key], first["initial_pose"][key],
                                                       rtol=1e-10, atol=1e-10)
                                        for key in ("rotation", "translation")):
                    return False
    return True


def plot_budget_errors(runs: list[dict], config: dict, family: str, issues: list[dict]) -> None:
    """F3b/F5 reuse node-aware aggregation and existing output/coverage writers."""
    spec, options = config["figures"][family], config["metrics"]
    grid = family == "budget_sensitivity"
    budgets = sorted(spec["budget_grid"]["anchor_budgets"] if grid else spec["anchor_budgets"])
    if grid:
        styles = {s["pairwise_budget"]: s for s in spec["series_styles"]}
        series = [{"frontend": spec["frontend"], "pose_stage": "final", **styles[k]}
                  for k in spec["line_order"]]
    else:
        series = [{"pairwise_budget": spec["pairwise_budget"], "color": COLORS[i],
                   "marker": spec["marker"], "linestyle": "--" if s["pose_stage"] == "initial" else "-", **s}
                  for i, s in enumerate(spec["series"])]
    pool = [r for r in runs if r["config"]["anchor_budget_per_node"] in budgets
            and matches(r, anchor_selection_strategy=spec["anchor_selection_strategy"],
                        pairwise_selection_strategy=spec["pairwise_selection_strategy"])
            and any(matches(r, initialization_method=s["frontend"], pairwise_budget_per_node=s["pairwise_budget"])
                    for s in series)]
    groups = defaultdict(list)
    for run in pool:
        groups[invariant_key(run, {"anchor_budget_per_node", "pairwise_budget_per_node", "initialization_method"})].append(run)
    figure_id = "F5" if grid else "F3b"
    if not groups:
        issues.append({"figure": figure_id, "reason": "missing_configuration"})
    for group_id, group in groups.items():
        for nodes in _metric_nodes(group[0], options):
            for noise in options["noise_order"]:
                if spec["noise_levels"] != "all" and noise not in spec["noise_levels"]:
                    continue
                found_noise = [r for r in group if r["plot_export"]["context"]["noise_level_id"] == noise]
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
                rows, coverage, consistency = [], [], {}
                for budget in budgets:
                    found = [r for r in found_noise if r["config"]["anchor_budget_per_node"] == budget]
                    consistency[budget] = _controlled_initials(found, nodes, compare_pose=grid)
                    if not consistency[budget]:
                        issues.append({"figure": figure_id, "noise": noise, "anchor_budget": budget,
                                       "reason": "initial_consistency_mismatch; comparison omitted"})
                for s in series:
                    ys = [[], []]
                    for budget in budgets:
                        found = [r for r in found_noise if matches(r, anchor_budget_per_node=budget,
                                 pairwise_budget_per_node=s["pairwise_budget"], initialization_method=s["frontend"])]
                        stage = s["pose_stage"] + "_pose"
                        samples = [phase_sample(r, nodes, stage, spec["require_actual_budget"]) for r in found]
                        for sample in samples:
                            if not consistency[budget]:
                                sample["usable_error"] = False
                                sample["reason"] = "initial_consistency_mismatch"
                            coverage.append({"series": s["label"], "anchor_budget": budget, "pairwise_budget": s["pairwise_budget"], **sample})
                        if not found:
                            issue = {"figure": figure_id, "series": s["label"], "noise": noise,
                                     "anchor_budget": budget, "pairwise_budget": s["pairwise_budget"], "reason": "missing_configuration"}
                            issues.append(issue)
                            coverage.append(issue)
                        summary = summarize(samples, require_time=False)
                        rows.append({"series": s["label"], "noise": noise, "nodes": nodes, "stage": stage,
                                     "anchor_budget": budget, "pairwise_budget": s["pairwise_budget"],
                                     "initial_consistency": consistency[budget], **summary})
                        for values, metric in zip(ys, ("translation_m", "rotation_deg")):
                            value = summary[metric]
                            floor = config["r5"]["metrics"]["log_display_floor"][metric]
                            values.append(np.nan if value is None else max(value, floor) if spec["error_y_scale"] == "log" else value)
                    for axis, values in zip(axes, ys):
                        axis.plot(budgets, values, color=s["color"], marker=s["marker"],
                                  ls=s["linestyle"], label=s["label"])
                for axis, metric in zip(axes, ("Translation RMSE (m)", "Rotation RMSE (deg)")):
                    axis.set(xlabel="Requested anchor budget per node (KA)", ylabel=metric, xticks=budgets)
                    axis.set_yscale(spec["error_y_scale"])
                    if not grid:
                        axis.legend(fontsize=8)
                aggregation = "Equal-node RMSE" if len(nodes) > 1 else "Per-node error"
                title = "BO-EPnP + GNBP final estimates" if grid else "Initial methods and BO-EPnP + GNBP final (KP=4)"
                fig.suptitle(f"{figure_id} · {noise} · {title}\n{aggregation}: {', '.join(nodes)}")
                if grid:
                    handles, labels = axes[0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .86), ncol=4, fontsize=9)
                valid = sum(r["n"] for r in rows)
                total = sum(r["logical_attempts"] for r in rows)
                fig.text(.5, -.02, f"Usable series samples {valid}/{total}; failures and budget shortfalls remain gaps. "
                         "Counts per KA/KP are in CSV.\nOne trial contributes one node-set RMSE; timing repeats are deduplicated. No interpolation.",
                         ha="center", fontsize=8)
                fig.tight_layout(rect=(0, 0, 1, .87) if grid else (0, 0, 1, 1))
                name = f"{figure_id}_{family}_{slug(noise)}_{slug('+'.join(nodes))}_{group_id}"
                save_figure(fig, name, rows, coverage, config, found_noise,
                            {"node_ids": nodes, "aggregation": aggregation, "pose_stage": "final" if grid else "initial/final",
                             "noise": noise, "budgets": budgets, "pairwise_budgets": sorted({s['pairwise_budget'] for s in series}),
                             "usable_series_samples": valid, "attempted_series_samples": total,
                             "initial_consistency": consistency, "initial_pose_tolerance": "rtol=atol=1e-10"})
                plt.close(fig)


def plot_accuracy_runtime(runs: list[dict], config: dict, issues: list[dict]) -> None:
    """F3 connects each method/configuration across explicitly ordered noise levels."""
    spec, options = config["figures"]["accuracy_runtime"], config["metrics"]
    pool = [
        r
        for r in runs
        if matches(
            r,
            suite=spec["suite"],
            anchor_budget_per_node=spec["anchor_budget"],
            anchor_selection_strategy=spec["anchor_strategy"],
            pairwise_selection_strategy=spec["pairwise_strategy"],
        )
    ]
    groups = defaultdict(list)
    for run in pool:
        groups[invariant_key(run, {"initialization_method", "pairwise_budget_per_node"})].append(
            run
        )
    if not groups:
        issues.append(
            {
                "figure": "accuracy_runtime",
                "reason": "missing_configuration; collect requested KA/methods",
            }
        )
    series = [
        (
            method.upper() + " initial",
            method,
            spec["frontend_reference_pairwise_budget"],
            "initial_pose",
        )
        for method in spec["frontend_methods"]
    ]
    series += [
        (f"BO-EPnP + GNBP (KP={budget})", spec["gnbp_initialization_method"], budget, "final_pose")
        for budget in spec["gnbp_pairwise_budgets"]
    ]
    noise_order = [
        n
        for n in options["noise_order"]
        if spec["noise_levels"] == "all" or n in spec["noise_levels"]
    ]
    for group_id, group in groups.items():
        cohorts = set()
        timing_sources = [r for r in group if any(matches(r, initialization_method=method,
                           pairwise_budget_per_node=budget) for _, method, budget, _ in series)]
        if "r5" in config and len({r["plot_export"]["timing_protocol"] for r in timing_sources}) > 1:
            raise ValueError("F3a cannot combine different timing protocols")
        for _, method, budget, _ in series:
            for noise in noise_order:
                found = [
                    r
                    for r in group
                    if matches(r, initialization_method=method, pairwise_budget_per_node=budget)
                    and r["plot_export"]["context"]["noise_level_id"] == noise
                ]
                if found:
                    cohorts.add(tuple(sorted({instance_key(r, across_noise=True) for r in found})))
        if len(cohorts) > 1:
            issues.append(
                {
                    "figure": "accuracy_runtime",
                    "reason": "unmatched_sequence_window_realization_trial_cohorts; figure_skipped",
                }
            )
            continue
        for nodes in _metric_nodes(group[0], options):
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
            rows, coverage, method_handles, zero = [], [], [], False
            cohorts = set()
            for number, (label, method, budget, stage) in enumerate(series):
                color, linestyle = COLORS[number % len(COLORS)], (
                    "--" if stage == "initial_pose" else "-"
                )
                method_handles.append(Line2D([], [], color=color, ls=linestyle, label=label))
                x, ys = [], [[], []]
                for ni, noise in enumerate(noise_order):
                    found = [
                        r
                        for r in group
                        if matches(
                            r, initialization_method=method, pairwise_budget_per_node=budget
                        )
                        and r["plot_export"]["context"]["noise_level_id"] == noise
                    ]
                    samples = [
                        phase_sample(r, nodes, stage, options["require_actual_budget"])
                        for r in found
                    ]
                    summary = summarize(samples, require_time=True)
                    cohorts.add(tuple(sorted({r["paired_noise_instance_id"] for r in samples})))
                    coverage.extend({"series": label, **s} for s in samples)
                    if not found:
                        coverage.append(
                            {"series": label, "noise": noise, "reason": "missing_configuration"}
                        )
                        issues.append(
                            {
                                "figure": "accuracy_runtime",
                                "series": label,
                                "noise": noise,
                                "reason": "missing_configuration",
                            }
                        )
                    summary.update(
                        series=label,
                        noise=noise,
                        stage=stage,
                        nodes=nodes,
                        anchor_budget=spec["anchor_budget"],
                        pairwise_budget=budget,
                    )
                    rows.append(summary)
                    factor = 1000 if "r5" in config else 1
                    if "r5" in config:
                        summary["wall_milliseconds"] = summary["wall_seconds"] * factor if summary["wall_seconds"] is not None else None
                    x.append(summary["wall_seconds"] * factor if summary["n"] else np.nan)
                    for axis, values, metric in zip(axes, ys, ("translation_m", "rotation_deg")):
                        value = summary[metric]
                        plotted = (
                            max(value, _display_floor(config, metric))
                            if value is not None and spec["y_scale"] == "log"
                            else value
                        )
                        values.append(plotted if summary["n"] else np.nan)
                        if summary["n"]:
                            zero |= value == 0
                            axis.scatter(
                                x[-1],
                                values[-1],
                                marker=MARKERS[ni % len(MARKERS)],
                                color=color,
                                s=40,
                                zorder=4,
                            )
                for axis, values in zip(axes, ys):
                    axis.plot(x, values, color=color, ls=linestyle, lw=1.4)
            noise_handles = []
            for ni, noise in enumerate(noise_order):
                context = next(
                    (
                        r["plot_export"]["context"]
                        for r in group
                        if r["plot_export"]["context"]["noise_level_id"] == noise
                    ),
                    None,
                )
                label = (
                    f"{noise}: σaz/σel={context['sigma_azimuth_deg']:g}°/{context['sigma_elevation_deg']:g}°"
                    if context
                    else f"{noise}: missing input"
                )
                noise_handles.append(
                    Line2D(
                        [],
                        [],
                        color="#555555",
                        ls="",
                        marker=MARKERS[ni % len(MARKERS)],
                        label=label,
                    )
                )
            for axis, label in zip(axes, ("Translation RMSE (m)", "Rotation RMSE (deg)")):
                axis.set(xlabel="Network wall time to estimate (ms)" if "r5" in config else "Network wall time to estimate (s)", ylabel=label)
                axis.set_xscale(spec["x_scale"])
                axis.set_yscale(spec["y_scale"])
            axes[0].legend(handles=method_handles, loc="best", fontsize=8)
            axes[1].legend(handles=noise_handles, loc="best", fontsize=8)
            mode = (
                "Equal-node RMSE"
                if len(nodes) > 1
                else "Node error; timing is whole-network"
            )
            fig.suptitle(f"{'F3a' if 'r5' in config else 'F3'} · {mode} · {', '.join(nodes)} · KA={spec['anchor_budget']}")
            caption = "Wall time of current common pipeline to each estimate. Initial: all-edge segmentation + initialization.\nFinal: initial pipeline + pairwise selection + GNBP. Export excluded; timing repeats are not accuracy samples."
            counts = []
            for noise in noise_order:
                values = [r for r in rows if r["noise"] == noise]
                patterns = {
                    (
                        r["n"],
                        r["logical_attempts"],
                        r["phase_successes"],
                        r["actual_budget_successes"],
                    )
                    for r in values
                }
                if len(patterns) == 1:
                    n, attempts, success, budget_ok = next(iter(patterns))
                    counts.append(
                        f"{noise}: n={n}/{attempts}, success={success}/{attempts}, budget={budget_ok}/{attempts}"
                    )
                else:
                    counts.append(
                        noise
                        + ": "
                        + "; ".join(
                            f"{r['series']} n={r['n']}/{r['logical_attempts']}, S={r['phase_successes']}, B={r['actual_budget_successes']}"
                            for r in values
                        )
                    )
            caption += "\n" + " | ".join(counts)
            if zero:
                caption += " Zero errors use the configured display floor; CSV retains zero."
            if len(cohorts - {()}) > 1:
                caption += " Unmatched experiment cohorts: see coverage."
                issues.append(
                    {
                        "figure": "accuracy_runtime",
                        "reason": "unmatched_sequence_window_trial_cohorts",
                    }
                )
            fig.text(0.5, -0.025, caption, ha="center", va="top", fontsize=8)
            fig.tight_layout()
            name = f"{'F3a' if 'r5' in config else 'F3'}_accuracy_runtime_all_noise_g{group[0]['plot_export']['graph_id'][:8]}_{slug('+'.join(nodes))}_KA{spec['anchor_budget']}_{group_id}"
            save_figure(
                fig,
                name,
                rows,
                coverage,
                config,
                group,
                {"zero_render_floor": {m: _display_floor(config, m) for m in ("translation_m", "rotation_deg")},
                 "noise_order": noise_order, "node_ids": nodes, "aggregation": mode,
                 "pose_stage": "initial frontends / final BO-EPnP+GNBP"},
            )
            plt.close(fig)


def plot_fixed_budget(runs: list[dict], config: dict, issues: list[dict]) -> None:
    """F4 one logical error per trial; insufficient samples use scatter, never boxes."""
    spec, options = config["figures"]["fixed_budget"], config["metrics"]
    pool = [
        r
        for r in runs
        if matches(
            r,
            suite=spec["suite"],
            initialization_method=spec["frontend"],
            anchor_budget_per_node=spec["anchor_budget"],
            pairwise_budget_per_node=spec["pairwise_budget"],
        )
    ]
    groups = defaultdict(list)
    for run in pool:
        groups[
            invariant_key(run, {"anchor_selection_strategy", "pairwise_selection_strategy"})
        ].append(run)
    if not groups:
        issues.append({"figure": "fixed_budget", "reason": "missing_configuration"})
    for group_id, group in groups.items():
        for nodes in _metric_nodes(group[0], options):
            for noise in options["noise_order"]:
                noise_runs = [
                    r for r in group if r["plot_export"]["context"]["noise_level_id"] == noise
                ]
                if not noise_runs:
                    issues.append(
                        {"figure": "fixed_budget", "noise": noise, "reason": "missing_input"}
                    )
                    continue
                # Preselection candidate IDs should agree across strategy pairs.
                paired_cohorts = set()
                for anchor, pairwise in spec["strategy_pairs"]:
                    found = [
                        r
                        for r in noise_runs
                        if matches(
                            r,
                            anchor_selection_strategy=anchor,
                            pairwise_selection_strategy=pairwise,
                        )
                    ]
                    if found:
                        paired_cohorts.add(tuple(sorted({instance_key(r) for r in found})))
                if len(paired_cohorts) > 1:
                    issues.append(
                        {
                            "figure": "fixed_budget",
                            "noise": noise,
                            "reason": "unpaired_input_window_trial_cohorts; figure_skipped",
                        }
                    )
                    continue
                candidates_by_instance = defaultdict(list)
                for r in noise_runs:
                    candidates_by_instance[instance_key(r)].append(
                        {n: r["nodes"][n]["candidates"] for n in nodes}
                    )
                equal = all(
                    all(c == values[0] for c in values)
                    for values in candidates_by_instance.values()
                )
                if not equal:
                    issues.append(
                        {
                            "figure": "fixed_budget",
                            "noise": noise,
                            "reason": "preselection_candidate_pool_differs",
                        }
                    )
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
                rows, coverage, labels, zero = [], [], [], False
                for pos, (anchor, pairwise) in enumerate(spec["strategy_pairs"], 1):
                    found = [
                        r
                        for r in noise_runs
                        if matches(
                            r,
                            anchor_selection_strategy=anchor,
                            pairwise_selection_strategy=pairwise,
                        )
                    ]
                    samples = [
                        phase_sample(
                            r, nodes, spec["pose_stage"], options["require_actual_budget"]
                        )
                        for r in found
                    ]
                    logical = reduce_repeats(samples)
                    good = [r for r in logical if r["usable_error"]]
                    coverage.extend({"strategy": f"{anchor}/{pairwise}", **r} for r in samples)
                    if not found:
                        coverage.append({"strategy": anchor, "reason": "missing_configuration"})
                    summary = summarize(samples, require_time=False)
                    box = len(good) >= options["minimum_boxplot_samples"]
                    labels.append(
                        f"{anchor}\n{pairwise}\nn={len(good)}/{len(logical)}; success={summary['phase_successes']}/{len(logical)}\nbudget={summary['actual_budget_successes']}/{len(logical)}"
                    )
                    rows.extend(
                        {
                            "strategy": f"{anchor}/{pairwise}",
                            "display": "box_with_points" if box else "scatter_only",
                            **r,
                        }
                        for r in logical
                    )
                    if not logical:
                        rows.append({"strategy": f"{anchor}/{pairwise}", **summary})
                    for axis, metric in zip(axes, ("translation_m", "rotation_deg")):
                        values = [r[metric] for r in good]
                        zero |= any(v == 0 for v in values)
                        shown = [
                            max(v, _display_floor(config, metric)) if spec["y_scale"] == "log" else v for v in values
                        ]
                        if box:
                            axis.boxplot([shown], positions=[pos], widths=0.42, showfliers=False)
                        jitter = (
                            np.linspace(-0.09, 0.09, len(shown))
                            if len(shown) > 1
                            else np.zeros(len(shown))
                        )
                        axis.scatter(pos + jitter, shown, color=COLORS[pos - 1], s=35, zorder=5)
                for axis, label in zip(
                    axes, ("Final translation error (m)", "Final rotation error (deg)")
                ):
                    axis.set_xticks(range(1, len(labels) + 1), labels)
                    axis.set_ylabel(label)
                    axis.set_yscale(spec["y_scale"])
                    axis.set_xlim(0.5, len(labels) + 0.5)
                fig.suptitle(
                    f"F4 · {noise} · KA={spec['anchor_budget']}, KP={spec['pairwise_budget']} · {', '.join(nodes)}"
                )
                caption = f"One sample = one input/window/selection trial; boxes require n≥{options['minimum_boxplot_samples']}.\nRepeated deterministic selections are not independent noise realizations. Coverage reports failures/budget shortfalls."
                if "r5" in config:
                    caption = (f"Final GNBP; {'equal-node RMSE' if len(nodes) > 1 else 'per-node error'} over {', '.join(nodes)}. "
                               + caption)
                if zero:
                    caption += " Zero errors use the configured display floor."
                fig.text(0.5, -0.03, caption, ha="center", fontsize=8)
                fig.tight_layout()
                name = f"F4_fixed_budget_{slug(noise)}_g{group[0]['plot_export']['graph_id'][:8]}_{slug('+'.join(nodes))}_KA{spec['anchor_budget']}_KP{spec['pairwise_budget']}_{group_id}"
                save_figure(
                    fig,
                    name,
                    rows,
                    coverage,
                    config,
                    noise_runs,
                    {"preselection_candidate_pools_equal": equal,
                     "zero_render_floor": {m: _display_floor(config, m) for m in ("translation_m", "rotation_deg")},
                     "node_ids": nodes, "aggregation": "Equal-node RMSE" if len(nodes) > 1 else "Per-node error",
                     "pose_stage": "final"},
                )
                plt.close(fig)
