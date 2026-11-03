"""Matched first-stage-only comparison on existing NPZs; writes tables only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import boepnp_gnbp_nodes as frontend
from src.bo_epnp_solver import BO_EPnP
from src.bearing_only_solver import bgpnp
from src.exp_node_gnbp import load_npz_inputs
from src.gnbp_keypoint_selection import GNBPKeypointSelectionConfig
from src.multi_agent_bearing_gabp import so3_log
from src.node_gnbp_backend import UAVNode
from src.structured_tls import StructuredTLSConfig, information_metrics

LOG = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    """Write arrays as copied JSON-compatible values."""
    def convert(v: object) -> object:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, default=convert, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save a rectangular numerical table without overwriting a prior run."""
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pose_errors(rotation: np.ndarray, translation: np.ndarray, truth: dict) -> tuple[float, float]:
    """Geodesic rotation degrees and absolute translation metres."""
    return (float(np.rad2deg(np.linalg.norm(so3_log(rotation @ truth["rotation_local_world"].T)))),
            float(np.linalg.norm(translation - truth["translation_local"])))


def bearing_errors(window: frontend.WindowInput, keys: list, pose: object) -> np.ndarray:
    """Angular residuals on normalized measured rays in the constraint frame."""
    world, local, measured = frontend._anchor_arrays(window, keys)
    predicted = (pose.rotation @ world.T).T + pose.translation - local
    predicted /= np.linalg.norm(predicted, axis=1)[:, None]
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted, measured), axis=1),
                                np.einsum("ij,ij->i", predicted, measured)))


def summarize(rows: list[dict]) -> list[dict]:
    """Report successful-sample RMSE and failures/convergence separately."""
    summary = []
    for noise, method in dict.fromkeys((r["noise"], r["method"]) for r in rows):
        group = [r for r in rows if (r["noise"], r["method"]) == (noise, method)]
        good = [r for r in group if r["success"]]
        result = dict(noise=noise, sigma_azimuth_deg=group[0]["sigma_azimuth_deg"],
            sigma_elevation_deg=group[0]["sigma_elevation_deg"], method=method,
            total=len(group), successful=len(good), failed=len(group)-len(good),
            stls_converged=sum(r["converged"] is True for r in group),
            stls_attempted=sum(r["converged"] is not None for r in group),
            gross_error_count=sum(r["rotation_deg"] > 10. or r["translation_m"] > 1. for r in good))
        for field in ("rotation_deg", "translation_m", "selected_bearing_rmse_deg",
                      "window_bearing_rmse_deg", "initial_rotation_deg", "initial_translation_m"):
            values = np.array([r[field] for r in good if r[field] is not None])
            result[field+"_rmse"] = float(np.sqrt(np.mean(values**2))) if len(values) else None
            if field in ("rotation_deg", "translation_m"):
                result[field+"_median"] = float(np.median(values)) if len(values) else None
                result[field+"_iqr"] = float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else None
        result["mean_selection_ms"] = float(np.mean([r["selection_ms"] for r in good])) if good else None
        result["mean_solver_ms"] = float(np.mean([r["solver_ms"] for r in good])) if good else None
        summary.append(result)
    return summary


def run(plan_path: Path) -> Path:
    """Run only segment_node and initialize_node with backend calls guarded."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    values = json.loads((ROOT / plan["base_config"]).read_text(encoding="utf-8"))
    if "schur" in values:
        values["schur"] = GNBPKeypointSelectionConfig(**values["schur"])
    cfg = frontend.NodeGNBPConfig(**values)
    methods_path = ROOT / plan["methods_config"]
    specs = json.loads(methods_path.read_text(encoding="utf-8"))
    methods = [replace(cfg, **spec, record_frontend_diagnostics=True) for spec in specs]
    if any(c.frontend_use_ml_refine is not False for c in methods):
        raise ValueError("Matched first-stage comparison requires explicit ML off for every method")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / plan["output_root"] / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output / "execution.log", encoding="utf-8"), logging.StreamHandler()])
    paths = list((ROOT / "src").glob("*.py")) + [Path(__file__), plan_path,
        ROOT / plan["base_config"], methods_path] + [ROOT / p for p in plan["observations"]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(output / "resolved_config.json", dict(plan=plan, methods=[asdict(c) for c in methods],
        python=sys.executable, numpy=np.__version__, input_and_source_hashes=hashes))
    rows, history = [], []
    with ExitStack() as stack:
        for name in ("select_pairwise", "refine_nodes", "solve_window"):
            stack.enter_context(patch.object(frontend, name, side_effect=AssertionError("Stage two forbidden")))
        stack.enter_context(patch.object(BO_EPnP, "_ml_refinement_russell",
                                         side_effect=AssertionError("ML forbidden in matched comparison")))
        for observation in plan["observations"]:
            path = ROOT / observation
            arguments, truth, metadata = load_npz_inputs(path)
            for start in plan["window_starts"]:
                stop = start + plan["window_size"]
                if stop > metadata["sample_count"]:
                    raise ValueError("Window exceeds available samples")
                window = frontend.WindowInput(**arguments, absolute_indices=np.arange(start, stop),
                    window_id=f"{metadata['sha256'][:12]}:{start}:{stop}")
                common = {}
                for method in methods:
                    label = method.anchor_selection_strategy + "+" + method.bo_epnp_solver
                    LOG.info("%s start=%d %s", path.stem, start, label)
                    for name in window.estimated_ids:
                        node = UAVNode(name, window.estimated_position_local[name])
                        frontend.segment_node(node, (window, method))
                        # Cache one common geometry/reference across all five methods.
                        pool = node.candidates["anchor"]
                        if name not in common:
                            world, local, bearings = frontend._anchor_arrays(window, pool)
                            controls = BO_EPnP._select_control_points(world, "pca")
                            geometry = frontend.structured_anchor_geometry(window, pool, controls)
                            a, b = BO_EPnP._build_sphere_aware_linear_system(local, bearings,
                                                                          bgpnp.compute_alphas(world, controls))
                            reference = BO_EPnP._solve_wtls(a, b)
                            terms = geometry.linearize(reference, StructuredTLSConfig(**method.structured_tls), allow_invalid=True)
                            common[name] = (pool, terms["information"])
                        else:
                            assert pool == common[name][0], "Methods changed candidate extraction"
                        frontend.initialize_node(node, (window, method))
                        selection = node.diagnostics["anchor_selection"]
                        solver = node.diagnostics.get("frontend_solver", {})
                        stls = solver.get("structured_tls", {})
                        chosen = node.selected.get("anchor", [])
                        row = dict(dataset=path.stem, noise=path.stem.split("_seed")[0],
                            seed=int(path.stem.split("_seed")[-1]), window_start=start,
                            window_size=plan["window_size"], node=name, method=label,
                            sigma_azimuth_deg=metadata["sigma_azimuth_deg"],
                            sigma_elevation_deg=metadata["sigma_elevation_deg"],
                            candidate_count=len(pool), selected_count=len(chosen),
                            success=node.state is not None, failure_reason=node.diagnostics["initialization"].get("reason", ""),
                            rotation_deg=None, translation_m=None, initial_rotation_deg=None,
                            initial_translation_m=None, selected_bearing_rmse_deg=None,
                            window_bearing_rmse_deg=None, converged=stls.get("converged"),
                            stop_reason=stls.get("status", "closed_form"), iterations=stls.get("iterations", 0),
                            initial_cost=stls.get("initial_cost"), final_cost=stls.get("final_cost"),
                            common_information_logdet=None, common_information_min_eigenvalue=None,
                            selection_ms=1000*node.diagnostics.get("anchor_selection_seconds", 0.),
                            solver_ms=1000*node.diagnostics.get("frontend_seconds", 0.))
                        if node.state is not None:
                            row["rotation_deg"], row["translation_m"] = pose_errors(node.state.rotation, node.state.translation, truth[name])
                            row["selected_bearing_rmse_deg"] = float(np.sqrt(np.mean(bearing_errors(window, chosen, node.state)**2)))
                            row["window_bearing_rmse_deg"] = float(np.sqrt(np.mean(bearing_errors(window, pool, node.state)**2)))
                            if stls:
                                ri, ti = BO_EPnP._recover_pose(np.asarray(selection["control_points"]),
                                    np.asarray(stls["initial_control_points"]).reshape(4, 3))
                                row["initial_rotation_deg"], row["initial_translation_m"] = pose_errors(ri, ti, truth[name])
                            indices = [pool.index(k) for k in chosen]
                            h = common[name][1][indices].sum(axis=0)
                            metrics = information_metrics(h, method.anchor_rank_rtol, method.anchor_logdet_regularization)
                            row["common_information_logdet"] = metrics["logdet"]
                            row["common_information_min_eigenvalue"] = metrics["minimum_eigenvalue"]
                        for step in selection.get("history", []):
                            history.append(dict(dataset=path.stem, node=name, method=label,
                                id=str(step.get("id")), rank=step["rank"], logdet=step["logdet"],
                                minimum_eigenvalue=step.get("minimum_eigenvalue"),
                                maximum_eigenvalue=step.get("maximum_eigenvalue"), condition_number=step.get("condition_number")))
                        write_json(output / "records" / f"{path.stem}_{start}_{name}_{label}.json",
                            dict(row=row, config=asdict(method), diagnostics=node.diagnostics,
                                 pose=frontend.pose_dict(node.state), truth=truth[name]))
                        rows.append(row)
    summary = summarize(rows)
    write_csv(output / "node_metrics.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "selection_history.csv", history)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    write_json(output / "audit.json", dict(source_and_inputs_unchanged=unchanged,
        backend_calls=0, ml_calls=0, rows=len(rows), success=sum(r["success"] for r in rows)))
    if not unchanged:
        raise RuntimeError("Source or observation hash changed during execution")
    lines = ["|噪声 σaz/σel (°)|方法|有限输出/总数|旋转RMSE (°)|平移RMSE (m)|全窗口bearing RMSE (°)|STLS收敛|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        fmt = lambda x: "—" if x is None else f"{x:.6g}"
        lines.append(f"|{r['noise']} {r['sigma_azimuth_deg']}/{r['sigma_elevation_deg']}|{r['method']}|"
            f"{r['successful']}/{r['total']}|{fmt(r['rotation_deg_rmse'])}|{fmt(r['translation_m_rmse'])}|"
            f"{fmt(r['window_bearing_rmse_deg_rmse'])}|"
            + (f"{r['stls_converged']}/{r['stls_attempted']}|" if r['stls_attempted'] else "—|"))
    (output / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    LOG.info("OUTPUT %s", output)
    return output


def main() -> None:
    """CLI using the same method override JSON schema as the generic runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/structured_tls/experiment.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
