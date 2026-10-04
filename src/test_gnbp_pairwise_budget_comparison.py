"""Regression tests for the fixed-data pairwise-budget experiment.

Run the actual experiment explicitly with python -m src.exp_gnbp_pairwise_budget.
Discovery uses synthetic fixtures and never regenerates MILUV observations.
"""

from copy import deepcopy
from dataclasses import replace
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.bgPnp_GABP import MiluvSolveConfig, MiluvSolveEvaluator, SolverEstimate, WindowSampleBuilder
from src.exp_gnbp_pairwise_budget import (
    PairwiseBudgetExperimentConfig, build_initialization_sample, count_factors,
    save_records, validate_records, _record, _refinement_record,
)
from src.gnbp_keypoint_selection import (
    GNBPKeypointSelectionConfig, build_gnbp_candidate_factors, select_gnbp_keypoints,
)
from src.multi_agent_bearing_gabp import (
    GaBPSettings, PoseState, linearize_anchor_factor, linearize_pairwise_factor,
    refine_with_gabp,
)
from src.test_gnbp_keypoint_selection import synthetic_input


class TestPairwiseBudgetComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.batch, self.sample, self.states = synthetic_input()
        self.cfg = GNBPKeypointSelectionConfig(
            candidate_policy="pairwise_only", candidate_scope="anchor_union",
            max_total_factors_per_node=7,
        )

    def test_total_budget_overrides_extra_and_preserves_anchors(self) -> None:
        result = select_gnbp_keypoints(self.batch, self.sample, self.states,
            config=replace(self.cfg, extra_factors_per_node=0))
        for edge, retained in self.sample.anchor_edge_keypoint_indices.items():
            np.testing.assert_array_equal(result.refinement_indices[edge], retained)
        for node, order in result.selection_order.items():
            self.assertEqual(len(order), 3)
            self.assertTrue(all(key.target_id in self.states for key in order))
            self.assertEqual(result.diagnostics["nodes"][node]["total_owned_factor_count"], 7)
        self.assertEqual(len(result.anchor_factors), 8)
        self.assertEqual(len(result.pairwise_factors), 6)

    def test_zero_and_infeasible_and_exhausted_budget(self) -> None:
        zero = select_gnbp_keypoints(self.batch, self.sample, self.states,
            config=replace(self.cfg, max_total_factors_per_node=4))
        self.assertEqual(len(zero.pairwise_factors), 0)
        with self.assertRaisesRegex(ValueError, "below retained"):
            select_gnbp_keypoints(self.batch, self.sample, self.states,
                config=replace(self.cfg, max_total_factors_per_node=3))
        full = select_gnbp_keypoints(self.batch, self.sample, self.states,
            config=replace(self.cfg, max_total_factors_per_node=100))
        self.assertTrue(all(d["stop_reason"] == "no_eligible_candidates"
                            for d in full.diagnostics["nodes"].values()))
        self.assertEqual(len(full.pairwise_factors), 16)

    def test_candidate_scope_and_matched_factor_equality(self) -> None:
        all_factors = build_gnbp_candidate_factors(self.batch, self.sample, self.states, config=self.cfg)
        selected = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        union = set(self.sample.keypoint_indices)
        for edge, indices in all_factors.refinement_indices.items():
            self.assertTrue(set(indices) <= union)
            self.assertTrue(set(selected.refinement_indices.get(edge, [])) <= set(indices))
        def keyed(result):
            lookup = {}
            for edge, indices in result.refinement_indices.items():
                observer, target = edge.split("_")
                factors = ([f for f in result.anchor_factors if f.observer_id == observer]
                           if target == "A" else [f for f in result.pairwise_factors
                            if f.observer_id == observer and f.target_id == target])
                lookup.update({(edge, int(i)): f for i, f in zip(indices, factors)})
            return lookup
        reference = keyed(all_factors)
        for key, factor in keyed(selected).items():
            other = reference[key]
            np.testing.assert_array_equal(factor.measured_angles, other.measured_angles)
            np.testing.assert_array_equal(factor.observer_extrinsic_rotation, other.observer_extrinsic_rotation)
            self.assertEqual(factor.sigma_azimuth, other.sigma_azimuth)
            self.assertEqual(factor.sigma_elevation, other.sigma_elevation)
        full = build_gnbp_candidate_factors(self.batch, self.sample, self.states,
            config=replace(self.cfg, candidate_scope="full_window"))
        self.assertEqual(len(full.pairwise_factors), 32)
        self.assertEqual(len(full.anchor_factors), 8)

    def test_invalid_optional_candidates_and_missing_edge(self) -> None:
        self.batch.bearing_observation_b4["B_C"][:, 0] = np.nan
        self.batch.bearing_observation.pop("C_B")
        selected = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        full = build_gnbp_candidate_factors(self.batch, self.sample, self.states, config=self.cfg)
        self.assertEqual(selected.diagnostics["rejected_candidates"], full.diagnostics["rejected_candidates"])
        self.assertNotIn(0, selected.refinement_indices.get("B_C", []))
        self.assertEqual(selected.selection_order["C"], ())

    def test_owned_counts_reverse_edges_and_spacing(self) -> None:
        result = select_gnbp_keypoints(self.batch, self.sample, self.states,
            config=replace(self.cfg, minimum_index_separation=3, max_extra_factors_per_edge=2))
        counts = count_factors(result.anchor_factors, result.pairwise_factors,
            self.sample.estimated_nodes, result.refinement_indices, result.initialization_indices)
        self.assertEqual(counts["graph"]["observation_total"],
                         sum(c["total_owned"] for c in counts["by_observer"].values()))
        self.assertEqual(counts["graph"]["pairwise"], 4)
        for edge in ("B_C", "C_B"):
            self.assertTrue(np.all(np.diff(result.refinement_indices[edge]) >= 3))

    def test_three_estimated_nodes_share_one_outgoing_budget(self) -> None:
        self.states["D"] = PoseState(self.states["C"].rotation.copy(), self.states["C"].translation + .7)
        self.batch.estimated_position_l["D"] = self.batch.estimated_position_l["C"].copy()
        self.batch.observer_rotation_b3_b4["D"] = self.batch.observer_rotation_b3_b4["C"].copy()
        self.batch.estimated_nodes = self.sample.estimated_nodes = ("B", "C", "D")
        self.batch.robot_ids = ("A", "B", "C", "D")
        self.sample.anchor_edge_keypoint_indices["D_A"] = np.array([2, 5, 8, 11])
        for observer, state in self.states.items():
            for target in ("A", *self.states):
                if target == observer:
                    continue
                world = (self.batch.anchor_position_w["A"] if target == "A" else
                    self.states[target].rotation.T @ (self.batch.estimated_position_l[target]
                                                      - self.states[target].translation[:, None]))
                relative = state.rotation @ world + state.translation[:, None] - self.batch.estimated_position_l[observer]
                bearing = relative / np.linalg.norm(relative, axis=0)
                edge = f"{observer}_{target}"
                self.batch.bearing_observation[edge] = bearing
                self.batch.bearing_observation_b4[edge] = np.einsum(
                    "nji,jn->in", self.batch.observer_rotation_b3_b4[observer], bearing)
        result = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        self.assertEqual(len(result.pairwise_factors), 9)
        self.assertTrue(all(len(order) == 3 for order in result.selection_order.values()))

    def test_same_initialization_selection_without_plot_writes(self) -> None:
        with patch("src.bgPnp_GABP.plot_curves"):
            original = WindowSampleBuilder().build(self.batch, 6, "linear_system_condition")
        actual = build_initialization_sample(self.batch, 6)
        for edge in original.anchor_edge_keypoint_indices:
            np.testing.assert_array_equal(original.anchor_edge_keypoint_indices[edge],
                                          actual.anchor_edge_keypoint_indices[edge])

    def test_truth_independence_and_state_no_aliasing(self) -> None:
        before = deepcopy(self.states)
        first = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        self.batch.estimated_se3 = {"B": "must not be read"}
        self.sample.estimated_se3_gt = {"C": "must not be read"}
        second = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        self.assertEqual(first.selection_order, second.selection_order)
        refined = refine_with_gabp(self.states, first.anchor_factors, first.pairwise_factors,
                                    settings=GaBPSettings(max_gn_iterations=1))
        refined.states["B"].translation[0] += 9
        for node in self.states:
            np.testing.assert_array_equal(before[node].translation, self.states[node].translation)

    def test_final_residual_and_stop_reason(self) -> None:
        factors = build_gnbp_candidate_factors(self.batch, self.sample, self.states, config=self.cfg)
        initial = deepcopy(self.states)
        initial["B"].translation += .01
        result = refine_with_gabp(initial, factors.anchor_factors, factors.pairwise_factors,
                                  settings=GaBPSettings(max_gn_iterations=1, gn_tolerance=1e-30))
        expected = np.sqrt(sum(linearize_anchor_factor(f, result.states).residual_norm**2 for f in factors.anchor_factors)
            + sum(linearize_pairwise_factor(f, result.states).residual_norm**2 for f in factors.pairwise_factors))
        self.assertAlmostEqual(result.final_residual_norm, expected)
        self.assertEqual(result.stop_reason, "max_gn_iterations")
        self.assertNotAlmostEqual(result.final_residual_norm, result.history[-1].residual_norm)

    def test_all_node_pose_persistence_roundtrip_and_failure(self) -> None:
        self.batch.estimated_se3 = {node: {"rotation_local_world": s.rotation.copy(),
            "translation_local": s.translation.copy()} for node, s in self.states.items()}
        record = _record(self.batch, self.sample, "synthetic", "pairwise_selected", 7, .1)
        cfg = PairwiseBudgetExperimentConfig(primary_anchor_id="A")
        record = _refinement_record(self.batch, self.sample, self.states, {}, .2, record, cfg)
        self.assertEqual(record["status"], "success")
        self.assertEqual(set(record["poses"]), {"B", "C"})
        with TemporaryDirectory() as tmp:
            save_records(Path(tmp), [record])
            with np.load(Path(tmp) / "results.npz", allow_pickle=True) as archive:
                loaded = archive["results"].tolist()[0]
            for node in self.states:
                np.testing.assert_array_equal(loaded["poses"][node]["initial"]["R_world_to_local"], self.states[node].rotation)
                self.assertLess(loaded["poses"][node]["errors_final"]["translation_abs_m"], 1e-8)
            with (Path(tmp) / "per_agent_metrics.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(int(row["total_owned"]) for row in rows), record["counts"]["graph"]["observation_total"])
        failed = _record(self.batch, self.sample, "synthetic", "pairwise_all", None, .1)
        failed = _refinement_record(self.batch, self.sample, {"B": self.states["B"]},
                                     {"C": "test failure"}, .1, failed, cfg)
        self.assertEqual(failed["status"], "failed")
        self.assertIsNotNone(failed["poses"]["B"]["initial"])
        self.assertTrue(all(p["final"] is None for p in failed["poses"].values()))

    def test_existing_evaluator_persists_all_nodes_and_final_counts(self) -> None:
        factors = select_gnbp_keypoints(self.batch, self.sample, self.states, config=self.cfg)
        result = refine_with_gabp(self.states, factors.anchor_factors, factors.pairwise_factors)
        initial = {node: SolverEstimate(s.rotation.copy(), s.translation.copy(), .01)
                   for node, s in self.states.items()}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "placeholder.npz"
            path.touch()
            evaluator = MiluvSolveEvaluator(MiluvSolveConfig(
                observation_path=path, primary_anchor_id="A", target_agent_id="B",
                gnbp_selection_config=self.cfg,
            ))
            with patch.object(evaluator.gabp_solver, "solve", return_value=(initial, result)):
                estimate = evaluator._solve_bgpnp_gabp(self.batch, self.sample)
        self.assertEqual(set(estimate.diagnostics["poses"]), {"B", "C"})
        self.assertEqual(estimate.diagnostics["factor_counts"]["observation_total"], 14)
        np.testing.assert_array_equal(estimate.diagnostics["poses"]["C"]["initial"]["t_world_to_local"],
                                      self.states["C"].translation)

    def test_saved_comparison_validation_detects_budget_violation(self) -> None:
        self.batch.estimated_se3 = {node: {"rotation_local_world": s.rotation.copy(),
            "translation_local": s.translation.copy()} for node, s in self.states.items()}
        cfg = PairwiseBudgetExperimentConfig(primary_anchor_id="A")
        records = []
        for method, budget in (("pairwise_all", None), ("pairwise_selected", 7)):
            record = _record(self.batch, self.sample, "synthetic", method, budget, .1)
            records.append(_refinement_record(self.batch, self.sample, self.states, {}, .2, record, cfg))
        self.assertEqual(validate_records(records)["paired_selected_records"], 1)
        records[-1]["identity"]["T"] = 4
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            validate_records(records)

    def test_invalid_policy_and_scope(self) -> None:
        for values in ({"candidate_policy": "other"}, {"candidate_scope": "other"},
                       {"max_total_factors_per_node": -1}):
            with self.assertRaises(ValueError):
                GNBPKeypointSelectionConfig(**values)


if __name__ == "__main__":
    unittest.main()
