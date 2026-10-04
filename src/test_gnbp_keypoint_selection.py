"""Synthetic, dataset-independent tests of the second-stage selection contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.bgPnp_GABP import (
    BgpnpGabpSolver,
    LoadedObservationBatch,
    MiluvObservationReader,
    MultiAgentWindowSample,
)
from src.gnbp_keypoint_selection import (
    GNBPKeypointSelectionConfig,
    _schur,
    select_gnbp_keypoints,
)
from src.multi_agent_bearing_gabp import (
    GaBPRefinementResult,
    GaBPSettings,
    PoseState,
    linearize_anchor_factor,
    linearize_pairwise_factor,
    refine_with_gabp,
    so3_exp,
)


def synthetic_input() -> tuple[
    LoadedObservationBatch, MultiAgentWindowSample, dict[str, PoseState]
]:
    rng = np.random.default_rng(19)
    n = 16
    states = {
        "B": PoseState(so3_exp(np.array([.2, -.1, .05])), np.array([-.4, .2, -.1])),
        "C": PoseState(so3_exp(np.array([-.1, .05, .12])), np.array([-.9, -.3, .15])),
    }
    local = {node: rng.normal(size=(3, n)) * .5 for node in states}
    anchor = rng.normal(size=(3, n)) + np.array([[4.], [2.], [1.]])
    rotations = {
        node: np.array([so3_exp(np.array([.3 * np.sin(k), .2, .1])).T for k in range(n)])
        for node in states
    }
    b3, b4 = {}, {}
    for observer, pose in states.items():
        for target in ("A", *states):
            if target == observer:
                continue
            world = anchor if target == "A" else states[target].rotation.T @ (
                local[target] - states[target].translation[:, None]
            )
            relative = pose.rotation @ world + pose.translation[:, None] - local[observer]
            bearing = relative / np.linalg.norm(relative, axis=0)
            edge = f"{observer}_{target}"
            b3[edge] = bearing
            b4[edge] = np.einsum("nji,jn->in", rotations[observer], bearing)
    batch = LoadedObservationBatch(
        experiment_dir=Path("."), timestamps=np.arange(n, dtype=float),
        window_sizes=np.array([n]), sigma_azimuth_deg=.1, sigma_elevation_deg=.2,
        anchor_nodes=("A",), estimated_nodes=("B", "C"), robot_ids=("A", "B", "C"),
        adjacency_matrix=np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]]),
        anchor_position_w={"A": anchor}, estimated_position_l=local,
        estimated_position_w={}, estimated_se3={}, bearing_observation=b3,
        bearing_observation_b4=b4, observer_rotation_b3_b4=rotations,
    )
    sample = MultiAgentWindowSample(
        0, n, np.arange(n), {"B_A": np.array([0, 3, 6, 9]), "C_A": np.array([1, 4, 7, 10])},
        ("A",), ("B", "C"), {},
    )
    return batch, sample, states


class TestGNBPKeypointSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.batch, self.sample, self.states = synthetic_input()

    def test_zero_budget_retains_exact_anchor_sets_and_noise(self) -> None:
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states,
            config=GNBPKeypointSelectionConfig(extra_factors_per_node=0),
        )
        self.assertEqual(len(result.anchor_factors), 8)
        self.assertEqual(result.pairwise_factors, [])
        for edge, values in self.sample.anchor_edge_keypoint_indices.items():
            np.testing.assert_array_equal(values, result.refinement_indices[edge])
        for factor in result.anchor_factors:
            self.assertAlmostEqual(factor.sigma_azimuth, np.deg2rad(.1))
            self.assertAlmostEqual(factor.sigma_elevation, np.deg2rad(.2))
            info = linearize_anchor_factor(factor, self.states)
            self.assertLess(info.residual_norm, 1e-10)
        self.assertEqual(result.diagnostics["nodes"]["B"]["initial"]["rank"], 6)

    def test_exact_budget_indices_and_no_mutation(self) -> None:
        before = deepcopy((self.batch, self.sample, self.states))
        cfg = GNBPKeypointSelectionConfig(extra_factors_per_node=3)
        result = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        repeated = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        self.assertEqual(result.selection_order, repeated.selection_order)
        self.assertEqual(len(result.anchor_factors) + len(result.pairwise_factors), 14)
        for node in self.states:
            self.assertEqual(len(result.selection_order[node]), 3)
            np.testing.assert_array_equal(self.states[node].rotation, before[2][node].rotation)
            np.testing.assert_array_equal(
                self.states[node].translation, before[2][node].translation
            )
            trace = result.diagnostics["nodes"][node]
            self.assertEqual(trace["stop_reason"], "budget_exhausted")
            values = [trace["initial"]["lambda_min"]] + [x["lambda_min"] for x in trace["history"]]
            self.assertTrue(np.all(np.diff(values) >= -1e-7))
        for edge, points in self.batch.bearing_observation.items():
            np.testing.assert_array_equal(points, before[0].bearing_observation[edge])
        for edge, points in self.sample.anchor_edge_keypoint_indices.items():
            np.testing.assert_array_equal(points, before[1].anchor_edge_keypoint_indices[edge])
            self.assertTrue(set(points) <= set(result.refinement_indices[edge]))

    def test_pairwise_candidates_outside_union_and_local_marginal(self) -> None:
        # Only one previously unselected time is available beyond retained anchors.
        union = np.unique(np.concatenate(list(self.sample.anchor_edge_keypoint_indices.values())))
        self.sample.window_indices = np.r_[union, 15]
        # Candidate anchor observations are invalid, leaving only owned pairwise candidates.
        for edge, retained in self.sample.anchor_edge_keypoint_indices.items():
            missing = np.setdiff1d(self.sample.window_indices, retained)
            self.batch.bearing_observation_b4[edge][:, missing] = np.nan
        cfg = GNBPKeypointSelectionConfig(extra_factors_per_node=20)
        result = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        self.assertIn(15, result.refinement_indices["B_C"])
        self.assertIn(15, result.refinement_indices["C_B"])
        self.assertEqual(len(result.anchor_factors), 8)
        # Reference: aggregate all B-owned selected factors first, then eliminate C.
        base = {node: np.zeros((6, 6)) for node in self.states}
        for factor in result.anchor_factors:
            base[factor.observer_id] += linearize_anchor_factor(factor, self.states).information
        joint = np.block([[base["B"], np.zeros((6, 6))], [np.zeros((6, 6)), base["C"]]])
        for factor in result.pairwise_factors:
            if factor.observer_id == "B":
                info = linearize_pairwise_factor(factor, self.states)
                joint += np.block([[info.omega_ii, info.omega_ij], [info.omega_ji, info.omega_jj]])
        expected = np.linalg.inv(np.linalg.inv(joint)[:6, :6])
        actual = result.diagnostics["nodes"]["B"]["scaled_marginal_information"]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-6)

    def test_node_selection_does_not_use_neighbors_extra_measurements(self) -> None:
        cfg = GNBPKeypointSelectionConfig(extra_factors_per_node=3)
        result = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        # Remove C's candidate measurements but keep C's original base unchanged.
        self.batch.bearing_observation_b4["C_B"][:] = np.nan
        unused = np.setdiff1d(
            self.sample.window_indices,
            self.sample.anchor_edge_keypoint_indices["C_A"],
        )
        self.batch.bearing_observation_b4["C_A"][:, unused] = np.nan
        other = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        self.assertEqual(result.selection_order["B"], other.selection_order["B"])

    def test_edge_budget_and_time_separation(self) -> None:
        cfg = GNBPKeypointSelectionConfig(
            extra_factors_per_node=8, max_extra_factors_per_edge=1, minimum_index_separation=3
        )
        result = select_gnbp_keypoints(self.batch, self.sample, self.states, config=cfg)
        for node, order in result.selection_order.items():
            self.assertLessEqual(len(order), 2)
            self.assertEqual(len({key.edge_key for key in order}), len(order))
        for edge, indices in result.refinement_indices.items():
            self.assertTrue(np.all(np.diff(indices) >= 3), edge)

    def test_threshold_stops_before_first_candidate(self) -> None:
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states,
            config=GNBPKeypointSelectionConfig(
                stop_mode="information_threshold", information_threshold=1e-12
            ),
        )
        self.assertTrue(all(not keys for keys in result.selection_order.values()))
        self.assertEqual(result.diagnostics["nodes"]["B"]["stop_reason"], "information_threshold")

    def test_explicit_legacy_frame_and_invalid_inputs(self) -> None:
        self.batch.bearing_observation_b4 = None
        with self.assertRaisesRegex(ValueError, "B4 selection requires"):
            select_gnbp_keypoints(self.batch, self.sample, self.states)
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states, measurement_mode="b3_diagonal_approx"
        )
        self.assertEqual(result.diagnostics["measurement_mode"], "b3_diagonal_approx")
        self.sample.anchor_edge_keypoint_indices["B_A"] = np.array([0, 0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_gnbp_keypoints(
                self.batch, self.sample, self.states,
                measurement_mode="b3_diagonal_approx",
            )

    def test_invalid_retained_observation_fails_instead_of_dropping_it(self) -> None:
        self.batch.bearing_observation_b4["B_A"][:, 0] = 0
        with self.assertRaisesRegex(ValueError, "Invalid retained observation"):
            select_gnbp_keypoints(self.batch, self.sample, self.states)

    def test_rank_deficient_start_still_selects_points(self) -> None:
        self.sample.anchor_edge_keypoint_indices = {"B_A": np.array([0]), "C_A": np.array([1])}
        # Restrict to anchors to isolate rank-first scoring from singular neighbors.
        for edge in ("B_C", "C_B"):
            del self.batch.bearing_observation[edge]
            del self.batch.bearing_observation_b4[edge]
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states,
            config=GNBPKeypointSelectionConfig(extra_factors_per_node=4),
        )
        for node in self.states:
            trace = result.diagnostics["nodes"][node]
            self.assertEqual(trace["initial"]["rank"], 2)
            self.assertEqual(trace["history"][0]["mode"], "rank")
            self.assertEqual(trace["final"]["rank"], 6)

    def test_selected_factors_work_with_existing_refiner_without_prior(self) -> None:
        initial = {node: PoseState(so3_exp(np.array([.002, -.003, .001])) @ pose.rotation,
                                  pose.translation + np.array([.005, -.003, .002]))
                   for node, pose in self.states.items()}
        result = select_gnbp_keypoints(
            self.batch, self.sample, initial,
            config=GNBPKeypointSelectionConfig(extra_factors_per_node=2),
        )
        refined = refine_with_gabp(
            initial, result.anchor_factors, result.pairwise_factors,
            settings=GaBPSettings(max_gn_iterations=8, max_bp_iterations=80),
        )
        self.assertLess(refined.history[-1].residual_norm, refined.history[0].residual_norm)
        for node in initial:
            self.assertLess(
                np.linalg.norm(
                    refined.states[node].translation - self.states[node].translation
                ),
                np.linalg.norm(initial[node].translation - self.states[node].translation),
            )

    def test_solver_integration_uses_selected_factors_and_metadata(self) -> None:
        returns = [((pose.rotation, pose.translation, None), 0.0) for pose in self.states.values()]
        with patch("src.bgPnp_GABP.bgpnp.solve", side_effect=returns), patch(
            "src.bgPnp_GABP.refine_with_gabp", return_value=GaBPRefinementResult(self.states)
        ) as refine, patch(
            "src.bgPnp_GABP.get_pairwise_observation_by_union",
            side_effect=AssertionError,
        ):
            _, output = BgpnpGabpSolver().solve(
                self.batch, self.sample, "A",
                selection_config=GNBPKeypointSelectionConfig(extra_factors_per_node=2),
            )
        self.assertIsNone(refine.call_args.kwargs["prior_covariance"])
        kwargs = refine.call_args.kwargs
        self.assertEqual(len(kwargs["anchor_factors"]) + len(kwargs["pairwise_factors"]), 12)
        self.assertEqual(output.selection_metadata["diagnostics"]["method"], "local_schur")
        with self.assertRaisesRegex(ValueError, "prior_covariance=None"):
            BgpnpGabpSolver().solve(self.batch, self.sample, "A", np.eye(6),
                                   selection_config=GNBPKeypointSelectionConfig())

    def test_schur_scalar_example_and_singular_neighbor(self) -> None:
        cfg = GNBPKeypointSelectionConfig()
        one, _ = _schur(np.array([[14.]]), {"C": np.array([[2.]])}, {"C": np.array([[8.]])}, cfg)
        two, _ = _schur(np.array([[18.]]), {"C": np.array([[4.]])}, {"C": np.array([[11.]])}, cfg)
        self.assertAlmostEqual(one.item(), 13.5)
        self.assertAlmostEqual(two.item(), 16.545454545454547)
        marginal, used = _schur(np.eye(2), {"C": np.diag([1., 0.])}, {"C": np.diag([2., 0.])}, cfg)
        np.testing.assert_allclose(marginal, np.diag([.5, 1.]))
        self.assertTrue(used)
        with self.assertRaisesRegex(ValueError, "outside"):
            _schur(np.eye(2), {"C": np.eye(2)}, {"C": np.diag([2., 0.])}, cfg)

    def test_config_validation(self) -> None:
        for kwargs in ({"extra_factors_per_node": -1}, {"extra_factors_per_node": 1.5},
                       {"translation_scale": 0}, {"rank_atol": np.nan},
                       {"stop_mode": "information_threshold"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GNBPKeypointSelectionConfig(**kwargs)

    def test_state_scaling_matches_dense_joint_reference(self) -> None:
        cfg = GNBPKeypointSelectionConfig(
            extra_factors_per_node=3, translation_scale=5.0
        )
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states, config=cfg
        )
        zero = select_gnbp_keypoints(
            self.batch, self.sample, self.states,
            config=GNBPKeypointSelectionConfig(extra_factors_per_node=0),
        )
        joint = np.zeros((12, 12))
        # B's local model retains C's base, not C's newly selected factors.
        for factor in zero.anchor_factors:
            section = slice(0, 6) if factor.observer_id == "B" else slice(6, 12)
            joint[section, section] += linearize_anchor_factor(
                factor, self.states
            ).information
        for key in result.selection_order["B"]:
            if key.target_id == "A":
                offset = list(result.refinement_indices["B_A"]).index(key.time_index)
                factor = [f for f in result.anchor_factors if f.observer_id == "B"][offset]
                joint[:6, :6] += linearize_anchor_factor(factor, self.states).information
            else:
                offset = list(result.refinement_indices["B_C"]).index(key.time_index)
                factor = [f for f in result.pairwise_factors if f.observer_id == "B"][offset]
                info = linearize_pairwise_factor(factor, self.states)
                joint += np.block([
                    [info.omega_ii, info.omega_ij],
                    [info.omega_ji, info.omega_jj],
                ])
        scale = np.diag([1., 1., 1., 5., 5., 5.])
        marginal = joint[:6, :6] - joint[:6, 6:] @ np.linalg.solve(
            joint[6:, 6:], joint[6:, :6]
        )
        np.testing.assert_allclose(
            result.diagnostics["nodes"]["B"]["scaled_marginal_information"],
            scale.T @ marginal @ scale, rtol=1e-8, atol=1e-6,
        )

    def test_singular_base_with_pairwise_candidates_is_supported(self) -> None:
        self.sample.anchor_edge_keypoint_indices = {
            "B_A": np.array([0]), "C_A": np.array([1])
        }
        result = select_gnbp_keypoints(
            self.batch, self.sample, self.states,
            config=GNBPKeypointSelectionConfig(extra_factors_per_node=3),
        )
        self.assertEqual(len(result.selection_order["B"]), 3)
        self.assertEqual(result.diagnostics["nodes"]["B"]["final"]["rank"], 6)

    def test_pole_candidate_is_reported_and_bad_window_is_rejected(self) -> None:
        self.batch.bearing_observation_b4["B_A"][:, 15] = [0., 0., 1.]
        result = select_gnbp_keypoints(self.batch, self.sample, self.states)
        rejected = result.diagnostics["rejected_candidates"]
        self.assertTrue(any(x["edge"] == "B_A" and x["index"] == 15 for x in rejected))
        self.assertNotIn(15, result.refinement_indices["B_A"])
        self.sample.window_indices = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "outside the candidate window"):
            select_gnbp_keypoints(self.batch, self.sample, self.states)

    def test_reader_preserves_b4_fields_and_legacy_compatibility(self) -> None:
        obs = {"anchor_position_w": self.batch.anchor_position_w,
               "estimated_position_l": self.batch.estimated_position_l,
               "bearing_observation": self.batch.bearing_observation,
               "bearing_observation_b4": self.batch.bearing_observation_b4,
               "observer_rotation_b3_b4": self.batch.observer_rotation_b3_b4}
        arrays = {name: getattr(self.batch, name) for name in (
            "timestamps", "window_sizes", "sigma_azimuth_deg", "sigma_elevation_deg",
            "anchor_nodes", "estimated_nodes", "robot_ids", "adjacency_matrix")}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "batch.npz"
            for legacy in (False, True):
                if legacy:
                    obs.pop("bearing_observation_b4")
                    obs.pop("observer_rotation_b3_b4")
                reference = {"estimated_position_w": {}, "estimated_se3": {}}
                np.savez(
                    path, experiment_dir=".", observation_data=np.array(obs, dtype=object),
                    reference_data=np.array(reference, dtype=object), **arrays,
                )
                loaded = MiluvObservationReader.load(path)
                if legacy:
                    self.assertIsNone(loaded.bearing_observation_b4)
                else:
                    np.testing.assert_allclose(
                        loaded.bearing_observation_b4["B_C"],
                        obs["bearing_observation_b4"]["B_C"], rtol=1e-14, atol=1e-14,
                    )


if __name__ == "__main__":
    unittest.main()
