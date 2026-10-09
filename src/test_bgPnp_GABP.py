from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.bgPnp_GABP import (
    get_anchor_edge_observation,
    LoadedObservationBatch,
    MiluvSolveConfig,
    UniformKeypointSelector,
    WindowSampleBuilder,
    exp_miluv_bo_epnp_gnbp,
)
from src.exp_miluv_bearing import (
    MiluvBearingExperimentConfig,
    build_observation_matrix,
    generate_miluv_observation_batches,
)


MILUV_EXPERIMENT_DIR = Path(
    r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0"
)


def _noise_tag(value: float) -> str:
    """Match the production filename tag for one noise value."""
    return f"{value:.3f}".replace(".", "")


def _build_synthetic_batch(num_points: int = 24) -> LoadedObservationBatch:
    timestamps = np.arange(num_points, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    anchor = np.vstack(
        [
            3.0 * np.cos(theta),
            2.0 * np.sin(theta),
            1.0 + 0.5 * np.sin(2.0 * theta),
        ]
    )
    local = np.vstack(
        [
            np.linspace(-1.0, 1.0, num_points),
            0.4 * np.sin(theta),
            0.2 * np.cos(theta),
        ]
    )
    delta = anchor - local
    bearing = delta / np.linalg.norm(delta, axis=0, keepdims=True)

    return LoadedObservationBatch(
        experiment_dir=Path("."),
        timestamps=timestamps,
        window_sizes=np.array([num_points], dtype=int),
        sigma_azimuth_deg=0.1,
        sigma_elevation_deg=0.2,
        anchor_nodes=("ifo001",),
        estimated_nodes=("ifo002",),
        robot_ids=("ifo001", "ifo002"),
        adjacency_matrix=np.array([[0, 1], [0, 0]], dtype=int),
        anchor_position_w={"ifo001": anchor},
        estimated_position_l={"ifo002": local},
        estimated_position_w={"ifo002": anchor + 0.2},
        estimated_se3={
            "ifo002": {
                "rotation_local_world": np.eye(3),
                "translation_local": np.zeros(3),
                "rotation_wb_start": np.eye(3),
                "position_w_start": np.zeros(3),
            }
        },
        bearing_observation={"ifo002_ifo001": bearing},
    )


class TestKeypointSelector(unittest.TestCase):
    def test_observability_keypoint_methods_build_samples(self) -> None:
        batch = _build_synthetic_batch()
        for method in (
            "uniform",
            "bearing_diversity",
            "trajectory_geometry",
            "linear_system_condition",
        ):
            selector = UniformKeypointSelector(
                method=method,
                observer_id="ifo002",
                anchor_id="ifo001",
            )
            sample = WindowSampleBuilder(selector).build(batch, keypoint_count=8)
            edge_key = "ifo002_ifo001"
            self.assertIn(edge_key, sample.anchor_edge_keypoint_indices)

            edge_indices = sample.anchor_edge_keypoint_indices[edge_key]
            self.assertEqual(edge_indices.size, 8)
            self.assertTrue(np.all(np.diff(edge_indices) >= 0))

            self.assertEqual(sample.keypoint_indices.size, 8)
            self.assertTrue(np.all(np.diff(sample.keypoint_indices) >= 0))

            anchor_points, observer_points, bearings = get_anchor_edge_observation(
                batch,
                sample,
                observer_id="ifo002",
                anchor_id="ifo001",
            )
            self.assertEqual(anchor_points.shape, (3, 8))
            self.assertEqual(observer_points.shape, (3, 8))
            self.assertEqual(bearings.shape, (3, 8))

    def test_build_can_override_selector_method(self) -> None:
        batch = _build_synthetic_batch()
        selector = UniformKeypointSelector(
            method="uniform",
            observer_id="ifo002",
            anchor_id="ifo001",
        )
        sample = WindowSampleBuilder(selector).build(
            batch,
            keypoint_count=7,
            keypoint_selection_method="bearing_diversity",
        )
        edge_key = "ifo002_ifo001"
        self.assertIn(edge_key, sample.anchor_edge_keypoint_indices)
        self.assertEqual(sample.anchor_edge_keypoint_indices[edge_key].size, 7)
        self.assertEqual(sample.keypoint_indices.size, 7)
        self.assertTrue(np.all(np.diff(sample.keypoint_indices) >= 0))


@unittest.skipUnless(MILUV_EXPERIMENT_DIR.exists(), "MILUV experiment directory not available.")
class TestBgPnpGABP(unittest.TestCase):
    def test_observation_to_bgpnp_gabp_flow_runs(self) -> None:
        observation_root = Path("data/miluv_bgpnp_gabp_test")
        adjacency_matrix, _node_order = build_observation_matrix(
            ["ifo001"],
            ["ifo002", "ifo003"],
        )
        generation_config = MiluvBearingExperimentConfig(
            experiment_dir=MILUV_EXPERIMENT_DIR,
            output_root=observation_root,
            adjacency_matrix=tuple(tuple(int(v) for v in row) for row in adjacency_matrix),
            anchor_node=("ifo001",),
            estimated_node=("ifo002", "ifo003"),
            window_sizes=(20,),
            sigma_azimuth_deg=(0.1, 0.2),
            sigma_elevation_deg=(0.2, 0.3),
            window_stride=10,
            keypoint_counts=(6, 10),
            dt=0.05,
            random_seed=7,
            force_recompute=True,
            max_samples=800,
            start_point=2000,
            end_point=6000,
        )

        observation_paths = generate_miluv_observation_batches(generation_config)
        self.assertEqual(len(observation_paths), 2)
        for observation_path in observation_paths:
            self.assertTrue(observation_path.exists())

        for observation_path in observation_paths:
            solve_config = MiluvSolveConfig(
                observation_path=observation_path,
                output_root=observation_root,
                figure_dir=Path("docs/figures"),
                keypoint_counts=(6, 12, 18),
                primary_anchor_id="ifo001",
                target_agent_id="ifo002",   ## 对于多机情况，为便于对比，选择ifo02无人机的位姿作为对比项
                
                # uniform
                # trajectory_geometry
                # linear_system_condition
                # bearing_diversity
                keypoint_selection_method="uniform",
                run_bls=True,
                run_bsdp=True,
                run_bo_epnp=True,
                run_bgpnp=True,
                run_bgpnp_gabp=True,
            )

            artifacts = exp_miluv_bo_epnp_gnbp(solve_config)

            self.assertTrue(Path(artifacts["results_path"]).exists())
            for figure_path in artifacts["figure_paths"]:
                self.assertTrue(Path(figure_path).exists())

            results = artifacts["results"]
            self.assertGreaterEqual(len(results), 2)
            metadata = results[0]
            self.assertEqual(metadata["target_agent_id"], "ifo002")
            self.assertEqual(metadata["estimated_nodes"], ["ifo002", "ifo003"])

            expected_algorithms = set()
            if solve_config.run_bls:
                expected_algorithms.add("bls")
            if solve_config.run_bsdp:
                expected_algorithms.add("bsdp")
            if solve_config.run_bo_epnp:
                expected_algorithms.add("bo_epnp")
            if solve_config.run_bgpnp:
                expected_algorithms.add("bgpnp")
            if solve_config.run_bgpnp_gabp:
                expected_algorithms.add("bgpnp_gabp")
            for entry in results[1:]:
                self.assertEqual(set(entry["errors"].keys()), expected_algorithms)
                self.assertEqual(set(entry["times"].keys()), expected_algorithms)
                self.assertEqual(set(entry["failures"].keys()), expected_algorithms)

            az_tag = _noise_tag(float(metadata["sigma_azimuth_deg"]))
            el_tag = _noise_tag(float(metadata["sigma_elevation_deg"]))
            figure_names = {Path(path).name for path in artifacts["figure_paths"]}
            self.assertIn(f"bgpnp_gabp_keypoints_3d_az{az_tag}_el{el_tag}.png", figure_names)
            self.assertIn(f"bgpnp_gabp_rot_error_ifo002_az{az_tag}_el{el_tag}.png", figure_names)
            self.assertIn(f"bgpnp_gabp_tra_error_ifo002_az{az_tag}_el{el_tag}.png", figure_names)


if __name__ == "__main__":
    unittest.main()
