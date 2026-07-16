from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.exp_miluv_bearing import MiluvBearingExperimentConfig, exp_syenthesis_bearing, build_observation_matrix


MILUV_EXPERIMENT_DIR = Path(
    r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0"
)

 
@unittest.skipUnless(MILUV_EXPERIMENT_DIR.exists(), "MILUV experiment directory not available.")
class TestExpMiluvBearing(unittest.TestCase):
    def test_first_stage_harness_runs_and_outputs_files(self) -> None:
        output_root = Path("data/miluv_bearing_experiment/test_default_3_random_0")
        adjmatrix, order_node = build_observation_matrix(["ifo001"], ["ifo002", "ifo003"])
        config = MiluvBearingExperimentConfig(
            experiment_dir=MILUV_EXPERIMENT_DIR,
            output_root=output_root,
            adjacency_matrix=adjmatrix,
            anchor_node=("ifo001",),
            estimated_node=("ifo002", "ifo003"),
            window_sizes=(20,),
            sigma_azimuth_deg=(0.1,),
            sigma_elevation_deg=(0.5,),
            window_stride=10,
            keypoint_counts=(6, 12, 18),
            dt=0.05,
            random_seed=7,
            force_recompute=True,
            max_samples=800,  # 原始数据中一共有8100历元，时间间隔0.01s左右，生成的观测量需要对齐时间戳，按照dt采样时间生成时间序列，设置采样40s数据，即800个点
            max_windows=1,
            start_point=2000,
            end_point=6000,
        )

        artifacts = exp_syenthesis_bearing(config)

        self.assertEqual(len(artifacts["observation_paths"]), 1)
        observation_path = artifacts["observation_paths"][0]
        self.assertTrue(observation_path.exists())
        self.assertTrue(Path(artifacts["results_path"]).exists())

        loaded_results = np.load(artifacts["results_path"], allow_pickle=True)["results"].tolist()
        self.assertEqual(len(loaded_results), 1)
        batch_group = loaded_results[0]
        self.assertGreaterEqual(len(batch_group), 2)
        metadata = batch_group[0]
        self.assertEqual(metadata["window_size"], 20)

        figure_names = {Path(path).name for path in artifacts["figure_paths"]}
        self.assertIn("miluv_v1_keypoint_selection.png", figure_names)

        window_entries = batch_group[1:]
        self.assertEqual(len(window_entries), 3)
        for entry in window_entries:
            with self.subTest(keypoint_count=entry["keypoint_count"]):
                self.assertIn("bls", entry["errors"])
                self.assertIn("bsdp", entry["errors"])
                self.assertIn("bo_epnp", entry["errors"])
                self.assertIn("bls", entry["times"])
                self.assertIn("bsdp", entry["times"])
                self.assertIn("bo_epnp", entry["times"])

        for figure_path in artifacts["figure_paths"]:
            self.assertTrue(Path(figure_path).exists())


if __name__ == "__main__":
    unittest.main()
