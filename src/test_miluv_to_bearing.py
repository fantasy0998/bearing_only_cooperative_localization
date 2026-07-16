from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.miluv_to_bearing import synthesize_three_robot_bearing_observations


MILUV_EXPERIMENT_DIR = Path(
    r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0"
)


@unittest.skipUnless(MILUV_EXPERIMENT_DIR.exists(), "MILUV experiment directory not available.")
class TestMiluvToBearing(unittest.TestCase):
    def test_synthesize_three_robot_bearings(self) -> None:
        dataset = synthesize_three_robot_bearing_observations(
            experiment_dir=MILUV_EXPERIMENT_DIR,
            dt=0.05,
            max_samples=25,
            reference_index=0,
            random_seed=7,
        )

        self.assertEqual(dataset["anchor_id"], "ifo001")
        self.assertEqual(set(dataset["robot_ids"]), {"ifo001", "ifo002", "ifo003"})
        self.assertEqual(len(dataset["observations"]), 4)
        self.assertGreaterEqual(dataset["timestamps"].size, 10)
        self.assertEqual(dataset["reference_index"], 0)

        for edge_key, observation in dataset["observations"].items():
            with self.subTest(edge=edge_key):
                bearing = observation["bearing"]
                bearing_noiseless = observation["bearing_noiseless"]
                bearing_b3 = observation["bearing_b3"]
                bearing_b3_noiseless = observation["bearing_b3_noiseless"]
                bearing_b4 = observation["bearing_b4"]
                bearing_b4_noiseless = observation["bearing_b4_noiseless"]
                angles = observation["angles"]
                angles_b4 = observation["angles_b4"]
                rotation_b3_b4 = observation["rotation_b3_b4"]

                self.assertEqual(bearing.shape[0], 3)
                self.assertEqual(bearing_noiseless.shape[0], 3)
                self.assertEqual(bearing_b3.shape[0], 3)
                self.assertEqual(bearing_b3_noiseless.shape[0], 3)
                self.assertEqual(bearing_b4.shape[0], 3)
                self.assertEqual(bearing_b4_noiseless.shape[0], 3)
                self.assertEqual(angles.shape[0], 2)
                self.assertEqual(angles_b4.shape[0], 2)
                self.assertEqual(bearing_b4.shape[1], dataset["timestamps"].size)
                self.assertEqual(rotation_b3_b4.shape[0], dataset["timestamps"].size)

                final_norm = np.linalg.norm(bearing, axis=0)
                final_noiseless_norm = np.linalg.norm(bearing_noiseless, axis=0)
                b3_norm = np.linalg.norm(bearing_b3, axis=0)
                b3_noiseless_norm = np.linalg.norm(bearing_b3_noiseless, axis=0)
                noisy_norm = np.linalg.norm(bearing_b4, axis=0)
                noiseless_norm = np.linalg.norm(bearing_b4_noiseless, axis=0)
                self.assertTrue(np.allclose(final_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(final_noiseless_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(b3_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(b3_noiseless_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(noisy_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(noiseless_norm, 1.0, atol=1e-6))
                self.assertTrue(np.allclose(bearing, bearing_b3, atol=1e-10))
                self.assertTrue(np.allclose(bearing_noiseless, bearing_b3_noiseless, atol=1e-10))
                self.assertTrue(
                    np.allclose(bearing_b3[:, 0], bearing_b4[:, 0], atol=1e-10)
                )
                self.assertTrue(
                    np.allclose(bearing_b3_noiseless[:, 0], bearing_b4_noiseless[:, 0], atol=1e-10)
                )
                self.assertTrue(
                    np.allclose(rotation_b3_b4[0], np.eye(3), atol=1e-10)
                )


if __name__ == "__main__":
    unittest.main()
