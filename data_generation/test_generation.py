"""Regression tests for the standalone generator; no solver required by tests."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from data_generation import generate as gen
from data_generation import miluv_bearing_core as core


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root / "experiment"
        for i, node in enumerate(("ifo001", "ifo002", "ifo003", "ifo004")):
            folder = self.experiment / node
            folder.mkdir(parents=True)
            t = np.linspace(0, 4, 81) + i * 0.001
            positions = np.column_stack(
                (i * 2 + np.sin(t + i), np.cos(t * 0.7 + i), 0.2 * t + 0.3 * i)
            )
            rotations = Rotation.from_euler(
                "xyz", np.column_stack((0.1 * np.sin(t), 0.2 * t + i * 0.1, 0.4 * t + i * 0.3))
            )
            data = np.column_stack((t, positions, rotations.as_quat()))
            pd.DataFrame(
                data,
                columns=["timestamp", *core.MOCAP_POSITION_COLUMNS, *core.MOCAP_QUATERNION_COLUMNS],
            ).to_csv(folder / "mocap.csv", index=False)
        (self.experiment / "timeshift.yaml").write_text("timeshift_s: 1\ntimeshift_ns: 500000000\n")
        self.cfg = copy.deepcopy(gen.DEFAULTS)
        self.cfg.update(
            experiment_dir=str(self.experiment),
            output_root=str(self.root / "out"),
            max_samples=40,
            window_sizes=[20],
            reference_index=7,
        )
        self.config_path = self.root / "config.json"
        gen.write_json(self.config_path, self.cfg)

    def dataset(self, seed: int = 7, sigma: float = 0.1) -> dict:
        return core.synthesize_three_robot_bearing_observations(
            self.experiment,
            dt=0.05,
            max_samples=40,
            reference_index=7,
            sigma_azimuth_deg=sigma,
            sigma_elevation_deg=sigma * 2,
            random_seed=seed,
        )

    def test_default_none_indices_and_no_extrapolation(self) -> None:
        frames = core.load_three_robot_mocap(self.experiment)
        t = core.build_common_timestamps(frames, dt=0.073)
        self.assertGreaterEqual(t[0], 0.002)
        self.assertLessEqual(t[-1], 4)
        for frame in frames.values():
            p, q, r = core.interpolate_robot_state(frame, t)
            self.assertEqual(p.shape, (3, len(t)))
            self.assertTrue(np.isfinite(q).all() and np.isfinite(r).all())

    def test_index_bounds_and_auto_dt(self) -> None:
        frames = core.load_three_robot_mocap(self.experiment)
        t = core.build_common_timestamps(frames, start_point=10, end_point=30, max_samples=8)
        self.assertEqual(len(t), 8)
        self.assertAlmostEqual(t[0], 0.5)
        np.testing.assert_allclose(np.diff(t), 0.05, atol=1e-14)

    def test_invalid_time_ranges(self) -> None:
        frames = core.load_three_robot_mocap(self.experiment)
        for kwargs in (
            {"dt": 0},
            {"dt": float("nan")},
            {"max_samples": 1},
            {"start_point": -1},
            {"end_point": 81},
            {"start_point": 40, "end_point": 10},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                core.build_common_timestamps(frames, **kwargs)

    def test_invalid_csv_is_rejected(self) -> None:
        path = self.experiment / "ifo001" / "mocap.csv"
        data = pd.read_csv(path)
        for kind in ("nan", "zero_quaternion", "duplicate_times"):
            bad = data.copy()
            if kind == "nan":
                bad.loc[0, "pose.position.x"] = np.nan
            elif kind == "zero_quaternion":
                bad.loc[0, core.MOCAP_QUATERNION_COLUMNS] = 0
            else:
                bad["timestamp"] = 0
            bad.to_csv(path, index=False)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.load_miluv_mocap_csv(self.experiment, "ifo001")

    def test_nonzero_reference_frames_match_world_geometry(self) -> None:
        data = self.dataset(sigma=0)
        gen.validate_config(self.cfg)
        arrays = gen.build_solver_arrays(data, self.cfg, self.cfg["noise_levels"][0], 7)
        obs = arrays["observation_data"].item()
        truth = arrays["reference_data"].item()["estimated_se3"]
        for node in self.cfg["estimated_nodes"]:
            np.testing.assert_allclose(obs["estimated_position_l"][node][:, 7], 0, atol=1e-14)
            np.testing.assert_allclose(
                obs["observer_rotation_b3_b4"][node][7], np.eye(3), atol=1e-14
            )
            np.testing.assert_allclose(
                data["estimated_pose"][node]["translation_local"], truth[node]["translation_local"]
            )
        for edge in data["observations"].values():
            observer, target = edge["observer_id"], edge["target_id"]
            p = data["states"][target]["position_w"] - data["states"][observer]["position_w"]
            expected = truth[observer]["rotation_local_world"] @ p
            expected /= np.linalg.norm(expected, axis=0)
            np.testing.assert_allclose(edge["bearing_b3"], expected, atol=1e-13)
            actual_b4 = edge["bearing_b4"]
            for k in (0, 7, 39):
                expected_b4 = data["states"][observer]["rotation_wb"][k].T @ p[:, k]
                expected_b4 /= np.linalg.norm(expected_b4)
                np.testing.assert_allclose(actual_b4[:, k], expected_b4, atol=1e-13)

    def test_noise_seed_reproducibility_and_truth_invariance(self) -> None:
        first, repeated, other = self.dataset(), self.dataset(), self.dataset(seed=8)
        for key, edge in first["observations"].items():
            np.testing.assert_array_equal(
                edge["bearing_b4"], repeated["observations"][key]["bearing_b4"]
            )
            np.testing.assert_array_equal(
                edge["bearing_b4_noiseless"], other["observations"][key]["bearing_b4_noiseless"]
            )
            self.assertFalse(
                np.array_equal(edge["bearing_b4"], other["observations"][key]["bearing_b4"])
            )

    def test_noise_statistics_units_and_scale(self) -> None:
        count = 20000
        observer = np.zeros((3, count))
        target = np.repeat(np.array([[1.0], [0.3], [0.2]]), count, axis=1)
        edge = core.synthesize_directed_bearing_edge(
            observer,
            np.repeat(np.eye(3)[None], count, axis=0),
            target,
            sigma_azimuth_deg=0.2,
            sigma_elevation_deg=0.5,
            rng=np.random.default_rng(17),
        )
        edge.update(observer_id="a", target_id="b")
        stats = gen.dataset_statistics(
            {"timestamps": np.arange(count), "states": {}, "observations": {"a<-b": edge}}, [0.5]
        )
        result = stats["edges"]["a<-b"]
        self.assertAlmostEqual(result["azimuth_noise_deg"]["std_ddof0"], 0.2, delta=0.006)
        self.assertAlmostEqual(result["elevation_noise_deg"]["std_ddof0"], 0.5, delta=0.015)
        self.assertLess(result["max_unit_norm_error"], 1e-14)

    def test_cli_overrides_and_config_relative_paths(self) -> None:
        self.cfg["experiment_dir"] = "experiment"
        self.cfg["output_root"] = "relative_output"
        self.cfg.pop("statistics")
        gen.write_json(self.config_path, self.cfg)
        parser = gen.build_argparser()
        cfg = gen.resolve_config(
            parser.parse_args(
                [
                    "--config",
                    str(self.config_path),
                    "--dt",
                    "auto",
                    "--max-samples",
                    "none",
                    "--no-statistics",
                    "--seeds",
                    "10",
                    "11",
                    "--sigma-azimuth-deg",
                    ".1",
                    ".2",
                    "--sigma-elevation-deg",
                    ".3",
                    ".4",
                    "--edges",
                    "ifo002:ifo001",
                ]
            )
        )
        self.assertEqual(cfg["experiment_dir"], str(self.experiment.resolve()))
        self.assertEqual(cfg["output_root"], str((self.root / "relative_output").resolve()))
        self.assertIsNone(cfg["dt"])
        self.assertIsNone(cfg["max_samples"])
        self.assertEqual(len(cfg["noise_levels"]), 2)
        self.assertEqual(cfg["random_seeds"], [10, 11])
        self.assertEqual(cfg["directed_edges"], [["ifo002", "ifo001"]])
        self.assertFalse(cfg["statistics"]["enabled"])
        default = gen.resolve_config(parser.parse_args(["--config", str(self.config_path)]))
        self.assertTrue(default["statistics"]["enabled"])

    def test_invalid_config_and_unpaired_sigmas(self) -> None:
        parser = gen.build_argparser()
        with self.assertRaisesRegex(ValueError, "both sigma"):
            gen.resolve_config(
                parser.parse_args(["--config", str(self.config_path), "--sigma-azimuth-deg", ".1"])
            )
        for override in (
            {"dt": -1},
            {"random_seeds": [1, 1]},
            {"estimated_nodes": ["../bad"]},
            {"directed_edges": [["ifo001", "ifo002"]]},
            {"statistics": {"enabled": True, "quantiles": [2]}},
        ):
            cfg = copy.deepcopy(self.cfg)
            cfg.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                gen.validate_config(cfg)
        self.cfg["typo"] = 1
        gen.write_json(self.config_path, self.cfg)
        with self.assertRaisesRegex(ValueError, "Unknown config"):
            gen.resolve_config(parser.parse_args(["--config", str(self.config_path)]))

    def test_invalid_reference_and_window_before_output(self) -> None:
        for override in ({"reference_index": 40}, {"window_sizes": [41]}):
            cfg = copy.deepcopy(self.cfg)
            cfg.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                gen.generate(cfg)
            self.assertFalse(Path(cfg["output_root"]).exists())

    def test_zero_noise_rejected_for_solver_compatibility(self) -> None:
        self.cfg["noise_levels"][0]["sigma_azimuth_deg"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            gen.generate(self.cfg)
        self.assertFalse(Path(self.cfg["output_root"]).exists())

    def test_failed_generation_is_recorded(self) -> None:
        shutil.copy2(self.experiment / "ifo001/mocap.csv", self.experiment / "ifo002/mocap.csv")
        with self.assertRaisesRegex(ValueError, "coincident"):
            gen.generate(self.cfg)
        manifests = list(Path(self.cfg["output_root"]).glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("coincident", manifest["failure_reason"])

    def test_default_edge_order_preserves_harness_rng_assignment(self) -> None:
        gen.validate_config(self.cfg)
        self.assertEqual(
            self.cfg["directed_edges"],
            [
                ["ifo002", "ifo001"],
                ["ifo003", "ifo001"],
                ["ifo003", "ifo002"],
                ["ifo002", "ifo003"],
            ],
        )

    def test_multi_anchor_sparse_graph_round_trip(self) -> None:
        self.cfg.update(
            anchor_nodes=["ifo001", "ifo004"],
            directed_edges=[["ifo002", "ifo004"], ["ifo003", "ifo002"]],
        )
        output = gen.generate(self.cfg)
        with np.load(output / "observations/noise_1_seed7.npz", allow_pickle=True) as raw:
            obs = raw["observation_data"].item()
            self.assertEqual(set(obs["anchor_position_w"]), {"ifo001", "ifo004"})
            self.assertEqual(set(obs["estimated_position_l"]), {"ifo002", "ifo003"})
            self.assertEqual(set(obs["bearing_observation_b4"]), {"ifo002_ifo004", "ifo003_ifo002"})
            adjacency = np.zeros((4, 4), dtype=int)
            adjacency[1, 2] = adjacency[2, 3] = 1
            np.testing.assert_array_equal(raw["adjacency_matrix"], adjacency)

    def test_batch_product_manifest_and_no_overwrite(self) -> None:
        self.cfg["noise_levels"].append(
            {"noise_id": "higher", "sigma_azimuth_deg": 0.2, "sigma_elevation_deg": 0.4}
        )
        self.cfg["random_seeds"] = [7, 8]
        first = gen.generate(self.cfg)
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["files"]), 4)
        self.assertEqual(len(list(first.glob("observations/*.statistics.json"))), 4)
        self.cfg["statistics"]["enabled"] = False
        second = gen.generate(self.cfg)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        self.assertEqual(len(list(second.glob("observations/*.statistics.json"))), 0)

    def test_standalone_copy_and_dry_run(self) -> None:
        standalone = self.root / "standalone"
        standalone.mkdir()
        for name in ("generate.py", "miluv_bearing_core.py"):
            shutil.copy2(Path(gen.__file__).parent / name, standalone / name)
        gen.write_json(standalone / "config.json", self.cfg)
        command = [sys.executable, str(standalone / "generate.py")]
        for extra in (["--help"], ["--dry-run"], []):
            process = subprocess.run(
                command + extra, cwd=standalone, capture_output=True, text=True
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            if extra:
                self.assertFalse(Path(self.cfg["output_root"]).exists())
        self.assertEqual(len(list(Path(self.cfg["output_root"]).glob("*/observations/*.npz"))), 1)

    def test_copied_noise_kernel_is_unchanged(self) -> None:
        original = Path(__file__).resolve().parents[1] / "src/miluv_to_bearing.py"
        if not original.is_file():
            self.skipTest("Original source is absent from a standalone copy")

        def functions(path: Path) -> dict:
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
                if isinstance(node, ast.FunctionDef)
            }

        source, copied = functions(original), functions(Path(core.__file__))
        for name in (
            "wrap_to_pi",
            "bearing_to_angles",
            "angles_to_bearing",
            "compute_b3_from_b4_rotation_sequence",
            "rotate_bearing_sequence",
            "synthesize_directed_bearing_edge",
        ):
            self.assertEqual(source[name], copied[name], name)


if __name__ == "__main__":
    unittest.main()
