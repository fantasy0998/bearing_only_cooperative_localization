from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.bo_epnp_solver import BO_EPnP
from src.bearing_only_solver import bearing_linear_solver
from src.miluv_to_bearing import (
    compute_b3_from_b4_rotation_sequence,
    rotate_bearing_sequence,
    synthesize_three_robot_bearing_observations,
)
from src.multi_agent_bearing_gabp import so3_log
from src.bgPnp_GABP import plot_key_point

LOGGER = logging.getLogger(__name__)

def build_observation_matrix(anchor_nodes, estimated_nodes):
    """
    构建多机协同定位观测邻接矩阵。

    矩阵定义：
        行：被观测对象 target
        列：观测者 observer
        A[i, j] = 1 表示第 j 个节点可以观测第 i 个节点

    规则：
        1. 锚节点不作为观测者，即锚节点对应的列全为 0
        2. 待估节点可以观测锚节点
        3. 待估节点之间互相观测
        4. 节点自身不观测自身

    Parameters
    ----------
    anchor_nodes : list[str]
        锚节点列表，例如 ['A']
    estimated_nodes : str or list[str]
        待估节点，例如 'BC' 或 ['B', 'C']

    Returns
    -------
    adjacency_matrix : np.ndarray
        观测邻接矩阵
    node_order : list[str]
        矩阵对应的节点顺序
    """

    if isinstance(anchor_nodes, str):
        anchor_nodes = [anchor_nodes]
    if isinstance(estimated_nodes, str):
        estimated_nodes = [estimated_nodes]

    anchor_nodes = [str(node).strip().strip("'\"") for node in anchor_nodes]
    estimated_nodes = [str(node).strip().strip("'\"") for node in estimated_nodes]

    node_order = list(anchor_nodes) + list(estimated_nodes)

    if len(node_order) != len(set(node_order)):
        raise ValueError("节点名称存在重复，请检查 anchor_nodes 和 estimated_nodes")

    n = len(node_order)
    adjacency_matrix = np.zeros((n, n), dtype=int)

    estimated_set = set(estimated_nodes)

    for observer_idx, observer in enumerate(node_order):
        # 只有待估节点可以作为观测者
        if observer not in estimated_set:
            continue

        for target_idx, target in enumerate(node_order):
            # 无人机自身不观测自身
            if observer == target:
                continue
            # 待估节点可以观测锚节点和其他待估节点
            adjacency_matrix[target_idx, observer_idx] = 1

    return adjacency_matrix, node_order

@dataclass(frozen=True)
class MiluvBearingExperimentConfig:
    """Top-level configuration for the MILUV bearing harness."""

    experiment_dir: Path
    output_root: Path = Path("data/miluv_bearing_experiment")
    anchor_node: tuple[str, ...] = ("ifo001",)
    estimated_node: tuple[str, ...] = ("ifo002", "ifo003")
    adjacency_matrix: tuple[tuple[int, ...], ...] = (
    (0, 1, 1),
    (0, 0, 1),
    (0, 1, 0),
) 
    window_sizes: tuple[int, ...] = (20, 30)
    sigma_azimuth_deg: tuple[float, ...] = (0.1, 0.2, 0.3)
    sigma_elevation_deg: tuple[float, ...] = (0.5, 0.6, 0.7)
    window_stride: int = 10
    keypoint_counts: tuple[int, ...] = (12, 18, 24)
    dt: float = 0.05
    random_seed: int = 7
    force_recompute: bool = False
    max_samples: int | None = None
    max_windows: int | None = None
    figure_dir: Path = Path("docs/figures")
    start_point: int =2000
    end_point:   int =6000


    def __post_init__(self) -> None:
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"experiment_dir does not exist: {self.experiment_dir}")
        anchor_node = _normalize_node_tuple(self.anchor_node)
        estimated_node = _normalize_node_tuple(self.estimated_node)
        object.__setattr__(self, "anchor_node", anchor_node)
        object.__setattr__(self, "estimated_node", estimated_node)
        if not self.window_sizes:
            raise ValueError("window_sizes must not be empty.")
        if len(self.sigma_azimuth_deg) != len(self.sigma_elevation_deg):
            raise ValueError("sigma_azimuth_deg and sigma_elevation_deg must have the same length.")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if any(size <= 1 for size in self.window_sizes):
            raise ValueError("All window sizes must be greater than 1.")
        if any(count <= 0 for count in self.keypoint_counts):
            raise ValueError("All keypoint counts must be positive.")
        if len(set(anchor_node + estimated_node)) != len(anchor_node) + len(estimated_node):
            raise ValueError("anchor_node and estimated_node must not contain duplicate node ids.")
        adjacency = np.asarray(self.adjacency_matrix, dtype=int)
        expected_node_count = len(anchor_node) + len(estimated_node)
        if adjacency.shape != (expected_node_count, expected_node_count):
            raise ValueError(
                "adjacency_matrix must have shape "
                f"({expected_node_count}, {expected_node_count}), got {adjacency.shape}."
            )


@dataclass(frozen=True)
class ObservationBatchSpec:
    """One observation-file specification."""

    window_size: int
    sigma_azimuth_deg: float
    sigma_elevation_deg: float

    @property
    def stem(self) -> str:
        return (
            f"miluv_bearing_az{_noise_tag(self.sigma_azimuth_deg)}"
            f"_el{_noise_tag(self.sigma_elevation_deg)}"
        )


@dataclass
class SingleWindowSample:
    """AB-only sample used by the first-stage baselines."""

    start: int
    stop: int
    window_indices: np.ndarray
    keypoint_indices: np.ndarray
    p_A_world: np.ndarray
    p_B_local: np.ndarray
    b_BA: np.ndarray
    R_B_gt: np.ndarray
    t_B_gt: np.ndarray


class MiluvObservationGenerator:
    """Create or reuse MILUV observation batches."""

    def __init__(self, config: MiluvBearingExperimentConfig) -> None:
        self.config = config
        self.observation_dir = config.output_root / "observations"
        self.observation_dir.mkdir(parents=True, exist_ok=True)
# ObservationBatchSpec对象是一个数据结构类，专门存储窗口大小和噪声水平
# 将输入的窗口大小和噪声水平组合成多个ObservationBatchSpec对象，并将它们存储在一个列表中返回
    def specs(self) -> list[ObservationBatchSpec]:
        """Enumerate all observation-batch combinations."""
        specs: list[ObservationBatchSpec] = []
        default_window_size = int(self.config.window_sizes[0])
        for sigma_az, sigma_el in zip(
            self.config.sigma_azimuth_deg, self.config.sigma_elevation_deg
        ):
            specs.append(
                ObservationBatchSpec(
                    window_size=default_window_size,
                    sigma_azimuth_deg=float(sigma_az),
                    sigma_elevation_deg=float(sigma_el),
                )
            )
        return specs

    def observation_path(self, spec: ObservationBatchSpec) -> Path:
        """Return the output path for one observation batch."""
        return self.observation_dir / f"{spec.stem}.npz"

    def generate_or_reuse(self, spec: ObservationBatchSpec) -> Path:
        """Generate one observation file or reuse an existing one."""
        output_path = self.observation_path(spec)
        # 判断噪声和观测窗口对应的文件是否存在 
        if output_path.exists() and not self.config.force_recompute:
            LOGGER.info("Reusing observation batch: %s", output_path)
            return output_path

        robot_ids = self.config.anchor_node + self.config.estimated_node
        directed_edges = _directed_edges_from_adjacency(
            robot_ids,
            np.asarray(self.config.adjacency_matrix, dtype=int),
        )
        dataset = synthesize_three_robot_bearing_observations(
            experiment_dir=self.config.experiment_dir,
            anchor_id=self.config.anchor_node[0],
            robot_ids=robot_ids,
            directed_edges=directed_edges,
            dt=self.config.dt,
            max_samples=self.config.max_samples,
            reference_index=0,
            sigma_azimuth_deg=spec.sigma_azimuth_deg,
            sigma_elevation_deg=spec.sigma_elevation_deg,
            random_seed=self.config.random_seed,
            start_point=self.config.start_point,
            end_point=self.config.end_point,
        )

        # tra={}
        # for nodes in robot_ids:
        #     tra[nodes]=dataset["states"][nodes]["position_w"]
        # output_path=Path('D:/3-1cooperativeLocalization/code/bearing_only_hu/taes-bearing-only-cooperative-localization/docs/figures/tra_3d.png')
        # original_path = Path(r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0")
        # plot_key_point(trajectory=tra,
        #                outputpath=output_path,
        #                original_tra_path=original_path,)

        states = dataset["states"]
        observations = dataset["observations"]
        anchor_position_w: dict[str, np.ndarray] = {}
        estimated_position_l: dict[str, np.ndarray] = {}
        estimated_position_w: dict[str, np.ndarray] = {}
        estimated_se3: dict[str, dict[str, np.ndarray]] = {}

        for node_id in self.config.anchor_node:
            anchor_position_w[node_id] = np.asarray(states[node_id]["position_w"], dtype=float)

        for node_id in self.config.estimated_node:
            position_w = np.asarray(states[node_id]["position_w"], dtype=float)
            rotation_wb = np.asarray(states[node_id]["rotation_wb"], dtype=float)
            position_l, rotation_local_world, translation_local = _build_window_local_points(
                position_w,
                rotation_wb,
                0,
            )
            estimated_position_l[node_id] = position_l
            estimated_position_w[node_id] = position_w
            estimated_se3[node_id] = {
                "rotation_local_world": rotation_local_world,
                "translation_local": translation_local.reshape(3),
                "rotation_wb_start": rotation_wb[0],
                "position_w_start": position_w[:, 0],
            }

        bearing_observation = {
            edge_key.replace("<-", "_"): np.asarray(edge_value["bearing_b3"], dtype=float)
            for edge_key, edge_value in observations.items()
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        observation_data = {
            "anchor_position_w": anchor_position_w,
            "estimated_position_l": estimated_position_l,
            "bearing_observation": bearing_observation,
            "bearing_observation_b4": {
                edge.replace("<-", "_"): np.asarray(value["bearing_b4"], dtype=float)
                for edge, value in observations.items()
            },
            "observer_rotation_b3_b4": {
                edge.split("<-")[0]: np.asarray(value["rotation_b3_b4"], dtype=float)
                for edge, value in observations.items()
            },
            "bearing_reference_index": 0,
            "observation_schema_version": 2,
        }
        reference_data = {
            "estimated_position_w": estimated_position_w,
            "estimated_se3": estimated_se3,
        }

        arrays: dict[str, Any] = {
            "experiment_dir": str(self.config.experiment_dir),
            "window_sizes": np.asarray(self.config.window_sizes, dtype=int),
            "window_size": np.array(spec.window_size, dtype=int),
            "sigma_azimuth_deg": np.array(spec.sigma_azimuth_deg, dtype=float),
            "sigma_elevation_deg": np.array(spec.sigma_elevation_deg, dtype=float),
            "timestamps": dataset["timestamps"],
            "anchor_nodes": np.asarray(self.config.anchor_node, dtype=object),
            "estimated_nodes": np.asarray(self.config.estimated_node, dtype=object),
            "robot_ids": np.asarray(robot_ids, dtype=object),
            "observation_data": np.array(observation_data, dtype=object),
            "reference_data": np.array(reference_data, dtype=object),
        }

        np.savez_compressed(
            output_path,
            **arrays,
        )
        LOGGER.info("Generated observation batch: %s", output_path)
        return output_path

    def generate_all(self) -> list[Path]:
        """Generate or reuse all requested batches."""
        # 在这里循环生成所有边的结果
        return [self.generate_or_reuse(spec) for spec in self.specs()]


class UniformKeypointSelector:
    """Uniformly distribute keypoints across one window."""

    def select(self, window_indices: np.ndarray, keypoint_count: int) -> np.ndarray:
        """Select approximately uniform keypoints."""
        indices = np.asarray(window_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            raise ValueError("window_indices must not be empty.")
        if keypoint_count <= 0:
            raise ValueError("keypoint_count must be positive.")
        if keypoint_count >= indices.size:
            return indices.copy()

        linspace_positions = np.linspace(0, indices.size - 1, keypoint_count)
        selected = indices[np.round(linspace_positions).astype(int)]
        return np.unique(selected)


class BaselineEvaluator:
    """Run first-stage AB-only baselines."""

    def __init__(self, keypoint_selector: UniformKeypointSelector | None = None) -> None:
        self.keypoint_selector = keypoint_selector or UniformKeypointSelector()

    def evaluate_observation_batch(
        self,
        observation_path: Path,
        config: MiluvBearingExperimentConfig,
    ) -> list[dict[str, Any]]:
        """Evaluate one saved observation batch."""
        data = self._load_observation_batch(observation_path)
        metadata = {
            "observation_file": observation_path.name,
            "window_size": int(data["window_size"]),
            "sigma_azimuth_deg": float(data["sigma_azimuth_deg"]),
            "sigma_elevation_deg": float(data["sigma_elevation_deg"]),
            "keypoint_counts": list(config.keypoint_counts),
            "experiment_dir": str(data["experiment_dir"]),
        }

        results_for_batch: list[dict[str, Any]] = [metadata]
        window_size = int(data["window_size"])
        starts = list(range(0, data["timestamps"].size - window_size + 1, config.window_stride))
        if config.max_windows is not None:
            starts = starts[: config.max_windows]

        for start in starts:
            for keypoint_count in config.keypoint_counts:
                sample = self._build_ab_sample(data, start, window_size, keypoint_count)
                results_for_batch.append(self._evaluate_sample(sample))

        return results_for_batch

    @staticmethod
    def _load_observation_batch(observation_path: Path) -> dict[str, Any]:
        data = np.load(observation_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def _build_ab_sample(
        self,
        data: dict[str, Any],
        start: int,
        window_size: int,
        keypoint_count: int,
    ) -> SingleWindowSample:
        stop = start + window_size
        window_indices = np.arange(start, stop, dtype=int)
        keypoint_indices = self.keypoint_selector.select(window_indices, keypoint_count)

        anchor_position_w = np.asarray(data["anchor_position_w"], dtype=float)
        b_position_w = np.asarray(data["B_position_w"], dtype=float)
        b_rotation_wb = np.asarray(data["B_rotation_wb"], dtype=float)
        b_from_a_bearing_b4 = np.asarray(data["B_from_A_bearing_b4"], dtype=float)

        p_B_local_full, R_B_gt, t_B_gt = _build_window_local_points(
            b_position_w, b_rotation_wb, start
        )
        b_BA_full = _build_window_b3_bearing(
            b_from_a_bearing_b4, b_rotation_wb, start, stop
        )

        return SingleWindowSample(
            start=start,
            stop=stop,
            window_indices=window_indices,
            keypoint_indices=keypoint_indices,
            p_A_world=anchor_position_w[:, keypoint_indices],
            p_B_local=p_B_local_full[:, keypoint_indices],
            b_BA=b_BA_full[:, keypoint_indices - start],
            R_B_gt=R_B_gt,
            t_B_gt=t_B_gt,
        )

    def _evaluate_sample(self, sample: SingleWindowSample) -> dict[str, Any]:
        result = {
            "start": sample.start,
            "stop": sample.stop,
            "keypoint_count": int(sample.keypoint_indices.size),
            "keypoint_indices": sample.keypoint_indices.copy(),
            "errors": {},
            "times": {},
            "failures": {"bls": 0, "bsdp": 0, "bo_epnp": 0},
        }

        for solver_name, solver_fn in (
            ("bls", self._solve_bls),
            ("bsdp", self._solve_bsdp),
            ("bo_epnp", self._solve_bo_epnp),
        ):
            try:
                estimate = solver_fn(sample)
                result["times"][solver_name] = estimate["runtime"]
                result["errors"][solver_name] = _pose_error(
                    estimate["rotation"],
                    estimate["translation"],
                    sample.R_B_gt,
                    sample.t_B_gt,
                )
            except Exception as exc:
                LOGGER.warning(
                    "%s failed for window [%d, %d): %s",
                    solver_name,
                    sample.start,
                    sample.stop,
                    exc,
                )
                result["failures"][solver_name] = 1
                result["times"][solver_name] = np.nan
                result["errors"][solver_name] = {
                    "rot": np.nan,
                    "rot_log": np.nan,
                    "tra": np.nan,
                }

        return result

    @staticmethod
    def _solve_bls(sample: SingleWindowSample) -> dict[str, Any]:
        (rotation, translation), runtime = bearing_linear_solver.solve(
            sample.p_A_world,
            sample.p_B_local,
            sample.b_BA,
        )
        return {
            "rotation": np.asarray(rotation, dtype=float),
            "translation": np.asarray(translation, dtype=float).reshape(3),
            "runtime": float(runtime),
        }

    @staticmethod
    def _solve_bsdp(sample: SingleWindowSample) -> dict[str, Any]:
        (rotation, translation), runtime = bearing_linear_solver.solve_with_sdp_sdr(
            sample.p_A_world,
            sample.p_B_local,
            sample.b_BA,
        )
        return {
            "rotation": np.asarray(rotation, dtype=float),
            "translation": np.asarray(translation, dtype=float).reshape(3),
            "runtime": float(runtime),
        }

    @staticmethod
    def _solve_bo_epnp(sample: SingleWindowSample) -> dict[str, Any]:
        start_time = time.perf_counter()
        rotation, translation = BO_EPnP.solve(
            sample.p_A_world,
            sample.p_B_local,
            sample.b_BA,
            use_wtls=True,
            use_ml_refine=True,
            control_point_method="pca",
        )
        return {
            "rotation": np.asarray(rotation, dtype=float),
            "translation": np.asarray(translation, dtype=float).reshape(3),
            "runtime": float(time.perf_counter() - start_time),
        }


class ResultRepository:
    """Save results and create summary plots."""

    def __init__(self, config: MiluvBearingExperimentConfig) -> None:
        self.config = config
        self.results_dir = config.output_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = config.figure_dir
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    @property
    def results_path(self) -> Path:
        return self.results_dir / "results.npz"

    def save_results(self, results: list[list[dict[str, Any]]]) -> Path:
        """Save grouped results to NPZ."""
        np.savez_compressed(self.results_path, results=np.array(results, dtype=object))
        return self.results_path

    def load_results(self) -> list[list[dict[str, Any]]]:
        """Load grouped results from NPZ."""
        loaded = np.load(self.results_path, allow_pickle=True)["results"]
        return loaded.tolist()

    def plot_summary(self, results: list[list[dict[str, Any]]]) -> list[Path]:
        """Create summary plots for the current results."""
        plot_specs = [
            ("rot_log", "Mean SO(3) Log Rotation Error", "Rotation Error [rad]", "miluv_v1_mean_rot_log_error.png"),
            ("tra", "Mean Normalized Translation Error", "Translation Error", "miluv_v1_mean_translation_error.png"),
            ("runtime", "Mean Runtime", "Runtime [s]", "miluv_v1_runtime.png"),
        ]
        output_paths: list[Path] = []
        for metric_key, title, ylabel, filename in plot_specs:
            output_paths.append(
                self._plot_metric(results, metric_key, title, ylabel, self.figure_dir / filename)
            )
        output_paths.append(
            self._plot_keypoint_selection(
                results,
                self.figure_dir / "miluv_v1_keypoint_selection.png",
            )
        )
        return output_paths

    def _plot_metric(
        self,
        grouped_results: list[list[dict[str, Any]]],
        metric_key: str,
        title: str,
        ylabel: str,
        output_path: Path,
    ) -> Path:
        fig, axes = plt.subplots(
            nrows=len(grouped_results),
            ncols=1,
            figsize=(7.0, max(4.0, 4.0 * len(grouped_results))),
            squeeze=False,
        )

        for row_idx, batch_results in enumerate(grouped_results):
            ax = axes[row_idx, 0]
            metadata = batch_results[0]
            entries = batch_results[1:]
            aggregation = _aggregate_metric(entries, metric_key)
            for solver_name, by_keypoint in aggregation.items():
                x_values = sorted(by_keypoint)
                y_values = [np.nanmean(by_keypoint[key]) for key in x_values]
                ax.plot(x_values, y_values, marker="o", linewidth=1.6, label=solver_name)

            ax.set_title(
                f"batch={metadata['window_size']} az={metadata['sigma_azimuth_deg']:.3f} "
                f"el={metadata['sigma_elevation_deg']:.3f}"
            )
            ax.set_xlabel("Keypoint Count")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def _plot_keypoint_selection(
        self,
        grouped_results: list[list[dict[str, Any]]],
        output_path: Path,
    ) -> Path:
        """Plot the selected keypoints along the observer trajectory."""
        keypoint_counts = tuple(int(count) for count in self.config.keypoint_counts)
        fig, axes = plt.subplots(
            nrows=len(grouped_results),
            ncols=len(keypoint_counts),
            figsize=(5.2 * len(keypoint_counts), 4.8 * len(grouped_results)),
            squeeze=False,
        )

        for row_idx, batch_results in enumerate(grouped_results):
            metadata = batch_results[0]
            entries = batch_results[1:]
            observation_path = self.config.output_root / "observations" / metadata["observation_file"]
            batch_data = np.load(observation_path, allow_pickle=True)
            b_position_w = np.asarray(batch_data["B_position_w"], dtype=float)
            batch_label = (
                f"batch={metadata['window_size']} "
                f"az={metadata['sigma_azimuth_deg']:.3f} "
                f"el={metadata['sigma_elevation_deg']:.3f}"
            )

            for col_idx, keypoint_count in enumerate(keypoint_counts):
                ax = axes[row_idx, col_idx]
                entry = self._first_entry_for_keypoint(entries, keypoint_count)
                start = int(entry["start"])
                stop = int(entry["stop"])
                keypoint_indices = np.asarray(entry["keypoint_indices"], dtype=int)
                trajectory = b_position_w[:, start:stop]
                keypoint_positions = b_position_w[:, keypoint_indices]

                ax.plot(
                    trajectory[0],
                    trajectory[1],
                    color="tab:blue",
                    linewidth=2.0,
                    label="B trajectory",
                )
                ax.scatter(
                    keypoint_positions[0],
                    keypoint_positions[1],
                    s=70,
                    color="red",
                    edgecolors="darkred",
                    linewidths=1.6,
                    zorder=3,
                    label="Selected keypoints",
                )
                ax.scatter(
                    trajectory[0, 0],
                    trajectory[1, 0],
                    s=55,
                    color="green",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=4,
                    label="Window start" if col_idx == 0 else None,
                )
                ax.scatter(
                    trajectory[0, -1],
                    trajectory[1, -1],
                    s=55,
                    color="orange",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=4,
                    label="Window end" if col_idx == 0 else None,
                )
                ax.set_title(f"{batch_label}\nkeypoints={keypoint_count}, window=[{start}, {stop})")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.axis("equal")
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="best")

        fig.suptitle("Selected Keypoints on the B-UAV Trajectory")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return output_path

    @staticmethod
    def _first_entry_for_keypoint(
        entries: list[dict[str, Any]],
        keypoint_count: int,
    ) -> dict[str, Any]:
        """Return the earliest window entry for one keypoint count."""
        matching_entries = [
            entry for entry in entries if int(entry["keypoint_count"]) == int(keypoint_count)
        ]
        if not matching_entries:
            raise ValueError(f"No result entry found for keypoint_count={keypoint_count}")
        return min(matching_entries, key=lambda entry: int(entry["start"]))


def generate_miluv_observation_batches(
    config: MiluvBearingExperimentConfig,
) -> list[Path]:
    """Generate or reuse all MILUV observation batches."""
    generator = MiluvObservationGenerator(config)
    return generator.generate_all()


def _normalize_node_tuple(nodes: Iterable[str] | str) -> tuple[str, ...]:
    """Normalize node ids to a clean tuple."""
    if isinstance(nodes, str):
        nodes = (nodes,)
    return tuple(str(node).strip().strip("'\"") for node in nodes)


def _noise_tag(value: float) -> str:
    """Convert a noise standard deviation to a compact filename token."""
    compact = f"{float(value):g}"
    return compact.replace(".", "")


def _directed_edges_from_adjacency(
    node_order: Iterable[str],
    adjacency_matrix: np.ndarray,
) -> tuple[tuple[str, str], ...]:
    """Convert an observation adjacency matrix to directed (observer, target) edges."""
    ordered_nodes = tuple(node_order)
    adjacency = np.asarray(adjacency_matrix, dtype=int)
    if adjacency.shape != (len(ordered_nodes), len(ordered_nodes)):
        raise ValueError(
            "adjacency_matrix shape does not match node_order, got "
            f"{adjacency.shape} for {len(ordered_nodes)} nodes."
        )

    directed_edges: list[tuple[str, str]] = []
    for target_idx, target_id in enumerate(ordered_nodes):
        for observer_idx, observer_id in enumerate(ordered_nodes):
            if target_id == observer_id:
                continue
            if int(adjacency[target_idx, observer_idx]) != 0:
                directed_edges.append((observer_id, target_id))
    return tuple(directed_edges)


def evaluate_miluv_observation_batches(
    observation_paths: list[Path],
    config: MiluvBearingExperimentConfig,
) -> tuple[list[list[dict[str, Any]]], Path, list[Path]]:
    """Evaluate all saved observation batches."""
    evaluator = BaselineEvaluator()
    repository = ResultRepository(config)
    grouped_results = [
        evaluator.evaluate_observation_batch(observation_path, config)
        for observation_path in observation_paths
    ]
    results_path = repository.save_results(grouped_results)
    figure_paths = repository.plot_summary(grouped_results)
    return grouped_results, results_path, figure_paths


def exp_syenthesis_bearing(
    config: MiluvBearingExperimentConfig,
) -> dict[str, Any]:
    """Run the first-stage MILUV bearing experiment harness."""
    observation_paths = generate_miluv_observation_batches(config)
    grouped_results, results_path, figure_paths = evaluate_miluv_observation_batches(
        observation_paths, config
    )
    return {
        "observation_paths": observation_paths,
        "results": grouped_results,
        "results_path": results_path,
        "figure_paths": figure_paths,
    }


def _build_window_local_points(
    position_w: np.ndarray,
    rotation_wb: np.ndarray,
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build local trajectory points and the ground-truth world-to-local pose."""
    pos = np.asarray(position_w, dtype=float)
    rot = np.asarray(rotation_wb, dtype=float)
    if pos.ndim != 2 or pos.shape[0] != 3:
        raise ValueError(f"position_w must have shape (3, N), got {pos.shape}")
    if rot.ndim != 3 or rot.shape[1:] != (3, 3) or rot.shape[0] != pos.shape[1]:
        raise ValueError(
            "rotation_wb must have shape (N, 3, 3) matching position_w, got "
            f"{rot.shape}"
        )
    if not 0 <= start < pos.shape[1]:
        raise ValueError(f"start index {start} is out of bounds for {pos.shape[1]} points")

    rotation_local_world = rot[start].T
    translation_local = -rotation_local_world @ pos[:, start]
    local_points = rotation_local_world @ pos + translation_local.reshape(3, 1)
    return local_points, rotation_local_world, translation_local


def _build_window_b3_bearing(
    bearing_b4: np.ndarray,
    rotation_wb: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Convert a B4 bearing sequence into the window-start frozen B3 frame."""
    bearing_slice = np.asarray(bearing_b4, dtype=float)[:, start:stop]
    rotation_slice = np.asarray(rotation_wb, dtype=float)[start:stop]
    rotation_b3_b4 = compute_b3_from_b4_rotation_sequence(rotation_slice, reference_index=0)
    return rotate_bearing_sequence(bearing_slice, rotation_b3_b4)


def _pose_error(
    rotation_est: np.ndarray,
    translation_est: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
) -> dict[str, float]:
    """Compute first-stage pose errors."""
    translation_gt_norm = max(float(np.linalg.norm(translation_gt)), 1e-12)
    return {
        "rot": float(np.linalg.norm(rotation_est - rotation_gt)),
        "rot_log": float(np.linalg.norm(so3_log(rotation_est @ rotation_gt.T))),
        "tra": float(np.linalg.norm(translation_est - translation_gt) / translation_gt_norm),
    }


def _aggregate_metric(
    entries: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, dict[int, list[float]]]:
    """Aggregate one metric by solver and keypoint count."""
    aggregation: dict[str, dict[int, list[float]]] = {}
    for entry in entries:
        keypoint_count = int(entry["keypoint_count"])
        for solver_name, metrics in entry["errors"].items():
            if metric_key == "runtime":
                value = float(entry["times"].get(solver_name, np.nan))
            else:
                value = float(metrics.get(metric_key, np.nan))
            aggregation.setdefault(solver_name, {}).setdefault(keypoint_count, []).append(value)
    return aggregation
