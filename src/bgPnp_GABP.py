from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.miluv_to_bearing import load_three_robot_mocap
from src.bearing_only_solver import bearing_linear_solver, bgpnp
from src.bo_epnp_solver import BO_EPnP
from src.multi_agent_bearing_gabp import (
    AnchorBearingFactor,
    GaBPRefinementResult,
    GaBPSettings,
    PairwiseBearingFactor,
    PoseState,
    refine_with_gabp,
    so3_log,
    vector_to_az_el,
)

LOGGER = logging.getLogger(__name__)


def _noise_tag(value: float) -> str:
    """Format one noise value for filenames."""
    return f"{value:.3f}".replace(".", "")


@dataclass(frozen=True)
class MiluvSolveConfig:
    """Configuration for reading one observation file and running solvers."""

    observation_path: Path
    output_root: Path = Path("data/miluv_bgpnp_gabp")
    figure_dir: Path = Path("docs/figures")
    keypoint_counts: tuple[int, ...] = (6, 12, 18)
    primary_anchor_id: str = "ifo001"
    target_agent_id: str = "ifo002"
    run_bls: bool = True
    run_bsdp: bool = True
    run_bo_epnp: bool = True
    run_bgpnp: bool = True
    run_bgpnp_gabp: bool = True
    sigma_prior_rotation_deg: float = 5.0
    sigma_prior_translation: float = 0.5
    keypoint_selection_method: str = "uniform"
    gabp_settings: GaBPSettings = GaBPSettings(
        max_gn_iterations=8,
        max_bp_iterations=30,
        gn_tolerance=1e-7,
        bp_tolerance=1e-8,
        message_damping=0.6,
        step_damping=1.0,
        diagonal_regularization=1e-9,
    )

    def __post_init__(self) -> None:
        if not self.observation_path.exists():
            raise FileNotFoundError(f"observation_path does not exist: {self.observation_path}")
        if not self.keypoint_counts:
            raise ValueError("keypoint_counts must not be empty.")
        if any(count <= 0 for count in self.keypoint_counts):
            raise ValueError("All keypoint_counts must be positive.")


@dataclass
class LoadedObservationBatch:
    """Expanded view of one generated MILUV observation file."""

    experiment_dir: Path
    timestamps: np.ndarray
    window_sizes: np.ndarray
    sigma_azimuth_deg: float
    sigma_elevation_deg: float
    anchor_nodes: tuple[str, ...]
    estimated_nodes: tuple[str, ...]
    robot_ids: tuple[str, ...]
    adjacency_matrix: np.ndarray
    anchor_position_w: dict[str, np.ndarray]
    estimated_position_l: dict[str, np.ndarray]
    estimated_position_w: dict[str, np.ndarray]
    estimated_se3: dict[str, dict[str, np.ndarray]]
    bearing_observation: dict[str, np.ndarray]


@dataclass
class MultiAgentWindowSample:
    """Index-only window sample used before solver-specific data slicing."""

    start: int
    stop: int
    window_indices: np.ndarray
    anchor_edge_keypoint_indices: dict[str, np.ndarray]
    anchor_nodes: tuple[str, ...]
    estimated_nodes: tuple[str, ...]
    estimated_se3_gt: dict[str, dict[str, np.ndarray]]
    
    # Used for compatibility with old code. property is a method,but it can be called as a variable
    @property
    def key_point_selection(self) -> dict[str, np.ndarray]:
        """Backward-compatible alias for edge-wise anchor keypoints."""
        return self.anchor_edge_keypoint_indices

    # indices of all keypoints used in gabp
    @property
    def keypoint_indices(self) -> np.ndarray:
        """Backward-compatible union of all anchor-edge keypoint indices."""
        return union_anchor_edge_keypoints(self)


@dataclass
class FactorObservationSample:
    """Observation arrays sliced for GABP factor construction."""

    anchor_nodes: tuple[str, ...]
    estimated_nodes: tuple[str, ...]
    keypoint_indices: np.ndarray
    anchor_position_w: dict[str, np.ndarray]
    estimated_position_l: dict[str, np.ndarray]
    bearing_observation: dict[str, np.ndarray]


@dataclass
class SolverEstimate:
    """Pose estimate and timing for one solver."""

    rotation: np.ndarray
    translation: np.ndarray
    runtime: float


class MiluvObservationReader:
    """Load one strict-harness observation file."""

    @staticmethod
    def load(observation_path: Path) -> LoadedObservationBatch:
        """Load one observation batch from NPZ and expand object fields."""
        raw = np.load(observation_path, allow_pickle=True)
        observation_data = raw["observation_data"].item()
        reference_data = raw["reference_data"].item()

        adjacency_matrix = (
            np.asarray(raw["adjacency_matrix"], dtype=int)
            if "adjacency_matrix" in raw.files
            else _infer_adjacency_matrix(
                tuple(raw["robot_ids"].tolist()),
                tuple(raw["anchor_nodes"].tolist()),
                tuple(raw["estimated_nodes"].tolist()),
                observation_data["bearing_observation"],
            )
        )

        return LoadedObservationBatch(
            experiment_dir=Path(str(raw["experiment_dir"].item())),
            timestamps=np.asarray(raw["timestamps"], dtype=float),
            window_sizes=np.asarray(raw["window_sizes"], dtype=int),
            sigma_azimuth_deg=float(np.asarray(raw["sigma_azimuth_deg"]).reshape(())),
            sigma_elevation_deg=float(np.asarray(raw["sigma_elevation_deg"]).reshape(())),
            anchor_nodes=tuple(str(node) for node in raw["anchor_nodes"].tolist()),
            estimated_nodes=tuple(str(node) for node in raw["estimated_nodes"].tolist()),
            robot_ids=tuple(str(node) for node in raw["robot_ids"].tolist()),
            adjacency_matrix=adjacency_matrix,
            anchor_position_w={
                node: _as_points_3xn(array, f"anchor_position_w[{node}]")
                for node, array in observation_data["anchor_position_w"].items()
            },
            estimated_position_l={
                node: _as_points_3xn(array, f"estimated_position_l[{node}]")
                for node, array in observation_data["estimated_position_l"].items()
            },
            estimated_position_w={
                node: _as_points_3xn(array, f"estimated_position_w[{node}]")
                for node, array in reference_data["estimated_position_w"].items()
            },
            estimated_se3={
                node: {
                    "rotation_local_world": _as_rotation(
                        value["rotation_local_world"],
                        f"estimated_se3[{node}].rotation_local_world",
                    ),
                    "translation_local": _as_vector3(
                        value["translation_local"],
                        f"estimated_se3[{node}].translation_local",
                    ),
                    "rotation_wb_start": _as_rotation(
                        value["rotation_wb_start"],
                        f"estimated_se3[{node}].rotation_wb_start",
                    ),
                    "position_w_start": _as_vector3(
                        value["position_w_start"],
                        f"estimated_se3[{node}].position_w_start",
                    ),
                }
                for node, value in reference_data["estimated_se3"].items()
            },
            bearing_observation={
                edge: _normalize_bearing_sequence(
                    _as_points_3xn(array, f"bearing_observation[{edge}]")
                )
                for edge, array in observation_data["bearing_observation"].items()
            },
        )


class UniformKeypointSelector:
    """Select keypoints with uniform or observability-inspired criteria."""

    VALID_METHODS = {
        "uniform",
        "bearing_diversity",
        "trajectory_geometry",
        "linear_system_condition",
    }

    def __init__(
        self,
        method: str = "uniform",
        observer_id: str | None = None,
        anchor_id: str | None = None,
        epsilon: float = 1e-9,
        force_endpoints: bool = True,
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown keypoint selection method '{method}'. "
                f"Expected one of {sorted(self.VALID_METHODS)}."
            )
        self.method = method
        self.observer_id = observer_id
        self.anchor_id = anchor_id
        self.epsilon = float(epsilon)
        self.force_endpoints = bool(force_endpoints)

    def select(
        self,
        window_indices: np.ndarray,
        keypoint_count: int,
        batch: LoadedObservationBatch | None = None,
        edge_tuple: tuple[str, str] | None = None,
        method: str | None = None,
    ) -> np.ndarray:
        """Return keypoint indices according to the configured method."""
        effective_method = method or self.method
        if effective_method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown keypoint selection method '{effective_method}'. "
                f"Expected one of {sorted(self.VALID_METHODS)}."
            )

        if effective_method == "uniform":
            return self.select_uniform(window_indices, keypoint_count)
        if batch is None:
            raise ValueError(f"batch is required for method '{effective_method}'.")
        if effective_method == "bearing_diversity":
            return self.select_by_bearing_diversity(
                window_indices,
                keypoint_count,
                batch,
                edge_tuple,
            )
        if effective_method == "trajectory_geometry":
            return self.select_by_trajectory_geometry(
                window_indices,
                keypoint_count,
                batch,
                edge_tuple,
            )
        if effective_method == "linear_system_condition":
            return self.select_by_linear_system_condition(
                window_indices,
                keypoint_count,
                batch,
                edge_tuple,
            )
        raise RuntimeError(f"Unhandled keypoint selection method: {effective_method}")

    @staticmethod
    def select_uniform(window_indices: np.ndarray, keypoint_count: int) -> np.ndarray:
        """Return approximately uniform keypoint indices."""
        indices = _validate_candidate_indices(window_indices, keypoint_count)
        if keypoint_count >= indices.size:
            return indices.copy()
        positions = np.linspace(0, indices.size - 1, keypoint_count)
        selected = indices[np.round(positions).astype(int)]
        return np.unique(selected)

    def select_by_bearing_diversity(
        self,
        window_indices: np.ndarray,
        keypoint_count: int,
        batch: LoadedObservationBatch,
        edge_tuple: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Greedily select keypoints maximizing bearing-direction diversity."""
        if edge_tuple is None:
            edge_tuple = self._resolve_observer_anchor(batch)
        edge_key = f"{edge_tuple[0]}_{edge_tuple[1]}"
        if edge_key not in batch.bearing_observation:
            raise KeyError(f"Missing bearing edge for keypoint selection: {edge_key}")
        bearings = batch.bearing_observation[edge_key]

        return self._greedy_select(
            window_indices,
            keypoint_count,
            lambda selected: self._bearing_diversity_score(bearings[:, selected]),
        )

    def select_by_trajectory_geometry(
        self,
        window_indices: np.ndarray,
        keypoint_count: int,
        batch: LoadedObservationBatch,
        edge_tuple: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Greedily select keypoints maximizing local trajectory spread."""
        if edge_tuple is None:
            edge_tuple = self._resolve_observer_anchor(batch)
        observer_id = edge_tuple[0]
        local_points = batch.estimated_position_l[observer_id]

        return self._greedy_select(
            window_indices,
            keypoint_count,
            lambda selected: self._trajectory_geometry_score(local_points[:, selected]),
        )

    def select_by_linear_system_condition(
        self,
        window_indices: np.ndarray,
        keypoint_count: int,
        batch: LoadedObservationBatch,
        edge_tuple: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Greedily select keypoints improving the BO-EPnP proxy matrix rank."""
        if edge_tuple is None:
            edge_tuple = self._resolve_observer_anchor(batch)
        observer_id, anchor_id = edge_tuple
        edge_key = f"{observer_id}_{anchor_id}"
        if edge_key not in batch.bearing_observation:
            raise KeyError(f"Missing bearing edge for keypoint selection: {edge_key}")
        anchor_points = batch.anchor_position_w[anchor_id]
        local_points = batch.estimated_position_l[observer_id]
        bearings = batch.bearing_observation[edge_key]

        return self._greedy_select(
            window_indices,
            keypoint_count,
            lambda selected: self._linear_system_score(
                anchor_points[:, selected],
                local_points[:, selected],
                bearings[:, selected],
            ),
        )

    def _resolve_observer_anchor(self, batch: LoadedObservationBatch) -> tuple[str, str]:
        observer_id = self.observer_id or batch.estimated_nodes[0]
        anchor_id = self.anchor_id or batch.anchor_nodes[0]
        if observer_id not in batch.estimated_position_l:
            raise KeyError(f"Unknown observer_id for keypoint selection: {observer_id}")
        if anchor_id not in batch.anchor_position_w:
            raise KeyError(f"Unknown anchor_id for keypoint selection: {anchor_id}")
        return observer_id, anchor_id

    def _greedy_select(
        self,
        window_indices: np.ndarray,
        keypoint_count: int,
        score_fn: Any, # lambda fuction is an input parameter of this method
    ) -> np.ndarray:
        indices = _validate_candidate_indices(window_indices, keypoint_count)
        if keypoint_count >= indices.size:
            return indices.copy()

        selected: list[int] = []
        if self.force_endpoints and keypoint_count >= 2:
            selected = [int(indices[0]), int(indices[-1])]
        elif self.force_endpoints and keypoint_count == 1:
            selected = [int(indices[0])]

        remaining = [int(idx) for idx in indices if int(idx) not in selected]
        while len(selected) < keypoint_count and remaining:
            best_idx = remaining[0]
            best_score = -np.inf
            for candidate in remaining:
                trial = np.asarray(selected + [candidate], dtype=int)
                score = float(score_fn(trial))
                if score > best_score:
                    best_score = score
                    best_idx = candidate
            selected.append(best_idx)
            remaining.remove(best_idx)

        #return np.asarray(sorted(selected), dtype=int)
        return np.asarray(selected, dtype=int)

    def _bearing_diversity_score(self, bearings: np.ndarray) -> float:
        bearings = _normalize_bearing_sequence(_as_points_3xn(bearings, "bearings"))
        information = np.zeros((3, 3), dtype=float)
        for idx in range(bearings.shape[1]):
            b = bearings[:, idx]
            information += np.eye(3) - np.outer(b, b)
        #return _safe_logdet(information, self.epsilon)
        singular_values = np.linalg.svd(information, compute_uv=False)
        return float(np.min(singular_values))


    def _trajectory_geometry_score(self, local_points: np.ndarray) -> float:
        points = _as_points_3xn(local_points, "local_points")
        centered = points - np.mean(points, axis=1, keepdims=True)
        covariance = centered @ centered.T
        return _safe_logdet(covariance, self.epsilon)

    def _linear_system_score(
        self,
        anchor_points: np.ndarray,
        local_points: np.ndarray,
        bearings: np.ndarray,
    ) -> float:
        _as_points_3xn(anchor_points, "anchor_points")
        points = _as_points_3xn(local_points, "local_points")
        normalized_bearings = _normalize_bearing_sequence(
            _as_points_3xn(bearings, "bearings")
        )
        matrix = self._build_bo_epnp_proxy_matrix(points, normalized_bearings)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size == 0:
            return -np.inf
        return float(np.log(singular_values[-1] ** 2 + self.epsilon))

    def _build_bo_epnp_proxy_matrix(
        self,
        local_points: np.ndarray,
        bearings: np.ndarray,
    ) -> np.ndarray:
        control_points = _build_local_control_points(local_points, self.epsilon)
        alpha = _compute_barycentric_coordinates(local_points, control_points)
        rows: list[np.ndarray] = []
        for idx in range(local_points.shape[1]):
            bearing = bearings[:, idx]
            projection = np.eye(3) - np.outer(bearing, bearing)
            basis = np.kron(alpha[:, idx].reshape(1, 4), np.eye(3))
            rows.append(projection @ basis)
        return np.vstack(rows)


class WindowSampleBuilder:
    """Build one full-window sample from a loaded observation batch."""

    def __init__(self, keypoint_selector: UniformKeypointSelector | None = None) -> None:
        self.keypoint_selector = keypoint_selector or UniformKeypointSelector()

    def build(
        self,
        batch: LoadedObservationBatch,
        keypoint_count: int,
        keypoint_selection_method: str | None = None,
    ) -> MultiAgentWindowSample:
        """Build one sample using the full observation span as a single window."""
        window_indices = np.arange(batch.timestamps.size, dtype=int)
        anchor_edge_keypoint_indices: dict[str, np.ndarray] = {}
        for anchor_id in batch.anchor_nodes:
            for observer_id in batch.estimated_nodes:
                edge_key = f"{observer_id}_{anchor_id}"
                if edge_key not in batch.bearing_observation:
                    continue
                x=batch.bearing_observation[edge_key][0,:]
                y=batch.bearing_observation[edge_key][1,:]
                z=batch.bearing_observation[edge_key][2,:]
                az=np.rad2deg(np.arctan2(y, x))
                horizontal_norm = np.sqrt(x**2 + y**2)
                el = np.rad2deg(np.arctan2(z, horizontal_norm))
                meas={f"{edge_key}_az":az,
                      f"{edge_key}_el":el}
                outputpath="D:\\3-1cooperativeLocalization\\code\\bearing_only_hu\\taes-bearing-only-cooperative-localization\\docs\\figures\\bearing_direction.png"
                plot_curves(meas,outputpath)
                anchor_edge_keypoint_indices[edge_key] = self.keypoint_selector.select(
                    window_indices,
                    keypoint_count,
                    batch,
                    edge_tuple=(observer_id, anchor_id),
                    method=keypoint_selection_method,
                )

        if not anchor_edge_keypoint_indices:
            raise ValueError("No anchor-edge keypoints were selected from the observation batch.")

        return MultiAgentWindowSample(
            start=0,
            stop=batch.timestamps.size,
            window_indices=window_indices,
            anchor_edge_keypoint_indices=anchor_edge_keypoint_indices,
            anchor_nodes=batch.anchor_nodes,
            estimated_nodes=batch.estimated_nodes,
            estimated_se3_gt=batch.estimated_se3,
        )


class BoEpnpFrontEnd:
    """Anchor-based BO-EPnP front-end wrapper."""

    @staticmethod
    def solve_for_agent(
        batch: LoadedObservationBatch,
        sample: MultiAgentWindowSample,
        observer_id: str,
        anchor_id: str,
    ) -> SolverEstimate:
        """Solve one observer pose with BO-EPnP against one anchor."""
        edge_key = f"{observer_id}_{anchor_id}"
        if edge_key not in batch.bearing_observation:
            raise KeyError(f"Missing bearing observation edge: {edge_key}")
        p1, p2, bearing=get_anchor_edge_observation(batch,sample,observer_id,anchor_id)

        start_time = time.perf_counter()
        rotation, translation = BO_EPnP.solve(
            p1,
            p2,
            bearing,
            use_wtls=True,
            use_ml_refine=True,
            control_point_method="pca",
        )
        return SolverEstimate(
            rotation=np.asarray(rotation, dtype=float),
            translation=np.asarray(translation, dtype=float).reshape(3),
            runtime=float(time.perf_counter() - start_time),
        )


class BgpnpGabpSolver:
    """Run BGPnP initialization followed by GABP refinement."""

    def __init__(self, settings: GaBPSettings | None = None) -> None:
        self.settings = settings or GaBPSettings()

    def solve(
        self,
        batch: LoadedObservationBatch,
        sample: MultiAgentWindowSample,
        primary_anchor_id: str,
        prior_covariance: np.ndarray,
    ) -> tuple[dict[str, SolverEstimate], GaBPRefinementResult]:
        """Initialize all estimated nodes with BGPnP and refine them jointly with GABP."""
        initial_estimates: dict[str, SolverEstimate] = {}
        initial_states: dict[str, PoseState] = {}

        for node_id in sample.estimated_nodes:
            
            p1, p2, bearing=get_anchor_edge_observation(batch,sample,node_id,primary_anchor_id)
            # p1 = sample.anchor_position_w[primary_anchor_id].T
            # p2 = sample.estimated_position_l[node_id].T
            # bearing = sample.bearing_observation[edge_key].T
            start_time = time.perf_counter()
            (rotation, translation, _), _runtime = bgpnp.solve(
                p1.T,
                p2.T,
                bearing.T,
                sol_iter=True,
                enforce_manifold=True,
            )
            elapsed = float(time.perf_counter() - start_time)
            estimate = SolverEstimate(
                rotation=np.asarray(rotation, dtype=float),
                translation=np.asarray(translation, dtype=float).reshape(3),
                runtime=elapsed,
            )
            initial_estimates[node_id] = estimate
            initial_states[node_id] = PoseState(
                rotation=estimate.rotation.copy(),
                translation=estimate.translation.copy(),
            )
        sample_keypoint = get_pairwise_observation_by_union(batch, sample)
        anchor_factors = self._build_anchor_factors(sample_keypoint, primary_anchor_id)
        pairwise_factors = self._build_pairwise_factors(sample_keypoint)
        refinement = refine_with_gabp(
            initial_states=initial_states,
            anchor_factors=anchor_factors,
            pairwise_factors=pairwise_factors,
            prior_states=initial_states,
            prior_covariance=prior_covariance,
            settings=self.settings,
        )
        return initial_estimates, refinement

    @staticmethod
    def _build_anchor_factors(
        sample: FactorObservationSample,
        primary_anchor_id: str,
    ) -> list[AnchorBearingFactor]:
        """Create one anchor bearing factor per keypoint and observer."""
        factors: list[AnchorBearingFactor] = []
        for observer_id in sample.estimated_nodes:
            edge_key = f"{observer_id}_{primary_anchor_id}"
            if edge_key not in sample.bearing_observation:
                continue
            bearings = sample.bearing_observation[edge_key]
            anchor_points = sample.anchor_position_w[primary_anchor_id]
            observer_points = sample.estimated_position_l[observer_id]
            for idx in range(bearings.shape[1]):
                factors.append(
                    AnchorBearingFactor(
                        observer_id=observer_id,
                        anchor_position_world=anchor_points[:, idx],
                        measured_angles=vector_to_az_el(bearings[:, idx]),
                        observer_point_local=observer_points[:, idx],
                    )
                )
        return factors

    @staticmethod
    def _build_pairwise_factors(sample: FactorObservationSample) -> list[PairwiseBearingFactor]:
        """Create pairwise bearing factors from all estimated-to-estimated edges."""
        factors: list[PairwiseBearingFactor] = []
        for edge_key, bearings in sample.bearing_observation.items():
            observer_id, target_id = edge_key.split("_", maxsplit=1)
            if observer_id not in sample.estimated_nodes or target_id not in sample.estimated_nodes:
                continue
            observer_points = sample.estimated_position_l[observer_id]
            target_points = sample.estimated_position_l[target_id]
            for idx in range(bearings.shape[1]):
                factors.append(
                    PairwiseBearingFactor(
                        observer_id=observer_id,
                        target_id=target_id,
                        target_point_local=target_points[:, idx],
                        measured_angles=vector_to_az_el(bearings[:, idx]),
                        observer_point_local=observer_points[:, idx],
                    )
                )
        return factors


class MiluvSolveEvaluator:
    """Evaluate front-end and bgpnp+gabp solvers on one observation file."""

    def __init__(self, config: MiluvSolveConfig) -> None:
        self.config = config
        self.reader = MiluvObservationReader()
        self.builder = WindowSampleBuilder(
            UniformKeypointSelector(
                observer_id=config.target_agent_id,
                anchor_id=config.primary_anchor_id,
            )
        )
        self.bo_epnp_frontend = BoEpnpFrontEnd()
        self.gabp_solver = BgpnpGabpSolver(config.gabp_settings)

    def run(self) -> tuple[list[dict[str, Any]], list[Path]]:
        """Run all enabled solvers and produce figures."""# get observations as the structure of MiluvObservationReader
        batch = self.reader.load(self.config.observation_path)
        
        tra={}
        for nodes in batch.anchor_nodes:
            tra[nodes]=batch.anchor_position_w[nodes]
        for nodes in batch.estimated_nodes:
            tra[nodes]=batch.estimated_position_w[nodes]
        #output_path=Path('D:/3-1cooperativeLocalization/code/bearing_only_hu/taes-bearing-only-cooperative-localization/docs/figures/tra_3d.png')
        #plot_key_point(tra,output_path)
        
        # list for results saving, define the structure of results
        results: list[dict[str, Any]] = [
            {
                "observation_file": self.config.observation_path.name,
                "sigma_azimuth_deg": batch.sigma_azimuth_deg,
                "sigma_elevation_deg": batch.sigma_elevation_deg,
                "keypoint_counts": list(self.config.keypoint_counts),
                "target_agent_id": self.config.target_agent_id,
                "estimated_nodes": list(batch.estimated_nodes),
            }
        ]

        figure_paths: list[Path] = []
        sample_for_plot: MultiAgentWindowSample | None = None
        # select key points functions 
        sample = self.builder.build(
            batch,
            max(self.config.keypoint_counts),
            keypoint_selection_method=self.config.keypoint_selection_method,
        )
        # keypoint plot #
        kp={}
        edge_key=f"{batch.estimated_nodes[0]}_{batch.anchor_nodes[0]}"
        indices = np.asarray(sample.anchor_edge_keypoint_indices[edge_key], dtype=int)
        for nodes in sample.anchor_nodes:
            kp[nodes]=batch.anchor_position_w[nodes][:, indices]
        kp[batch.estimated_nodes[0]]=batch.estimated_position_w[batch.estimated_nodes[0]][:, indices]
        output_path=Path(f"{self.config.figure_dir}/tra_3d.png")
        plot_key_point(tra,output_path,keypoint=kp)
        for keypoint_count in self.config.keypoint_counts:
            keypoint_num = np.arange(keypoint_count, dtype=int)
            sample_a = slice_window_sample(sample, keypoint_num)

            
            # kp={}
            # for nodes in sample.anchor_nodes:
            #     kp[nodes]=batch.anchor_position_w[nodes][:, indices],
            # for nodes in sample.estimated_nodes:
            #     kp[nodes]=sample.estimated_position_w_gt[nodes]

            if sample_for_plot is None:
                sample_for_plot = sample_a
            entry = self._evaluate_sample(batch, sample_a)
            results.append(entry)

        if sample_for_plot is None:
            raise RuntimeError("No sample built during evaluation.")

        # scatter plot of keypoints on the complete trajectory 
        figure_paths.append(self._plot_keypoints_3d(batch, sample_for_plot))
        figure_paths.extend(self._plot_error_curves(results))
        return results, figure_paths

    def _evaluate_sample(self,
                         batch: LoadedObservationBatch, 
                         sample: MultiAgentWindowSample) -> dict[str, Any]:
        target_id = self.config.target_agent_id
        anchor_edge_keypoint_indices = {
            edge_key: indices.copy()
            for edge_key, indices in sample.anchor_edge_keypoint_indices.items()
        }
        entry: dict[str, Any] = {
            # number of keypoints of every edge
            "anchor_edge_keypoint_indices":anchor_edge_keypoint_indices,
            "keypoint_count": int(next(iter(sample.anchor_edge_keypoint_indices.values())).size),
            # indices of all keypoints used in gabp
            "keypoint_indices": sample.keypoint_indices.copy(),
            "errors": {},
            "times": {},
            "failures": {},
        }

        gt_rotation = sample.estimated_se3_gt[target_id]["rotation_local_world"]
        gt_translation = sample.estimated_se3_gt[target_id]["translation_local"]

        algorithm_calls: list[tuple[str, Any]] = []
        if self.config.run_bls:
            algorithm_calls.append(("bls", self._solve_bls))
        if self.config.run_bsdp:
            algorithm_calls.append(("bsdp", self._solve_bsdp))
        if self.config.run_bo_epnp:
            algorithm_calls.append(("bo_epnp", self._solve_bo_epnp))
        if self.config.run_bgpnp:
            algorithm_calls.append(("bgpnp", self._solve_bgpnp))
        if self.config.run_bgpnp_gabp:
            algorithm_calls.append(("bgpnp_gabp", self._solve_bgpnp_gabp))

        for algorithm_name, solver in algorithm_calls:
            try:
                estimate = solver(batch,sample)
                entry["times"][algorithm_name] = estimate.runtime
                entry["errors"][algorithm_name] = _pose_error(
                    estimate.rotation,
                    estimate.translation,
                    gt_rotation,
                    gt_translation,
                )
                entry["failures"][algorithm_name] = 0
            except Exception as exc:  # pragma: no cover - error path still recorded in results
                LOGGER.exception("%s failed for keypoint_count=%d", algorithm_name, sample.keypoint_indices.size)
                entry["times"][algorithm_name] = np.nan
                entry["errors"][algorithm_name] = {"rot": np.nan, "rot_log": np.nan, "tra": np.nan}
                entry["failures"][algorithm_name] = 1
                entry.setdefault("messages", {})[algorithm_name] = str(exc)

        return entry

    def _solve_bls(self, batch: LoadedObservationBatch, sample: MultiAgentWindowSample) -> SolverEstimate:
        return _solve_anchor_only_algorithm(
            batch,
            sample,
            self.config.target_agent_id,
            self.config.primary_anchor_id,
            "bls",
        )

    def _solve_bsdp(self, batch: LoadedObservationBatch,sample: MultiAgentWindowSample) -> SolverEstimate:
        return _solve_anchor_only_algorithm(
            batch,
            sample,
            self.config.target_agent_id,
            self.config.primary_anchor_id,
            "bsdp",
        )

    def _solve_bo_epnp(self, batch: LoadedObservationBatch,sample: MultiAgentWindowSample) -> SolverEstimate:
        return self.bo_epnp_frontend.solve_for_agent(
            batch,
            sample,
            self.config.target_agent_id,
            self.config.primary_anchor_id,
        )

    def _solve_bgpnp(self, batch: LoadedObservationBatch,sample: MultiAgentWindowSample) -> SolverEstimate:
        return _solve_anchor_only_algorithm(
            batch,
            sample,
            self.config.target_agent_id,
            self.config.primary_anchor_id,
            "bgpnp",
        )

    def _solve_bgpnp_gabp(self, batch: LoadedObservationBatch,sample: MultiAgentWindowSample) -> SolverEstimate:
        covariance = _build_prior_covariance(
            sigma_rotation_deg=self.config.sigma_prior_rotation_deg,
            sigma_translation=self.config.sigma_prior_translation,
        )
        start_time = time.perf_counter()
        _initial_estimates, refinement = self.gabp_solver.solve(
            batch,
            sample,
            primary_anchor_id=self.config.primary_anchor_id,
            prior_covariance=covariance,
        )
        state = refinement.states[self.config.target_agent_id]
        return SolverEstimate(
            rotation=state.rotation.copy(),
            translation=state.translation.copy(),
            runtime=float(time.perf_counter() - start_time),
        )

    def _plot_keypoints_3d(
        self,
        batch: LoadedObservationBatch,
        sample: MultiAgentWindowSample,
    ) -> Path:
        """Plot full 3D trajectories and the selected keypoints."""
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)
        az_tag = _noise_tag(batch.sigma_azimuth_deg)
        el_tag = _noise_tag(batch.sigma_elevation_deg)
        figure_path = self.config.figure_dir / f"bgpnp_gabp_keypoints_3d_az{az_tag}_el{el_tag}.png"
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")

        color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for idx, node_id in enumerate(batch.anchor_nodes + batch.estimated_nodes):
            if node_id in batch.anchor_position_w:
                trajectory = batch.anchor_position_w[node_id]
            else:
                trajectory = batch.estimated_position_w[node_id]
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(trajectory[0], trajectory[1], trajectory[2], color=color, linewidth=1.8, label=node_id)
            if node_id in batch.estimated_position_w:
                selected = batch.estimated_position_w[node_id][:, sample.keypoint_indices]
                ax.scatter(
                    selected[0],
                    selected[1],
                    selected[2],
                    s=42,
                    color="red" if node_id == self.config.target_agent_id else "purple",
                    linewidths=1.0,
                    edgecolors="black",
                    zorder=5,
                )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("MILUV trajectories with selected keypoints")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)
        return figure_path

    def _plot_error_curves(self, results: list[dict[str, Any]]) -> list[Path]:
        """Plot pose-error curves versus keypoint count for the target agent."""
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)
        entries = results[1:]
        algorithm_names = list(entries[0]["errors"].keys()) if entries else []
        keypoint_counts = [int(entry["keypoint_count"]) for entry in entries]
        metadata = results[0]
        az_tag = _noise_tag(float(metadata["sigma_azimuth_deg"]))
        el_tag = _noise_tag(float(metadata["sigma_elevation_deg"]))
        target_agent_id = str(metadata["target_agent_id"])
        figure_specs = [
            (
                "rot_log",
                "SO(3) log rotation error [rad]",
                f"bgpnp_gabp_rot_error_{target_agent_id}_az{az_tag}_el{el_tag}.png",
            ),
            (
                "tra",
                "normalized translation error",
                f"bgpnp_gabp_tra_error_{target_agent_id}_az{az_tag}_el{el_tag}.png",
            ),
        ]
        figure_paths: list[Path] = []

        for metric_key, ylabel, filename in figure_specs:
            figure_path = self.config.figure_dir / filename
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for algorithm_name in algorithm_names:
                values = [
                    float(entry["errors"][algorithm_name][metric_key])
                    for entry in entries
                ]
                ax.plot(
                    keypoint_counts,
                    values,
                    marker="o",
                    linewidth=1.8,
                    label=algorithm_name,
                )
            ax.set_xlabel("Keypoint count")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(figure_path, dpi=220)
            plt.close(fig)
            figure_paths.append(figure_path)

        return figure_paths


class ResultRepository:
    """Persist results of one observation-file solve run."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.results_dir = output_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def results_path(self) -> Path:
        return self.results_dir / "bgpnp_gabp_results.npz"

    def save(self, results: list[dict[str, Any]]) -> Path:
        """Save results to NPZ."""
        np.savez_compressed(self.results_path, results=np.array(results, dtype=object))
        return self.results_path


def exp_miluv_bo_epnp_gnbp(config: MiluvSolveConfig) -> dict[str, Any]:
    """Run the MILUV observation -> BO-EPnP/GABP solve flow."""
    evaluator = MiluvSolveEvaluator(config)
    results, figure_paths = evaluator.run()
    repository = ResultRepository(config.output_root)
    results_path = repository.save(results)
    return {
        "observation_path": config.observation_path,
        "results": results,
        "results_path": results_path,
        "figure_paths": figure_paths,
    }


def get_anchor_edge_observation(
    batch: LoadedObservationBatch,
    sample: MultiAgentWindowSample,
    observer_id: str,
    anchor_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Args:
            batch: complete measurements
            sample: indices of keypoint in sample
            observer_id: observer agent id
            anchor_id: anchor agent id
        Returns:
            (anchor position in global coordinate,
             estiamated position in local coordinate
             bearing estimates between anchor and estimated nodes) """
    
    """Slice one anchor-edge observation using its selected keypoint indices."""
    edge_key = f"{observer_id}_{anchor_id}"
    if edge_key not in sample.anchor_edge_keypoint_indices:
        raise KeyError(f"Missing anchor-edge keypoints: {edge_key}")
    if anchor_id not in batch.anchor_position_w:
        raise KeyError(f"Unknown anchor node: {anchor_id}")
    if observer_id not in batch.estimated_position_l:
        raise KeyError(f"Unknown estimated observer node: {observer_id}")
    if edge_key not in batch.bearing_observation:
        raise KeyError(f"Missing bearing observation edge: {edge_key}")

    indices = np.asarray(sample.anchor_edge_keypoint_indices[edge_key], dtype=int)
    return (
        batch.anchor_position_w[anchor_id][:, indices],
        batch.estimated_position_l[observer_id][:, indices],
        batch.bearing_observation[edge_key][:, indices],
    )


def union_anchor_edge_keypoints(sample: MultiAgentWindowSample) -> np.ndarray:
    """Return the sorted union of all selected anchor-edge keypoint indices."""
    if not sample.anchor_edge_keypoint_indices:
        raise ValueError("anchor_edge_keypoint_indices must not be empty.")
    # .values可以用于提取字典的键值，concatenate可以将每行首尾拼接成一个大的一维数组 unique排除这个数组中重复的元素
    merged = np.concatenate(
        [
            np.asarray(indices, dtype=int).reshape(-1)
            for indices in sample.anchor_edge_keypoint_indices.values()
        ]
    )
    return np.unique(merged)


def get_pairwise_observation_by_union(
    batch: LoadedObservationBatch,
    sample: MultiAgentWindowSample,
) -> FactorObservationSample:
    """Slice GABP factor observations using the union of anchor-edge keypoints."""
    keypoint_indices = union_anchor_edge_keypoints(sample)

    anchor_position_w = {
        anchor_id: batch.anchor_position_w[anchor_id][:, keypoint_indices]
        for anchor_id in sample.anchor_nodes
        if anchor_id in batch.anchor_position_w
    }
    estimated_position_l = {
        node_id: batch.estimated_position_l[node_id][:, keypoint_indices]
        for node_id in sample.estimated_nodes
        if node_id in batch.estimated_position_l
    }

    bearing_observation: dict[str, np.ndarray] = {}
    for edge_key, bearings in batch.bearing_observation.items():
        observer_id, target_id = edge_key.split("_", maxsplit=1)
        is_anchor_edge = (
            observer_id in sample.estimated_nodes
            and target_id in sample.anchor_nodes
        )
        is_pairwise_edge = (
            observer_id in sample.estimated_nodes
            and target_id in sample.estimated_nodes
        )
        if is_anchor_edge or is_pairwise_edge:
            bearing_observation[edge_key] = bearings[:, keypoint_indices]

    if not bearing_observation:
        raise ValueError("No anchor or pairwise bearing observations were found for GABP.")

    return FactorObservationSample(
        anchor_nodes=sample.anchor_nodes,
        estimated_nodes=sample.estimated_nodes,
        keypoint_indices=keypoint_indices,
        anchor_position_w=anchor_position_w,
        estimated_position_l=estimated_position_l,
        bearing_observation=bearing_observation,
    )


def _solve_anchor_only_algorithm(
    batch: LoadedObservationBatch,
    sample: MultiAgentWindowSample,
    observer_id: str,
    anchor_id: str,
    algorithm_name: str,
) -> SolverEstimate:
    

    p1_col, p2_col, bearing_col=get_anchor_edge_observation(batch,sample,observer_id,anchor_id)


    start_time = time.perf_counter()
    if algorithm_name == "bls":
        (rotation, translation), _ = bearing_linear_solver.solve(p1_col, p2_col, bearing_col)
    elif algorithm_name == "bsdp":
        (rotation, translation), _ = bearing_linear_solver.solve_with_sdp_sdr(p1_col, p2_col, bearing_col)
    elif algorithm_name == "bgpnp":
        (rotation, translation, _), _ = bgpnp.solve(
            p1_col.T,
            p2_col.T,
            bearing_col.T,
            sol_iter=True,
            enforce_manifold=True,
        )
    else:
        raise ValueError(f"Unknown anchor-only algorithm: {algorithm_name}")

    return SolverEstimate(
        rotation=np.asarray(rotation, dtype=float),
        translation=np.asarray(translation, dtype=float).reshape(3),
        runtime=float(time.perf_counter() - start_time),
    )


def _build_prior_covariance(
    sigma_rotation_deg: float,
    sigma_translation: float,
) -> np.ndarray:
    sigma_rotation = np.deg2rad(float(sigma_rotation_deg))
    sigma_translation = float(sigma_translation)
    return np.diag(
        [
            sigma_rotation**2,
            sigma_rotation**2,
            sigma_rotation**2,
            sigma_translation**2,
            sigma_translation**2,
            sigma_translation**2,
        ]
    )


def _infer_adjacency_matrix(
    robot_ids: tuple[str, ...],
    anchor_nodes: tuple[str, ...],
    estimated_nodes: tuple[str, ...],
    bearing_observation: dict[str, np.ndarray],
) -> np.ndarray:
    """Infer adjacency from saved edge keys when the field is absent in NPZ."""
    node_to_idx = {node: idx for idx, node in enumerate(robot_ids)}
    adjacency = np.zeros((len(robot_ids), len(robot_ids)), dtype=int)
    for edge_key in bearing_observation:
        observer_id, target_id = edge_key.split("_", maxsplit=1)
        if observer_id in node_to_idx and target_id in node_to_idx:
            adjacency[node_to_idx[target_id], node_to_idx[observer_id]] = 1
    for anchor_id in anchor_nodes:
        if anchor_id in estimated_nodes:
            raise ValueError(f"Node {anchor_id} appears in both anchor_nodes and estimated_nodes.")
    return adjacency

# 判定输入的points为三行n列
def _as_points_3xn(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3, N), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr

# 判定rotation是一个旋转矩阵
def _as_rotation(rotation: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(rotation, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr

# 判定vector3只包含三个元素
def _as_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _normalize_bearing_sequence(bearing_3xn: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(bearing_3xn, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("bearing sequence contains zero vectors")
    return bearing_3xn / norms


def _validate_candidate_indices(window_indices: np.ndarray, keypoint_count: int) -> np.ndarray:
    indices = np.asarray(window_indices, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError("window_indices must not be empty.")
    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive.")
    return indices


def _safe_logdet(matrix: np.ndarray, epsilon: float) -> float:
    regularized = np.asarray(matrix, dtype=float) + float(epsilon) * np.eye(matrix.shape[0])
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return float(logdet)


def _build_local_control_points(local_points: np.ndarray, epsilon: float) -> np.ndarray:
    points = _as_points_3xn(local_points, "local_points")
    centroid = np.mean(points, axis=1)
    centered = points - centroid.reshape(3, 1)
    covariance = centered @ centered.T / max(points.shape[1], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scales = np.sqrt(np.maximum(eigenvalues, float(epsilon)))
    control_points = np.column_stack(
        [
            centroid,
            centroid + scales[0] * eigenvectors[:, 0],
            centroid + scales[1] * eigenvectors[:, 1],
            centroid + scales[2] * eigenvectors[:, 2],
        ]
    )
    return control_points


def _compute_barycentric_coordinates(
    local_points: np.ndarray,
    control_points: np.ndarray,
) -> np.ndarray:
    points = _as_points_3xn(local_points, "local_points")
    controls = _as_points_3xn(control_points, "control_points")
    if controls.shape[1] != 4:
        raise ValueError(f"control_points must have shape (3, 4), got {controls.shape}")
    control_aug = np.vstack([controls, np.ones((1, 4))])
    point_aug = np.vstack([points, np.ones((1, points.shape[1]))])
    try:
        return np.linalg.solve(control_aug, point_aug)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(control_aug) @ point_aug


def _pose_error(
    rotation_est: np.ndarray,
    translation_est: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
) -> dict[str, float]:
    translation_norm = max(float(np.linalg.norm(translation_gt)), 1e-12)
    return {
        "rot": float(np.linalg.norm(rotation_est - rotation_gt)),
        "rot_log": float(np.linalg.norm(so3_log(rotation_est @ rotation_gt.T))),
        "tra": float(np.linalg.norm(translation_est - translation_gt) / translation_norm),
    }

def slice_window_sample(
    sample_full: MultiAgentWindowSample,
    keypoint_num: np.ndarray,
) -> MultiAgentWindowSample:
    """Slice only selected keypoint indices for each anchor edge."""
    sub_list = np.asarray(keypoint_num, dtype=int).reshape(-1)
    anchor_edge_keypoint_indices: dict[str, np.ndarray] = {}
    for edge_key, indices in sample_full.anchor_edge_keypoint_indices.items():
        edge_indices = np.asarray(indices, dtype=int).reshape(-1)
        if np.any(sub_list < 0) or np.any(sub_list >= edge_indices.size):
            raise IndexError(
                f"sub_indices out of range for edge {edge_key}: "
                f"edge has {edge_indices.size} keypoints."
            )
        anchor_edge_keypoint_indices[edge_key] = edge_indices[sub_list]

    return MultiAgentWindowSample(
        start=sample_full.start,
        stop=sample_full.stop,
        window_indices=sample_full.window_indices,
        anchor_nodes=sample_full.anchor_nodes,
        estimated_nodes=sample_full.estimated_nodes,
        anchor_edge_keypoint_indices=anchor_edge_keypoint_indices,
        estimated_se3_gt=sample_full.estimated_se3_gt,
    )


def plot_key_point(
    trajectory: dict[str, np.ndarray],
    outputpath: Path,
    keypoint:dict[str, np.ndarray] | None=None,
    original_tra_path: Path |None=None,
)->str :
    def _normalize_points(points: np.ndarray) -> np.ndarray:
        """
        统一转换为 (N, 3) 形式。
        """
        points = np.asarray(points, dtype=float)

        if points.ndim != 2:
            raise ValueError(f"输入点数组必须是二维，当前 shape={points.shape}")

        if points.shape[1] == 3:
            return points
        elif points.shape[0] == 3:
            return points.T
        else:
            raise ValueError(
                f"输入点数组必须为 (N,3) 或 (3,N) 形状，当前 shape={points.shape}"
            )

    outputpath = str(outputpath)
    Path(outputpath).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 颜色池
    colors = plt.cm.tab10.colors

    # 为每个无人机分配颜色
    color_map: dict[str, tuple[float, ...]] = {}
    for idx, uav_id in enumerate(trajectory.keys()):
        color_map[uav_id] = colors[idx % len(colors)]

    # 先画连续轨迹
    for uav_id, traj in trajectory.items():
        traj = _normalize_points(traj)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=color_map[uav_id],
            linewidth=2.0,
            label=f"{uav_id} trajectory",
        )
        # 起点：红色三角形
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            traj[0, 2],
            c="red",
            marker="^",
            s=100,
            edgecolors="black",
            linewidths=1.2,
            label="Start point",
            depthshade=True,
        )

    # 终点：红色正方形
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            c="red",
            marker="s",
            s=100,
            edgecolors="black",
            linewidths=1.2,
            label="End point",
            depthshade=True,
        )

    # 再画筛选点
    if keypoint is not None:
        for uav_id, pts in keypoint.items():
            pts = _normalize_points(pts)

            # 如果 keypoint 中的 id 不在 trajectory 里，就给它一个默认红色
            point_color = color_map.get(uav_id, "red")

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[point_color],
                s=60,              # 可以再大一点
                edgecolors="black",
                linewidths=1.5,
                label=f"{uav_id} selected",
                depthshade=True,
            )
   
    if original_tra_path is not None:
        org_tra=load_three_robot_mocap(original_tra_path,list(trajectory.keys()))
        POSITION_COLUMNS = [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        ]
        positions_dict: dict[str, np.ndarray] = {}
        for robot_id, frame in org_tra.items():
            positions_dict[robot_id] = frame[POSITION_COLUMNS].to_numpy(dtype=float)
        for uav_id,trjo in positions_dict.items():
            trjo = _normalize_points(trjo)
            ax.plot(
            trjo[:, 0],
            trjo[:, 1],
            trjo[:, 2],
            color=color_map[uav_id],
            linewidth=1.0,
            label=f"{uav_id} trajectory",
            )

    ax.set_title("3D UAV Trajectories and Selected Points")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()
    plt.savefig(outputpath, dpi=300)
    plt.close(fig)

    return outputpath

def plot_curves(
    data: dict[str, np.ndarray],
    output_path: str | Path,
    x: np.ndarray | None = None,
    title: str = "Curve comparison",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 300,
) -> str:
    """
    绘制多条曲线，并自动为不同曲线分配颜色。

    Parameters
    ----------
    data : dict[str, np.ndarray]
        输入曲线数据。
        key 为曲线名称，value 为一维数组 y。
        例如：
        {
            "method A": np.array([...]),
            "method B": np.array([...])
        }

    output_path : str | Path
        图片保存路径。

    x : np.ndarray | None
        横坐标。如果为 None，则默认使用 0, 1, ..., N-1。

    title : str
        图标题。

    xlabel : str
        x 轴名称。

    ylabel : str
        y 轴名称。

    figsize : tuple[int, int]
        图像大小。

    dpi : int
        图片分辨率。

    Returns
    -------
    str
        保存后的图片路径。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # 自动颜色池
    colors = plt.cm.tab10.colors

    for idx, (name, y) in enumerate(data.items()):
        y = np.asarray(y, dtype=float).reshape(-1)

        if x is None:
            x_plot = np.arange(len(y))
        else:
            x_plot = np.asarray(x, dtype=float).reshape(-1)
            if len(x_plot) != len(y):
                raise ValueError(
                    f"x 和 y 的长度不一致：curve={name}, len(x)={len(x_plot)}, len(y)={len(y)}"
                )

        ax.plot(
            x_plot,
            y,
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=name,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return str(output_path)
