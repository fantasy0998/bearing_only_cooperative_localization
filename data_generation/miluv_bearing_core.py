"""Copied MILUV geometry/noise core; independent of localization solvers.

Origin: src/miluv_to_bearing.py. See source_provenance.json for the copy hash.
Local fixes validate the time grid and consistently use reference_index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

LOGGER = logging.getLogger(__name__)

MOCAP_POSITION_COLUMNS = [
    "pose.position.x",
    "pose.position.y",
    "pose.position.z",
]
MOCAP_QUATERNION_COLUMNS = [
    "pose.orientation.x",
    "pose.orientation.y",
    "pose.orientation.z",
    "pose.orientation.w",
]
DEFAULT_ROBOT_IDS = ("ifo001", "ifo002", "ifo003")
DEFAULT_DIRECTED_EDGES = (
    ("ifo002", "ifo001"),
    ("ifo003", "ifo001"),
    ("ifo002", "ifo003"),
    ("ifo003", "ifo002"),
)


def load_experiment_timeshift_seconds(experiment_dir: Path) -> float:
    """Return the MILUV experiment time shift in seconds."""
    timeshift_path = experiment_dir / "timeshift.yaml"
    if not timeshift_path.exists():
        LOGGER.warning("timeshift.yaml not found under %s; using 0.0 s.", experiment_dir)
        return 0.0

    timeshift: dict[str, float] = {}
    with timeshift_path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            timeshift[key.strip()] = float(value.strip())

    if "timeshift_s" not in timeshift or "timeshift_ns" not in timeshift:
        raise ValueError(f"Invalid timeshift.yaml content: {timeshift_path}")

    return timeshift["timeshift_s"] + timeshift["timeshift_ns"] / 1e9


def load_miluv_mocap_csv(experiment_dir: Path, robot_id: str) -> pd.DataFrame:
    """Load and sanitize one robot's mocap.csv file."""
    mocap_path = experiment_dir / robot_id / "mocap.csv"
    if not mocap_path.exists():
        raise FileNotFoundError(f"Missing mocap.csv for {robot_id}: {mocap_path}")

    data = pd.read_csv(mocap_path)
    required_columns = ["timestamp", *MOCAP_POSITION_COLUMNS, *MOCAP_QUATERNION_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{mocap_path} is missing columns: {missing_columns}")

    data = data[required_columns].copy()
    if len(data) < 2 or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f"{mocap_path} must contain at least two finite samples")
    if np.any(np.linalg.norm(data[MOCAP_QUATERNION_COLUMNS].to_numpy(), axis=1) < 1e-12):
        raise ValueError(f"Zero quaternion in {mocap_path}")
    data.drop_duplicates(subset="timestamp", inplace=True)
    data.sort_values(by="timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    if len(data) < 2:
        raise ValueError(f"{mocap_path} must contain at least two distinct timestamps")
    return data


def load_three_robot_mocap(
    experiment_dir: Path,
    robot_ids: Iterable[str] = DEFAULT_ROBOT_IDS,
) -> dict[str, pd.DataFrame]:
    """Load mocap.csv files for the requested robots."""
    return {robot_id: load_miluv_mocap_csv(experiment_dir, robot_id) for robot_id in robot_ids}


def estimate_common_dt(mocap_frames: dict[str, pd.DataFrame]) -> float:
    """Estimate a conservative common sampling interval from mocap timestamps."""
    median_steps = []
    for frame in mocap_frames.values():
        timestamps = frame["timestamp"].to_numpy(dtype=float)
        if timestamps.size < 2:
            continue
        diffs = np.diff(timestamps)
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size == 0:
            continue
        median_steps.append(float(np.median(positive_diffs)))

    if not median_steps:
        raise ValueError("Unable to estimate a common sampling interval from mocap timestamps.")

    return float(max(median_steps))


def build_common_timestamps(
    mocap_frames: dict[str, pd.DataFrame],
    dt: float | None = None,
    max_samples: int | None = None,
    start_point: int | None = None,
    end_point: int | None = None,
) -> np.ndarray:
    """Build the common synchronized timestamp grid for all robots."""
    timestamps_per_robot = [
        frame["timestamp"].to_numpy(dtype=float) for frame in mocap_frames.values()
    ]
    # Find the largest common time interval across all robots.
    t_min = max(float(timestamps[0]) for timestamps in timestamps_per_robot)
    t_max = min(float(timestamps[-1]) for timestamps in timestamps_per_robot)

    if t_max <= t_min:
        raise ValueError("No overlapping mocap interval exists across the selected robots.")

    # Indices refer to the sanitized first robot CSV; end_point is inclusive.
    first = timestamps_per_robot[0]
    for name, index in (("start_point", start_point), ("end_point", end_point)):
        if index is not None and (type(index) is not int or not 0 <= index < len(first)):
            raise ValueError(f"{name} must be None or an index in [0,{len(first)})")
    t_start = t_min if start_point is None else max(t_min, float(first[start_point]))
    t_end = t_max if end_point is None else min(t_max, float(first[end_point]))
    if t_end <= t_start:
        raise ValueError("Selected interval does not overlap all robots or is reversed")
    sync_dt = estimate_common_dt(mocap_frames) if dt is None else float(dt)
    if not np.isfinite(sync_dt) or sync_dt <= 0:
        raise ValueError("dt must be finite and positive")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 2):
        raise ValueError("max_samples must be None or an integer >= 2")
    timestamps = np.arange(t_start, t_end + 0.5 * sync_dt, sync_dt, dtype=float)
    timestamps = timestamps[timestamps <= t_end]

    if max_samples is not None:
        timestamps = timestamps[:max_samples]

    if timestamps.size < 2:
        raise ValueError("Synchronized timestamp grid is too short for bearing synthesis.")

    return timestamps


def interpolate_robot_state(
    mocap_frame: pd.DataFrame,
    query_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate mocap position and quaternion to query timestamps."""
    timestamps = mocap_frame["timestamp"].to_numpy(dtype=float)
    positions = mocap_frame[MOCAP_POSITION_COLUMNS].to_numpy(dtype=float)
    quaternions = mocap_frame[MOCAP_QUATERNION_COLUMNS].to_numpy(dtype=float)

    position_interp = interp1d(
        timestamps,
        positions,
        axis=0,
        kind="linear",
        bounds_error=True,
    )
    positions_interp = position_interp(query_timestamps)
    #
    rotations = Rotation.from_quat(quaternions)
    quaternion_interp = Slerp(timestamps, rotations)(query_timestamps)
    quaternions_interp = quaternion_interp.as_quat()
    rotation_matrices = quaternion_interp.as_matrix()
    return positions_interp.T, quaternions_interp.T, rotation_matrices


def sync_three_robot_states(
    experiment_dir: Path,
    robot_ids: Iterable[str] = DEFAULT_ROBOT_IDS,
    dt: float | None = None,
    max_samples: int | None = None,
    start_point: int | None = None,
    end_point: int | None = None,
) -> dict[str, Any]:
    """Synchronize three MILUV robot mocap trajectories on a common time axis."""
    mocap_frames = load_three_robot_mocap(experiment_dir, robot_ids)
    timestamps = build_common_timestamps(
        mocap_frames,
        dt=dt,
        max_samples=max_samples,
        start_point=start_point,
        end_point=end_point,
    )

    states: dict[str, dict[str, np.ndarray]] = {}
    for robot_id, mocap_frame in mocap_frames.items():
        position_w, quat_wb_xyzw, rotation_wb = interpolate_robot_state(mocap_frame, timestamps)
        states[robot_id] = {
            "position_w": position_w,
            "quat_wb_xyzw": quat_wb_xyzw,
            "rotation_wb": rotation_wb,
        }

    return {
        "timestamps": timestamps,
        "states": states,
        "timeshift_s": load_experiment_timeshift_seconds(
            experiment_dir
        ),  ## 在对齐与imu等传感器的时间戳时才会用到，在动捕结果中还用不到
    }


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def bearing_to_angles(bearing_b4: np.ndarray) -> np.ndarray:
    """Convert unit bearing vectors into azimuth/elevation angles."""
    if bearing_b4.shape[0] != 3:
        raise ValueError(f"bearing_b4 must have shape (3, N), got {bearing_b4.shape}")

    azimuth = np.arctan2(bearing_b4[1], bearing_b4[0])
    elevation = np.arcsin(np.clip(bearing_b4[2], -1.0, 1.0))
    return np.vstack((azimuth, elevation))


def angles_to_bearing(angles: np.ndarray) -> np.ndarray:
    """Convert azimuth/elevation angles to unit bearing vectors."""
    if angles.shape[0] != 2:
        raise ValueError(f"angles must have shape (2, N), got {angles.shape}")

    azimuth = angles[0]
    elevation = angles[1]
    cos_elevation = np.cos(elevation)
    return np.vstack(
        (
            np.cos(azimuth) * cos_elevation,
            np.sin(azimuth) * cos_elevation,
            np.sin(elevation),
        )
    )


def compute_b3_from_b4_rotation_sequence(
    observer_rotation_wb: np.ndarray,
    reference_index: int = 0,
) -> np.ndarray:
    """Compute the per-epoch relative rotation from B4(k) to frozen B3(k0)."""
    if observer_rotation_wb.ndim != 3 or observer_rotation_wb.shape[1:] != (3, 3):
        raise ValueError(
            "observer_rotation_wb must have shape (N, 3, 3), got " f"{observer_rotation_wb.shape}"
        )
    if not 0 <= reference_index < observer_rotation_wb.shape[0]:
        raise ValueError(
            f"reference_index={reference_index} is out of bounds for "
            f"{observer_rotation_wb.shape[0]} rotations."
        )

    rotation_wb3 = observer_rotation_wb[reference_index]
    return np.einsum("ij,njk->nik", rotation_wb3.T, observer_rotation_wb)


def rotate_bearing_sequence(
    bearing_vectors: np.ndarray,
    rotation_target_source: np.ndarray,
) -> np.ndarray:
    """Rotate a sequence of 3D vectors by a sequence of rotation matrices."""
    if bearing_vectors.ndim != 2 or bearing_vectors.shape[0] != 3:
        raise ValueError(f"bearing_vectors must have shape (3, N), got {bearing_vectors.shape}")
    if rotation_target_source.ndim != 3 or rotation_target_source.shape[1:] != (3, 3):
        raise ValueError(
            "rotation_target_source must have shape (N, 3, 3), got "
            f"{rotation_target_source.shape}"
        )
    if rotation_target_source.shape[0] != bearing_vectors.shape[1]:
        raise ValueError(
            "rotation_target_source must match the number of bearing samples, got "
            f"{rotation_target_source.shape[0]} rotations for {bearing_vectors.shape[1]} vectors"
        )

    rotated = np.einsum("nij,jn->in", rotation_target_source, bearing_vectors)
    norms = np.linalg.norm(rotated, axis=0)
    if np.any(norms <= 1e-12):
        raise ValueError("Encountered a near-zero rotated bearing vector.")
    return rotated / norms


def synthesize_directed_bearing_edge(
    observer_position_w: np.ndarray,
    observer_rotation_wb: np.ndarray,
    target_position_w: np.ndarray,
    reference_index: int = 0,
    sigma_azimuth_deg: float = 0.5,
    sigma_elevation_deg: float = 2.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Synthesize one directed bearing sequence, with final observations in frozen B3(k0)."""
    if observer_position_w.shape != target_position_w.shape:
        raise ValueError(
            "observer_position_w and target_position_w must have the same shape, got "
            f"{observer_position_w.shape} and {target_position_w.shape}"
        )
    if observer_position_w.shape[0] != 3:
        raise ValueError(f"Positions must have shape (3, N), got {observer_position_w.shape}")
    if observer_rotation_wb.shape[0] != observer_position_w.shape[1]:
        raise ValueError(
            "observer_rotation_wb must have one rotation per timestamp, got "
            f"{observer_rotation_wb.shape[0]} rotations for {observer_position_w.shape[1]} states"
        )

    relative_w = (target_position_w - observer_position_w).T
    norms = np.linalg.norm(relative_w, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(
            "Encountered coincident observer/target positions during bearing synthesis."
        )

    relative_b4 = np.einsum(
        "nij,nj->ni",
        np.transpose(observer_rotation_wb, (0, 2, 1)),
        relative_w,
    )
    bearing_noiseless = (relative_b4 / norms[:, None]).T

    angles_noiseless = bearing_to_angles(bearing_noiseless)
    sigma_azimuth = np.deg2rad(sigma_azimuth_deg)
    sigma_elevation = np.deg2rad(sigma_elevation_deg)
    generator = np.random.default_rng() if rng is None else rng
    noise = np.vstack(
        (
            generator.normal(0.0, sigma_azimuth, size=angles_noiseless.shape[1]),
            generator.normal(0.0, sigma_elevation, size=angles_noiseless.shape[1]),
        )
    )
    angles_noisy = angles_noiseless + noise
    angles_noisy[0] = wrap_to_pi(angles_noisy[0])
    angles_noisy[1] = np.clip(angles_noisy[1], -0.5 * np.pi + 1e-6, 0.5 * np.pi - 1e-6)

    bearing_noisy = angles_to_bearing(angles_noisy)
    rotation_b3_b4 = compute_b3_from_b4_rotation_sequence(
        observer_rotation_wb, reference_index=reference_index
    )
    bearing_b3_noiseless = rotate_bearing_sequence(bearing_noiseless, rotation_b3_b4)
    bearing_b3 = rotate_bearing_sequence(bearing_noisy, rotation_b3_b4)
    angles_b3_noiseless = bearing_to_angles(bearing_b3_noiseless)
    angles_b3 = bearing_to_angles(bearing_b3)
    return {
        "bearing": bearing_b3,
        "bearing_noiseless": bearing_b3_noiseless,
        "angles": angles_b3,
        "angles_noiseless": angles_b3_noiseless,
        "bearing_b3": bearing_b3,
        "bearing_b3_noiseless": bearing_b3_noiseless,
        "angles_b3": angles_b3,
        "angles_b3_noiseless": angles_b3_noiseless,
        "bearing_b4": bearing_noisy,
        "bearing_b4_noiseless": bearing_noiseless,
        "angles_b4": angles_noisy,
        "angles_b4_noiseless": angles_noiseless,
        "rotation_b3_b4": rotation_b3_b4,
        "reference_index": np.array(reference_index, dtype=int),
        "range_w": norms,
    }


def synthesize_three_robot_bearing_observations(
    experiment_dir: Path,
    anchor_id: str = "ifo001",
    robot_ids: Iterable[str] = DEFAULT_ROBOT_IDS,
    directed_edges: Iterable[tuple[str, str]] = DEFAULT_DIRECTED_EDGES,
    dt: float | None = None,
    max_samples: int | None = None,
    reference_index: int = 0,
    sigma_azimuth_deg: float = 0.5,
    sigma_elevation_deg: float = 2.0,
    random_seed: int | None = None,
    start_point: int | None = None,
    end_point: int | None = None,
) -> dict[str, Any]:
    """Generate synchronized three-robot B4-frame bearing observations from MILUV mocap GT."""
    sync_data = sync_three_robot_states(
        experiment_dir=experiment_dir,
        robot_ids=robot_ids,
        dt=dt,
        max_samples=max_samples,
        start_point=start_point,
        end_point=end_point,
    )
    # states 是一个字典，键是机器人id，值包含该机器人在每个时间戳的插值位置、四元数和旋转矩阵
    states = sync_data["states"]
    timestamps = sync_data["timestamps"]

    if anchor_id not in states:
        raise ValueError(f"Anchor robot {anchor_id} is not part of the synchronized state set.")

    generator = np.random.default_rng(random_seed)
    observations: dict[str, dict[str, np.ndarray | str]] = {}
    for observer_id, target_id in directed_edges:
        if observer_id not in states or target_id not in states:
            raise ValueError(f"Edge ({observer_id}, {target_id}) references unknown robots.")

        edge_key = f"{observer_id}<-{target_id}"
        edge_observation = synthesize_directed_bearing_edge(
            observer_position_w=states[observer_id]["position_w"],
            observer_rotation_wb=states[observer_id]["rotation_wb"],
            target_position_w=states[target_id]["position_w"],
            reference_index=reference_index,  ## 参考坐标系
            sigma_azimuth_deg=sigma_azimuth_deg,
            sigma_elevation_deg=sigma_elevation_deg,
            rng=generator,
        )
        observations[edge_key] = {
            "observer_id": observer_id,
            "target_id": target_id,
            **edge_observation,
        }

    estimated_nodes = tuple(robot_id for robot_id in robot_ids if robot_id != anchor_id)
    estimated_pose: dict[str, dict[str, np.ndarray]] = {}
    for robot_id in estimated_nodes:
        position_w = np.asarray(states[robot_id]["position_w"], dtype=float)
        rotation_wb = np.asarray(states[robot_id]["rotation_wb"], dtype=float)
        rotation_local_world = rotation_wb[reference_index].T
        translation_local = -rotation_local_world @ position_w[:, reference_index]
        estimated_pose[robot_id] = {
            "rotation_local_world": rotation_local_world,
            "translation_local": translation_local.reshape(3),
        }

    return {
        "experiment_dir": str(experiment_dir),
        "timeshift_s": sync_data["timeshift_s"],
        "anchor_id": anchor_id,
        "estimated_nodes": estimated_nodes,
        "robot_ids": tuple(robot_ids),
        "estimated_pose": estimated_pose,
        "reference_index": reference_index,
        "timestamps": timestamps,
        "states": states,
        "observations": observations,
    }
