from __future__ import annotations

import argparse
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    missing_columns  = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{mocap_path} is missing columns: {missing_columns}")

    data = data[required_columns].copy()
    data.drop_duplicates(subset="timestamp", inplace=True)
    data.sort_values(by="timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
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
    start_point:int  | None = None,
    end_point:int  | None = None,
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

    t_start=timestamps_per_robot[0][start_point]
    t_end =timestamps_per_robot[0][end_point]
    sync_dt = estimate_common_dt(mocap_frames) if dt is None else float(dt)
    timestamps = np.arange(t_start, t_end + 0.5 * sync_dt, sync_dt, dtype=float)
    timestamps = timestamps[timestamps <= t_max + 1e-9]

    if max_samples is not None:
        timestamps = timestamps[: max_samples]

    if timestamps.size < 2:
        raise ValueError("Synchronized timestamp grid is too short for bearing synthesis.")

    return timestamps

# 返回位置插值结果，四元数插值结果和旋转矩阵插值结果
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
        bounds_error=False,
        fill_value="extrapolate",
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
    timestamps = build_common_timestamps(mocap_frames, dt=dt, max_samples=max_samples,
                                          start_point=start_point,
                                          end_point=end_point,
                                          )

    states: dict[str, dict[str, np.ndarray]] = {}
    for robot_id, mocap_frame in mocap_frames.items():
        position_w, quat_wb_xyzw, rotation_wb = interpolate_robot_state(
            mocap_frame, timestamps
        )
        states[robot_id] = {
            "position_w": position_w,
            "quat_wb_xyzw": quat_wb_xyzw,
            "rotation_wb": rotation_wb,
        }

    return {
        "timestamps": timestamps,
        "states": states,
        "timeshift_s": load_experiment_timeshift_seconds(experiment_dir), ## 在对齐与imu等传感器的时间戳时才会用到，在动捕结果中还用不到
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
            "observer_rotation_wb must have shape (N, 3, 3), got "
            f"{observer_rotation_wb.shape}"
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
        raise ValueError(
            f"bearing_vectors must have shape (3, N), got {bearing_vectors.shape}"
        )
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
        raise ValueError(
            f"Positions must have shape (3, N), got {observer_position_w.shape}"
        )
    if observer_rotation_wb.shape[0] != observer_position_w.shape[1]:
        raise ValueError(
            "observer_rotation_wb must have one rotation per timestamp, got "
            f"{observer_rotation_wb.shape[0]} rotations for {observer_position_w.shape[1]} states"
        )

    relative_w = (target_position_w - observer_position_w).T
    norms = np.linalg.norm(relative_w, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Encountered coincident observer/target positions during bearing synthesis.")

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
    bearing_b3_noiseless = rotate_bearing_sequence(
        bearing_noiseless, rotation_b3_b4
    )
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

def plot_positionw_keypoints(
    experiment_dir: Path,
    selected_points: dict[str, np.ndarray],
    output_path: Path,
    robot_ids: Iterable[str] = DEFAULT_ROBOT_IDS,
) -> str:
    """Plot global trajectories and highlighted selected keypoints in one 3D figure.

    Parameters
    ----------
    experiment_dir:
        MILUV experiment directory containing the robot `mocap.csv` files.
    selected_points:
        Mapping from robot id to selected global points. Each value must have shape
        `(3, N)` or `(N, 3)`.
    output_path:
        Output image path.
    robot_ids:
        Robot ids whose global trajectories should be drawn.
    """
    mocap_frames = load_three_robot_mocap(experiment_dir, robot_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    for robot_id, mocap_frame in mocap_frames.items():
        trajectory = mocap_frame[MOCAP_POSITION_COLUMNS].to_numpy(dtype=float)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linewidth=1.8,
            label=f"{robot_id} trajectory",
        )

    for robot_id, points in selected_points.items():
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2:
            raise ValueError(
                f"selected_points['{robot_id}'] must be a 2D array, got {points_array.shape}"
            )
        if points_array.shape[0] == 3:
            xyz = points_array
        elif points_array.shape[1] == 3:
            xyz = points_array.T
        else:
            raise ValueError(
                f"selected_points['{robot_id}'] must have shape (3, N) or (N, 3), got {points_array.shape}"
            )

        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=90,
            c="red",
            edgecolors="darkred",
            linewidths=1.8,
            depthshade=False,
            label=f"{robot_id} selected points",
            zorder=4,
        )

    ax.set_title("MILUV Global Trajectories with Selected Keypoints")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    LOGGER.info("Saved global trajectory/keypoint figure to %s", output_path)
    return str(output_path)

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
    start_point: int| None = None,
    end_point: int| None = None,
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
            raise ValueError(
                f"Edge ({observer_id}, {target_id}) references unknown robots."
            )

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
        rotation_local_world = rotation_wb[0].T
        translation_local = -rotation_local_world @ position_w[:, 0]
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
        "timestamps": timestamps,
        "states": states,
        "observations": observations,
    } 


def save_three_robot_bearing_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Save a synthesized bearing dataset to a compressed NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "experiment_dir": dataset["experiment_dir"],
        "timeshift_s": dataset["timeshift_s"],
        "anchor_id": dataset["anchor_id"],
        "robot_ids": np.array(dataset["robot_ids"], dtype=object),
        "reference_index": dataset["reference_index"],
        "timestamps": dataset["timestamps"],
    }

    for robot_id, state in dataset["states"].items():
        arrays[f"{robot_id}_position_w"] = state["position_w"]
        arrays[f"{robot_id}_quat_wb_xyzw"] = state["quat_wb_xyzw"]
        arrays[f"{robot_id}_rotation_wb"] = state["rotation_wb"]

    for edge_key, observation in dataset["observations"].items():
        prefix = edge_key.replace("<-", "__from__")
        arrays[f"{prefix}_observer_id"] = observation["observer_id"]
        arrays[f"{prefix}_target_id"] = observation["target_id"]
        arrays[f"{prefix}_bearing"] = observation["bearing"]
        arrays[f"{prefix}_bearing_noiseless"] = observation["bearing_noiseless"]
        arrays[f"{prefix}_angles"] = observation["angles"]
        arrays[f"{prefix}_angles_noiseless"] = observation["angles_noiseless"]
        arrays[f"{prefix}_bearing_b3"] = observation["bearing_b3"]
        arrays[f"{prefix}_bearing_b3_noiseless"] = observation["bearing_b3_noiseless"]
        arrays[f"{prefix}_angles_b3"] = observation["angles_b3"]
        arrays[f"{prefix}_angles_b3_noiseless"] = observation["angles_b3_noiseless"]
        arrays[f"{prefix}_bearing_b4"] = observation["bearing_b4"]
        arrays[f"{prefix}_bearing_b4_noiseless"] = observation["bearing_b4_noiseless"]
        arrays[f"{prefix}_angles_b4"] = observation["angles_b4"]
        arrays[f"{prefix}_angles_b4_noiseless"] = observation["angles_b4_noiseless"]
        arrays[f"{prefix}_rotation_b3_b4"] = observation["rotation_b3_b4"]
        arrays[f"{prefix}_reference_index"] = observation["reference_index"]
        arrays[f"{prefix}_range_w"] = observation["range_w"]

    np.savez_compressed(output_path, **arrays)


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Synthesize MILUV three-robot bearing observations.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to one MILUV experiment directory, e.g. .../data/3robot/default_3_random_0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output path.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Common synchronized sampling interval in seconds.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum number of synchronized samples.")
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Frozen B3 reference index on the synchronized timeline.",
    )
    parser.add_argument("--sigma-azimuth-deg", type=float, default=0.5)
    parser.add_argument("--sigma-elevation-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    args = build_argparser().parse_args()

    dataset = synthesize_three_robot_bearing_observations(
        experiment_dir=args.experiment_dir,
        dt=args.dt,
        max_samples=args.max_samples,
        reference_index=args.reference_index,
        sigma_azimuth_deg=args.sigma_azimuth_deg,
        sigma_elevation_deg=args.sigma_elevation_deg,
        random_seed=args.seed,
    )

    LOGGER.info(
        "Synthesized %d synchronized samples and %d directed edges from %s.",
        dataset["timestamps"].size,
        len(dataset["observations"]),
        args.experiment_dir,
    )

    if args.output is not None:
        save_three_robot_bearing_dataset(dataset, args.output)
        LOGGER.info("Saved synthesized bearing dataset to %s.", args.output)


if __name__ == "__main__":
    main()
