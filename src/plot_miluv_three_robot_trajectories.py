import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot")
DEFAULT_EXPERIMENT_DIRS = (
    DEFAULT_DATA_ROOT / "default_3_random_0",
    DEFAULT_DATA_ROOT / "default_3_random_0b",
    DEFAULT_DATA_ROOT / "default_3_zigzag_0",
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "figures" / "miluv_three_experiments_multiview.png"
)
DEFAULT_ROBOTS = ("ifo001", "ifo002", "ifo003")
REQUIRED_COLUMNS = (
    "timestamp",
    "pose.position.x",
    "pose.position.y",
    "pose.position.z",
)


def load_mocap_positions(robot_dir: Path) -> pd.DataFrame:
    """Load timestamped mocap positions for one robot."""
    mocap_path = robot_dir / "mocap.csv"
    if not mocap_path.exists():
        raise FileNotFoundError(f"mocap file not found: {mocap_path}")

    df = pd.read_csv(mocap_path, usecols=list(REQUIRED_COLUMNS))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {mocap_path}: {missing}")

    return df.sort_values("timestamp").reset_index(drop=True)



def _compute_axis_limits(xyz_points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = xyz_points.min(axis=0)
    maxs = xyz_points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 0.5
    return (
        (centers[0] - radius, centers[0] + radius),
        (centers[1] - radius, centers[1] + radius),
        (centers[2] - radius, centers[2] + radius),
    )



def _plot_robot_trace(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: np.ndarray) -> None:
    ax.plot(x, y, label=label, linewidth=1.5, color=color)
    ax.scatter(x[0], y[0], marker="o", color=color, s=20)
    ax.scatter(x[-1], y[-1], marker="^", color=color, s=24)



def _plot_single_experiment_row(
    axes_row: Sequence[plt.Axes],
    experiment_dir: Path,
    robot_ids: Sequence[str],
) -> None:
    traces: dict[str, pd.DataFrame] = {}
    stacked_positions: list[np.ndarray] = []

    for robot_id in robot_ids:
        robot_dir = experiment_dir / robot_id
        if not robot_dir.exists():
            raise FileNotFoundError(f"robot directory not found: {robot_dir}")
        df = load_mocap_positions(robot_dir)
        traces[robot_id] = df
        stacked_positions.append(
            df[["pose.position.x", "pose.position.y", "pose.position.z"]].to_numpy(dtype=float)
        )

    all_points = np.vstack(stacked_positions)
    (xlim, ylim, zlim) = _compute_axis_limits(all_points)
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))

    ax3d, ax_front, ax_side, ax_top = axes_row

    for color, robot_id in zip(colors, robot_ids):
        df = traces[robot_id]
        x = df["pose.position.x"].to_numpy(dtype=float)
        y = df["pose.position.y"].to_numpy(dtype=float)
        z = df["pose.position.z"].to_numpy(dtype=float)

        ax3d.plot(x, y, z, label=robot_id, linewidth=1.4, color=color)
        ax3d.scatter(x[0], y[0], z[0], marker="o", color=color, s=18)
        ax3d.scatter(x[-1], y[-1], z[-1], marker="^", color=color, s=22)

        _plot_robot_trace(ax_front, x, z, robot_id, color)
        _plot_robot_trace(ax_side, y, z, robot_id, color)
        _plot_robot_trace(ax_top, x, y, robot_id, color)

    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_title(f"{experiment_dir.name}\n3D View")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=22, azim=-58)
    ax3d.legend(loc="best", fontsize=8)

    ax_front.set_title("Front View (x-z)")
    ax_front.set_xlabel("x [m]")
    ax_front.set_ylabel("z [m]")
    ax_front.set_xlim(*xlim)
    ax_front.set_ylim(*zlim)
    ax_front.set_aspect("equal", adjustable="box")
    ax_front.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax_side.set_title("Side View (y-z)")
    ax_side.set_xlabel("y [m]")
    ax_side.set_ylabel("z [m]")
    ax_side.set_xlim(*ylim)
    ax_side.set_ylim(*zlim)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax_top.set_title("Top View (x-y)")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(*ylim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)



def plot_three_robot_trajectories(
    experiment_dirs: Sequence[Path] | Path,
    robot_ids: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot one or more three-robot MILUV experiments with 3D/front/side/top views."""
    if isinstance(experiment_dirs, Path):
        experiment_dir_list = [experiment_dirs]
    else:
        experiment_dir_list = list(experiment_dirs)

    if not experiment_dir_list:
        raise ValueError("experiment_dirs must contain at least one experiment")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(experiment_dir_list)
    fig = plt.figure(figsize=(18, 4.8 * n_rows))
    axes: list[list[plt.Axes]] = []

    for row_idx in range(n_rows):
        row_axes: list[plt.Axes] = []
        for col_idx in range(4):
            subplot_index = row_idx * 4 + col_idx + 1
            if col_idx == 0:
                ax = fig.add_subplot(n_rows, 4, subplot_index, projection="3d")
            else:
                ax = fig.add_subplot(n_rows, 4, subplot_index)
            row_axes.append(ax)
        axes.append(row_axes)

    for row_axes, experiment_dir in zip(axes, experiment_dir_list):
        _plot_single_experiment_row(row_axes, experiment_dir, robot_ids)

    fig.suptitle("MILUV Three-Robot Trajectories Across Experiments", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved trajectory figure to %s", output_path)
    return output_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize three-robot MILUV mocap trajectories.")
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_EXPERIMENT_DIRS),
        help="One or more experiment directories containing ifo001/ifo002/ifo003 subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--robots",
        nargs=3,
        default=list(DEFAULT_ROBOTS),
        help="Three robot IDs to plot.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    plot_three_robot_trajectories(args.experiment_dirs, args.robots, args.output)


if __name__ == "__main__":
    main()

