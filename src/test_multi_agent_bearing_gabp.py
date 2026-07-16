from __future__ import annotations

from pathlib import Path
import unittest
import warnings

warnings.filterwarnings("ignore", message=".*deprecated.*", module="matplotlib.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", module="pyparsing.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.multi_agent_bearing_gabp import (
    AnchorBearingFactor,
    GaBPSettings,
    PairwiseBearingFactor,
    PoseState,
    predict_anchor_angles,
    predict_pairwise_angles,
    refine_with_gabp,
    so3_exp,
    so3_log,
    wrap_to_pi,
)


class TestMultiAgentBearingGaBP(unittest.TestCase):
    def test_refinement_reduces_three_agent_bearing_cost(self) -> None:
        true_states = {
            "B": PoseState(np.eye(3), np.array([-0.3, 0.2, -0.1])),
            "C": PoseState(so3_exp(np.array([0.03, -0.02, 0.01])), np.array([-1.2, -0.4, 0.1])),
        }
        initial_states = {
            "B": PoseState(so3_exp(np.array([0.04, -0.03, 0.02])), np.array([-0.22, 0.13, -0.06])),
            "C": PoseState(
                so3_exp(np.array([-0.02, 0.04, -0.03])) @ true_states["C"].rotation,
                np.array([-1.10, -0.49, 0.15]),
            ),
        }

        b_local_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.03]),
            np.array([0.2, -0.05, 0.04]),
            np.array([0.32, -0.02, 0.02]),
            np.array([0.45, 0.08, 0.05]),
            np.array([0.60, 0.12, 0.03]),
        ]
        c_local_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.08, 0.12, 0.02]),
            np.array([-0.16, 0.22, 0.05]),
            np.array([-0.20, 0.34, 0.03]),
            np.array([-0.14, 0.48, 0.04]),
            np.array([-0.02, 0.62, 0.02]),
        ]
        anchor_points = [
            np.array([2.0, -0.1, 0.4]),
            np.array([2.1, 0.2, 0.5]),
            np.array([1.9, 0.55, 0.3]),
            np.array([2.3, 0.75, 0.45]),
            np.array([2.6, 0.8, 0.6]),
            np.array([2.8, 1.0, 0.55]),
        ]

        anchor_factors: list[AnchorBearingFactor] = []
        pairwise_factors: list[PairwiseBearingFactor] = []
        for p_a, p_b_local, p_c_local in zip(anchor_points, b_local_points, c_local_points):
            # AnchorBearingFactor 类的执行原理
            anchor_factors.append(
                AnchorBearingFactor(
                    observer_id="B",
                    anchor_position_world=p_a,
                    measured_angles=predict_anchor_angles(true_states["B"], p_a, p_b_local),
                    observer_point_local=p_b_local,
                )
            )
            anchor_factors.append(
                AnchorBearingFactor(
                    observer_id="C",
                    anchor_position_world=p_a,
                    # measured_angles是角度还是向量？是角度，predict_anchor_angles函数的输出是一个包含两个元素的数组，分别表示水平和垂直方向的角度。
                    measured_angles=predict_anchor_angles(true_states["C"], p_a, p_c_local),
                    observer_point_local=p_c_local,
                )
            )
            pairwise_factors.append(
                PairwiseBearingFactor(
                    observer_id="B",
                    target_id="C",
                    target_point_local=p_c_local,
                    measured_angles=predict_pairwise_angles(
                        true_states["B"], true_states["C"], p_b_local, p_c_local
                    ),
                    observer_point_local=p_b_local,
                )
            )
            pairwise_factors.append(
                PairwiseBearingFactor(
                    observer_id="C",
                    target_id="B",
                    target_point_local=p_b_local,
                    measured_angles=predict_pairwise_angles(
                        true_states["C"], true_states["B"], p_c_local, p_b_local
                    ),
                    observer_point_local=p_c_local,
                )
            )

        initial_cost = _bearing_cost(initial_states, anchor_factors, pairwise_factors)
        initial_rot_err, initial_trans_err = _pose_error(initial_states, true_states)
        result = refine_with_gabp(
            initial_states=initial_states,
            anchor_factors=anchor_factors,
            pairwise_factors=pairwise_factors,
            settings=GaBPSettings(
                max_gn_iterations=8,
                max_bp_iterations=50,
                message_damping=0.7,
                gn_tolerance=1e-10,
                bp_tolerance=1e-10,
                step_damping=1.0,
            ),
        )
        final_cost = _bearing_cost(result.states, anchor_factors, pairwise_factors)
        final_rot_err, final_trans_err = _pose_error(result.states, true_states)
        _save_residual_history_plot(
            [entry.residual_norm for entry in result.history],
            final_cost,
            Path("docs/figures/multi_agent_gabp_residual_history.png"),
        )

        self.assertLess(final_cost, 0.05 * initial_cost)
        self.assertLess(final_rot_err, initial_rot_err)
        self.assertLess(final_trans_err, initial_trans_err)
        self.assertGreaterEqual(len(result.history), 1)


def _bearing_cost(
    states: dict[str, PoseState],
    anchor_factors: list[AnchorBearingFactor],
    pairwise_factors: list[PairwiseBearingFactor],
) -> float:
    residuals = []
    for factor in anchor_factors:
        pred = predict_anchor_angles(
            states[factor.observer_id],
            factor.anchor_position_world,
            factor.observer_point_local,
            factor.observer_extrinsic_rotation,
            factor.observer_extrinsic_translation,
        )
        residuals.extend(
            [
                wrap_to_pi(factor.measured_angles[0] - pred[0]),
                factor.measured_angles[1] - pred[1],
            ]
        )
    for factor in pairwise_factors:
        pred = predict_pairwise_angles(
            states[factor.observer_id],
            states[factor.target_id],
            factor.observer_point_local,
            factor.target_point_local,
            factor.observer_extrinsic_rotation,
            factor.observer_extrinsic_translation,
        )
        residuals.extend(
            [
                wrap_to_pi(factor.measured_angles[0] - pred[0]),
                factor.measured_angles[1] - pred[1],
            ]
        )
    return float(np.linalg.norm(np.asarray(residuals)))


def _pose_error(
    states: dict[str, PoseState],
    true_states: dict[str, PoseState],
) -> tuple[float, float]:
    rotation_errors = []
    translation_errors = []
    for state_id, true_state in true_states.items():
        state = states[state_id]
        rotation_errors.append(np.linalg.norm(so3_log(state.rotation @ true_state.rotation.T)))
        translation_errors.append(np.linalg.norm(state.translation - true_state.translation))
    return float(np.mean(rotation_errors)), float(np.mean(translation_errors))


def _save_residual_history_plot(
    residual_history: list[float],
    final_residual: float,
    output_path: Path,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        y_values = residual_history + [final_residual]
        x_values = list(range(len(y_values)))

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(x_values, y_values, marker="o", linewidth=1.6)
        ax.set_title("GaBP Residual History")
        ax.set_xlabel("GN iteration")
        ax.set_ylabel("Bearing residual norm")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale("log")
        ax.annotate("final", xy=(x_values[-1], y_values[-1]), xytext=(5, 5), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
