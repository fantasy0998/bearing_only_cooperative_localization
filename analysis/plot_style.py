"""Consistent publication styles and explicit zero-error rendering."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
ZERO_FLOOR = 1e-12


def configure_style() -> None:
    """Keep fonts embedded, axes clean, and noise/method encodings stable."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.17,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )
