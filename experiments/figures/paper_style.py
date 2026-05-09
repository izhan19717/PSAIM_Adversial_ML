from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output" / "figures"
COLORBLIND_PALETTE = sns.color_palette("colorblind", 8)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.0,
            "grid.alpha": 0.25,
        }
    )


def label_panels(axes: Iterable[plt.Axes], x: float = -0.12, y: float = 1.08) -> None:
    for label, ax in zip(ascii_uppercase, axes):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )


def save_figure(fig: plt.Figure, name: str, output_dir: Path | None = None) -> None:
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{name}.{extension}", bbox_inches="tight")
