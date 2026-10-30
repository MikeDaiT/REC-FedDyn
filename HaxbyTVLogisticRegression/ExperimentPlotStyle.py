"""Shared Matplotlib style for every experiment figure in this project."""

from __future__ import annotations

import matplotlib as mpl


LINE_WIDTH = 3.5
FONT_SIZE = 17
AXES_LABEL_SIZE = 19
AXES_TITLE_SIZE = 22
TICK_LABEL_SIZE = 17
LEGEND_FONT_SIZE = 16
FIGURE_TITLE_SIZE = 25


def apply_experiment_plot_style() -> None:
    """Apply the project-wide, publication-readable plotting defaults."""

    mpl.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": AXES_LABEL_SIZE,
            "axes.titlesize": AXES_TITLE_SIZE,
            "axes.linewidth": 1.2,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.frameon": False,
            "figure.titlesize": FIGURE_TITLE_SIZE,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": 7.0,
        }
    )
