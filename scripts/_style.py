"""Shared plotting style and repository paths for paper figures.

Output resolution is 600 dpi, matching the AIP Publishing combination-art
requirement used for the published figures.
"""

from pathlib import Path

import matplotlib as mpl


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data"


def require_public_input(path, figure_name):
    """Validate data inputs that are not bundled in the public subset."""
    path = Path(path)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{figure_name} requires '{path}'. This raw input is excluded from "
        "the public repository; use the pre-rendered PNG in figures/ or "
        "request the raw field package from the author."
    )


# Journal figure palette.
COLOR_BLACK = "#000000"
COLOR_RED = "#C0271E"
COLOR_BLUE = "#1F4E79"    # 진한 파랑
COLOR_GRAY = "#7F7F7F"


def apply_style():
    """Apply shared matplotlib settings."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 100,
        "mathtext.default": "regular",
        "mathtext.fontset": "stix",
    })


def save(fig, path, *, transparent=False):
    """Save at the standard figure resolution."""
    fig.savefig(path, dpi=600, bbox_inches="tight", transparent=transparent)
