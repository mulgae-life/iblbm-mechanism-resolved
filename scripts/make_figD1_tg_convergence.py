"""Appendix D Fig. D.1 (file: figD1_tg_convergence.png) — Taylor-Green convergence.

Hard-coded convergence table:
- Pure LBM, DF, MDF (adaptive [5,20]), DFC L2 errors at D = 10, 20, 40, 80
- Convergence orders: 1.920, 1.988, 1.960, 2.157
- Peskin 4-point delta function only.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE,
                    COLOR_GRAY)
from _style import FIGURE_DIR


FIG_DIR = FIGURE_DIR

D = np.array([10, 20, 40, 80], dtype=float)
DATA = {
    "Pure LBM": {"L2": [1.634e-2, 4.109e-3, 9.173e-4, 3.188e-4], "order": 1.920,
                 "color": COLOR_GRAY,  "marker": "x", "ls": "-",  "lw": 1.0,
                 "ms": 5.5, "z": 2},
    "DF":       {"L2": [4.282e-2, 1.069e-2, 2.674e-3, 6.872e-4], "order": 1.988,
                 "color": COLOR_BLACK, "marker": "o", "ls": "-",  "lw": 1.2,
                 "ms": 5.5, "z": 3},
    r"MDF (adaptive $N \in [5,20]$)": {"L2": [4.348e-2, 1.077e-2, 2.696e-3, 7.453e-4], "order": 1.960,
                 "color": COLOR_RED,   "marker": "s", "ls": "--", "lw": 1.2,
                 "ms": 5.5, "z": 3},
    "DFC":      {"L2": [4.345e-2, 1.093e-2, 2.616e-3, 4.795e-4], "order": 2.157,
                 "color": COLOR_BLUE,  "marker": "^", "ls": "-",  "lw": 2.0,
                 "ms": 8.0, "z": 5},
}


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    dx = 1.0 / D  # delta x = 1/D (grid resolution per cylinder)

    for name, d in DATA.items():
        ax.loglog(dx, d["L2"], marker=d["marker"], ls=d["ls"],
                  color=d["color"], mfc=d["color"], mec=d["color"],
                  ms=d["ms"], lw=d["lw"], zorder=d["z"],
                  label=f"{name} (order {d['order']:.2f})")

    # 2nd-order reference slope guide
    ref_x = np.array([dx[0], dx[-1]])
    ref_y = ref_x ** 2 * (DATA["DF"]["L2"][0] / dx[0] ** 2) * 0.5
    ax.loglog(ref_x, ref_y, ls=(0, (4, 4)), color=COLOR_GRAY, lw=0.8,
              zorder=1, label="2nd-order reference")

    dfc_l2_80 = DATA["DFC"]["L2"][-1]
    ax.annotate(r"DFC: order 2.16",
                xy=(dx[-1], dfc_l2_80),
                xytext=(9.5e-3, 3e-3),
                fontsize=7.5, color=COLOR_BLUE, fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=COLOR_BLUE, lw=0.7),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.0,
                                shrinkA=4, shrinkB=4,
                                connectionstyle="arc3,rad=0.45"))

    ax.set_xlabel(r"$\Delta x = 1/D$  (grid spacing)")
    ax.set_ylabel(r"$L_2$ relative error")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(8e-3, 2.0e-1)
    ax.set_ylim(2e-4, 1e-1)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "figD1_tg_convergence.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
