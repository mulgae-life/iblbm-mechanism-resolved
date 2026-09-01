"""Fig. 3 (file: fig3_slip_error.png) — Boundary slip error.

Hard-coded table values for Re ∈ {20, 40, 100, 200}.
2 panel: (a) hat (b) Peskin 4-point. log-y, vs Re. DF / MDF (adaptive [5,20]) / DFC 3 series.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE)
from _style import FIGURE_DIR


FIG_DIR = FIGURE_DIR


# (Re, kernel, DF, MDF10, DFC)
TABLE4 = {
    "hat": {
        "Re": [20, 40, 100, 200],
        "DF":    [6.02e-3, 5.04e-3, 5.29e-3, 5.55e-3],
        "MDF10": [6.00e-4, 6.17e-4, 7.08e-4, 6.71e-4],
        "DFC":   [1.99e-3, 1.97e-3, 2.47e-3, 2.32e-3],
    },
    "p4": {
        "Re": [20, 40, 100, 200],
        "DF":    [1.23e-2, 9.99e-3, 9.73e-3, 1.04e-2],
        "MDF10": [9.88e-5, 1.25e-4, 1.76e-4, 1.53e-4],
        "DFC":   [1.39e-4, 1.72e-4, 2.75e-4, 2.55e-4],
    },
}


def panel(ax, kernel, label):
    d = TABLE4[kernel]
    Re = np.array(d["Re"], dtype=float)
    # DF: 검정 실선 + 원, MDF10: 빨강 점선 + 삼각, DFC: 파랑 대시선 + 다이아몬드 (Fig.3/5와 통일)
    ax.plot(Re, d["DF"], "-o", color=COLOR_BLACK, mfc=COLOR_BLACK,
            mec=COLOR_BLACK, ms=5.5, lw=1.2, label="DF")
    ax.plot(Re, d["MDF10"], ":^", color=COLOR_RED, mfc=COLOR_RED,
            mec=COLOR_RED, ms=5.5, lw=1.4, label=r"MDF (adaptive $N \in [5,20]$)")
    ax.plot(Re, d["DFC"], "--D", color=COLOR_BLUE, mfc=COLOR_BLUE,
            mec=COLOR_BLUE, ms=5.5, lw=1.2, label="DFC")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([20, 40, 100, 200])
    ax.set_xticklabels(["20", "40", "100", "200"])
    ax.set_xlim(15, 250)
    ax.set_ylim(5e-5, 3e-2)
    ax.set_xlabel(r"$\mathrm{Re}$")
    if kernel == "hat":
        ax.set_ylabel(r"$\varepsilon_{\mathrm{slip,mean}}$")
    title = "(a) hat" if kernel == "hat" else "(b) Peskin 4-point"
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    panel(axes[0], "hat", "(a)")
    panel(axes[1], "p4", "(b)")
    # legend를 (b) panel 중간 빈 공간 (DF top vs MDF/DFC bottom 2 dex gap)으로
    # 이동하여 데이터 라인 위 그려지지 않도록 분리. frameon + white box로 가독성 보장.
    axes[1].legend(loc="center right", frameon=True, fontsize=8,
                   facecolor="white", edgecolor="#888888", framealpha=0.95,
                   borderaxespad=0.6).set_zorder(10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_slip_error.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
