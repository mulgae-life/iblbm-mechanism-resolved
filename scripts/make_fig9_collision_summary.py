"""Fig. 9 (file: fig9_collision_summary.png) — Collision-model summary.

Two-panel summary of the collision-model controls:
  (a) Fixed-cylinder C_d collision spread (BGK / TRT / CM-MRT) on four
      representative cases that span the 12-case spread range
      (main Table VI / Appendix B Table B.4).
  (b) Two-particle wake-interaction terminal Re_{f,max} collision spread on the
      DF / Peskin-4 / Velocity-Verlet / explicit-history reference case
      (heavy rho_r=1.5, light rho_r=1.25; main Table VI). Reported on the
      fluid-density basis Re_f = U D / nu (Majumder/Uhlmann convention),
      consistent with §5.2 / Table V.

Re=200 cells use the converged tail (steps 30,200-60,000, ~10 vortex-shedding
cycles at St≈0.20) of the 3001×2401 baseline grid (NN=2401 with xmax=1.25 /
ymax=1.0); the values match `Cd_avg_50tail` in
`data/fixed_cylinder/re200/re200_summary_50tail.json` (50% tail post-processing).
cf. main §2.9.1 Table I and Appendix A §A.1. Re=100 cells use status.json
`Cd_avg` (paper Table II matches `Cd_avg` for Re ≤ 100).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE
from _style import FIGURE_DIR


FIG_DIR = FIGURE_DIR


PANEL_A_CASES = [
    # (label, BGK, TRT, CM-MRT, spread_pct, annotation)
    ("Re=100\nDF + hat",  1.3965, 1.3834, 1.3947, 0.94, "max"),
    ("Re=100\nMDF + hat", 1.3849, 1.3720, 1.3829, 0.93, ""),
    ("Re=100\nDFC + P4",  1.4051, 1.4021, 1.4046, 0.21, ""),
    ("Re=200\nMDF + P4",  1.2982, 1.2971, 1.2984, 0.10, "min"),
]

# Re_f = Re_p / rho_r per particle (heavy: 1.5, light: 1.25)
PANEL_B = {
    "heavy": (277.60, 278.19, 277.62, 0.21, 277.81),  # BGK, TRT, CM-MRT, spread%, mean
    "light": (238.62, 238.24, 239.89, 0.69, 238.91),
}

def plot_panel_a(ax):
    n_cases = len(PANEL_A_CASES)
    x = np.arange(n_cases)
    width = 0.25

    bgk = [v[1] for v in PANEL_A_CASES]
    trt = [v[2] for v in PANEL_A_CASES]
    mrt = [v[3] for v in PANEL_A_CASES]
    spreads = [v[4] for v in PANEL_A_CASES]
    annot = [v[5] for v in PANEL_A_CASES]
    labels = [v[0] for v in PANEL_A_CASES]

    ax.bar(x - width, bgk, width, color="white", edgecolor=COLOR_BLACK,
           lw=1.2, label="BGK", zorder=3)
    ax.bar(x,         trt, width, color="white", edgecolor=COLOR_RED,
           lw=1.2, label="TRT", zorder=3, hatch="///")
    ax.bar(x + width, mrt, width, color="white", edgecolor=COLOR_BLUE,
           lw=1.2, label="CM-MRT", zorder=3, hatch="...")

    # Spread annotations above each lane group
    for i, (sp, an) in enumerate(zip(spreads, annot)):
        y_top = max(bgk[i], trt[i], mrt[i])
        tag = f"{sp:.2f}%"
        if an:
            tag += f"\n({an})"
        ax.text(i, y_top + 0.012, tag, ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\bar{C}_d$")
    ax.set_ylim(1.20, 1.50)
    ax.set_title("(a) Fixed-cylinder collision spread", fontsize=10,
                 loc="left")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)


def plot_panel_b(ax):
    bgk_h, trt_h, mrt_h, sp_h, mean_h = PANEL_B["heavy"]
    bgk_l, trt_l, mrt_l, sp_l, mean_l = PANEL_B["light"]

    x = np.arange(2)  # heavy, light
    width = 0.25

    ax.bar(x - width, [bgk_h, bgk_l], width, color="white",
           edgecolor=COLOR_BLACK, lw=1.2, label="BGK", zorder=3)
    ax.bar(x,         [trt_h, trt_l], width, color="white",
           edgecolor=COLOR_RED,   lw=1.2, label="TRT", zorder=3, hatch="///")
    ax.bar(x + width, [mrt_h, mrt_l], width, color="white",
           edgecolor=COLOR_BLUE,  lw=1.2, label="CM-MRT", zorder=3,
           hatch="...")

    # Spread annotations
    for i, (vals, sp) in enumerate(
        [((bgk_h, trt_h, mrt_h), sp_h),
         ((bgk_l, trt_l, mrt_l), sp_l)]
    ):
        y_top = max(vals)
        ax.text(i, y_top + 3, f"spread {sp:.2f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Heavy ($\\rho_r$=1.5)", "Light ($\\rho_r$=1.25)"],
                       fontsize=8)
    ax.set_ylabel(r"$\mathrm{Re}_{f,\max}$ (fluid-density basis)")
    # Tight y-limits so the collision spread is the visual focus.
    ax.set_ylim(230, 290)
    ax.set_xlim(-0.5, 1.6)
    ax.set_title("(b) wake-interaction collision spread",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5),
                              gridspec_kw={'width_ratios': [1.4, 1.0]})
    plot_panel_a(axes[0])
    plot_panel_b(axes[1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)

    out = os.path.join(FIG_DIR, "fig9_collision_summary.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
