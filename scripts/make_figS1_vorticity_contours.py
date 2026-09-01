"""Supplementary §S3 Fig. S1 (file: figS1_vorticity_contours.png) — Wake vorticity comparison.

Re = 100, 200 × DF / MDF (adaptive [5,20]) / DFC, all with Peskin 4-point delta function.
6 panels (2 rows × 3 cols).

Wake-level visual comparison for fixed-cylinder cases.
The rendered PNG is bundled in figures/. Raw velocity fields are excluded
from the public subset.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save, COLOR_BLACK)
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR, require_public_input


FIG_DIR = FIGURE_DIR
DATA_BASE = PUBLIC_DATA_DIR / "raw_fields" / "fixed_cylinder"

# 본문 §2.9.1 / Table I production setup:
# Re=100: 2001 x 1601 grid, [0,1.25] x [0,1.0] domain, dx = 0.000625, D = 40dx
# Re=200: 3001 x 2401 grid, [0,1.25] x [0,1.0] domain, dx = 1/2400, D = 60dx
# cylinder_center = (0.4 L_x, 0.5 L_y).
LX, LY = 1.25, 1.0
CYL_FRAC_X, CYL_FRAC_Y = 0.4, 0.5


def load_field(re, method_dir):
    p = DATA_BASE / f"re{re}" / method_dir / "velocity_field.npz"
    d = np.load(require_public_input(p, "Fig. S1"), allow_pickle=True)
    return d["Eux"], d["Euy"], float(d["dx"]), float(d["dy"])


def compute_vorticity(u, v, dx, dy):
    """ω_z = ∂v/∂x - ∂u/∂y on Eulerian grid."""
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    return dvdx - dudy


def panel_vorticity(ax, re, method_dir, panel_label, levels, D_phys,
                    show_ylabel, show_xlabel):
    u, v, dx, dy = load_field(re, method_dir)
    ny, nx = u.shape
    omega = compute_vorticity(u, v, dx, dy)

    cx = CYL_FRAC_X * LX
    cy = CYL_FRAC_Y * LY

    half_w_lo = 3.0 * D_phys
    half_w_hi = 8.0 * D_phys
    half_h = 3.0 * D_phys
    ix_lo = max(int((cx - half_w_lo) / dx), 0)
    ix_hi = min(int((cx + half_w_hi) / dx), nx - 1)
    iy_lo = max(int((cy - half_h) / dy), 0)
    iy_hi = min(int((cy + half_h) / dy), ny - 1)

    sub = omega[iy_lo:iy_hi, ix_lo:ix_hi]
    # D-normalized 좌표 — wake range 가 Re=100/200 어느 쪽이든 -3 ~ +8 (x),
    # -3 ~ +3 (y) 동일.
    xs = (np.arange(ix_lo, ix_hi) * dx - cx) / D_phys
    ys = (np.arange(iy_lo, iy_hi) * dy - cy) / D_phys
    X, Y = np.meshgrid(xs, ys)

    # contourf (다중 level) + contour line overlay, 옅은 톤으로 통일.
    cf = ax.contourf(X, Y, sub, levels=levels, cmap="RdBu_r", extend="both",
                     alpha=0.85)
    ax.contour(X, Y, sub, levels=levels, colors="k",
               linewidths=0.2, alpha=0.25)

    # cylinder — D-normalized 좌표에서 center (0, 0), radius 0.5
    cyl = Circle((0, 0), 0.5,
                 facecolor="white", edgecolor=COLOR_BLACK,
                 linewidth=1.2, zorder=5)
    ax.add_patch(cyl)

    # 통일된 xlim/ylim/ticks (6 panel 모두 동일)
    ax.set_xlim(-3, 8)
    ax.set_ylim(-3, 3)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.set_yticks([-2, 0, 2])
    ax.set_aspect("equal", adjustable="box")

    if show_xlabel:
        ax.set_xlabel(r"$(x - x_c)/D$")
    if show_ylabel:
        ax.set_ylabel(r"$(y - y_c)/D$", labelpad=-2)

    # 코너 panel label (검정 반투명 박스 + 흰 글자)
    ax.text(0.04, 0.94, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="white",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="black",
                      ec="none", alpha=0.65))
    return cf


def main():
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.5),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.05})

    # D_phys: production §2.9.1 / Table I setup.
    # Both Re use cylinder_D_ratio = 1/40 relative to ymax = 1.0, so the physical
    # diameter is D_phys = 0.025 for both; only the grid resolution differs.
    # Re=100 → N=2001(x)×1601(y), dx=0.000625, D_lat=(1/40)×1600=40 → D_phys=0.025
    # Re=200 → N=3001(x)×2401(y), dx=1/2400,   D_lat=(1/40)×2400=60 → D_phys=0.025
    D_PHYS_RE100 = 40 * 0.000625    # 0.025
    D_PHYS_RE200 = 60 / 2400.0      # 0.025

    # 통일된 vorticity 레벨. VMAX=5 → 99th%(7)보다 약간 작아 강한 영역 일부만
    # saturate, 약한 영역은 그라데이션으로 부드럽게 표현.
    VMAX = 5.0
    levels = np.linspace(-VMAX, VMAX, 41)

    cases = [
        ("df_bgk_p4",  "a", "d"),
        ("mdf_bgk_p4", "b", "e"),
        ("dfc_bgk_p4", "c", "f"),
    ]
    cf = None
    for j, (md, lbl_top, lbl_bot) in enumerate(cases):
        cf = panel_vorticity(axes[0, j], 100, md, lbl_top, levels,
                             D_PHYS_RE100, show_ylabel=(j == 0),
                             show_xlabel=False)
        panel_vorticity(axes[1, j], 200, md, lbl_bot, levels,
                        D_PHYS_RE200, show_ylabel=(j == 0),
                        show_xlabel=True)

    # Column header (1행 panel 위)
    col_titles = ["DF", "MDF", "DFC"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=12, fontweight="bold",
                             color=COLOR_BLACK, pad=8)

    # Row label (좌측 외부, rotated)
    axes[0, 0].text(-0.32, 0.5, "Re = 100", transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight="bold", color=COLOR_BLACK,
                    ha="center", va="center", rotation=90)
    axes[1, 0].text(-0.32, 0.5, "Re = 200", transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight="bold", color=COLOR_BLACK,
                    ha="center", va="center", rotation=90)

    # 단일 colorbar (figure 우측, 라벨 "Vorticity ω")
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.025,
                        pad=0.02, shrink=0.92, aspect=28)
    cbar.set_ticks([-5, -2.5, 0, 2.5, 5])
    cbar.set_label(r"Vorticity $\omega$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    out = os.path.join(FIG_DIR, "figS1_vorticity_contours.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
