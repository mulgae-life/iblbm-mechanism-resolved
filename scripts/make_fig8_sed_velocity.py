"""Fig. 8 (file: fig8_sed_velocity.png) — wake-interaction internal-mass correction ladder.

Point/range plot of heavy and light terminal Re_{f,max} for:
  (1) Majumder reference (276.61 / 231.41)  — gray reference line (dashed)
  (2) Present explicit-history baseline (277.60 / 238.62)
  (3) Without-explicit-correction ablation (270.62 / 199.52)

Caption: 각 패널의 y축은 0에서 시작하지 않는 확대 축이다. 1% 미만 차이를
분해하기 위해서다.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_GRAY
from _style import FIGURE_DIR


FIG_DIR = FIGURE_DIR

LABELS = ["Majumder\nreference",
          "Present\nexplicit-history",
          "Without explicit\ncorrection"]
HEAVY = [276.61, 277.60, 270.62]
LIGHT = [231.41, 238.62, 199.52]
# 편차 퍼센트는 본문 Table V 의 무반올림 재계산 정본값을 그대로 고정한다.
# 위 HEAVY/LIGHT 는 소수 둘째 자리로 반올림된 표시값이므로, 이 배열을 표시값
# 에서 다시 계산하지 말 것 (예: 270.62/276.61 로 계산하면 -2.17 이 나와 본문
# Table V 의 -2.16 과 어긋난다).
HEAVY_GAP_PCT = [None, +0.36, -2.16]
LIGHT_GAP_PCT = [None, +3.12, -13.78]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)

    x = np.arange(3)
    Y_LABEL = r"$\mathrm{Re}_{f,\max}$"

    OFFSETS = [(4, 12), (0, 14), (-12, 0)]
    HA      = ["left", "center", "right"]
    VA      = ["bottom", "bottom", "center"]

    # Heavy panel — point + reference horizontal line
    ax = axes[0]
    ax.axhline(HEAVY[0], color=COLOR_GRAY, ls=":", lw=0.8,
               label="Majumder reference")
    ax.plot(x, HEAVY, "o-", ms=7, color=COLOR_BLACK, lw=1.2, zorder=5)
    for i, (xi, h, gap) in enumerate(zip(x, HEAVY, HEAVY_GAP_PCT)):
        label = f"{h:.2f}" if gap is None else f"{h:.2f} ({gap:+.2f}%)"
        ax.annotate(label, xy=(xi, h),
                    xytext=OFFSETS[i], textcoords="offset points",
                    ha=HA[i], va=VA[i], fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("(a) Heavy ($\\rho_s/\\rho_f = 1.5$)",
                 fontsize=9, loc="left")
    # ylim 264-288 (range 24 = 6 × 4): 안쪽 5 tick 균등, 양 끝과 한 칸씩 떨어짐.
    ax.set_ylim(264, 288)
    ax.set_yticks([268, 272, 276, 280, 284])
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    # y-axis truncation indicator
    ax.spines["bottom"].set_visible(True)

    # Light panel
    ax = axes[1]
    ax.axhline(LIGHT[0], color=COLOR_GRAY, ls=":", lw=0.8,
               label="Majumder reference")
    ax.plot(x, LIGHT, "s-", ms=7, color=COLOR_RED, lw=1.2, zorder=5)
    for i, (xi, h, gap) in enumerate(zip(x, LIGHT, LIGHT_GAP_PCT)):
        label = f"{h:.2f}" if gap is None else f"{h:.2f} ({gap:+.2f}%)"
        ax.annotate(label, xy=(xi, h),
                    xytext=OFFSETS[i], textcoords="offset points",
                    ha=HA[i], va=VA[i], fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("(b) Light ($\\rho_s/\\rho_f = 1.25$)",
                 fontsize=9, loc="left")
    # ylim 190-250 (range 60 = 6 × 10): 안쪽 5 tick 균등, 양 끝과 한 칸씩 떨어짐.
    ax.set_ylim(190, 250)
    ax.set_yticks([200, 210, 220, 230, 240])
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig8_sed_velocity.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
