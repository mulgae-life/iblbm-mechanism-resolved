"""Fig. 2 (file: fig2_cd_comparison.png) — Time-averaged drag coefficient.

Data source:
  data/fixed_cylinder/re*/{df,mdf,dfc}_bgk_{hat,p4}/status.json

각 panel = 한 Re. 각 panel에 hat (왼쪽) vs Peskin 4-point (오른쪽).
3개 method (DF/MDF/DFC) × 2 kernel = 6 marker per panel.
MDF는 §2.5.2 의 adaptive [5,20] 설정 한 가지만 사용.
Gray band = literature range.

Literature ranges from paper Table II (BGK row, midpoint = (lo+hi)/2):
  Re=20 : 2.076 – 2.16
  Re=40 : 1.555 – 1.62
  Re=100: 1.364 – 1.45
  Re=200: 1.349 – 1.44
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE,
                    COLOR_GRAY)
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR


FIG_DIR = FIGURE_DIR
DATA_ROOT = PUBLIC_DATA_DIR

LIT_RANGES = {
    20: (2.076, 2.16),
    40: (1.555, 1.62),
    100: (1.364, 1.45),
    200: (1.349, 1.44),
}


def _cd50_re200(method, kernel):
    """Re=200 50% tail mean (steps 30,000–60,000) from Cd_history.

    Status.json `Cd_avg` for Re=200 uses ~30% tail; the 50% tail aligns
    with the audit window of main Table II / Table VI / Table B.4.
    """
    p = _status_path(200, method, kernel)
    obj = json.load(open(p))
    hist = obj.get("Cd_history") or []
    interval = obj.get("config", {}).get("check_interval", 200)
    if not hist:
        return float(obj["Cd_avg"])
    start_step = 30000
    start_idx = max(0, int(start_step / interval) - 1)
    tail = [v for v in hist[start_idx:]
            if isinstance(v, (int, float)) and not (v != v)]
    if not tail:
        return float(obj["Cd_avg"])
    return float(np.mean(tail))


def load_cd_avg(re, method, kernel):
    if re == 200:
        return _cd50_re200(method, kernel)
    p = _status_path(re, method, kernel)
    return float(json.load(open(p))["Cd_avg"])


def _status_path(re, method, kernel):
    return (DATA_ROOT / "fixed_cylinder" / f"re{re}"
            / f"{method}_bgk_{kernel}" / "status.json")


def get_table(re):
    """Return (hat_vals, p4_vals, lit_lo, lit_hi). hat_vals = (DF, MDF, DFC)."""
    hat = tuple(load_cd_avg(re, m, "hat") for m in ["df", "mdf", "dfc"])
    p4 = tuple(load_cd_avg(re, m, "p4") for m in ["df", "mdf", "dfc"])
    lo, hi = LIT_RANGES[re]
    return hat, p4, lo, hi


METHOD_LABELS = ["DF", "MDF", "DFC"]
METHOD_COLORS = [COLOR_BLACK, COLOR_RED, COLOR_BLUE]
METHOD_MARKER = ["o",         "^",       "D"]


def panel(ax, re_val, panel_label, global_span=None):
    hat_vals, p4_vals, lit_lo, lit_hi = get_table(re_val)

    # x positions: hat group 0..2, gap, p4 group 4..6
    x_hat = np.array([0, 1, 2])
    x_p4 = np.array([4.0, 5.0, 6.0])
    x_all = np.concatenate([x_hat, x_p4])
    vals_all = list(hat_vals) + list(p4_vals)
    colors = METHOD_COLORS * 2
    markers = METHOD_MARKER * 2

    # Lit band
    ax.axhspan(lit_lo, lit_hi, color=COLOR_GRAY, alpha=0.18,
               zorder=1, label="Literature range")

    # Bars + 상단 마커
    for x, v, c, mk in zip(x_all, vals_all, colors, markers):
        ax.bar(x, v, width=0.60, color="white", edgecolor=c, lw=1.0,
               zorder=3)
        ax.plot(x, v, marker=mk, ms=6.0, mfc=c, mec=c, zorder=5)

    # Group label (hat / P4) — axes 좌표 상단에 고정
    ax.text(1.0, 0.92, "hat", fontsize=8, ha="center", va="top",
            style="italic", color=COLOR_BLACK,
            transform=ax.get_xaxis_transform())
    ax.text(5.0, 0.92, "Peskin 4-point", fontsize=8, ha="center", va="top",
            style="italic", color=COLOR_BLACK,
            transform=ax.get_xaxis_transform())
    # Group separator (vertical dashed line)
    ax.axvline(3.0, color=COLOR_GRAY, ls=":", lw=0.6, alpha=0.7, zorder=2)

    ax.set_xticks(np.concatenate([x_hat, x_p4]))
    ax.set_xticklabels(METHOD_LABELS * 2, rotation=30, fontsize=7,
                       ha="right")
    ax.set_xlim(-1.3, 7.3)

    mid = 0.5 * (lit_lo + lit_hi)
    span = (global_span if global_span is not None
            else max(max(vals_all) - min(vals_all), lit_hi - lit_lo))
    ax.set_ylim(mid - span * 1.7, mid + span * 1.9)

    ax.set_ylabel(r"$\bar{C}_d$")
    ax.set_title(f"({panel_label}) Re = {re_val}", fontsize=9, loc="left")
    ax.tick_params(axis="x", which="minor", bottom=False)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    panels = [(20, "a"), (40, "b"), (100, "c"), (200, "d")]
    all_spans = []
    for re_val, _ in panels:
        hv, pv, lo, hi = get_table(re_val)
        all_spans.append(max(max(hv + pv) - min(hv + pv), hi - lo))
    global_span = max(all_spans)
    for ax, (re_val, lbl) in zip(axes.flat, panels):
        panel(ax, re_val, lbl, global_span=global_span)

    # Legend (top-center of figure) — method별 marker/color
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=COLOR_GRAY, alpha=0.5,
                   ms=8, label="Literature range"),
        plt.Line2D([], [], marker="o", ls="", color=COLOR_BLACK,
                   ms=6, label="DF"),
        plt.Line2D([], [], marker="^", ls="", color=COLOR_RED,
                   ms=6, label=r"MDF (adaptive $N \in [5,20]$)"),
        plt.Line2D([], [], marker="D", ls="", color=COLOR_BLUE,
                   ms=6, label="DFC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG_DIR, "fig2_cd_comparison.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
