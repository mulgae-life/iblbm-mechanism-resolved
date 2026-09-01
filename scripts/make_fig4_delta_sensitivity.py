"""Fig. 4 (file: fig4_delta_sensitivity.png) — Kernel-reversal ΔC̄d sensitivity.

본문 §4 main message:
- ΔC̄d sensitivity = (C̄d_P4 − C̄d_hat) / C̄d_hat × 100% (time-averaged C̄d).
- DF / MDF (adaptive [5,20]) / DFC across Re ∈ {20, 40, 100, 200}.
- DFC가 Re=40↔100 사이 sign reversal, DF가 Re=100↔200 사이 sign reversal.
  → "DFC reverses earlier and stronger than DF."
- MDF는 |ΔC̄d| ≤ 1.15% — kernel-noise band 안.

Data source:
  data/fixed_cylinder/re*/{df,mdf,dfc}_bgk_{hat,p4}/status.json
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR


FIG_DIR = FIGURE_DIR
DATA_ROOT = PUBLIC_DATA_DIR

RE_LIST = [20, 40, 100, 200]
METHODS = [
    ("DF",                                   "df",  COLOR_BLACK, "o", "-"),
    (r"MDF (adaptive $N \in [5,20]$)",       "mdf", COLOR_RED,   "^", "-"),
    ("DFC",                                  "dfc", COLOR_BLUE,  "D", "-"),
]


def _cd50_re200(method, kernel):
    """Re=200 50% tail mean (steps 30,000–60,000) from Cd_history.

    The status.json `Cd_avg` for Re=200 uses ~30% tail (avg_start_step≈42,200).
    The 50% tail (≈9.5 vortex shedding cycles) is the audit-consistent
    averaging window adopted in main Table II / Table VI / Table B.4.
    """
    p = _status_path(200, method, kernel)
    obj = json.load(open(p))
    hist = obj.get("Cd_history") or []
    interval = obj.get("config", {}).get("check_interval", 200)
    if not hist:
        return float(obj["Cd_avg"])
    n = len(hist)
    start_step = 30000
    start_idx = max(0, int(start_step / interval) - 1)
    tail = [v for v in hist[start_idx:]
            if isinstance(v, (int, float)) and not (v != v)]
    if not tail:
        return float(obj["Cd_avg"])
    return float(np.mean(tail))


def load_cd(re, method, kernel):
    """Time-averaged Cd over converged window.

    Re ≤ 100 uses status.json `Cd_avg`; Re = 200 uses 50% tail recomputed
    from `Cd_history` to match the audit window of main Table II.
    """
    if re == 200:
        return _cd50_re200(method, kernel)
    p = _status_path(re, method, kernel)
    return float(json.load(open(p))["Cd_avg"])


def _status_path(re, method, kernel):
    return (DATA_ROOT / "fixed_cylinder" / f"re{re}"
            / f"{method}_bgk_{kernel}" / "status.json")


def delta_cd_pct(re, method):
    cd_hat = load_cd(re, method, "hat")
    cd_p4 = load_cd(re, method, "p4")
    return (cd_p4 - cd_hat) / cd_hat * 100.0


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    # 음영 (C̄d_P4 < C̄d_hat 영역, hat > P4 reversal 쪽)
    ax.axhspan(-10.0, 0, color=COLOR_BLUE, alpha=0.06, zorder=0)

    for label, key, color, marker, ls in METHODS:
        ys = [delta_cd_pct(re, key) for re in RE_LIST]
        ax.plot(RE_LIST, ys, ls=ls, color=color, lw=1.6, marker=marker,
                ms=8, mew=0.7, mec=color, mfc=color,
                label=label, zorder=4)

    # zero line — marker center가 line을 정확히 통과하도록 marker보다 낮은 zorder로 둔다.
    ax.axhline(0.0, color="#444444", lw=1.2, ls="-", alpha=0.85, zorder=2)

    def _reversal_re(key, re_lo, re_hi):
        d_lo = delta_cd_pct(re_lo, key)
        d_hi = delta_cd_pct(re_hi, key)
        # log(Re) 공간 linear interp — display 좌표상 zero-crossing 과 일치
        log_lo = np.log10(re_lo)
        log_hi = np.log10(re_hi)
        log_cross = log_lo + (log_hi - log_lo) * (0.0 - d_lo) / (d_hi - d_lo)
        return 10.0 ** log_cross

    dfc_cross = _reversal_re("dfc", 40, 100)
    ax.plot([dfc_cross], [0.0], "o", mfc="none", mec=COLOR_BLUE,
            ms=12, mew=1.8, zorder=6)
    ax.annotate(
        "DFC reversal\n(40 < Re < 100)",
        xy=(dfc_cross, 0.0),
        xytext=(22, -3.5),
        fontsize=7.5, color=COLOR_BLUE,
        ha="left", va="center", zorder=10,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                  edgecolor=COLOR_BLUE, lw=0.4, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=0.8,
                        shrinkA=4, shrinkB=12),
    )

    # DF reversal marker — explicit callout (검은 원 의미 명시)
    df_cross = _reversal_re("df", 100, 200)
    ax.plot([df_cross], [0.0], "o", mfc="none", mec=COLOR_BLACK,
            ms=14, mew=1.6, zorder=5)
    ax.annotate(
        "DF reversal\n(100 < Re < 200)",
        xy=(df_cross, 0.0),
        xytext=(130, 2.0),
        fontsize=7.5, color=COLOR_BLACK,
        ha="left", va="center", zorder=10,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                  edgecolor=COLOR_BLACK, lw=0.4, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLACK, lw=0.8,
                        shrinkA=4, shrinkB=12),
    )

    ax.set_xscale("log")
    ax.set_xticks(RE_LIST)
    ax.set_xticklabels([str(r) for r in RE_LIST])
    ax.set_xlim(15, 270)
    ax.set_ylim(-10.5, 4.5)
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"$\Delta \bar{C}_d$ sensitivity "
                  r"$(\bar{C}_{d,\,P4} - \bar{C}_{d,\,hat})"
                  r" / \bar{C}_{d,\,hat} \times 100\%$")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    out = os.path.join(FIG_DIR, "fig4_delta_sensitivity.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
