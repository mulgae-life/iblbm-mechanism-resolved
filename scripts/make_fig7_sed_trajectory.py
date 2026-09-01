"""Fig. 7 (file: fig7_sed_trajectory.png) — overtaking trajectory and Re history.

Panels:
- (a) overtaking-trajectory phase plane showing trailing-particle
      trajectory deflecting around leading-particle wake.
- (b) Heavy/light particle terminal Re_{f,max} history; markers indicate locked
      correction-inclusive baseline (ρ_r=1.5 heavy, ρ_r=1.25 light) with
      Majumder reference (276.61 / 231.41) as horizontal dashed lines.

Data source:
  data/two_particle_sedimentation/method_matrix/
  df_bgk_verlet_explicit_history/{sedimentation_history.json,status.json}

Note:
- gravity direction = 'right' (+x); +x is the settling direction.
- For visual presentation we use vertical-down convention: plot -x as ordinate.
- Re_f,max anchor: heavy 277.60, light 238.62.
- Reference (Majumder, fluid-density basis = Re_f): 276.61 / 231.41.
The rendered PNG is bundled in figures/. The two-particle trajectory and
Reynolds-history scalar time series are bundled under data/; only the raw
velocity fields remain in the optional raw-field package.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_GRAY
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR, require_public_input


FIG_DIR = FIGURE_DIR
DATA_DIR = (PUBLIC_DATA_DIR / "two_particle_sedimentation"
            / "method_matrix" / "df_bgk_verlet_explicit_history")

REF_HEAVY = 276.61
REF_LIGHT = 231.41
LOCKED_HEAVY = 277.60
LOCKED_LIGHT = 238.62


def load():
    hist_path = require_public_input(
        os.path.join(DATA_DIR, "sedimentation_history.json"),
        "Fig. 7",
    )
    status_path = require_public_input(
        os.path.join(DATA_DIR, "status.json"),
        "Fig. 7",
    )
    with open(hist_path) as f:
        h = json.load(f)
    with open(status_path) as f:
        s = json.load(f)
    return h, s


def panel_a(ax, h):
    """Phase plane: trajectories of both particles. y vs x (settling direction)
    with x_axis = transverse position, y_axis = settling distance (positive
    downward for visual intuition).

    Trajectory data x ≈ 0.4~0.6 cm (좁은 lateral 변화). xlim을 trajectory
    영역으로 좁혀 plot 가독성 강화. 채널 벽은 caption에서 명시.
    """
    t = np.array([s["particles"][0]["t_star"] for s in h])
    x0 = np.array([s["particles"][0]["x"] for s in h])
    y0 = np.array([s["particles"][0]["y"] for s in h])
    x1 = np.array([s["particles"][1]["x"] for s in h])
    y1 = np.array([s["particles"][1]["y"] for s in h])

    gap = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    kiss_idx = int(np.argmin(gap))

    # heavy (black solid) / light (red dashed); labeled by density, since the
    # upstream/downstream roles exchange during the overtaking / role-exchange sequence
    ax.plot(y0, -x0, "-", color=COLOR_BLACK, lw=1.2,
            label=r"heavy ($\rho_r=1.5$)")
    ax.plot(y1, -x1, "--", color=COLOR_RED, lw=1.2,
            label=r"light ($\rho_r=1.25$)")

    # initial / kiss / final markers
    ax.plot(y0[0], -x0[0], "o", color=COLOR_BLACK, mfc="white", ms=6, mew=1.0)
    ax.plot(y1[0], -x1[0], "o", color=COLOR_RED, mfc="white", ms=6, mew=1.0)
    ax.plot(y0[kiss_idx], -x0[kiss_idx], "D", color=COLOR_BLACK, ms=5)
    ax.plot(y1[kiss_idx], -x1[kiss_idx], "D", color=COLOR_RED, ms=5)
    ax.plot(y0[-1], -x0[-1], "o", color=COLOR_BLACK, ms=6)
    ax.plot(y1[-1], -x1[-1], "o", color=COLOR_RED, ms=6)

    # trajectory data 범위로 xlim 좁힘 (trajectory에 집중)
    y_all = np.concatenate([y0, y1])
    y_min = float(y_all.min()) - 0.05
    y_max = float(y_all.max()) + 0.05

    phase_specs = [
        ("pre-overtaking",   y0[0],         -x0[0]),
        ("closest approach", y0[kiss_idx],  -x0[kiss_idx]),
        ("post-overtaking",  y0[-1],        -x0[-1]),
    ]
    for label_str, xy_x, xy_y in phase_specs:
        ax.annotate(label_str,
                    xy=(xy_x, xy_y), xycoords="data",
                    xytext=(1.04, xy_y),
                    textcoords=("axes fraction", "data"),
                    fontsize=7.5, color=COLOR_GRAY, style="italic",
                    fontweight="bold", va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=COLOR_GRAY, lw=0.4, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=0.5,
                                    shrinkA=3, shrinkB=3, alpha=0.75),
                    zorder=1)

    ax.set_xlabel(r"transverse position $y$")
    ax.set_ylabel(r"settling distance $-x$")
    ax.set_xlim(y_min, y_max)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02),
              fontsize=6.5, framealpha=0.92, handlelength=2, ncol=1)
    ax.set_title("(a) overtaking trajectory", fontsize=9, loc="left")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)


def panel_b(ax, h, s):
    """Re instantaneous history for heavy (id 0) and light (id 1).

    Uses raw lattice velocity sqrt(vx**2 + vy**2) consistent with monitor's
    Re_standard definition (lattice speed × D/ν). vx_star is per-particle
    normalised by its own U_g (ρ-dependent), so multiplying by config["Re"]
    (heavy U_g basis) breaks the light particle definition.
    """
    t = np.array([sn["particles"][0]["t_star"] for sn in h])
    v0_lat = np.array([np.sqrt(sn["particles"][0]["vx"] ** 2
                                + sn["particles"][0]["vy"] ** 2)
                       for sn in h])
    v1_lat = np.array([np.sqrt(sn["particles"][1]["vx"] ** 2
                                + sn["particles"][1]["vy"] ** 2)
                       for sn in h])
    # D/ν 를 격자 정의에서 직접 산출 (Sec 2.9.4: d=0.2, Δx=0.0025, τ=0.59).
    # peak 이 manuscript anchor(Table V) 및 status.json 측정값을 재현하는지
    # assert 로 교차검증한다 (LOCKED_* 는 주입값이 아니라 검증 대상이다).
    CS2 = 1.0 / 3.0
    D_LAT = 0.2 / 0.0025                 # d/Δx = 80
    NU_LAT = CS2 * (0.59 - 0.5)          # c_s²(τ-1/2) = 0.03
    D_over_nu = D_LAT / NU_LAT           # = 2666.7
    Re0 = v0_lat * D_over_nu
    Re1 = v1_lat * D_over_nu
    re_peak_status = float(s.get("Re_standard_peak", LOCKED_HEAVY))
    assert abs(Re0.max() - LOCKED_HEAVY) < 1.0, \
        f"heavy peak {Re0.max():.2f} != Table V anchor {LOCKED_HEAVY}"
    assert abs(Re1.max() - LOCKED_LIGHT) < 1.0, \
        f"light peak {Re1.max():.2f} != Table V anchor {LOCKED_LIGHT}"
    assert abs(Re0.max() - re_peak_status) < 1.0, \
        f"heavy peak {Re0.max():.2f} != status Re_standard_peak {re_peak_status:.2f}"

    # 인스턴트 Re_f(t) 곡선 — raw history (참고용; fluid-basis와 동일 표기)
    ax.plot(t, Re0, "-", color=COLOR_BLACK, lw=1.0, alpha=0.85,
            label=r"heavy $\mathrm{Re}_f(t)$ instantaneous")
    ax.plot(t, Re1, "-", color=COLOR_RED, lw=1.0, alpha=0.85,
            label=r"light $\mathrm{Re}_f(t)$ instantaneous")

    ax.axhline(REF_HEAVY, color=COLOR_BLACK, ls="--", lw=1.2, alpha=0.85)
    ax.axhline(REF_LIGHT, color=COLOR_RED,   ls="--", lw=1.2, alpha=0.85)
    t_heavy_peak_x = float(t[int(np.argmax(Re0))])
    t_light_peak_x = float(t[int(np.argmax(Re1))])
    ax.plot([t_heavy_peak_x], [LOCKED_HEAVY], marker="*",
            ms=12, color=COLOR_BLACK, mfc=COLOR_BLACK, mec=COLOR_BLACK,
            zorder=8)
    ax.plot([t_light_peak_x], [LOCKED_LIGHT], marker="*",
            ms=12, color=COLOR_RED, mfc=COLOR_RED, mec=COLOR_RED, zorder=8)

    # 두 박스 모두 패널 상단 여백에 나란히 놓고(Heavy 왼쪽 / Light 오른쪽),
    # annotate(arrowstyle="-") 로 마커 없이 얇은 리더 라인만 각자의 ★ 로 그린다.
    t_end = float(t.max())
    # Heavy 박스 — heavy ★ 바로 위 (수직 짧은 리더 라인).
    # 상단 여백의 가로 폭이 두 박스를 나란히 놓기에 딱 맞으므로, Heavy 박스는
    # 자기 별표 위에서 왼쪽을 차지하고 오른쪽은 Light 박스에 내준다.
    ax.annotate(
        f"Heavy\n  - - -  Majumder = {REF_HEAVY:.2f}\n  $\\bigstar$    peak = {LOCKED_HEAVY:.2f}",
        xy=(t_heavy_peak_x, LOCKED_HEAVY),
        xytext=(t_heavy_peak_x + 1.5, LOCKED_HEAVY + 6),
        fontsize=7, color=COLOR_BLACK, ha="center", va="bottom",
        fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec=COLOR_BLACK, lw=0.4, alpha=0.92),
        arrowprops=dict(arrowstyle="-", lw=0.5, color=COLOR_GRAY,
                        alpha=0.75, shrinkA=1, shrinkB=4),
        zorder=10)
    # Light 박스 위치는 legend/xlim/ylim 확정 후 지정하므로 annotate 호출을
    # 아래로 미룬다 (아래 참조).

    ax.set_xlabel(r"$t^* = t \, u_{g,\mathrm{heavy}} / D$")
    ax.set_ylabel(r"$\mathrm{Re}_{f}(t)$")
    h_heavy_re   = Line2D([], [], color=COLOR_BLACK, ls="-",  lw=1.0,
                          label=r"heavy $\mathrm{Re}_f(t)$ instantaneous")
    h_light_re   = Line2D([], [], color=COLOR_RED,   ls="-",  lw=1.0,
                          label=r"light $\mathrm{Re}_f(t)$ instantaneous")
    h_ref_heavy  = Line2D([], [], color=COLOR_BLACK, ls="--", lw=1.2)
    h_ref_light  = Line2D([], [], color=COLOR_RED,   ls="--", lw=1.2)
    h_peak_heavy = Line2D([], [], color=COLOR_BLACK, marker="*", ms=10,
                          linestyle="None")
    h_peak_light = Line2D([], [], color=COLOR_RED,   marker="*", ms=10,
                          linestyle="None")
    ax.legend(
        [h_heavy_re, h_light_re,
         (h_ref_heavy, h_ref_light),
         (h_peak_heavy, h_peak_light)],
        [r"heavy $\mathrm{Re}_f(t)$ instantaneous",
         r"light $\mathrm{Re}_f(t)$ instantaneous",
         r"Majumder reference (heavy / light)",
         r"present peak ($\rho_r=1.50$ / $1.25$)"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="lower right", fontsize=6.5, framealpha=0.92,
        handlelength=2.5)
    ax.set_title("(b) Particle Reynolds-number history",
                 fontsize=9, loc="left")
    ax.set_ylim(0, max(Re0.max(), Re1.max()) * 1.22)
    # 박스가 별표 옆이라 우측 마진은 1.05 로 충분하다.
    ax.set_xlim(0, float(t.max()) * 1.05)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    # Light 박스 — 패널 상단 우측의 빈 영역.
    # t* ≳ 25 구간에서 두 곡선의 최대값은 약 263 이고 축 상한은 약 339 이므로,
    # 상단 우측(Re ≈ 280–337)은 곡선·기준선·legend 어디와도 겹치지 않는다.
    # x 는 축 우측 끝 안쪽(axes fraction 0.985, ha="right"), y 는 Heavy 박스와
    # 같은 밑변(LOCKED_HEAVY + 6, va="bottom")에 맞춰 두 박스를 같은 띠에
    # 나란히 놓는다. 리더 라인은 light peak 별표로 그대로 유지한다.
    ax.annotate(
        f"Light\n  - - -  Majumder = {REF_LIGHT:.2f}\n  $\\bigstar$    peak = {LOCKED_LIGHT:.2f}",
        xy=(t_light_peak_x, LOCKED_LIGHT),
        xytext=(0.985, LOCKED_HEAVY + 6),
        textcoords=("axes fraction", "data"),
        fontsize=7, color=COLOR_RED, ha="right", va="bottom",
        fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec=COLOR_RED, lw=0.4, alpha=0.92),
        arrowprops=dict(arrowstyle="-", lw=0.5, color=COLOR_GRAY,
                        alpha=0.75, shrinkA=1, shrinkB=4),
        zorder=10)


def main():
    apply_style()
    # panel (a) trajectory와 panel (b) Re history의 영역 비를 1:1.6로 두어
    # (a)가 좁고 깊은 trajectory를 더 큰 가로 폭으로 확보.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.6]})
    h, s = load()
    panel_a(axes[0], h)
    panel_b(axes[1], h, s)
    fig.tight_layout()
    # panel (a) 우측 외부 phase 박스 컬럼 (transAxes x=1.04) — wspace 확장으로
    # phase 박스 가로폭(약 0.10 ax frac) 이 (b) panel ylabel 과 겹치지 않도록
    # wspace 를 확장한다.
    fig.subplots_adjust(left=0.08, wspace=0.58)
    out = os.path.join(FIG_DIR, "fig7_sed_trajectory.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
