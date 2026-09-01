"""Fig. 6 (file: fig6_sedimentation_summary.png) — Single-particle sedimentation summary.

Panels:
- (a) Seven evenly spaced particle positions for all three density ratios in
      channel frame; darker circles = later times. Centerline.
- (b)(c)(d) Particle-centred streamlines near terminal state for
      ρ_s/ρ_f = 1.01, 1.1, 1.5 respectively.

Data source:
  data/raw_fields/single_particle_sedimentation/baseline/{rho101,rho110,rho150}/
  - sedimentation_history.json (positions over time)
  - velocity_field.npz (terminal state velocity field)

The rendered PNG is bundled in figures/. Raw velocity fields are excluded
from the public subset.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE,
                    COLOR_GRAY)
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR, require_public_input


FIG_DIR = FIGURE_DIR
DATA_DIR = PUBLIC_DATA_DIR / "raw_fields" / "single_particle_sedimentation" / "baseline"

# domain Ly ≈ 3 cm, Lx ≈ 1 cm, particle D not directly needed


def load_history(rho_dir):
    path = require_public_input(
        os.path.join(DATA_DIR, rho_dir, "sedimentation_history.json"),
        "Fig. 6",
    )
    with open(path) as f:
        h = json.load(f)
    return h


def load_velocity(rho_dir):
    path = require_public_input(
        os.path.join(DATA_DIR, rho_dir, "velocity_field.npz"),
        "Fig. 6",
    )
    return np.load(path, allow_pickle=True)


def panel_a(ax, row_label_x=0.06, vy_label_x=0.895):
    """Seven evenly-spaced positions, three density ratios in a horizontal
    channel (rotated for landscape layout).

    채널은 직사각형 outline 대신 top/bottom 벽만 axhline으로 표시한다. row
    label 박스와 terminal-velocity 박스는 axes-fraction 좌표로 axes 외부에
    배치하고, 시간 화살표("earlier"→"later")도 axes 상단 외부(transAxes
    y>1.0)에 둔다.

    row_label_x / vy_label_x 는 main()에서 (b)/(d) axes spine 의 figure 좌표를
    측정해서 (a) transAxes 로 변환한 값을 받는다 (정확한 열 정렬).
    """
    # data source(rho 디렉토리)는 라벨과 정합한다: rho101→1.01, rho110→1.10,
    # rho150→1.50 (status.json 및 Table IV 기준; raw |vy|_max 가 rho101<rho110
    # <rho150 이므로 디렉토리명이 실제 density 와 일치). panel (b)-(d) streamline
    # 매핑과 동일 순서로 정렬한다.
    rho_list = [
        ("rho101", r"$\rho_s/\rho_f = 1.01$", COLOR_BLUE,  0.82, "lightest", 0.88),
        ("rho110", r"$\rho_s/\rho_f = 1.10$", COLOR_RED,   0.50, None,       1.10),
        ("rho150", r"$\rho_s/\rho_f = 1.50$", COLOR_BLACK, 0.18, "heaviest", 1.22),
    ]

    chan_x_lo = 0.0
    chan_x_hi = 1.0
    Ly_chan = 1.0
    n_samp = 7
    ROW_LABEL_X = row_label_x
    VY_LABEL_X = vy_label_x
    dots_lo = 0.14
    dots_hi = 0.74
    inner_margin = 0.01

    # 채널 벽 (top/bottom)
    ax.axhline(0.0, color=COLOR_BLACK, lw=1.0, zorder=2)
    ax.axhline(Ly_chan, color=COLOR_BLACK, lw=1.0, zorder=2)
    for spine_loc in ("left", "right", "top"):
        ax.spines[spine_loc].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for rho_dir, label, color, y_off, weight_tag, vy_term in rho_list:
        h = load_history(rho_dir)
        y_all = np.array([s["y"] for s in h])
        idx_samp = np.linspace(0, len(h) - 1, n_samp).astype(int)
        for i, idx in enumerate(idx_samp):
            x_disp = (dots_lo + inner_margin
                      + (dots_hi - dots_lo - 2 * inner_margin) * (
                          1.0 - y_all[idx] / 2.0))
            frac = i / (n_samp - 1)
            alpha = 0.30 + 0.70 * frac
            sz = 25 + 45 * frac          # dots 크기 (axes 폭 넓음)
            ax.scatter(x_disp, y_off, s=sz, c=[color],
                       alpha=alpha, edgecolors=color, linewidths=0.5,
                       zorder=3)
        # row label — (b) ylabel 위치 정렬 (transAxes x≈-0.025, ha="left")
        short_label = label.replace(r"\rho_s/\rho_f", r"\rho_r")
        ax.text(ROW_LABEL_X, y_off, short_label, fontsize=7.0, color=color,
                va="center", ha="left", fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor=color, lw=0.5))
        # |v_y*| 박스 — (d) 우측 spine 정렬 (transAxes x=1.0, ha="right")
        vy_text = rf"$|v_y^*|{{\approx}}{vy_term:.2f}$"
        if weight_tag is not None:
            vy_text = vy_text + f" ({weight_tag})"
        ax.text(VY_LABEL_X, y_off, vy_text,
                fontsize=8.0, color=color, va="center", ha="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec=color, lw=0.5, alpha=0.92))

    # 시간 화살표 — axes 상단 외부 (transAxes y>1), dots 영역 위쪽으로
    arrow_y = 1.10
    arrow_x_lo = dots_lo + inner_margin + 0.02
    arrow_x_hi = dots_hi - inner_margin - 0.02
    ax.annotate("", xy=(arrow_x_hi, arrow_y),
                xytext=(arrow_x_lo, arrow_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(arrow_x_lo + 0.02, arrow_y + 0.04, "earlier",
            fontsize=6.5, color=COLOR_GRAY, ha="left", va="bottom",
            style="italic", transform=ax.transAxes)
    ax.text(arrow_x_hi - 0.02, arrow_y + 0.04, "later",
            fontsize=6.5, color=COLOR_GRAY, ha="right", va="bottom",
            style="italic", transform=ax.transAxes)

    ax.set_xlim(chan_x_lo, chan_x_hi)
    ax.set_ylim(0, Ly_chan)
    ax.set_aspect("auto")
    ax.set_xticks([])
    # yticks/ylabel 제거 — row label 박스가 좌측 외부에서 채널 wall (0~1 cm)
    # 정보를 충분히 표현. yticklabel 잔존 시 row label과 시각 충돌.
    ax.set_yticks([])
    ax.set_ylabel("")
    # title pad — earlier/later 화살표(arrow_y≈1.18+0.04) 와 거리 확보
    ax.set_title("(a) Particle trajectories (time-ordered snapshots)",
                 fontsize=9, loc="left", pad=24)


def panel_streamlines(ax, rho_dir, panel_label, title_text, show_ylabel=True,
                       ref_speed=None):
    """Particle-centred streamlines at terminal state.

    Magnitude shaded with a sequential gray ramp; arrows kept as solid black
    streamlines. Shading lifts the contrast between weak/strong wake regimes
    visible in (b)–(d).
    """
    d = load_velocity(rho_dir)
    Eux = d["Eux"]; Euy = d["Euy"]
    dx = float(d["dx"]); dy = float(d["dy"])
    px, py = d["particle_pos"]
    Ny, Nx = Eux.shape

    h = load_history(rho_dir)
    vy_term = np.mean([s["vy"] for s in h[-10:]])
    Euy_pf = Euy - vy_term

    # 단일입자 code-space 직경 D=0.125 (confinement d/W=0.125; particle_pos.y
    # ≈0.0625=입자 반지름이므로 overlay circle 반지름 0.5*D=0.0625 가 실제 입자
    # 와 일치).
    D = 0.125
    half_w = 3.0 * D
    # 입자 위쪽 wake 강조 + 입자 아래 channel bottom까지만 (흰 영역 제거)
    half_h_up = 4.3 * D
    half_h_dn = min(0.7 * D, py)   # py ≈ 0.06 cm 이라 ~0.62 D
    ix_lo = max(int((px - half_w) / dx), 0)
    ix_hi = min(int((px + half_w) / dx), Nx - 1)
    iy_lo = max(int((py - half_h_dn) / dy), 0)
    iy_hi = min(int((py + half_h_up) / dy), Ny - 1)

    step = max(1, (iy_hi - iy_lo) // 130)
    ys = np.arange(iy_lo, iy_hi, step) * dy - py
    xs = np.arange(ix_lo, ix_hi, step) * dx - px
    Ux = Eux[iy_lo:iy_hi:step, ix_lo:ix_hi:step]
    Uy = Euy_pf[iy_lo:iy_hi:step, ix_lo:ix_hi:step]
    X, Y = np.meshgrid(xs / D, ys / D)

    # 이 그림의 주장은 streamline topology 의 regime 진행(Stokes-like →
    # recirculation → symmetric wake)이므로 배경 shading 대신 streamline
    # linewidth 를 속도에 비례시켜 wake 강도 차이를 선폭으로 전달한다.
    # 공통 스케일 (ref_speed) 로 normalize 하므로 panel 간 wake 강도 비교
    # 가능: ρ_r=1.01 은 전반적으로 가는 선, ρ_r=1.50 은 wake 부근에서 두꺼움.
    speed = np.sqrt(Ux ** 2 + Uy ** 2)
    ref = ref_speed if ref_speed is not None else speed.max()
    speed_norm = np.clip(speed / max(float(ref), 1e-12), 0.0, 1.0)
    # streamplot 의 linewidth 는 X/Y grid 와 같은 shape 의 2D 배열 허용.
    # 0.30 (가는 선) ~ 0.75 (적당히 굵은 선) 좁은 범위 — 잉크 양 절제.
    lw_field = 0.30 + 0.45 * speed_norm
    ax.streamplot(X, Y, Ux, Uy, color=COLOR_BLACK, density=1.2,
                  linewidth=lw_field, arrowsize=0.8)
    p_circ = Circle((0, 0), 0.5, facecolor="white",
                     edgecolor=COLOR_BLACK, lw=1.2, zorder=4)
    ax.add_patch(p_circ)
    # channel bottom wall hatch (입자 아래) — 텍스트는 caption에서 명시
    y_bot = -py / D
    ax.axhspan(-half_h_dn / D - 0.05, y_bot, facecolor=COLOR_GRAY,
               alpha=0.30, hatch="///", edgecolor=COLOR_GRAY, lw=0.0,
               zorder=2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-half_h_dn / D - 0.05, 4.3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$(x - x_p)/D$")
    if show_ylabel:
        ax.set_ylabel(r"$(y - y_p)/D$")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title(f"({panel_label}) {title_text}",
                 fontsize=9, loc="left")


def main():
    apply_style()
    fig = plt.figure(figsize=(6.6, 4.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.40, 1.7],
                          hspace=-0.10, wspace=0.10)
    ax_a = fig.add_subplot(gs[0, :])  # 전체 3 col — (a) 폭 = (b)+(c)+(d) 폭
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)
    ax_d = fig.add_subplot(gs[1, 2], sharey=ax_b)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    plt.setp(ax_d.get_yticklabels(), visible=False)

    # 공통 정규화 기준 — 3개 panel 의 95-percentile 중 최대값.
    # max 대신 q95 를 쓰는 이유: ρ_r=1.50 panel 에 입자 직후 매우 좁은 영역의
    # spike 가 있어 그것을 reference 로 잡으면 세 panel 모두 mid-tone 영역이
    # 압축되어 평탄해 보임. q95 는 wake 본체를 reference 로 잡아 광범위한
    # 명암 분포가 보존됨. ρ_r=1.01 panel 의 q95 가 ρ_r=1.50 의 ~70% 수준이라
    # (b) 는 자연히 약간 더 light 하게 표현되어 침강 약화의 물리적 의미가
    # 시각에 반영됨.
    def _q95(rho_dir):
        dd = load_velocity(rho_dir)
        h = load_history(rho_dir)
        vy_term = np.mean([s["vy"] for s in h[-10:]])
        sp = np.sqrt(dd["Eux"] ** 2 + (dd["Euy"] - vy_term) ** 2)
        return float(np.quantile(sp, 0.95))
    ref_global = max(_q95("rho101"), _q95("rho110"), _q95("rho150"))

    # streamline
    panel_streamlines(ax_b, "rho101", "b",
                      r"$\rho_s/\rho_f = 1.01$", show_ylabel=True,
                      ref_speed=ref_global)
    panel_streamlines(ax_c, "rho110", "c",
                      r"$\rho_s/\rho_f = 1.10$", show_ylabel=False,
                      ref_speed=ref_global)
    panel_streamlines(ax_d, "rho150", "d",
                      r"$\rho_s/\rho_f = 1.50$", show_ylabel=False,
                      ref_speed=ref_global)

    # axes 위치 측정 — colorbar 적용 후 최종 좌표
    fig.canvas.draw()
    a_pos = ax_a.get_position()
    b_pos = ax_b.get_position()
    d_pos = ax_d.get_position()
    # (b) y-spine x → (a) transAxes (좌측 박스 좌측 끝 정렬)
    b_spine_in_a = (b_pos.x0 - a_pos.x0) / a_pos.width
    # 박스 좌측 padding 보정 (boxstyle pad → transAxes 약 0.012)
    row_label_x = b_spine_in_a + 0.012
    # 우측 박스 ha="center" 중심 x = (d) plot 우측 spine - 박스 반폭.
    # 박스 우측 끝이 (d) 우측 spine 가까이, 박스가 dots 영역과 안 겹치도록.
    d_right_in_a = (d_pos.x1 - a_pos.x0) / a_pos.width
    vy_label_x = d_right_in_a - 0.14    # 박스 반폭 보정 + 좌측 shift -0.04

    panel_a(ax_a, row_label_x=row_label_x, vy_label_x=vy_label_x)

    out = os.path.join(FIG_DIR, "fig6_sedimentation_summary.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
