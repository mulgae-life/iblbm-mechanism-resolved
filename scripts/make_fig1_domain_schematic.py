"""Fig. 1 — Computational setup of four canonical cases.

본문 Sec. 3 / Sec. 5 case 매핑:
  (a) Fixed cylinder benchmark    — Sec. 3.1, Re ∈ {20, 40, 100, 200}
  (b) Oscillating cylinder        — Sec. 3.3, Re=100, KC=5 (Dütsch et al. [37])
  (c) Single-particle sedimentation — Sec. 5.1, ρ_r ∈ {1.01, 1.10, 1.50}
  (d) Two-particle wake-interaction sedimentation — Sec. 5.2, heavy ρ_r=1.5 + light ρ_r=1.25
                                      (Majumder [23, Sec. 4.4])

각 panel은 도메인 + 경계조건 + 입자/실린더 + 핵심 파라미터를 명시한다.

레이아웃 컨벤션:
  (a), (b) — horizontal flow setup (가로 박스, inflow / 진동 등 가로 진행)
  (c), (d) — vertical sedimentation setup (세로 채널, gravity ↓)
            * (c) Wang Fig. 3 스타일: 단일 vertical channel + 다중 시점 스냅샷
            * (d) Glowinski Fig. 8.16 스타일: 3 vertical sub-channels 가로 나열
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE, COLOR_GRAY
from _style import FIGURE_DIR


FIG_DIR = FIGURE_DIR


def _hatch_walls_vertical(ax, y0, y1, x_left, x_right, thickness=0.05):
    """Left/right no-slip wall hatched bands (vertical channel panels)."""
    ax.add_patch(Rectangle((x_left - thickness, y0), thickness, y1 - y0,
                           facecolor=COLOR_GRAY, hatch="\\\\",
                           alpha=0.4, lw=0.0, clip_on=False))
    ax.add_patch(Rectangle((x_right, y0), thickness, y1 - y0,
                           facecolor=COLOR_GRAY, hatch="\\\\",
                           alpha=0.4, lw=0.0, clip_on=False))


def panel_a(ax):
    """(a) Fixed-cylinder benchmark — Sec. 3.1."""
    ax.set_aspect("equal")
    Lx, Ly = 4.0, 2.4
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec=COLOR_BLACK, lw=1.0))
    # far-field Dirichlet (자유류 prescribed u_x=u_inf, u_y=0) 경계 라벨 —
    # 캡션·본문("free-stream far-field" / "rather than a stationary no-slip wall")과 일치.
    # solid-wall 빗금(hatching) 비표시: far-field 경계는 막힌 벽이 아니므로.
    ax.text(Lx / 2, Ly + 0.17, "free-stream far-field", fontsize=7,
            color=COLOR_GRAY, ha="center", style="italic")
    ax.text(Lx / 2, -0.18, "free-stream far-field", fontsize=7,
            color=COLOR_GRAY, ha="center", style="italic")

    # inflow arrows
    for y in np.linspace(0.30, Ly - 0.30, 6):
        ax.annotate("", xy=(0.55, y), xytext=(0.05, y),
                    arrowprops=dict(arrowstyle="->", color=COLOR_BLACK, lw=0.6))
    # U_inf inlet / Neumann outflow 텍스트 — va="top" 으로 panel 상단 라인과 0.10 띄운다
    ax.text(0.05, Ly - 0.10, r"$U_\infty$ inlet",
            fontsize=7.5, ha="left", va="top", fontweight="bold",
            color=COLOR_BLACK)
    ax.text(Lx - 0.05, Ly - 0.10, "Neumann outflow",
            fontsize=7, ha="right", va="top", style="italic",
            color=COLOR_GRAY)

    # cylinder + Lagrangian markers
    cx, cy, r = 1.4, Ly / 2, 0.30
    cyl = Circle((cx, cy), r, fill=True, fc="white", ec=COLOR_BLACK, lw=1.0)
    ax.add_patch(cyl)
    th = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    mx = cx + r * np.cos(th)
    my = cy + r * np.sin(th)
    ax.plot(mx, my, "o", ms=2.2, color=COLOR_RED, zorder=5)
    # Lagrangian marker callout: X_k 라벨
    k_idx = 1  # 우상단 마커
    ax.annotate(r"$\mathbf{X}_k$ (Lagrangian)",
                xy=(mx[k_idx], my[k_idx]),
                xytext=(mx[k_idx] + 0.40, my[k_idx] + 0.45),
                fontsize=7, color=COLOR_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.7),
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=COLOR_RED, lw=0.4))
    # Eulerian grid callout: x + h spacing.
    # 본문은 아래첨자 i 를 격자 방향 인덱스 전용으로 쓰므로, 격자점 자체를
    # 가리키는 이 라벨에서는 아래첨자를 떼고 x (Eulerian grid) 로 표기한다.
    gx0, gy0 = cx - r - 0.45, cy + 0.20
    gh = 0.10
    for ii in range(3):
        for jj in range(3):
            ax.plot(gx0 + ii * gh, gy0 + jj * gh, "+",
                    ms=4, color=COLOR_GRAY, mew=0.5, zorder=4)
    ax.annotate(r"$\mathbf{x}$ (Eulerian grid),"
                "\n"
                r"spacing $h$",
                xy=(gx0 + 2 * gh, gy0 + 2 * gh),
                xytext=(cx - 0.50, cy + r + 0.50),
                fontsize=7, color=COLOR_GRAY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=0.5),
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=COLOR_GRAY, lw=0.4))
    # D annotation
    ax.annotate("", xy=(cx + r, cy - r - 0.10),
                xytext=(cx - r, cy - r - 0.10),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLACK, lw=0.6))
    ax.text(cx, cy - r - 0.24, r"$D$", fontsize=8, ha="center")

    # wake annotation
    ax.annotate("", xy=(Lx - 0.15, cy), xytext=(cx + r + 0.05, cy),
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY,
                                lw=0.7, ls="--"))
    ax.text((cx + r + Lx) * 0.5, cy + 0.20, "wake",
            fontsize=7, color=COLOR_GRAY, ha="center", style="italic")

    # parameter box (한 줄; "fixed" 는 panel header 와 중복으로 제거)
    ax.text(0.10, 0.20,
            "Re $\\in$ {20, 40, 100, 200}",
            fontsize=7, ha="left", va="bottom", color=COLOR_BLACK,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=COLOR_BLACK, lw=0.5))

    ax.set_title("(a) Fixed cylinder benchmark",
                 fontsize=9, loc="left")
    ax.set_xlim(-0.05, Lx + 0.05)
    # ylim 위·아래 패딩 확보: free-stream far-field 텍스트(위 Ly+0.17 / 아래 -0.18)와 axes 끝 마진
    ax.set_ylim(-0.24, Ly + 0.32)
    ax.set_xticks([]); ax.set_yticks([])


def panel_b(ax):
    """(b) Oscillating cylinder — Sec. 3.3, Dütsch et al. [37]."""
    ax.set_aspect("equal")
    Lx, Ly = 4.0, 2.4
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec=COLOR_BLACK, lw=1.0))

    # (a) free-stream far-field 텍스트 y 조정과 동일 위치
    ax.text(Lx / 2, Ly + 0.17, "open boundary (all four sides)",
            fontsize=7, color=COLOR_GRAY, ha="center", style="italic")

    cx, cy, r = Lx / 2, Ly / 2, 0.32
    cyl = Circle((cx, cy), r, fill=True, fc="white", ec=COLOR_BLACK, lw=1.0)
    ax.add_patch(cyl)
    cyl_phantom = Circle((cx + 0.55, cy), r, fill=False,
                          ec=COLOR_BLACK, lw=0.6, ls=":")
    ax.add_patch(cyl_phantom)
    cyl_phantom2 = Circle((cx - 0.55, cy), r, fill=False,
                           ec=COLOR_BLACK, lw=0.6, ls=":")
    ax.add_patch(cyl_phantom2)
    # marker dot은 (a)에서만 표시한다. (b)에서는 X_b/X_k 라벨과 중복이다.
    ax.annotate("", xy=(cx + 0.85, cy + 0.50),
                xytext=(cx - 0.85, cy + 0.50),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLUE, lw=1.0))
    # 텍스트는 양방향 화살표와 시각 간격을 두고 배치
    ax.text(cx, cy + 0.78, r"$x_c(t) = -A\sin(2\pi f t)$",
            fontsize=7, color=COLOR_BLUE, ha="center", fontweight="bold")

    ax.text(cx, cy - r - 0.20, r"$D$", fontsize=8, ha="center")

    # parameter box — 우측 하단 (한 줄; Dütsch 인용은 caption 에)
    ax.text(Lx - 0.12, 0.10,
            "Re = 100,  KC = 5",
            fontsize=7, ha="right", va="bottom", color=COLOR_BLACK,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=COLOR_BLACK, lw=0.5))

    ax.set_title("(b) Oscillating cylinder",
                 fontsize=9, loc="left")
    ax.set_xlim(-0.05, Lx + 0.05)
    # (a)와 ylim 일치 → 위 row 가로 폭 동일
    ax.set_ylim(-0.24, Ly + 0.32)
    ax.set_xticks([]); ax.set_yticks([])


def panel_c(ax):
    """(c) Single-particle sedimentation — Sec. 5.1, Wang [16] / Glowinski [38].

    Vertical channel matching Wang Fig. 3 schematic convention with
    multi-snapshot rendering of the settling particle: gravity downward
    (panel-down = physical $-\\hat{e}_y$ in Glowinski/Wang frame). Five
    time snapshots shown by alpha gradient (earlier faint at top, later
    dark at bottom).
    """
    ax.set_aspect("equal")
    # Vertical channel — axes 가로 가운데 정렬 (xlim 0~4.50 중심 2.25)
    # H·g·입자·W 위치는 모두 chan_y_top 변수를 따른다.
    chan_x_left, chan_x_right = 1.75, 2.75
    chan_y_bot, chan_y_top = 0.15, 3.65
    chan_w = chan_x_right - chan_x_left
    chan_h = chan_y_top - chan_y_bot
    ax.add_patch(Rectangle((chan_x_left, chan_y_bot), chan_w, chan_h,
                            fill=False, ec=COLOR_BLACK, lw=1.0))

    wall_thick = 0.04
    _hatch_walls_vertical(ax, chan_y_bot, chan_y_top,
                          chan_x_left, chan_x_right, thickness=wall_thick)

    # W width annotation — 채널 박스 위 외부. axes top과 거리를 확보한다.
    w_ann_y = chan_y_top + 0.12
    ax.annotate("", xy=(chan_x_right, w_ann_y),
                xytext=(chan_x_left, w_ann_y),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLACK, lw=0.6))
    ax.text((chan_x_left + chan_x_right) / 2, w_ann_y + 0.05, r"$W$",
            fontsize=8, ha="center", va="bottom")

    # 채널 높이 annotation (좌측 외부). 기호는 본문 Sec. 2.9.3 의 L = 6 cm
    # (채널 높이) 를 따른다.
    h_ann_x = chan_x_left - wall_thick - 0.22
    ax.annotate("", xy=(h_ann_x, chan_y_top),
                xytext=(h_ann_x, chan_y_bot),
                arrowprops=dict(arrowstyle="<->", color=COLOR_BLACK, lw=0.6))
    ax.text(h_ann_x - 0.06, (chan_y_top + chan_y_bot) / 2, r"$L$",
            fontsize=8, ha="right", va="center")

    # 단일 입자 — 시간 진행 panel-down (alpha 옅음→진함)
    # 채널 압축에 맞춰 y 재배치 (5개 균등): top→bottom
    rp = 0.10
    cx_mid = (chan_x_left + chan_x_right) / 2
    snapshot_ys = np.linspace(chan_y_top - 0.30, chan_y_bot + 0.20, 5)
    snapshot_alphas = [0.35, 0.55, 0.75, 0.90, 1.00]
    snapshots = list(zip(snapshot_ys, snapshot_alphas))
    # 입자 색: 검정 (a,b,d 컨벤션과 통일 — (d) heavy 톤). alpha 그라데이션으로 시간 진행 표현
    for y_panel, alpha in snapshots:
        ax.add_patch(Circle((cx_mid, y_panel), rp,
                             fc=COLOR_BLACK, ec=COLOR_BLACK,
                             alpha=alpha, lw=0.6))

    # earlier / later 라벨 — H 화살표 좌측
    ax.text(h_ann_x - 0.18, snapshots[0][0] + 0.12, "earlier",
            fontsize=7.5, color=COLOR_BLACK,
            ha="right", va="bottom", style="italic")
    ax.text(h_ann_x - 0.18, snapshots[-1][0] - 0.12, "later",
            fontsize=7.5, color=COLOR_BLACK,
            ha="right", va="top", style="italic")

    # gravity 화살표 — 우측 외부, panel-down
    arrow_x = chan_x_right + wall_thick + 0.28
    ax.annotate("", xy=(arrow_x, chan_y_bot + 0.30),
                xytext=(arrow_x, chan_y_top - 0.30),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLACK, lw=1.0))
    ax.text(arrow_x + 0.10, (chan_y_top + chan_y_bot) / 2,
            r"$\mathbf{g}$",
            fontsize=10, color=COLOR_BLACK, fontweight="bold",
            va="center")

    # parameter box — 채널 아래 외부, 채널과 간격 0.15
    ax.text(cx_mid, chan_y_bot - 0.15,
            r"$\rho_r \in \{1.01, 1.10, 1.50\}$",
            fontsize=6.5, ha="center", va="top", color=COLOR_BLACK,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=COLOR_BLACK, lw=0.5))

    ax.set_title("(c) Single-particle sedimentation",
                 fontsize=9, loc="left")
    # axes ylim 위: 4.05 (chan_y_top 3.65 + W ann 0.17 + text margin 0.23)
    ax.set_xlim(0, 4.50)
    ax.set_ylim(-0.30, 4.05)
    ax.set_xticks([]); ax.set_yticks([])


def panel_d(ax):
    """(d) Two-particle wake-interaction — Sec. 5.2, Majumder et al. [23], Glowinski-style.

    Three vertical sub-channels arranged horizontally (pre-overtaking / closest approach
    / post-overtaking) following Glowinski Fig. 8.16 + Majumder Fig. 21 convention.
    Time progresses panel-right (separate sub-channels for $t_1, t_2, t_3$).
    Within each sub-channel, gravity acts panel-down. The underlying lattice
    has $g$ along $+x$ (Majumder); axis remapping per Fig. 7 caption.
    """
    ax.set_aspect("equal")

    # 3 sub-channels 가로 나열 — sub_h는 (c)의 채널 높이와 동기화
    sub_w, sub_h = 0.85, 3.05
    sub_y_bot = 0.30
    sub_y_top = sub_y_bot + sub_h
    gap = 0.30
    sub_x_lefts = [
        0.50,
        0.50 + sub_w + gap,
        0.50 + 2 * (sub_w + gap),
    ]
    phase_names = ["pre-\novertaking", "closest\napproach", "post-\novertaking"]

    rp = 0.09
    wall_thick = 0.04
    sub_w_half = sub_w / 2

    # 도식 narrative (Sec. 2.9.4 좌표 + Sec. 5.2.1/5.2.3 본문 + Fig. 7 캡션):
    #   paper 셋업: trailing=heavy at lattice (0.8, -0.13) / leading=light at (1.2, +0.13)
    #              중력이 +x 이므로 x 가 큰 light 가 앞서고 heavy 가 뒤에서 따라붙는다
    #              g = +x (lattice) → 도식 panel-down 방향
    #   도식 매핑: lattice +x → panel-down(matplotlib y 작은), y_perp → 도식 horizontal
    #              heavy x_lat=0.8 (upstream) → panel-up(y 큰); y_perp=-0.13 → 좌
    #              light x_lat=1.2 (downstream) → panel-down(y 작은); y_perp=+0.13 → 우
    # transverse offset ±0.18 = Sec. 2.9.4 실제 ±0.13 의 visual exaggeration
    # 균형: light/heavy 모두 panel 중심 ±0.18 이내 (panel boundary 와 마진 ≥ 0.13)
    # wake ellipse 제거: Glowinski Fig. 8.16 / Majumder Fig. 21 schematic 관행과 정합.
    # 메커니즘 (pre-overtaking/post-overtaking wake interaction)은 Sec. 5.2.1 본문 + Table IX (Sec. 5.2.3)
    # 입자 y 좌표는 sub_y_top 기준 top-offset 비례로 배치한다.
    cfg_per_sub = [
        # (hx_off, hy, lx_off, ly)
        # 거리 균형: pre-overtaking/post-overtaking 거리 ≈ 0.35 (직경 2.4×), closest approach 거리 0.10 (직경 0.8×, 접촉)
        # closest approach 공 2개 — 실제 schematic 에서는 두 입자가 살짝 떨어진 (접촉 직전)
        # 상태가 더 자연스러우므로 x offset ±0.10 으로 분리 (closest approach
        # 국면에서도 두 입자는 실제로 접촉하지 않는다). rp=0.09 라 거리 0.20
        # ≈ 1.1× 직경 (직경 0.18 보다 약간 큼 — 실제로 살짝 떨어진 접촉 직전).
        (-0.18, 2.81, 0.18, 2.57),   # pre-overtaking: heavy panel-up+좌, light panel-down+우
        (-0.10, 2.05, 0.10, 2.15),   # closest approach: heavy 좌+아래, light 우+위 — x 거리 0.20
        (-0.18, 1.11, 0.12, 1.36),   # post-overtaking: heavy 추월 panel-down
    ]

    for idx in range(3):
        x_left = sub_x_lefts[idx]
        ph = phase_names[idx]
        hx_off, hy, lx_off, ly = cfg_per_sub[idx]

        # Box outline
        ax.add_patch(Rectangle((x_left, sub_y_bot), sub_w, sub_h,
                                fill=False, ec=COLOR_BLACK, lw=0.8))
        # Walls
        _hatch_walls_vertical(ax, sub_y_bot, sub_y_top,
                              x_left, x_left + sub_w, thickness=wall_thick)

        cx_sub = x_left + sub_w_half
        xH = cx_sub + hx_off
        xL = cx_sub + lx_off

        # heavy 검정 (trailing)
        ax.add_patch(Circle((xH, hy), rp,
                             fc="white", ec=COLOR_BLACK, lw=1.0, zorder=3))
        # light 빨강 (leading)
        ax.add_patch(Circle((xL, ly), rp,
                             fc="white", ec=COLOR_RED, lw=1.0, zorder=3))

        # phase 라벨 (sub-channel 위; offset 0.10)
        ax.text(cx_sub, sub_y_top + 0.10, ph,
                fontsize=6.0, ha="center", va="bottom", linespacing=0.9,
                color=COLOR_BLACK, fontweight="bold")

    # heavy / light legend — (d) panel 좌하단 외부, 박스 형태
    # legend_w 1.85 (텍스트 좌우 마진), legend_h 0.40 (위·아래 텍스트 튀어나옴 방지)
    # fontsize 6.5pt ≈ 0.07 data units (axes 1.4 data/inch 기준) — entry_h 0.20 이상 필요
    legend_w = 1.85
    legend_h = 0.40
    # panel 가운데 정렬 — 상단 time arrow 와 같은 가로 중심선 (시각 대칭)
    legend_cx = (sub_x_lefts[0] + sub_x_lefts[-1] + sub_w) / 2
    legend_x0 = legend_cx - legend_w / 2
    legend_y0 = sub_y_bot - 0.10             # 0.20 (panel 아래 마진 0.10)
    legend_top = legend_y0
    legend_bot = legend_y0 - legend_h

    # background box
    ax.add_patch(Rectangle((legend_x0, legend_bot), legend_w, legend_h,
                            fill=True, fc="white", ec=COLOR_BLACK, lw=0.4))

    # entries (세로 배열): heavy 위, light 아래
    entry_h = legend_h / 2
    entry_marker_x = legend_x0 + 0.08
    entry_text_x = legend_x0 + 0.16

    heavy_y = legend_top - 0.5 * entry_h
    ax.add_patch(Circle((entry_marker_x, heavy_y), 0.035,
                         fc="white", ec=COLOR_BLACK, lw=1.0))
    ax.text(entry_text_x, heavy_y,
            r"heavy ($\rho_r = 1.5$)",
            fontsize=6.5, ha="left", va="center",
            color=COLOR_BLACK, fontweight="bold")

    light_y = legend_top - 1.5 * entry_h
    ax.add_patch(Circle((entry_marker_x, light_y), 0.035,
                         fc="white", ec=COLOR_RED, lw=1.0))
    ax.text(entry_text_x, light_y,
            r"light ($\rho_r = 1.25$)",
            fontsize=6.5, ha="left", va="center",
            color=COLOR_RED, fontweight="bold")

    # time arrow — 3 sub-channel 위, 가로 (phase label과 거리 확보; offset 0.40)
    time_y = sub_y_top + 0.40
    ax.annotate("", xy=(sub_x_lefts[-1] + sub_w, time_y),
                xytext=(sub_x_lefts[0], time_y),
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY,
                                lw=0.7, ls="--"))
    ax.text((sub_x_lefts[0] + sub_x_lefts[-1] + sub_w) / 2, time_y + 0.08,
            "time $\\to$", fontsize=7, ha="center",
            color=COLOR_GRAY, style="italic")

    # gravity arrow — panel 우측, panel-down (sub-channel 내부 gravity 방향)
    grav_x = sub_x_lefts[-1] + sub_w + wall_thick + 0.30
    ax.annotate("", xy=(grav_x, sub_y_bot + 0.30),
                xytext=(grav_x, sub_y_top - 0.30),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLACK, lw=1.0))
    ax.text(grav_x + 0.08, (sub_y_bot + sub_y_top) / 2, r"$\mathbf{g}$",
            fontsize=10, color=COLOR_BLACK, fontweight="bold",
            va="center")
    # 축 remapping 정보는 caption ("$g$ along $+x$, axis remapping per Fig. 7 caption") 으로 충분

    ax.set_title("(d) Two-particle wake-interaction sedimentation",
                 fontsize=9, loc="left")
    # (c)와 동일 ylim
    ax.set_xlim(0, 4.50)
    ax.set_ylim(-0.30, 4.05)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    apply_style()
    # 위 row (a)(b): data ratio 4.10:2.96 ≈ 1.385. 아래 row (c)(d): 4.50:5.00 = 0.90.
    # axes_w 동일 조건에서 height_ratios 자연비율 ≈ 1:1.54 (1.385/0.90).
    # figsize (7.5, 7.0): aspect="equal" 외부 padding 보상
    # height_ratios [1, 1.34]: 자연 비율 재계산 (위 row 1.385, 아래 row xlim/ylim=4.50/4.35=1.034)
    # hspace 0.08: (c)(d)를 위 row에 가깝게
    # wspace 0.08: (a)↔(b), (c)↔(d) 가로 여백 압축
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[1, 1.34],
                          hspace=0.08, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    out = os.path.join(FIG_DIR, "fig1_domain_schematic.png")
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
