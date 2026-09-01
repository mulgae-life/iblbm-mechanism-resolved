"""Fig. 5 (file: fig5_dfc_local_diagnostics.png) — DFC kernel-sensitivity local diagnostics.

Four-panel summary of the DFC kernel sensitivity at the Lagrangian-marker level
that supports the §4 correction-redistribution interpretation:

  (a) lambda_k / mean(lambda_k) vs azimuthal angle
      — correction redistribution evidence at the marker level.
  (b) marker correction-force magnitude |δf|_k.
  (c) azimuthal slip residual |u_slip| / U_inf — local no-slip residual signature.
  (d) near-boundary pressure deviation Delta p_k = (rho_k - 1) c_s^2.

Each panel overlays three cases:
  Re=40 hat    : 검 (#000000) — low-Re reference (no reversal).
  Re=100 hat   : 빨 (#C0271E) — reversal-occurring kernel.
  Re=100 P4    : 파 (#1F4E79) — non-reversal kernel.

Data source:
  data/raw_fields/fixed_cylinder/dfc_diag/{re40_hat,re100_hat,re100_p4}/
    - dfc_diag.npz   (Lx, Ly, dfc_force_lagr, lambda_k, rho, dx, dy, cylinder_*)
    - velocity_field.npz (Eux, Euy, dx, dy)

Computation:
  azimuthal angle theta_k = atan2(Ly_k - cy, Lx_k - cx) wrapped to [0, 360].
  lambda_norm_k = lambda_k / mean_k(lambda_k).
  |F_corr,k| = ||dfc_force_lagr[k]||_2.
  |u_slip,k| = ||u_fluid(Lx_k, Ly_k)||_2 (cylinder is stationary; bilinear interp).
  Delta p_k = (rho(Lx_k, Ly_k) - 1) * c_s^2 with c_s^2 = 1/3.

The rendered PNG is bundled in figures/. Raw diagnostic fields are excluded
from the public subset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, save, COLOR_BLACK, COLOR_RED, COLOR_BLUE, COLOR_GRAY
from _style import FIGURE_DIR, DATA_DIR as PUBLIC_DATA_DIR, require_public_input


DFC_DIAG_ROOT = PUBLIC_DATA_DIR / "raw_fields" / "fixed_cylinder" / "dfc_diag"
FIG_DIR = FIGURE_DIR

CASE_SPEC = [
    {"key": "re40_hat",
     "label": r"$\mathrm{Re}=40$, hat",
     "color": COLOR_BLACK, "ls": "-"},
    {"key": "re100_hat",
     "label": r"$\mathrm{Re}=100$, hat",
     "color": COLOR_RED, "ls": "-"},
    {"key": "re100_p4",
     "label": r"$\mathrm{Re}=100$, Peskin 4-point",
     "color": COLOR_BLUE, "ls": "-"},
]

U_INLET = 0.1  # lattice units; scenarios.steady 의 inflow_u
CS2 = 1.0 / 3.0


def _bilinear_at_markers(field: np.ndarray, Lx: np.ndarray, Ly: np.ndarray,
                          dx: float, dy: float) -> np.ndarray:
    """field shape (ny, nx). markers in physical units → grid index → bilinear interp."""
    ny, nx = field.shape
    ix = Lx / dx
    iy = Ly / dy
    # clip floored index first, then compute weights against the clipped index.
    # otherwise out-of-domain markers would yield weights outside [0, 1] and
    # produce silent extrapolation. cylinder markers stay well inside the
    # domain in the present datasets, but the clip ordering is the safe form.
    ix0 = np.clip(np.floor(ix).astype(np.int64), 0, nx - 2)
    iy0 = np.clip(np.floor(iy).astype(np.int64), 0, ny - 2)
    fx = np.clip(ix - ix0, 0.0, 1.0)
    fy = np.clip(iy - iy0, 0.0, 1.0)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    f00 = field[iy0, ix0]
    f10 = field[iy0, ix1]
    f01 = field[iy1, ix0]
    f11 = field[iy1, ix1]
    return ((1.0 - fx) * (1.0 - fy) * f00
            + fx * (1.0 - fy) * f10
            + (1.0 - fx) * fy * f01
            + fx * fy * f11)


def _load_diagnostics(case_dir: Path) -> dict:
    diag_path = case_dir / "dfc_diag.npz"
    vfld_path = case_dir / "velocity_field.npz"
    require_public_input(diag_path, "Fig. 5")
    require_public_input(vfld_path, "Fig. 5")

    diag = np.load(diag_path)
    vfld = np.load(vfld_path)

    Lx = diag["Lx"].astype(np.float64)
    Ly = diag["Ly"].astype(np.float64)
    cx, cy = np.asarray(diag["cylinder_center"], dtype=np.float64)
    dx = float(diag["dx"])
    dy = float(diag["dy"])

    # azimuthal angle wrapped to [0, 360]
    theta_rad = np.arctan2(Ly - cy, Lx - cx)
    theta_deg = np.rad2deg(theta_rad)
    theta_360 = np.where(theta_deg < 0, theta_deg + 360.0, theta_deg)
    sort_idx = np.argsort(theta_360)

    # (a) lambda_k / mean(lambda_k)
    lambda_k = diag["lambda_k"].astype(np.float64)
    lambda_mean = float(np.mean(lambda_k))
    lambda_norm = lambda_k / lambda_mean
    cv_pct = float(np.std(lambda_k) / lambda_mean * 100.0)

    # (b) correction magnitude
    F_mag = np.linalg.norm(diag["dfc_force_lagr"].astype(np.float64), axis=1)

    # (c) slip residual
    Eux = vfld["Eux"].astype(np.float64)
    Euy = vfld["Euy"].astype(np.float64)
    u = _bilinear_at_markers(Eux, Lx, Ly, dx, dy)
    v = _bilinear_at_markers(Euy, Lx, Ly, dx, dy)
    slip_norm = np.sqrt(u ** 2 + v ** 2) / U_INLET

    # (d) pressure deviation
    rho = diag["rho"].astype(np.float64)
    rho_at_marker = _bilinear_at_markers(rho, Lx, Ly, dx, dy)
    dp = (rho_at_marker - 1.0) * CS2

    return {
        "theta": theta_360[sort_idx],
        "lambda_norm": lambda_norm[sort_idx],
        "F_mag": F_mag[sort_idx],
        "slip_norm": slip_norm[sort_idx],
        "dp": dp[sort_idx],
        "cv_lambda_pct": cv_pct,
        "n_markers": int(lambda_k.size),
    }


def _setup_panel(ax, ylabel: str, title_letter: str, title: str,
                 reference_y: float | None = 0.0):
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    if reference_y is not None:
        ax.axhline(reference_y, color="#888888", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"azimuthal angle $\theta$ [deg]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({title_letter}) {title}", loc="left")
    ax.grid(True, alpha=0.3)


def main():
    apply_style()

    diags = {}
    for spec in CASE_SPEC:
        diags[spec["key"]] = _load_diagnostics(DFC_DIAG_ROOT / spec["key"])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    # 방위각 θ 는 주기량(θ=0 과 θ=360 이 같은 마커 이웃)이므로 순환 패딩 후
    # 평균한다. np.convolve(mode="same") 는 배열 바깥을 0으로 채워 양 끝
    # 창(window)의 평균이 0 쪽으로 끌려 내려가고, 그 결과 오버레이가 θ=0/360
    # 경계에서 데이터에 없는 급락을 만든다. 앞뒤를 서로의 반대쪽 끝값으로
    # 감싼 뒤 유효 구간만 취하면 경계에서도 연속적인 추세선이 된다.
    def _movavg(y, w=11):
        if y.size < w:
            return y
        k = np.ones(w) / w
        pad = w // 2
        y_wrapped = np.r_[y[-pad:], y, y[:pad]]
        return np.convolve(y_wrapped, k, mode="valid")

    # Panel (b)(c)(d): 게재 논문 그림과 동일한 stacking — 그리는 순서대로 (re40 → re100
    # → P4) 위로 쌓이고 lw=1.0 통일. label은 figure-level legend용으로 (b)에서만 부여.
    for spec in CASE_SPEC:
        d = diags[spec["key"]]
        kw = dict(color=spec["color"], linestyle=spec["ls"], linewidth=1.0)
        kw_avg = dict(color=spec["color"], linestyle="-",
                      linewidth=2.5, alpha=0.25, label="_nolegend_")
        axes[0, 1].plot(d["theta"], d["F_mag"], label=spec["label"], **kw)
        axes[0, 1].plot(d["theta"], _movavg(d["F_mag"]), **kw_avg)
        axes[1, 0].plot(d["theta"], d["slip_norm"], **kw)
        axes[1, 0].plot(d["theta"], _movavg(d["slip_norm"]), **kw_avg)
        axes[1, 1].plot(d["theta"], d["dp"], **kw)
        axes[1, 1].plot(d["theta"], _movavg(d["dp"]), **kw_avg)

    # Panel (a): λ_k는 marker geometry + kernel만의 함수(Tao 2019 Eq. 23,
    # iblbm/ibm/dfc.py:184)라 Re=40 hat / Re=100 hat 곡선이 bit-for-bit 동일.
    # 두 case가 모두 보이도록 phase-offset dashed pattern (검 phase=0, 빨 phase=6)
    # 으로 빨-검-빨-검 교대 표시. P4는 실선 + lw=1.8로 §4 평탄성 메시지 강조.
    A_STYLE = {
        "re40_hat":  dict(linestyle=(0, (6, 6)), linewidth=1.4, zorder=4),
        "re100_hat": dict(linestyle=(6, (6, 6)), linewidth=1.4, zorder=3),
        "re100_p4":  dict(linestyle="-",         linewidth=1.8, zorder=5),
    }
    for spec in CASE_SPEC:
        d = diags[spec["key"]]
        style = A_STYLE[spec["key"]]
        axes[0, 0].plot(d["theta"], d["lambda_norm"],
                        color=spec["color"], **style)
        axes[0, 0].plot(d["theta"], _movavg(d["lambda_norm"]),
                        color=spec["color"], linestyle="-",
                        linewidth=2.5, alpha=0.25,
                        zorder=style["zorder"] - 1)

    _setup_panel(
        axes[0, 0],
        r"$\lambda_k / \overline{\lambda}$",
        "a", r"Normalized DFC scaling factor",
        reference_y=1.0,
    )
    # Legend는 figure-level 상단 공통. (a) 패널은 hat 두 곡선이 dashed phase-offset
    # 표시라 legend handle 출처는 실선인 (b) 패널 사용.
    handles, labels = axes[0, 1].get_legend_handles_labels()

    _setup_panel(
        axes[0, 1],
        r"$|\mathbf{F}_{s,k}|$  [lattice units]",
        "b", "marker correction-force magnitude",
    )

    _setup_panel(
        axes[1, 0],
        r"$|\mathbf{u}_{\mathrm{slip}}| / U_{\infty}$",
        "c", "azimuthal slip residual",
    )

    _setup_panel(
        axes[1, 1],
        r"$\Delta p_k = (\rho_k - 1)\,c_s^2$  [lattice units]",
        "d", "near-boundary pressure deviation",
    )

    # figure-top title은 caption에서 담당한다 (PoF 양식)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.08)
    # Figure-level 공통 legend (3 cases × 1 row, 패널 외부 상단 중앙)
    # axes top 0.84 + legend y 0.965 → legend와 (a)/(b) 제목 사이 충분한 여백.
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.965), fontsize=10, frameon=True,
               facecolor="white", framealpha=1.0, edgecolor=COLOR_GRAY)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig5_dfc_local_diagnostics.png"
    save(fig, out)
    plt.close(fig)
    print(f"saved: {out}")
    print()
    print("lambda_k coefficient of variation (cross-check vs main §4.2):")
    for spec in CASE_SPEC:
        d = diags[spec["key"]]
        print(f"  {spec['key']:>10s}: CV = {d['cv_lambda_pct']:6.2f}%  "
              f"(n_markers={d['n_markers']})")


if __name__ == "__main__":
    main()
