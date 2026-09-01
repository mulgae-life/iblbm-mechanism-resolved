"""침강 런 진행 상황 플롯.

`monitor.sedimentation` 콜백에서 호출한다. 각 함수는 (output_dir, ...)를 받아
PNG를 저장한다.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _asnumpy(a):
    """CuPy 배열이면 NumPy로 변환, 아니면 그대로 반환."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _atomic_savefig(fig, path, **kwargs):
    """fig를 임시 파일에 저장 후 원자적으로 교체.

    대시보드 FileResponse가 쓰기 도중 불완전한 PNG를 서빙하는 문제 방지.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".plot_tmp_", suffix=".png", dir=directory)
    os.close(fd)
    try:
        save_kwargs = dict(kwargs)
        save_kwargs.setdefault("format", "png")
        fig.savefig(tmp, **save_kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def _auto_ylim(data, anchor=None, pad_ratio=0.2, min_range=0.05):
    """후반 50% 데이터 기반 y축 범위 계산.

    데이터가 20포인트 이상이면 후반 50%로 줌,
    그 이하면 전체 범위 사용.

    Args:
        data: 값 리스트
        anchor: 반드시 포함할 기준값 (예: target=1.60)
        pad_ratio: 범위 대비 상하 여백 비율
        min_range: 최소 y축 범위 (수렴 시 너무 좁아지는 것 방지)
    """
    if len(data) < 20:
        return None  # auto-scale 유지
    half = len(data) // 2
    recent = data[half:]
    lo, hi = min(recent), max(recent)
    if anchor is not None:
        lo = min(lo, anchor)
        hi = max(hi, anchor)
    span = max(hi - lo, min_range)
    pad = span * pad_ratio
    return (lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# 침강 전용 플롯
# ---------------------------------------------------------------------------

def _is_multiparticle_history(history):
    return history and "particles" in history[0]


def _particle_colors():
    return ["k", "r", "b", "g"]


def plot_sedimentation_velocity(output_dir, history):
    """vy*(t*) + vx*(t*) 시계열 플롯. 다입자: 입자별 선."""
    if not history:
        return

    if _is_multiparticle_history(history):
        n_p = len(history[0]["particles"])
        colors = _particle_colors()
        labels = ["heavy", "light"] if n_p == 2 else [f"p{i}" for i in range(n_p)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for pi in range(n_p):
            t = [r["particles"][pi]["t_star"] for r in history]
            vx = [r["particles"][pi]["vx_star"] for r in history]
            vy = [r["particles"][pi]["vy_star"] for r in history]
            c = colors[pi % len(colors)]
            ax1.plot(t, vx, f"{c}-", linewidth=1.0, label=labels[pi])
            ax2.plot(t, vy, f"{c}-", linewidth=1.0, label=labels[pi])
        ax1.set_ylabel("$v_x^*$ (settling)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel("$v_y^*$ (lateral)")
        ax2.set_xlabel("$t^*$")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        t = [r["t_star"] for r in history]
        vy = [r["vy_star"] for r in history]
        vx = [r["vx_star"] for r in history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(t, vy, "k-", linewidth=1.0)
        ax1.set_ylabel("$v_y^*$ (settling velocity)")
        ylim = _auto_ylim(vy)
        if ylim and all(np.isfinite(v) for v in ylim):
            ax1.set_ylim(ylim)
        ax1.grid(True, alpha=0.3)
        ax2.plot(t, vx, "b-", linewidth=1.0)
        ax2.set_ylabel("$v_x^*$ (lateral velocity)")
        ax2.set_xlabel("$t^*$")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Sedimentation velocity history")
    fig.tight_layout()
    _atomic_savefig(fig, os.path.join(output_dir, "sed_velocity.png"), dpi=100)
    plt.close(fig)


def plot_sedimentation_trajectory(output_dir, history):
    """입자 궤적 플롯. 다입자: 입자별 x(t) 궤적."""
    if not history:
        return

    if _is_multiparticle_history(history):
        n_p = len(history[0]["particles"])
        colors = _particle_colors()
        labels = ["heavy", "light"] if n_p == 2 else [f"p{i}" for i in range(n_p)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for pi in range(n_p):
            steps = [r["step"] for r in history]
            px = [r["particles"][pi]["x"] for r in history]
            py = [r["particles"][pi]["y"] for r in history]
            c = colors[pi % len(colors)]
            ax1.plot(steps, px, f"{c}-", linewidth=1.0, label=labels[pi])
            ax2.plot(steps, py, f"{c}-", linewidth=1.0, label=labels[pi])
        ax1.set_xlabel("step")
        ax1.set_ylabel("x (falling direction)")
        ax1.set_title("Position along gravity")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("step")
        ax2.set_ylabel("y (lateral)")
        ax2.set_title("Lateral position")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        t = [r["t_star"] for r in history]
        y = [r["y_star"] for r in history]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t, y, "k-", linewidth=1.2)
        ax1.set_xlabel("$t^*$")
        ax1.set_ylabel("$y^*$ (settling distance)")
        ax1.set_title("Sedimentation trajectory")
        ax1.grid(True, alpha=0.3)
        ax2 = None

    fig.tight_layout()
    _atomic_savefig(fig, os.path.join(output_dir, "sed_trajectory.png"), dpi=100)
    plt.close(fig)


def plot_sedimentation_particle(output_dir, state, frame_path=None,
                                speed_vmax=None, gravity_direction: str = "down"):
    """유동장 속도 컨투어 + 입자 위치 오버레이.

    gravity_direction="right"이면 x(낙하)→화면 아래, y(횡)→화면 좌우로 회전.
    다입자: 모든 입자를 원으로 표시 (heavy=회색, light=연회색).
    """
    ny, nx = state.ny, state.nx
    U = _asnumpy(state.U)
    speed = np.sqrt(U[:, 0]**2 + U[:, 1]**2).reshape(ny, nx)
    D_dom = 2.0 * state.r
    dx, dy = state.dx, state.dy
    x_phys = np.arange(nx) * dx
    y_phys = np.arange(ny) * dy

    gravity_right = gravity_direction == "right"

    def _single_particle_patch(center_x, center_y):
        return plt.Circle((center_x, center_y), 0.5, fill=False,
                          edgecolor="white", linewidth=1.5)

    if gravity_right:
        # 90° 회전: 화면 X=y(횡), 화면 Y=xmax-x(위→아래 낙하)
        screen_x = y_phys / D_dom
        screen_y = (x_phys[-1] - x_phys) / D_dom
        SX, SY = np.meshgrid(screen_x, screen_y)
        # speed[j,i]: j=y행, i=x열. Z[j,i]=speed at (x_phys[j], y_phys[i]) → speed.T
        speed_plot = speed.T

        fig_w = 5
        fig_h = min(fig_w * (screen_y.max() - screen_y.min()) / max(screen_x.max() - screen_x.min(), 1e-8), 24)
        fig_h = max(fig_h, 8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmax = speed_vmax if speed_vmax else max(float(np.nanmax(speed)) * 1.2, 1e-6)
        levels = np.linspace(0, vmax, 51)
        cf = ax.contourf(SX, SY, speed_plot, levels=levels, cmap="jet", extend="max")
        fig.colorbar(cf, ax=ax, label="|U|", shrink=0.4)

        if hasattr(state, 'particles') and state.particles is not None and len(state.particles) > 1:
            p_colors = ["#555555", "#aaaaaa", "#888888", "#cccccc"]
            for pi, p in enumerate(state.particles):
                pp = _asnumpy(p.pos)
                sy = (x_phys[-1] - pp[0]) / D_dom
                sx = pp[1] / D_dom
                c = plt.Circle((sx, sy), 0.5, fill=True,
                               facecolor=p_colors[pi % len(p_colors)],
                               edgecolor="white", linewidth=1.5, zorder=10)
                ax.add_patch(c)
        else:
            pos = _asnumpy(state.particle_pos) if state.particle_pos is not None else None
            if pos is not None:
                sy = (x_phys[-1] - pos[0]) / D_dom
                sx = pos[1] / D_dom
                ax.add_patch(_single_particle_patch(sx, sy))

        ax.set_aspect("equal")
        ax.set_xlabel("lateral (y/D)")
        ax.set_ylabel("falling direction (x/D, top=inlet)")
        ax.set_title("Velocity |U|")
    else:
        x_center = (x_phys[-1] + x_phys[0]) / 2.0
        x_norm = (x_phys - x_center) / D_dom
        y_norm = y_phys / D_dom
        X, Y = np.meshgrid(x_norm, y_norm)
        pos = _asnumpy(state.particle_pos) if state.particle_pos is not None else None

        domain_aspect = (y_norm[-1] - y_norm[0]) / max(x_norm[-1] - x_norm[0], 1e-8)
        fig_w = 6
        fig_h = min(fig_w * domain_aspect * 0.6, 24)
        fig_h = max(fig_h, 8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmax = speed_vmax if speed_vmax else max(float(np.nanmax(speed)) * 1.2, 1e-6)
        if not np.isfinite(vmax):
            vmax = 0.1
        levels = np.linspace(0, vmax, 51)
        cf = ax.contourf(X, Y, speed, levels=levels, cmap="jet", extend="max")
        fig.colorbar(cf, ax=ax, label="|U|")

        if pos is not None:
            circle_x = (pos[0] - x_center) / D_dom
            circle_y = pos[1] / D_dom
            ax.add_patch(_single_particle_patch(circle_x, circle_y))

        ax.set_aspect("equal")
        ax.set_title("Velocity magnitude |U|")
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")

    fig.tight_layout()
    _atomic_savefig(fig, os.path.join(output_dir, "sed_particle.png"), dpi=100)
    if frame_path is not None:
        _atomic_savefig(fig, frame_path, dpi=100)
    plt.close(fig)



def plot_sedimentation_summary(output_dir, history):
    """2x2 요약 패널. 다입자: 입자별 속도/궤적."""
    if not history:
        return

    if _is_multiparticle_history(history):
        n_p = len(history[0]["particles"])
        colors = _particle_colors()
        labels = ["heavy", "light"] if n_p == 2 else [f"p{i}" for i in range(n_p)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for pi in range(n_p):
            steps = [r["step"] for r in history]
            vx = [r["particles"][pi]["vx_star"] for r in history]
            vy = [r["particles"][pi]["vy_star"] for r in history]
            px = [r["particles"][pi]["x"] for r in history]
            py = [r["particles"][pi]["y"] for r in history]
            c = colors[pi % len(colors)]
            axes[0, 0].plot(steps, px, f"{c}-", lw=1.0, label=labels[pi])
            axes[0, 1].plot(steps, vx, f"{c}-", lw=1.0, label=labels[pi])
            axes[1, 0].plot(steps, vy, f"{c}-", lw=1.0, label=labels[pi])
            axes[1, 1].plot(py, px, f"{c}-", lw=1.0, label=labels[pi])
        axes[0, 0].set(xlabel="step", ylabel="x (falling)", title="Position")
        axes[0, 1].set(xlabel="step", ylabel="$v_x^*$ (settling)", title="Settling vel")
        axes[1, 0].set(xlabel="step", ylabel="$v_y^*$ (lateral)", title="Lateral vel")
        axes[1, 1].set(xlabel="y (lateral)", ylabel="x (falling)", title="Trajectory")
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    else:
        t = [r["t_star"] for r in history]
        y = [r["y_star"] for r in history]
        vy = [r["vy_star"] for r in history]
        vx = [r["vx_star"] for r in history]
        px = [r["x"] for r in history]
        py_pos = [r["y"] for r in history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(t, y, "k-", lw=1.0)
        axes[0, 0].set(xlabel="$t^*$", ylabel="$y^*$", title="Settling distance")
        axes[0, 1].plot(t, vy, "k-", lw=1.0)
        axes[0, 1].set(xlabel="$t^*$", ylabel="$v_y^*$", title="Terminal velocity")
        axes[1, 0].plot(t, vx, "b-", lw=1.0)
        axes[1, 0].set(xlabel="$t^*$", ylabel="$v_x^*$", title="Lateral velocity")
        axes[1, 1].plot(px, py_pos, "r-", lw=1.0)
        axes[1, 1].set(xlabel="x", ylabel="y", title="Trajectory")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

    fig.suptitle("Sedimentation Summary", fontsize=14)
    fig.tight_layout()
    _atomic_savefig(fig, os.path.join(output_dir, "sed_summary.png"), dpi=150)
    plt.close(fig)
