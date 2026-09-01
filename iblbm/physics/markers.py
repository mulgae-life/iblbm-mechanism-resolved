"""라그랑주 마커 기하(geometry) helper.

구성
  - `update_markers`          : 강체 변환 `X = R(θ)·(X_c − X_0) + x_p`
  - `compute_desired_velocity`: 경계 속도 `u_b = U_p + ω × (X − x_p)`
  - `create_circle_markers`   : 원형 마커 생성 + arc length Δs ≈ 2πR/L_b

retraction 규약
    Breugem (2012) 기반, 실효 직경을 맞추기 위해 `R_ret = R − r_d`만큼 내측으로 후퇴

        ┌───── r_d ─────┐
        │                ●   ← 실제 geometry surface
        │              ╱
        │            ╱
        ●         ╱           ← Lagrangian marker (retracted)
      X_l    X_l + r_d·n̂
"""
from __future__ import annotations

import numpy as np


def update_markers(Lx_c, Ly_c, cx_init, cy_init, pos_new, angle: float = 0.0):
    """강체 변환으로 마커 좌표 갱신.

    `X = R(θ) · (X_c − X_0) + x_p`
        - `X_c = (Lx_c, Ly_c)` 초기 마커 좌표
        - `X_0 = (cx_init, cy_init)` 초기 중심
        - `x_p = pos_new` 현재 중심 위치
        - `R(θ)` 2D 평면 회전
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rx = Lx_c - cx_init
    ry = Ly_c - cy_init
    Lx = cos_a * rx - sin_a * ry + pos_new[0]
    Ly = sin_a * rx + cos_a * ry + pos_new[1]
    return Lx, Ly


def compute_desired_velocity(vel, Lb: int, omega: float = 0.0, Lx=None, Ly=None, cx: float = 0.0, cy: float = 0.0, dx: float = 1.0):
    """경계 속도 `u_b = U_p + ω × r` (r = X − x_c) 생성.

    - `omega = 0` 또는 `Lx is None` 이면 병진 성분만 적용
    - 회전 성분 `ω × r`은 `dx`로 나누어 격자 단위로 환산
    """
    _xp = np
    if Lx is not None and type(Lx).__module__.startswith("cupy"):
        import cupy
        _xp = cupy
    desired_vel = _xp.empty((Lb, 2))
    desired_vel[:, 0] = float(vel[0])
    desired_vel[:, 1] = float(vel[1])
    if omega != 0.0 and Lx is not None:
        desired_vel[:, 0] += -omega * (Ly - cy) / dx
        desired_vel[:, 1] += omega * (Lx - cx) / dx
    return desired_vel


def create_circle_markers(cx: float, cy: float, r_domain: float, lattice_r: float, dx: float, spacing_factor: float, retraction_dx: float):
    """원형 경계 상의 Lagrangian marker set 생성.

    기하
      - arc length    Δs ≈ 2πR_lattice / L_b          (격자 단위)
      - retraction    R_ret = R_domain − r_d · Δx     (Breugem 2012 retraction)
      - marker 수     L_b ≈ 2πR_lattice / spacing_factor

    Returns
      `(Lx, Ly, Lb, Larea)` — 좌표, 마커 수, arc length Δs
    """
    dd = 2.0 * np.pi * lattice_r / spacing_factor
    step = 1.0 / dd
    Ldx = np.arange(0, 1.0 + step * 0.5, step)
    Ldx = Ldx[Ldx <= 1.0 + 1e-12]
    if len(Ldx) > 1 and np.abs(Ldx[-1] - 1.0) < step * 0.5:
        Ldx = Ldx[:-1]

    theta = 2.0 * np.pi * Ldx
    r_retracted = r_domain - retraction_dx * dx
    Lx = r_retracted * np.cos(theta) + cx
    Ly = r_retracted * np.sin(theta) + cy
    Lb = len(Lx)
    lattice_r_retracted = lattice_r - retraction_dx
    Larea = 2.0 * np.pi * lattice_r_retracted / Lb
    return Lx, Ly, Lb, Larea
