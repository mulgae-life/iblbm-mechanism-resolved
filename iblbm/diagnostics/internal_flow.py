"""침강 / 입자 내부 fictitious fluid 진단.

핵심 지표
  - 내부 잔류 유동
        ||ũ||_inside = {|u(x, y)| : signed_distance(x, y) < −κ Δx}
        · signed distance < −κ Δx: delta smearing 영역 제외한 깊은 내부 (κ ≥ 2)

  - 내부 선형 운동량
        P_int = ∫_{Ω_p} ρ u dV                           (프로젝트 고유 집계)

  - 내부 각운동량
        L_int,z = ∫_{Ω_p} (r × ρ u) · ẑ dV              (2D에서 z 성분만 존재)

침강 무차원 기준 속도 / 시간
  - u_g = √(|ρ_s/ρ_f − 1| g D)
  - t*  = t u_g / D
  - y*  = (y_0 − y) / D
  - v*  = −v_y / u_g,  u* = v_x / u_g

배열 레이아웃
  - `compute_*`  : 2D `(ny, nx)` 입력
  - `compute_l_int_z`: flat `(N, 2)`, `(N,)` solver/runtime 호환 wrapper

용도
  - 침강 후처리
  - runtime logging (각운동량 누적, inertia 진단)
"""
from __future__ import annotations

import numpy as np

def compute_inside_residual(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float,
    kappa: float = 2.0,
    u_ref: float = 0.1,
) -> dict:
    """고체 내부 잔류 유동 ||ũ||_inside 집계.

    식
      - signed distance d(x, y) = √((x−cx)² + (y−cy)²) − r
      - 마스크 d < −κ Δx  (κ ≥ 2로 delta smearing 영역 제외)
      - |u| = √(u_x² + u_y²)

    프로젝트 규약
      - κ buffer로 경계 근방 스미어 효과 배제는 현재 진단 규칙

    Args
      - Eux, Euy     2D 속도장 (ny, nx), lattice 단위
      - dx, dy       격자 간격 (도메인)
      - cx, cy       실린더 중심 (도메인)
      - D            실린더 직경 (도메인)
      - kappa        경계 버퍼 (lattice 단위, 기본 2.0)
      - u_ref        참조 속도 (lattice 단위, 기본 0.1)

    Returns
      - dict: `{'mean', 'max', 'n_points', 'mean_normalized', 'max_normalized'}`
    """
    ny, nx = Eux.shape
    r = D / 2.0

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    XX, YY = np.meshgrid(x, y)

    dist = np.sqrt((XX - cx)**2 + (YY - cy)**2) - r
    mask = dist < -kappa * dx

    n_pts = int(np.sum(mask))
    if n_pts == 0:
        return {
            'mean': 0.0, 'max': 0.0, 'n_points': 0,
            'mean_normalized': 0.0, 'max_normalized': 0.0,
        }

    u_mag = np.sqrt(Eux[mask]**2 + Euy[mask]**2)

    return {
        'mean': float(np.mean(u_mag)),
        'max': float(np.max(u_mag)),
        'n_points': n_pts,
        'mean_normalized': float(np.mean(u_mag) / u_ref),
        'max_normalized': float(np.max(u_mag) / u_ref),
    }

def compute_internal_momentum(
    Eux: np.ndarray, Euy: np.ndarray,
    ro: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, r: float,
) -> tuple[float, float]:
    """입자 내부 fictitious fluid 선형 운동량 P_int = ∫_{Ω_p} ρ u dV.

    식
      - 마스크:  (x − cx)² + (y − cy)² ≤ r²
      - dA = dx · dy
      - P_int,x = Σ_{mask} ρ u_x · dA
      - P_int,y = Σ_{mask} ρ u_y · dA

    Args
      - Eux, Euy   2D 속도장 (ny, nx), lattice 단위
      - ro         2D 밀도장 (ny, nx)
      - dx, dy     격자 간격 (도메인)
      - cx, cy     실린더 중심 (도메인)
      - r          실린더 반지름 (도메인)

    Returns
      - (P_int_x, P_int_y) — float, lattice 단위
    """
    ny, nx = Eux.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    XX, YY = np.meshgrid(x, y)
    mask = (XX - cx) ** 2 + (YY - cy) ** 2 <= r ** 2
    dA = dx * dy
    px = float(np.sum(ro[mask] * Eux[mask]) * dA)
    py = float(np.sum(ro[mask] * Euy[mask]) * dA)
    return px, py

def compute_internal_angular_momentum(
    Eux: np.ndarray, Euy: np.ndarray,
    ro: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, r: float,
) -> float:
    """입자 내부 fictitious fluid 각운동량 L_int,z = ∫_{Ω_p} (r × ρ u) · ẑ dV.

    식
      - r = (x − cx, y − cy)
      - r × ρu = (r_x ρ u_y − r_y ρ u_x) ẑ     (2D → z 성분만 존재)
      - 마스크: (x − cx)² + (y − cy)² ≤ r²
      - dA = dx · dy

    Args
      - Eux, Euy   2D 속도장 (ny, nx), lattice 단위
      - ro         2D 밀도장 (ny, nx)
      - dx, dy     격자 간격 (도메인)
      - cx, cy     실린더 중심 (도메인)
      - r          실린더 반지름 (도메인)

    Returns
      - L_int,z (float, lattice 단위)
    """
    ny, nx = Eux.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    XX, YY = np.meshgrid(x, y)
    mask = (XX - cx) ** 2 + (YY - cy) ** 2 <= r ** 2
    rx = XX[mask] - cx
    ry = YY[mask] - cy
    dA = dx * dy
    lz = float(np.sum((rx * ro[mask] * Euy[mask] - ry * ro[mask] * Eux[mask])) * dA)
    return lz

def record_sedimentation_state(
    particle_pos: np.ndarray,  # (2,) 도메인 좌표
    particle_vel: np.ndarray,  # (2,) 격자 단위
    t: int,                     # 스텝 번호
    d_lattice: float,          # 입자 직경 (격자 단위)
    g_lattice: float,          # 격자 중력
    rho_ratio: float,
    y0: float,                 # 초기 y 위치 (도메인 좌표)
    dx: float = 1.0,           # 격자 간격 (도메인 좌표)
) -> dict:
    """침강 상태 무차원화 레코드 생성.

    식
      - u_g  = √(|ρ_s/ρ_f − 1| · g · D)             (중력 기준 speed)
      - t*   = t · u_g / D
      - y*   = (y_0 − y) / D                        (낙하 거리, 아래가 양)
      - v*_y = −v_y / u_g                           (부호 규약: 아래 방향이 양)
      - v*_x =  v_x / u_g

    단위 규약
      - pos            도메인 좌표
      - vel            lattice 단위 (dx_lattice / dt_lattice)
      - d_lattice      lattice 단위 직경
      - D_domain       = d_lattice · dx (도메인 좌표 직경)

    엣지 케이스
      - u_g < 1e-15 → 0-division 방지용 zero placeholder 반환

    Args
      - particle_pos   (2,) 도메인 좌표
      - particle_vel   (2,) lattice 단위
      - t              스텝 번호
      - d_lattice      입자 직경 (lattice 단위)
      - g_lattice      격자 중력
      - rho_ratio      ρ_s / ρ_f
      - y0             초기 y 위치 (도메인)
      - dx             격자 간격 (도메인, 기본 1.0)

    Returns
      - dict — 무차원 + raw 값 혼합 레코드
    """
    u_g = np.sqrt(abs(rho_ratio - 1.0) * g_lattice * d_lattice)
    D_domain = d_lattice * dx  # 도메인 좌표 직경

    if u_g < 1e-15:
        return {
            'step': int(t),
            't_star': 0.0, 'y_star': 0.0, 'vy_star': 0.0, 'vx_star': 0.0,
            'x': float(particle_pos[0]), 'y': float(particle_pos[1]),
            'vx': float(particle_vel[0]), 'vy': float(particle_vel[1]),
        }

    return {
        'step': int(t),
        't_star': float(t * u_g / d_lattice),
        'y_star': float((y0 - particle_pos[1]) / D_domain),
        'vy_star': float(-particle_vel[1] / u_g),
        'vx_star': float(particle_vel[0] / u_g),
        'x': float(particle_pos[0]),
        'y': float(particle_pos[1]),
        'vx': float(particle_vel[0]),
        'vy': float(particle_vel[1]),
    }

def compute_l_int_z(U, ro, nx: int, ny: int, dx: float, cx: float, cy: float, r_domain: float) -> float:
    """L_int,z flat-array wrapper — solver/runtime 호환 helper.

    역할
      - solver/runtime이 쓰는 `(N, 2)` / `(N,)` 배열을 `(ny, nx)`로 reshape 후 집계
      - cupy / numpy 백엔드 자동 감지
      - r_lat + 2 box 내부만 마스킹하여 O(r²) 비용으로 절감

    식은 `compute_internal_angular_momentum`과 동일 (L_int,z = Σ (r_x ρ u_y − r_y ρ u_x)).
    """
    _xp = np
    if type(U).__module__.startswith("cupy") or type(ro).__module__.startswith("cupy"):
        import cupy as _xp

    Eux = U[:, 0].reshape(ny, nx)
    Euy = U[:, 1].reshape(ny, nx)
    ro_2d = ro.reshape(ny, nx)

    ci = cx / dx
    cj = cy / dx
    r_lat = r_domain / dx

    ci_int = int(round(ci))
    cj_int = int(round(cj))
    r_idx = int(r_lat) + 2
    i_lo = max(0, ci_int - r_idx)
    i_hi = min(nx, ci_int + r_idx + 1)
    j_lo = max(0, cj_int - r_idx)
    j_hi = min(ny, cj_int + r_idx + 1)

    x_lat = _xp.arange(i_lo, i_hi, dtype=float) - ci
    y_lat = _xp.arange(j_lo, j_hi, dtype=float) - cj
    RX, RY = _xp.meshgrid(x_lat, y_lat)
    mask = RX**2 + RY**2 <= r_lat**2

    sub_ro = ro_2d[j_lo:j_hi, i_lo:i_hi]
    sub_Eux = Eux[j_lo:j_hi, i_lo:i_hi]
    sub_Euy = Euy[j_lo:j_hi, i_lo:i_hi]
    return float(_xp.sum(RX[mask] * sub_ro[mask] * sub_Euy[mask] - RY[mask] * sub_ro[mask] * sub_Eux[mask]))
