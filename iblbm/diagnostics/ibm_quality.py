"""IBM 경계 충실도 진단 (slip error + leakage flux).

핵심 지표 (프로젝트 구현 진단, 원형 실린더 기준)
  - 슬립 오차 (boundary velocity error)
        ε_slip,mean = Σ_m ||ũ(X_m) − u_target(X_m)|| / (N_m · U_ref)
        ε_slip,max  = max_m ||ũ(X_m) − u_target(X_m)|| / U_ref

  - 누설 유속 (normal penetration)
        Φ_leak = ∮ ((ũ − u_target) · n) ds / (U_ref · π D)

경계 적분 개념

         n̂(θ)                               ┌──── u_target(X_m) 경계 prescribed
          ↑                                  │      (고정 실린더 → 0)
          │          marker X_m              │
        ┌─┴─┐      ● ● ● ● ●               ũ(X_m) = Σ_x u(x) δ_h(x − X_m)
        │ • │ — •              •            │      Eulerian 속도를 δ_h로 마커에 보간
        │ ← │•      cylinder   •
        │ • │ — •              •            Φ_leak는 (ũ − u_target)·n을 마커별 ds 가중으로
        └─┬─┘      ● ● ● ● ●                합하고 U_ref · π D로 정규화 (고정 경계이면 0 지향)
          │
          n̂

보간 규칙
  - `delta_type`: "hat" (2-point φ = max(1−|r|, 0)) | "peskin4pt" (Peskin 4-point)
  - marker 분포: 활성 marker spacing rule과 동일 개수 규칙, 진단용이라 inward retraction 미적용

계보
  - regularized δ_h 기반 보간: Peskin (2002) — `ibm/forcing.py`와 동일 계보
  - ε_slip / Φ_leak 자체 정의는 프로젝트 구현 (benchmark 후처리 전용)
"""
from __future__ import annotations

import numpy as np

def _delta_hat_np(r: np.ndarray) -> np.ndarray:
    """2-point hat δ_h: φ(r) = max(1 − |r|, 0) (support |r| ≤ 1)."""
    return np.maximum(1.0 - np.abs(r), 0.0)

def _delta_peskin4pt_np(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point δ_h (support |r| ≤ 2, 구간별 무리식)."""
    ar = np.abs(r)
    result = np.zeros_like(ar)
    mask1 = ar <= 1.0
    mask2 = (~mask1) & (ar <= 2.0)
    r1 = ar[mask1]
    result[mask1] = (1.0 / 8.0) * (
        3.0 - 2.0 * r1 + np.sqrt(np.maximum(1.0 + 4.0 * r1 - 4.0 * r1**2, 0.0))
    )
    r2 = ar[mask2]
    result[mask2] = (1.0 / 8.0) * (
        5.0 - 2.0 * r2 - np.sqrt(np.maximum(-7.0 + 12.0 * r2 - 4.0 * r2**2, 0.0))
    )
    return result

def _generate_cylinder_markers(
    cx: float, cy: float, r: float, NN: int,
    marker_spacing_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """진단용 원형 표면 마커 생성.

    규칙
      - 활성 marker spacing rule과 같은 개수 규칙 (`dd = 2π r_lat / spacing_factor`)
      - 진단용 → inward retraction 미적용 (경계선 정확히 위)

    Args
      - cx, cy                    실린더 중심 (도메인 좌표)
      - r                         반지름 (도메인 좌표)
      - NN                        격자 해상도 파라미터
      - marker_spacing_factor     마커 간격 (lattice 단위, 기본 0.5)

    Returns
      - (Lx, Ly, theta) — 마커 좌표 (도메인) + 각도 배열
    """
    lattice_r = r * (NN - 1)
    dd = 2.0 * np.pi * lattice_r / marker_spacing_factor
    step = 1.0 / dd
    Ldx = np.arange(0, 1.0 + step * 0.5, step)
    Ldx = Ldx[Ldx <= 1.0 + 1e-12]
    if len(Ldx) > 1 and np.abs(Ldx[-1] - 1.0) < step * 0.5:
        Ldx = Ldx[:-1]
    theta = 2.0 * np.pi * Ldx
    Lx = r * np.cos(theta) + cx
    Ly = r * np.sin(theta) + cy
    return Lx, Ly, theta

def _interpolate_to_markers(
    Eux: np.ndarray, Euy: np.ndarray,
    Lx: np.ndarray, Ly: np.ndarray,
    dx: float, dy: float, delta_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Eulerian → Lagrangian 속도 보간 (진단용 CPU 구현).

    식
      - ũ(X_m) = Σ_x u(x) δ_h(x − X_m)
      - δ_h(r) = φ(r/Δx) φ(r/Δy)   (product kernel)

    Args
      - Eux, Euy     2D 속도장 (ny, nx)
      - Lx, Ly       마커 좌표 (도메인)
      - dx, dy       격자 간격
      - delta_type   "hat" | "peskin4pt"

    Returns
      - (Lux, Luy) — 마커별 보간 속도 (lattice 단위)
    """
    ny, nx = Eux.shape
    delta_func, a = _DELTA_NP[delta_type]
    Lb = len(Lx)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)
    offsets = np.arange(-a, a + 1)

    Lux = np.zeros(Lb)
    Luy = np.zeros(Lb)

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            Lux += Eux[ej_c, ei_c] * w
            Luy += Euy[ej_c, ei_c] * w

    return Lux, Luy

def compute_slip_error(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float, NN: int,
    delta_type: str,
    u_target_x: np.ndarray | None = None,
    u_target_y: np.ndarray | None = None,
    u_ref: float = 0.1,
) -> dict:
    """경계면 슬립 오차 ε_slip 산출 (프로젝트 진단).

    식
      - ε_slip,mean = Σ_m ||ũ(X_m) − u_target(X_m)|| / (N_m · U_ref)
      - ε_slip,max  = max_m ||ũ(X_m) − u_target(X_m)|| / U_ref
      - u_target 기본값 = 0 (고정 실린더)

    단계
      1. 마커 Lx, Ly 생성 (inward retraction 없음)
      2. ũ(X_m) 보간
      3. slip 벡터 Δu = ũ − u_target → L2 크기 `slip_mag`
      4. mean/max를 U_ref로 무차원화

    Args
      - Eux, Euy                 2D 속도장 (ny, nx), lattice 단위
      - dx, dy                   격자 간격 (도메인)
      - cx, cy                   실린더 중심 (도메인)
      - D                        실린더 직경 (도메인)
      - NN                       격자 해상도 파라미터
      - delta_type               "hat" | "peskin4pt"
      - u_target_x, u_target_y   마커별 목표 속도 (None → 0)
      - u_ref                    참조 속도 (lattice 단위, 기본 0.1)

    Returns
      - dict: `{'mean', 'max', 'per_marker', 'theta'}`
    """
    r = D / 2.0
    Lx, Ly, theta = _generate_cylinder_markers(cx, cy, r, NN)
    Lb = len(Lx)

    Lux, Luy = _interpolate_to_markers(Eux, Euy, Lx, Ly, dx, dy, delta_type)

    if u_target_x is None:
        u_target_x = np.zeros(Lb)
    if u_target_y is None:
        u_target_y = np.zeros(Lb)

    slip_x = Lux - u_target_x
    slip_y = Luy - u_target_y
    slip_mag = np.sqrt(slip_x**2 + slip_y**2)

    return {
        'mean': float(np.mean(slip_mag) / u_ref),
        'max': float(np.max(slip_mag) / u_ref),
        'per_marker': slip_mag / u_ref,
        'theta': theta,
    }

def compute_leakage_flux(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float, NN: int,
    delta_type: str,
    u_target_x: np.ndarray | None = None,
    u_target_y: np.ndarray | None = None,
    u_ref: float = 0.1,
) -> dict:
    """경계면 관통 유량 Φ_leak 산출 (프로젝트 진단).

    식
      - Φ_leak = ∮ ((ũ − u_target) · n) ds
      - 정규화 : U_ref · π D
      - 원형 실린더: n̂_m = (cos θ_m, sin θ_m),  ds = 2π r / N_m

    해석
      - 고정 경계이면 정확 IBM에서 Φ_leak → 0
      - IBM의 누설 정도(경계 위반)를 단일 스칼라로 요약

    Args
      - Eux, Euy                 2D 속도장 (ny, nx), lattice 단위
      - dx, dy                   격자 간격 (도메인)
      - cx, cy                   실린더 중심 (도메인)
      - D                        실린더 직경 (도메인)
      - NN                       격자 해상도 파라미터
      - delta_type               "hat" | "peskin4pt"
      - u_target_x, u_target_y   마커별 목표 속도
      - u_ref                    참조 속도 (lattice 단위)

    Returns
      - dict: `{'total', 'total_raw', 'per_marker', 'theta'}`
    """
    r = D / 2.0
    Lx, Ly, theta = _generate_cylinder_markers(cx, cy, r, NN)
    Lb = len(Lx)

    Lux, Luy = _interpolate_to_markers(Eux, Euy, Lx, Ly, dx, dy, delta_type)

    if u_target_x is None:
        u_target_x = np.zeros(Lb)
    if u_target_y is None:
        u_target_y = np.zeros(Lb)

    slip_x = Lux - u_target_x
    slip_y = Luy - u_target_y

    # 외향 법선: n = (cos θ, sin θ)
    nx_vec = np.cos(theta)
    ny_vec = np.sin(theta)

    # 마커별 법선 유속
    un = slip_x * nx_vec + slip_y * ny_vec
    ds = 2.0 * np.pi * r / Lb

    flux_raw = float(np.sum(un * ds))
    flux_norm = flux_raw / (u_ref * np.pi * D)

    return {
        'total': flux_norm,
        'total_raw': flux_raw,
        'per_marker': un / u_ref,
        'theta': theta,
    }

_DELTA_NP = {
    "hat": (_delta_hat_np, 1),
    "peskin4pt": (_delta_peskin4pt_np, 2),
}
