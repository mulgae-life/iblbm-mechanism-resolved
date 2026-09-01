"""MDF-IBM relaxation parameter estimator (Zhang 2020 Eq. (39) + (46)).

참조
  - Zhang, Pan, Zhang, Haeri (2020)
    *A relaxed multi-direct-forcing immersed boundary-cascaded lattice*
    *Boltzmann method accelerated on GPU.*
    Computer Physics Communications 248, 106980.

수식 계보
  - Eq. (38)   linear form        X_k = X_{k-1} + (b − A X_{k-1})
  - Eq. (39)   coupling matrix    A_lm = Σ_x Φ(x − x_m) Φ(x − x_l) δs_m dx²
  - Eq. (12)   separable delta    Φ(x, y) = (1/dx²) φ(x/dx) φ(y/dx)
  - Eq. (46)   estimated optimum  ω̃ = ||A||_∞⁻¹ ≤ λ_max⁻¹ ≤ ω_opt
  - Eq. (47)   convergent range   0 < ω < 2 ||A||_∞⁻¹

Separable 분해 (Eq. (12) 대입 후 Eq. (39) 정리)

    Φ(x, y)         Eq. (12)          ┌ φ(x/dx) │
    ───────  =  ─────────────────  ≡  │ φ(y/dx) │ × (1/dx²)
     2D           rank-1 factorable   └         ┘

    A_lm = (δs_m / dx²) · S_x(l,m) · S_y(l,m)
        S_x(l,m) = Σ_i φ((i·dx − Lx_l)/dx) · φ((i·dx − Lx_m)/dx)
        S_y(l,m) = Σ_j φ((j·dx − Ly_l)/dx) · φ((j·dx − Ly_m)/dx)

  - 2D 이중합 → 1D inner product 두 번으로 환원
  - marker overlap 구간 밖에서는 S_x 또는 S_y = 0 → O(N²·stencil) 수준

운영 원칙
  - Solver hot-loop 밖: initialize 단계 1회 호출
  - 고정 경계(정지 cylinder 등): 1회 계산으로 충분
  - Moving rigid particle: marker가 particle frame에서 고정이면 ||A||_∞⁻¹는
    lattice 상대 위치에 따라 소폭 진동. 현 구현은 초기값 근사를 유지하며 매 스텝 재계산하지 않는다
  - 고정 상수 대신 시나리오별 marker 배치를 반영해 ω를 실측
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Backend-independent (pure numpy) delta kernels
# ---------------------------------------------------------------------------
# 본 모듈은 CPU 초기화 단계에서만 실행 → backend 무관 순수 numpy 경로 사용
# (`iblbm.ibm.common`의 delta는 numpy/cupy 중 활성 backend에 바인딩됨).

def _peskin4pt_np(r: np.ndarray) -> np.ndarray:
    ar = np.abs(r)
    result = np.zeros_like(ar)
    mask1 = ar <= 1.0
    mask2 = (~mask1) & (ar <= 2.0)
    r1 = ar[mask1]
    result[mask1] = 0.125 * (
        3.0 - 2.0 * r1 + np.sqrt(np.maximum(1.0 + 4.0 * r1 - 4.0 * r1 ** 2, 0.0))
    )
    r2 = ar[mask2]
    result[mask2] = 0.125 * (
        5.0 - 2.0 * r2 - np.sqrt(np.maximum(-7.0 + 12.0 * r2 - 4.0 * r2 ** 2, 0.0))
    )
    return result


def _hat_np(r: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - np.abs(r), 0.0)


_DELTA_NP = {
    "peskin4pt": (_peskin4pt_np, 2),  # 5-pt stencil, radius 2
    "hat":       (_hat_np, 1),         # 3-pt stencil, radius 1
}


def compute_A_norm_inf(
    Lx: np.ndarray,
    Ly: np.ndarray,
    delta_func: Callable[[np.ndarray], np.ndarray],
    stencil_radius: int,
    Larea: float,
    dx: float,
) -> tuple[float, np.ndarray]:
    """Zhang 2020 Eq. (39) 기반 `||A||_∞` 계산 (separable 최적화 경로).

    수식
        A_lm    = Σ_x Φ(x − x_m) Φ(x − x_l) δs_m dx²           Eq. (39)
        ||A||_∞ = max_l Σ_m A_lm

    구현 경로 (분리형)
        Φ(x,y) = (1/dx²) φ(x/dx) φ(y/dx)                       Eq. (12)
        →   A_lm = (δs_m / dx²) · S_x(l,m) · S_y(l,m)
        cutoff : |Lx_l − Lx_m| ≥ 2·stencil·dx  ⇒  A_lm = 0

    Parameters
    ----------
    Lx, Ly : (N,) ndarray
        Lagrangian marker 좌표 (lattice 단위, `i·dx`와 직접 비교 가능)
    delta_func : callable(r) -> ndarray
        1D normalized delta kernel φ
    stencil_radius : int
        kernel stencil radius  (peskin4pt → 2, hat → 1)
    Larea : float
        marker arc length δs_m (lattice units)
    dx : float
        lattice spacing (dy = dx 가정)

    Returns
    -------
    norm_inf : float
        `||A||_∞ = max_l Σ_m A_lm`
    row_sums : (N,) ndarray
        각 l의 row sum (진단용)
    """
    N = int(len(Lx))
    Lx = np.asarray(Lx, dtype=np.float64)
    Ly = np.asarray(Ly, dtype=np.float64)
    row_sums = np.zeros(N, dtype=np.float64)

    # overlap cutoff: |Lx_l - Lx_m| ≥ 2·stencil·dx 이면 A_lm = 0
    cutoff = 2.0 * stencil_radius * dx

    # Zhang Eq. 39 + Eq. 12 대입 후 factor = δs_m / dx²
    factor = float(Larea) / (dx * dx)

    for l in range(N):
        xl = Lx[l]
        yl = Ly[l]
        acc = 0.0
        for m in range(N):
            xm = Lx[m]
            ym = Ly[m]

            if abs(xl - xm) >= cutoff or abs(yl - ym) >= cutoff:
                continue

            # x-direction 1D inner product
            #   S_x = Σ_i φ((i·dx - xl)/dx) · φ((i·dx - xm)/dx)
            # overlap 범위:
            #   max(xl, xm) - stencil·dx ≤ i·dx ≤ min(xl, xm) + stencil·dx
            sx_lo = max(xl, xm) - stencil_radius * dx
            sx_hi = min(xl, xm) + stencil_radius * dx
            i_lo = int(np.ceil(sx_lo / dx))
            i_hi = int(np.floor(sx_hi / dx)) + 1
            if i_lo >= i_hi:
                continue
            i_arr = np.arange(i_lo, i_hi, dtype=np.float64) * dx
            phi_l_x = delta_func((i_arr - xl) / dx)
            phi_m_x = delta_func((i_arr - xm) / dx)
            sum_x = float(np.sum(phi_l_x * phi_m_x))
            if sum_x == 0.0:
                continue

            # y-direction 1D inner product
            sy_lo = max(yl, ym) - stencil_radius * dx
            sy_hi = min(yl, ym) + stencil_radius * dx
            j_lo = int(np.ceil(sy_lo / dx))
            j_hi = int(np.floor(sy_hi / dx)) + 1
            if j_lo >= j_hi:
                continue
            j_arr = np.arange(j_lo, j_hi, dtype=np.float64) * dx
            phi_l_y = delta_func((j_arr - yl) / dx)
            phi_m_y = delta_func((j_arr - ym) / dx)
            sum_y = float(np.sum(phi_l_y * phi_m_y))
            if sum_y == 0.0:
                continue

            acc += factor * sum_x * sum_y

        row_sums[l] = acc

    norm_inf = float(np.max(row_sums))
    return norm_inf, row_sums


def estimate_optimal_omega(
    Lx: np.ndarray,
    Ly: np.ndarray,
    delta_func: Callable[[np.ndarray], np.ndarray],
    stencil_radius: int,
    Larea: float,
    dx: float,
) -> float:
    """Zhang 2020 Eq. (46) estimated optimal relaxation parameter ω̃.

    수식
        ω̃ = ||A||_∞⁻¹     (Eq. (46):  ω̃ ≤ λ_max⁻¹ ≤ ω_opt)

    수렴 영역 (Eq. (47))
        0 < ω < 2 ||A||_∞⁻¹

    Returns
    -------
    omega : float
        `ω̃ = ||A||_∞⁻¹`
    """
    norm_inf, _ = compute_A_norm_inf(
        Lx=Lx, Ly=Ly,
        delta_func=delta_func, stencil_radius=stencil_radius,
        Larea=Larea, dx=dx,
    )
    return 1.0 / norm_inf


def estimate_optimal_omega_from_type(
    Lx,
    Ly,
    delta_type: str,
    Larea: float,
    dx: float,
) -> float:
    """Delta type 문자열 기준 `ω̃` 계산 (backend 독립 호출 경로).

    처리 순서
      1. CuPy → NumPy 변환          (backend 무관화)
      2. physical → lattice 정규화  (Lx, Ly 를 `/dx`; Larea는 이미 lattice)
      3. `compute_A_norm_inf` → `ω̃ = ||A||_∞⁻¹`

    Parameters
    ----------
    Lx, Ly : array-like
        marker 좌표 (physical units)
    delta_type : str
        `"peskin4pt"` | `"hat"`
    Larea : float
        marker arc length (lattice units; `init.initialize()` 에서 기 변환)
    dx : float
        lattice spacing (physical length)

    Returns
    -------
    omega : float
        `ω̃ = ||A||_∞⁻¹` (dimensionless, lattice units)
    """
    try:
        delta_func, stencil_radius = _DELTA_NP[delta_type]
    except KeyError as e:
        raise KeyError(
            f"a_norm: unknown delta_type '{delta_type}'. "
            f"registered: {list(_DELTA_NP.keys())}"
        ) from e

    # CuPy → NumPy 변환 (backend-independent)
    def _to_numpy(a):
        if hasattr(a, "get"):
            return np.asarray(a.get(), dtype=np.float64)
        return np.asarray(a, dtype=np.float64)

    Lx_np = _to_numpy(Lx)
    Ly_np = _to_numpy(Ly)

    # Lattice-unit 정규화 — state.Lx/Ly는 physical이므로 /dx 변환,
    # state.Larea는 init.initialize() 에서 이미 lattice units로 저장되어 그대로 사용.
    dx_phys = float(dx)
    if dx_phys <= 0.0:
        raise ValueError(f"a_norm: non-positive dx={dx_phys}")
    Lx_lu = Lx_np / dx_phys
    Ly_lu = Ly_np / dx_phys
    Larea_lu = float(Larea)  # 이미 lattice units
    dx_lu = 1.0

    norm_inf, _ = compute_A_norm_inf(
        Lx=Lx_lu, Ly=Ly_lu,
        delta_func=delta_func, stencil_radius=stencil_radius,
        Larea=Larea_lu, dx=dx_lu,
    )
    return 1.0 / norm_inf
