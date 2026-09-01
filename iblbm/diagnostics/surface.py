"""표면 응력 텐서 + 표면력 적분 진단.

핵심 식 (D2Q9 Chapman-Enskog 2차)
  - 압력 부분
        p δ_αβ = c_s² ρ δ_αβ
  - 점성 부분 (non-equilibrium 모멘트)
        τ_αβ^visc = −(1 − 1/(2τ)) Σ_i f_i^neq e_{i,α} e_{i,β}
  - Cauchy stress
        σ_αβ = −c_s² ρ δ_αβ + τ_αβ^visc
        c_s² = 1/3  (D2Q9)

표면력 적분
  - F_α = ∮_Γ σ_αβ n_β dl
      · Γ: 실린더 경계 (반지름 r, 중심 (cx, cy))
      · dl = r_lat · dθ
      · n̂(θ) = (cos θ, sin θ) — 외향 법선
  - sampling: `n_points`개 θ 등간격, 각 θ에서 bilinear interpolation으로 σ 샘플

용도
  - `sum(fib)` (IB 힘 합)과 별도로 Chapman-Enskog 응력 기반 유체력을 후처리에서 재산출
  - 두 경로의 일치도 (IB fidelity) 검정

계보
  - Chapman-Enskog non-equilibrium 응력 moment: 일반 LBM 분석 표준
"""
from __future__ import annotations

import numpy as np

def compute_stress_tensor(
    fstar: np.ndarray,   # (N, 9)
    feq: np.ndarray,     # (N, 9)
    ro: np.ndarray,      # (N,)
    tau: float,
    lattice,             # D2Q9 (e, w 포함)
    ny: int,
    nx: int,
) -> np.ndarray:
    """후처리용 D2Q9 Cauchy stress σ_αβ 산출 (pressure + viscous).

    식
      - f_i^neq = f_i* − f_i^eq
      - τ_αβ^visc = −(1 − 1/(2τ)) Σ_i f_i^neq e_{i,α} e_{i,β}
      - σ_αβ = −c_s² ρ δ_αβ + τ_αβ^visc
      - c_s² = 1/3 (D2Q9)

    Args
      - fstar     post-streaming 분포 (N, 9)
      - feq       평형 분포 (N, 9)
      - ro        밀도 (N,)
      - tau       완화 시간 τ
      - lattice   D2Q9 격자 (e: (9, 2), w: (9,))
      - ny, nx    격자 크기

    Returns
      - stress — (ny, nx, 2, 2) Cauchy stress tensor
    """
    cs2 = 1.0 / 3.0
    f_neq = fstar - feq  # (N, 9)

    # viscous factor: -(1 - 1/(2τ))
    visc_factor = -(1.0 - 1.0 / (2.0 * tau))

    e = np.asarray(lattice.e)  # (9, 2)

    stress = np.zeros((ny, nx, 2, 2))

    # non-equilibrium stress: Σ_i f_neq_i * e_iα * e_iβ
    for alpha in range(2):
        for beta in range(2):
            # Σ_i f_neq_i × e_iα × e_iβ
            contrib = np.sum(f_neq * (e[:, alpha] * e[:, beta])[None, :], axis=1)
            stress[:, :, alpha, beta] = visc_factor * contrib.reshape(ny, nx)

    # pressure term: -c_s² ρ δ_αβ
    ro_2d = ro.reshape(ny, nx)
    stress[:, :, 0, 0] -= cs2 * ro_2d
    stress[:, :, 1, 1] -= cs2 * ro_2d

    return stress

def integrate_surface_force(
    stress: np.ndarray,  # (ny, nx, 2, 2)
    cx: float,           # 실린더 중심 x (도메인 좌표)
    cy: float,           # 실린더 중심 y (도메인 좌표)
    r: float,            # 실린더 반지름 (도메인 좌표)
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    n_points: int = 360,
) -> np.ndarray:
    """응력 텐서 σ를 실린더 경계에 적분하여 유체력 F_α 산출.

    식
      - F_α = ∮_Γ σ_αβ n_β dl
      - θ_k = 2π k / n_points  (k = 0 .. n_points−1)
      - 샘플 좌표: (cx + r cos θ_k, cy + r sin θ_k)
      - dl = r_lat · dθ,     dθ = 2π / n_points
      - n̂_k = (cos θ_k, sin θ_k)   외향 법선
      - σ(θ_k)는 bilinear interpolation으로 샘플

    프로젝트 구현
      - 표면 sampling + bilinear interpolation 근사 (후처리 전용)

    Args
      - stress       (ny, nx, 2, 2) Cauchy stress (lattice 단위)
      - cx, cy       실린더 중심 (도메인)
      - r            반지름 (도메인)
      - dx, dy       격자 간격
      - nx, ny       격자 크기
      - n_points     적분 점 수 (기본 360)

    Returns
      - F — (2,) 표면 합력 (lattice 단위, `sum(fib)`과 동일 단위)
    """
    # 격자 좌표로 변환
    r_lat = r / dx  # 격자 단위 반지름

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dtheta = 2.0 * np.pi / n_points
    ds = r_lat * dtheta  # 격자 단위 호 길이

    # 적분 점 (도메인 좌표 → 격자 인덱스)
    xp_domain = cx + r * np.cos(theta)
    yp_domain = cy + r * np.sin(theta)

    # 법선 벡터 (외향)
    n_x = np.cos(theta)
    n_y = np.sin(theta)

    F = np.zeros(2)

    for k in range(n_points):
        xi = xp_domain[k] / dx
        yi = yp_domain[k] / dy

        i0 = int(np.floor(xi))
        j0 = int(np.floor(yi))
        i0 = max(0, min(i0, nx - 2))
        j0 = max(0, min(j0, ny - 2))

        fx = xi - i0
        fy = yi - j0

        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        for alpha in range(2):
            traction_alpha = 0.0
            for beta in range(2):
                n_beta = n_x[k] if beta == 0 else n_y[k]
                s_interp = (w00 * stress[j0, i0, alpha, beta]
                            + w10 * stress[j0, i0 + 1, alpha, beta]
                            + w01 * stress[j0 + 1, i0, alpha, beta]
                            + w11 * stress[j0 + 1, i0 + 1, alpha, beta])
                traction_alpha += s_interp * n_beta

            F[alpha] += traction_alpha * ds

    return F
