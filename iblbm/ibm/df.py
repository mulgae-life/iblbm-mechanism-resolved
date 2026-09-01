"""IBM Direct Forcing (DF) — Uhlmann 2005 계보.

3-단계 파이프라인

      Eulerian (ny × nx)                 Lagrangian (Lb markers)
  ┌─────────────────────────┐   (a)    ┌───────────────────────┐
  │  u(x),  ρ(x)            │  ─────▶  │  U*(X_l),  ρ(X_l)     │   보간
  └─────────────────────────┘   δ_h    └───────────────────────┘
                                           │      (b)  F_l
                                           │   = 2 ρ(X_l) (U_d − U*) / Δt   (compressible)
                                           │   = 2       (U_d − U*) / Δt    (incompressible)
                                           ▼
  ┌─────────────────────────┐   (c)    ┌───────────────────────┐
  │  f_ib(x)                │  ◀─────  │  F_l                  │   분산
  └─────────────────────────┘   δ_h·Δs └───────────────────────┘

수식 (Uhlmann 2005)
  - Eq. (5)    F = (U_d − Ũ) / Δt                     desired-velocity enforcement
  - Eq. (9a)   Ũ(X_l) = Σ_x ũ(x) δ_h(x − X_l) dx²    보간
  - Eq. (9b)   f(x)   = Σ_l F(X_l) δ_h(x − X_l) ΔV_l  분산

계보 / 분기
  - 밀도 가중 F = 2ρ (U_d − Ũ) / Δt    Wang, Fan, Luo (2008) Eq. (18) 계열
  - 비압축 분기 F = 2 (U_d − Ũ) / Δt   Majumder et al. (2023) Eq. (9)
  - LBM 힘 결합은 collision 모듈에서 Guo, Zheng, Shi (2002) Eq. (20)

전계수 `2`는 Wang/Majumder에서 no-slip enforcement를 1 sub-step에서 성취하기 위해
도입한 over-relaxation factor이며, Uhlmann 원식(Eq. 5)의 1/Δt와 대비됨.
"""

from __future__ import annotations

from ..backend import xp as np, add_at, _use_gpu
from .common import get_delta


def ibm_direct_forcing(
    Lx: np.ndarray, Ly: np.ndarray,
    desired_vel: np.ndarray,
    Eux: np.ndarray, Euy: np.ndarray, Ro: np.ndarray,
    dx: float, dy: float, dt: float,
    Larea: float, ny: int, nx: int,
    delta_type: str = "peskin4pt",
    incompressible: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IBM Direct Forcing (CPU 경로, 2·ρ 가중).

    3-단계 연산  (Uhlmann 2005 Eq. (9a)-(9b), 중간 힘 Eq. (5))
      (a) 보간      Ũ, ρ̃ = Σ_x {u, ρ}(x) δ_h(x − X_l)
      (b) 힘        F_l = 2 ρ̃ (U_d − Ũ) / Δt              (compressible)
                        = 2     (U_d − Ũ) / Δt              (incompressible)
      (c) 분산      f_ib(x) = Σ_l F_l δ_h(x − X_l) Δs

    2D 배열 규칙
        Eux, Euy, Ro : (ny, nx)   접근 arr[j, i]  (j = y-행, i = x-열)

    Args
        Lx, Ly       : (Lb,)  Lagrangian marker 좌표 (도메인 좌표계)
        desired_vel  : (Lb, 2) 목표 속도 U_d
        Eux, Euy     : (ny, nx) Eulerian 속도장
        Ro           : (ny, nx) 밀도장
        dx, dy       : 격자 간격
        dt           : 시간 스텝 Δt
        Larea        : marker arc length Δs (lattice units)
        ny, nx       : 격자 크기
        delta_type   : `"peskin4pt"` | `"hat"`
        incompressible : True → Majumder (2023) Eq. (9) 분기 (ρ-가중 없음)

    Returns
        fib       : (nodenums, 2)  IB 체적력 f_ib
        Lux, Luy  : (Lb,)          보간된 속도 Ũ
        Lfx, Lfy  : (Lb,)          Lagrangian 힘 F_l
        R         : (Lb,)          보간된 밀도 ρ̃
    """
    Lb = len(Lx)
    nodenums = ny * nx

    delta_func, a = get_delta(delta_type)

    if _use_gpu:
        from ..gpu_kernels import ibm_direct_forcing_gpu
        fib, Lux, Luy = ibm_direct_forcing_gpu(
            Lx, Ly, desired_vel, Eux, Euy, Ro,
            dx, dy, dt, Larea, ny, nx,
            delta_type=delta_type,
            return_interp_vel=True,
            incompressible=incompressible,
        )
        # GPU 경로는 Lfx/Lfy/R을 커널 내부로만 쓰고 반환하지 않음.
        # velocity residual 기준 수렴 판정은 Lux/Luy로 충분.
        return fib, Lux, Luy, None, None, None

    # 라그랑주 점의 가장 가까운 격자 인덱스 (0-base)
    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    offsets = np.arange(-a, a + 1)

    # --- 보간: Eulerian → Lagrangian ---
    Lux = np.zeros(Lb)
    Luy = np.zeros(Lb)
    R = np.zeros(Lb)

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj

            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)

            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy

            R += Ro[ej_c, ei_c] * w
            Lux += Eux[ej_c, ei_c] * w
            Luy += Euy[ej_c, ei_c] * w

    # --- 힘 계산 ---
    if incompressible:
        # Majumder Eq.(9): F = 2*(u_desired - u_interp)/dt (밀도 가중 없음)
        Lfx = 2.0 * (desired_vel[:, 0] - Lux) / dt
        Lfy = 2.0 * (desired_vel[:, 1] - Luy) / dt
    else:
        # 밀도 보간 방식: F = 2*R*(u_desired - u_interp)/dt
        Lfx = 2.0 * R * (desired_vel[:, 0] - Lux) / dt
        Lfy = 2.0 * R * (desired_vel[:, 1] - Luy) / dt

    # --- 분산: Lagrangian → Eulerian ---
    Efx = np.zeros((ny, nx))
    Efy = np.zeros((ny, nx))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj

            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)

            wx = delta_func((ei * dx - Lx) / dx)
            wy = delta_func((ej * dy - Ly) / dy)
            w = wx * wy * Larea

            # 중복 인덱스 누적 (일반 +=는 중복 무시)
            add_at(Efx, (ej_c, ei_c), Lfx * w)
            add_at(Efy, (ej_c, ei_c), Lfy * w)

    fib = np.zeros((nodenums, 2))
    fib[:, 0] = Efx.ravel()
    fib[:, 1] = Efy.ravel()

    return fib, Lux, Luy, Lfx, Lfy, R
