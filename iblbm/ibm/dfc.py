"""IBM Distribution Function Correction (DFC) — Tao 2019 non-iterative 경로.

파이프라인 (Tao et al. (2019) Eq. (15)-(25))

   Eulerian (ny·nx, 9)                           Lagrangian (Lb, 9)
  ┌──────────────────────┐    (a) Eq. (15)     ┌──────────────────────┐
  │  f_i*(x)             │  ─────────────────▶ │  f_i*(X_k)           │   보간
  └──────────────────────┘                     └──────────────────────┘
                                                       │
                                                       ▼ (b) Eq. (16) BB
                                               ┌──────────────────────┐
                                               │  f̄_i(X_k)            │   desired f_i
                                               │    = f_{opp[i]}*(X_k)│
                                               │      + 2 w_i ρ_f     │
                                               │        (e_i·u_w)/c_s²│
                                               └──────────────────────┘
                                                       │
                                                       ▼ (c) Eq. (22)-(23)
                                               ┌──────────────────────┐
                                               │  λ(k) = 1 / [2 ρ_f Δs·W_sum]
                                               └──────────────────────┘
                                                       │
                                                       ▼ (d) Eq. (17)
                                               ┌──────────────────────┐
                                               │  Δf_i(X_k)           │
                                               │   = λ(k)[f̄_i − f_i*] │
                                               └──────────────────────┘
                                                       │
  ┌──────────────────────┐    (e) Eq. (18)     ┌──────────────────────┐
  │  Δf_i(x)             │  ◀───────────────── │  Δf_i(X_k)           │   분산
  └──────────────────────┘                     └──────────────────────┘
          │ (f)
          ▼
      f_i ← f_i + Δf_i

핵심 성질 (Tao 2019)
  - non-iterative : λ(k) 해석해로 no-slip 직접 강제 (반복 불요)
  - Eq. (20)-(22) : λ 결정을 위한 scalar reduction  (approximation λ(k,i) = λ(k))
  - Eq. (25)      : 전체 고체력 F_s = −Σ_m Σ_i e_i λ(m)[f̄_i(X_m) − f_i*(X_m)]
                    + ρ_f V_s du_s/dt   (inertial mass 항 포함)

본 모듈은 Eq. (25) 우변의 분포 기여부 `−Σ e_i λ [f̄ − f*]`를 마커별로 집계한
`dfc_force`만 반환. `ρ_f V_s du_s/dt` 관성항은 상위 solver에서 합산.

참조
  - Tao, He, Chen, et al. (2019) — DFC 핵심 구성, Eq. (15)-(25)
"""

from __future__ import annotations

from ..backend import xp as np, add_at, _use_gpu
from .common import get_delta


def interpolate_f(Lx, Ly, fstar, dx, dy, ny, nx, delta_type):
    """Lagrangian 지점 9개 분포함수 보간 — Tao (2019) Eq. (15).

    수식
        f_i*(X_k) = Σ_x f_i(x) W(x − X_k) dx²

    df.py 보간과 동일한 stencil/δ_h 구조
      - 차이점 : `{u, ρ}` 대신 `f_i` (D2Q9 9개)를 동시에 보간
      - 동일 (di, dj) loop를 재사용하여 `f_interp` 누적

    Args
        Lx, Ly      : (Lb,) Lagrangian marker 좌표
        fstar       : (ny·nx, 9) post-streaming 분포함수
        dx, dy      : 격자 간격
        ny, nx      : 격자 크기
        delta_type  : `"hat"` | `"peskin4pt"`

    Returns
        f_interp : (Lb, 9)  보간된 분포함수 f_i*(X_k)
    """
    fstar_3d = fstar.reshape(ny, nx, 9)
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    f_interp = np.zeros((len(Lx), 9))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy  # (Lb,)
            f_interp += fstar_3d[ej_c, ei_c, :] * w[:, None]

    return f_interp


def bounce_back_fi(f_interp, rho_f, u_wall, lattice):
    """Lagrangian bounce-back — Tao (2019) Eq. (16).

    수식
        f̄_i(X_k) = f*_{opp(i)}(X_k) + 2 w_i ρ_f (e_i · u_wall) / c_s²

    경계 조건
      - u_wall = 0      → 표준 BB, no-slip 경계
      - u_wall ≠ 0      → moving boundary (Dirichlet)

    Args
        f_interp : (Lb, 9)  보간된 분포함수 f_i*(X_k)
        rho_f    : 전역 기준 밀도 (보통 1.0)
        u_wall   : (Lb, 2)  경계 desired velocity U_d
        lattice  : D2Q9 (opp, w, e 제공)

    Returns
        f_bb : (Lb, 9)  desired 분포함수 f̄_i(X_k)
    """
    cs2 = 1.0 / 3.0
    # f_opp[:, i] = f_interp[:, opp[i]]
    f_opp = f_interp[:, lattice.opp]            # (Lb, 9)
    # e_dot_u[:, i] = e_i . u_wall
    e_dot_u = u_wall @ lattice.e.T              # (Lb, 9)
    f_bb = f_opp + 2.0 * lattice.w[None, :] * rho_f * e_dot_u / cs2
    return f_bb


def compute_lambda(Lx, Ly, rho_f, Larea, dx, dy, ny, nx, delta_type):
    """Adjustment parameter λ(k) 해석해 — Tao (2019) Eq. (22)-(23).

    수식 (Eq. (23) 해석해)
        λ(k) = 1 / [ 2 ρ_f Δs · W_sum(k) · (dx)² ]      (lattice units dx² = 1)

    W_sum 산출 (2-pass: spread → interpolate)
        W_total(x) = Σ_{k'} W(x − X_{k'})                ← (a) spread
        W_sum(k)   = Σ_x W_total(x) W(x − X_k)           ← (b) interpolate

    Approximation (Tao 2019 §2.3)
        λ(k, i) = λ(k)     방향 독립성 가정 → 해석해 가능

    Args
        Lx, Ly      : (Lb,) Lagrangian marker 좌표
        rho_f       : 전역 밀도 (보통 1.0)
        Larea       : Δs (lattice units)
        dx, dy      : 격자 간격
        ny, nx      : 격자 크기
        delta_type  : `"hat"` | `"peskin4pt"`

    Returns
        lambda_k : (Lb,) marker 별 λ(k)
    """
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)
    Lb = len(Lx)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    # Step 1: spread -- W_total(x) = Sigma_{k'} W(x - X_{k'})
    W_total = np.zeros((ny, nx))
    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            add_at(W_total, (ej_c, ei_c), w)

    # Step 2: interpolate -- W_sum(k) = Sigma_x W_total(x) * W(x - X_k)
    W_sum = np.zeros(Lb)
    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            W_sum += W_total[ej_c, ei_c] * w

    # (dx)^2 = 1 in lattice units
    lambda_k = 1.0 / (2.0 * rho_f * Larea * W_sum)

    return lambda_k


def spread_delta_f(delta_f, Lx, Ly, Larea, dx, dy, ny, nx, delta_type):
    """Lagrangian Δf_i를 Eulerian 격자로 분산 — Tao (2019) Eq. (18).

    수식
        Δf_i(x) = Σ_k Δf_i(X_k) W(x − X_k) Δs

    9개 분포함수를 같은 δ_h·Δs 가중으로 동시 분산

    Args
        delta_f    : (Lb, 9)  Lagrangian Δf_i(X_k)
        Lx, Ly     : (Lb,)    Lagrangian marker 좌표
        Larea      : Δs (lattice units)
        dx, dy     : 격자 간격
        ny, nx     : 격자 크기
        delta_type : `"hat"` | `"peskin4pt"`

    Returns
        Ef_corr : (ny·nx, 9)  Eulerian 분포함수 보정 Δf_i(x)
    """
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    Ef_corr = np.zeros((ny, nx, 9))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy * Larea  # (Lb,)
            weighted = delta_f * w[:, None]  # (Lb, 9)
            for q in range(9):
                add_at(Ef_corr[:, :, q], (ej_c, ei_c), weighted[:, q])

    return Ef_corr.reshape(-1, 9)


def compute_dfc_fluid_force(deviation, lambda_k, Larea, lattice):
    """Marker 별 DFC 힘 기여 집계 — Tao (2019) Eq. (25) 분포 기여부.

    수식
        q_k = −Δs · Σ_i e_i · λ(k) · [f̄_i(X_k) − f_i*(X_k)]

    Eq. (25) 전체 식 대비 범위
        F_s = − Σ_m Σ_i e_i λ(m) [f̄_i − f_i*]   +   ρ_f V_s · du_s/dt
              └─────── 본 함수가 마커 단위로 집계 ──────┘ └── 관성항 (상위 solver) ──┘

    Args
        deviation : (Lb, 9)  `f̄_i − f_i*`  (λ 미적용 상태)
        lambda_k  : (Lb,)    λ(k) 값
        Larea     : Δs 적분 가중
        lattice   : D2Q9

    Returns
        force : (Lb, 2)  마커별 힘 기여 q_k
    """
    scaled_dev = lambda_k[:, None] * deviation * Larea  # (Lb, 9)
    force = -scaled_dev @ lattice.e                     # (Lb, 2)
    return force


def apply_dfc_correction(Lx, Ly, desired_vel, fstar,
                          dx, dy, Larea, ny, nx,
                          delta_type, lattice, lambda_cache=None,
                          correction_scale: float = 1.0):
    """DFC 분포함수 보정 통합 — Tao (2019) Eq. (15)-(25) full pipeline.

    수식
        f_i ← f_i + Δf_i
        Δf_i(X_k) = λ(k) · [f̄_i(X_k) − f_i*(X_k)]           Eq. (17)

    단계 (Tao 2019)
      (a) 보간         f_i*(X_k)       Eq. (15)       Euler → Lagrange
      (b) bounce-back  f̄_i(X_k)        Eq. (16)       desired 분포함수
      (c) λ(k)         λ(k)            Eq. (22)-(23)  해석해 또는 cache
      (d) 편차 · 스케일  Δf_i(X_k)                       λ · (f̄ − f*) · scale
      (e) 분산         Δf_i(x)         Eq. (18)       Lagrange → Euler
      (f) 보정         fstar += Δf_i
      (g) 유체력       dfc_force       Eq. (25) 분포부  마커별 집계

    Args
        Lx, Ly           : (Lb,)       Lagrangian marker 좌표
        desired_vel      : (Lb, 2)     경계 desired velocity U_d
        fstar            : (ny·nx, 9)  post-streaming 분포함수
        dx, dy           : 격자 간격
        Larea            : Δs (lattice units)
        ny, nx           : 격자 크기
        delta_type       : `"hat"` | `"peskin4pt"`
        lattice          : D2Q9
        lambda_cache     : (Lb,) 또는 None — 사전 계산된 λ(k) (없으면 내부 계산)
        correction_scale : Δf_i 스케일 팩터 (default 1.0; 프로젝트 확장)

    Returns
        fstar_corrected : (ny·nx, 9)  f_i + Δf_i
        dfc_force       : (Lb, 2)     Eq. (25) 분포 기여부 마커별 집계값
        lambda_k        : (Lb,)       λ(k) 값 (재사용용)
    """
    # (c) lambda(k)
    if lambda_cache is not None:
        lambda_k = lambda_cache
    elif _use_gpu:
        from ..gpu_kernels import compute_lambda_gpu
        lambda_k = compute_lambda_gpu(Lx, Ly, rho_f=1.0, Larea=Larea,
                                       dx=dx, dy=dy, ny=ny, nx=nx,
                                       delta_type=delta_type)
    else:
        lambda_k = compute_lambda(Lx, Ly, rho_f=1.0, Larea=Larea,
                                   dx=dx, dy=dy, ny=ny, nx=nx,
                                   delta_type=delta_type)

    # GPU: 전용 CUDA 커널 (interp + bb_lambda + spread 3개)
    if _use_gpu:
        from ..gpu_kernels import dfc_correction_gpu
        delta_f_euler, dfc_force = dfc_correction_gpu(
            Lx, Ly, desired_vel, fstar, lambda_k,
            dx, dy, Larea, ny, nx,
            delta_type=delta_type, lattice=lattice,
        )
        if correction_scale != 1.0:
            delta_f_euler = correction_scale * delta_f_euler
            dfc_force = correction_scale * dfc_force
        fstar_corrected = fstar + delta_f_euler
        return fstar_corrected, dfc_force, lambda_k

    # --- CPU 경로 ---

    # (a) f_i 보간: Euler -> Lagrange
    f_interp = interpolate_f(Lx, Ly, fstar, dx, dy, ny, nx, delta_type)

    # (b) Bounce-back: desired 분포함수
    f_bb = bounce_back_fi(f_interp, rho_f=1.0, u_wall=desired_vel,
                           lattice=lattice)

    # (d) 편차 + 스케일링
    deviation = f_bb - f_interp                    # (Lb, 9)
    delta_f_lagr = correction_scale * lambda_k[:, None] * deviation   # (Lb, 9)

    # (e) 분산: Lagrange -> Euler
    delta_f_euler = spread_delta_f(delta_f_lagr, Lx, Ly, Larea,
                                    dx, dy, ny, nx, delta_type)

    # (f) 보정
    fstar_corrected = fstar + delta_f_euler

    # (g) 마커별 힘 집계량 (Eq.25의 분포 기여부)
    dfc_force = correction_scale * compute_dfc_fluid_force(deviation, lambda_k, Larea, lattice)

    return fstar_corrected, dfc_force, lambda_k
