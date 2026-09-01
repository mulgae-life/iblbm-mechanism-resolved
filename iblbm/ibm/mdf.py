"""IBM Multi-Direct Forcing (MDF) — Wang 2008 + Zhang 2020 relaxation.

Iteration loop (residual 기반 조기 종료)

           ┌──────────────────────────── start (k = 0) ────────────────┐
           │   U_work = U,  fib_total = 0,  prev_residual = +∞         │
           └───────────────────────────────────────────────────────────┘
                               │
                               ▼
       ┌─────────────────── for k = 1 … n_iter ───────────────────┐
       │                                                         │
       │   (1) DF sub-step (df.ibm_direct_forcing):              │
       │         Ũ^(k)        ← interpolate(U_work)              │
       │         F_l^(k)      ← 2 ρ̃ (U_d − Ũ^(k)) / Δt          │
       │         fib^(k)      ← spread(F_l^(k))                  │
       │                                                         │
       │   (2) L∞ slip residual (interface-velocity error):     │
       │         r_k = max_l ||U_d − Ũ^(k)||₂                   │
       │                                                         │
       │   (3) 발산 guard:                                       │
       │         r_k > 1.5 · r_{k-1}   ⇒ break                   │
       │                                                         │
       │   (4) relaxed 누적 (Zhang 2020 Eq. (43)-(45)):          │
       │         fib_total  += ω · fib^(k)                       │
       │         U_work     += ω · fib^(k) · Δt / (2 ρ)          │
       │                                                         │
       │   (5) iterative ω-torque coupling (optional):           │
       │         desired_vel ← recompute with ω_est              │
       │                                                         │
       │   (6) 조기 종료:                                         │
       │         k ≥ min_iter  &&  r_k < tol   ⇒ break           │
       │                                                         │
       └─────────────────────────────────────────────────────────┘
                               │
                               ▼
                         return fib_total

Relaxation parameter ω — 이론 근거 (Zhang 2020)
  - Eq. (39)  A_lm = Σ_x Φ(x − x_m) Φ(x − x_l) δs_m dx²
  - Eq. (43)  X_k = X_{k-1} + ω (b − A X_{k-1})           Richardson iteration
  - Eq. (46)  ω̃   = ||A||_∞⁻¹ ≤ λ_max⁻¹ ≤ ω_opt           추정 최적
  - Eq. (47)  0 < ω < 2 ||A||_∞⁻¹                         수렴 영역

시나리오 의존성
  - ||A||_∞는 marker 배치(D/dx, Δs/dx) 및 delta kernel에 따라 변동
    (peskin4pt 예: D/dx = 50 → 2.593,  D/dx = 40 → 2.374)
  - 초기화 단계에서 `a_norm.estimate_optimal_omega_from_type` 1회 계산
  - Solver hot-loop에서는 state 캐시된 ω 사용

참조
  - Wang, Fan, Luo (2008) — MDF iteration 핵심 구조 Eq. (18)-(27)
  - Zhang et al. (2020)   — relaxation + ||A||_∞⁻¹ 추정
"""

from __future__ import annotations

from ..backend import xp as np, _use_gpu
from .df import ibm_direct_forcing


def ibm_multi_direct_forcing(
    Lx: np.ndarray, Ly: np.ndarray,
    desired_vel: np.ndarray,
    U: np.ndarray, ro: np.ndarray,
    dx: float, dy: float, dt: float,
    Larea: float, ny: int, nx: int,
    n_iter: int = 10,
    min_iter: int = 5,
    delta_type: str = "peskin4pt",
    omega: float = 1.0,
    tol: float = 1e-4,
    rotation_ctx: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """IBM Multi-Direct Forcing (MDF) — relaxed Richardson iteration.

    누적 표현
        f_ib,total = Σ_{k=1..N_iter} ω · f_ib^(k)

    반복 거동
      - DF sub-step × k회 누적 → no-slip 조건 강제력 강화
      - L∞ slip residual 기반 조기 종료 (`r_k < tol` + `k ≥ min_iter`)
      - 발산 guard (`r_k > 1.5 · r_{k-1}` 시 중단) → 반복 안정성

    Residual 정의
        r_k = max_l ||U_d − Ũ^(k)(X_l)||₂     interface-velocity L∞ error

    본 residual은 Wang 2008 Eq. (44) l_P2-norm 및 Kempe 2012 Table 1 공식과 정확히
    일치하지는 않으며, interface velocity error 계보만 공유한다.

    Args
        Lx, Ly      : (Lb,) Lagrangian marker 좌표
        desired_vel : (Lb, 2) 목표 속도 U_d
        U           : (nodenums, 2) 거시 속도
        ro          : (nodenums,) 현재 밀도 ρ
        dx, dy, dt  : 격자 간격, 시간 스텝
        Larea       : marker arc length Δs
        ny, nx      : 격자 크기
        n_iter      : 최대 반복 횟수
        min_iter    : 조기 종료 허용 최소 반복 횟수
        delta_type  : `"peskin4pt"` | `"hat"`
        omega       : relaxation parameter ω (Zhang 2020 Eq. (46) 추정값 권장)
        tol         : slip residual 수렴 임계값
        rotation_ctx: iterative ω-torque coupling 컨텍스트 (프로젝트 확장)

    Returns
        fib_total : (nodenums, 2) 누적 IB 체적력
        stats     : {"iter_count": int, "last_residual": float, "diverged": bool, "converged": bool}
                    iter_count = 실제 누적된 iteration 수
                    diverged   = 발산 break (residual×1.5 초과)로 종료한 경우 True
                    converged  = min_iter 충족 + residual<tol 로 조기종료한 경우 True
                    둘 다 False면 정상 종료 (n_iter 도달)
    """
    nodenums = ny * nx
    Ro = ro.reshape(ny, nx)
    Eux_work = np.ascontiguousarray(U[:, 0].reshape(ny, nx))
    Euy_work = np.ascontiguousarray(U[:, 1].reshape(ny, nx))
    velocity_scale = dt / (2.0 * Ro)
    desired_vel_work = desired_vel
    min_iter = max(1, min(int(min_iter), int(n_iter)))

    # slip residual 기준
    #   - Wang 2008 Eq. (44) / Kempe 2012 Table 1 공식과 정확 일치는 아님
    #   - interface velocity error 추적이라는 점만 계보상 공유
    prev_residual = float('inf')
    iter_count = 0
    last_residual = float('inf')
    diverged = False
    converged = False

    if _use_gpu and rotation_ctx is None:
        from ..gpu_kernels import ibm_direct_forcing_fields_gpu

        fib_total_x = np.zeros((ny, nx))
        fib_total_y = np.zeros((ny, nx))
        for i in range(n_iter):
            Efx, Efy, Lux, Luy = ibm_direct_forcing_fields_gpu(
                Lx, Ly, desired_vel_work, Eux_work, Euy_work, Ro,
                dx, dy, dt, Larea, ny, nx,
                delta_type=delta_type,
            )

            vel_diff_x = desired_vel_work[:, 0] - Lux
            vel_diff_y = desired_vel_work[:, 1] - Luy
            residual = float(np.max(np.sqrt(vel_diff_x * vel_diff_x + vel_diff_y * vel_diff_y)))
            last_residual = residual

            if i > 0 and residual > prev_residual * 1.5:
                diverged = True
                iter_count = i + 1
                break

            Efx *= omega
            Efy *= omega
            fib_total_x += Efx
            fib_total_y += Efy
            Eux_work += Efx * velocity_scale
            Euy_work += Efy * velocity_scale
            prev_residual = residual
            iter_count = i + 1

            if (i + 1) >= min_iter and residual < tol:
                converged = True
                break

        fib_total = np.empty((nodenums, 2))
        fib_total[:, 0] = fib_total_x.ravel()
        fib_total[:, 1] = fib_total_y.ravel()
        return fib_total, {
            "iter_count": iter_count,
            "last_residual": last_residual,
            "diverged": diverged,
            "converged": converged,
        }

    fib_total = np.zeros((nodenums, 2))
    for i in range(n_iter):
        fib, Lux, Luy, _, _, _ = ibm_direct_forcing(
            Lx, Ly, desired_vel_work, Eux_work, Euy_work, Ro,
            dx, dy, dt, Larea, ny, nx,
            delta_type=delta_type,
        )

        # 프로젝트 L∞ slip residual
        if Lux is None or Luy is None:
            raise RuntimeError("MDF residual 계산에 필요한 Lux/Luy 누락")
        vel_diff_x = desired_vel_work[:, 0] - Lux
        vel_diff_y = desired_vel_work[:, 1] - Luy
        residual = float(np.max(np.sqrt(vel_diff_x * vel_diff_x + vel_diff_y * vel_diff_y)))
        last_residual = residual

        # 발산 감지: 잔여 residual이 이전보다 커지면 반복 발산 → 즉시 중단
        if i > 0 and residual > prev_residual * 1.5:
            diverged = True
            iter_count = i + 1
            break

        fib *= omega
        fib_total += fib
        Eux_work += fib[:, 0].reshape(ny, nx) * velocity_scale
        Euy_work += fib[:, 1].reshape(ny, nx) * velocity_scale
        prev_residual = residual
        iter_count = i + 1

        # iterative ω coupling: 누적 fib로 토크 추정 → ω 갱신 → desired_vel 재계산
        if rotation_ctx is not None:
            ctx = rotation_ctx
            fib_x = fib_total[:, 0].reshape(ny, nx)
            fib_y = fib_total[:, 1].reshape(ny, nx)
            cx_lat, cy_lat = ctx["cx"] / dx, ctx["cy"] / dy
            II, JJ = ctx["II"], ctx["JJ"]
            T_est = -float(np.sum((II - cx_lat) * fib_y - (JJ - cy_lat) * fib_x))
            omega_est = float(ctx["omega_half"] + 0.5 * dt * T_est / ctx["I_particle"])
            desired_vel_work = ctx["compute_desired_velocity"](
                ctx["vel_half"], len(Lx),
                omega=omega_est, Lx=Lx, Ly=Ly,
                cx=ctx["cx"], cy=ctx["cy"], dx=dx,
            )

        # 최소 반복 횟수 이후 residual이 충분히 작으면 조기 종료
        if (i + 1) >= min_iter and residual < tol:
            converged = True
            break

    return fib_total, {
        "iter_count": iter_count,
        "last_residual": last_residual,
        "diverged": diverged,
        "converged": converged,
    }
