"""다입자 IBM helper: 입자별 DF/MDF closure 반복 + 유체력·토크 회수.

구조
  multi_particle_df           — 입자 순회, 입자별 DF closure (Uhlmann 2005 Eq. (9), (12c))
  multi_particle_mdf          — 단일 입자는 Wang 2008 반복, 다입자는 coupled 경로로
  multi_particle_mdf_coupled  — 다입자 MDF (Breugem 2012 Eq. (10a-d) 계보,
                                Zhang 2020 ω 추정 + residual 기반 adaptive)

공통 후처리
  F_h = −Σ_l F_l Δs         각 입자에 `particle._fib_force`
  T_h = −Σ_l r_l × F_l Δs   각 입자에 `particle._torque_new` (회전 활성 시)
  force_scale               `settling_inertia_model="full_volume"` empirical 0.58 등 프로젝트 확장 경로

참조 계보
  - Uhlmann (2005)      direct forcing 원전
  - Wang et al. (2008)  multi-direct forcing (고정 반복)
  - Breugem (2012)      식 (10) multidirect forcing 반복 구조 + retraction
  - Zhang et al. (2020) ω ≈ ‖A‖_∞⁻¹ 추정 및 residual 기반 조기 종료
"""
from __future__ import annotations

from ..backend import xp as np, _use_gpu
from .a_norm import estimate_optimal_omega_from_type
from .df import ibm_direct_forcing
from .forcing import ibm_df_particle_lagrangian_closure
from .mdf import ibm_multi_direct_forcing

# Stability cap for the coupled-iteration relaxation factor. Deep coupled
# forcing with an aggressive relaxation factor can over-drive the boundary
# fluid and resonate with the particle reaction; 0.8 is the value validated
# by the completed two-particle sedimentation production runs.
MDF_COUPLED_OMEGA_MAX = 0.8



def ibm_mdf_particle(state, cfg, Lx, Ly, desired_vel, Larea):
    """단일 입자 MDF 래퍼 — `ibm_multi_direct_forcing` 의 state/cfg 어댑터.

    Returns
        (fib_total, stats) — stats = {iter_count, last_residual, diverged, converged}
    """
    return ibm_multi_direct_forcing(
        Lx,
        Ly,
        desired_vel,
        state.U,
        state.ro,
        state.dx,
        state.dy,
        state.dt,
        Larea,
        state.ny,
        state.nx,
        n_iter=cfg.mdf_iterations,
        min_iter=cfg.mdf_min_iterations,
        delta_type=cfg.delta_type,
        omega=state._mdf_omega,
        tol=cfg.mdf_tolerance,
    )


def multi_particle_mdf_coupled(state, cfg):
    """다입자 MDF coupled iteration.

    Breugem (2012) Eq. (10) 계열의 반복 구조 + Zhang et al. (2020) ω 추정

      do s = 1, N_s
        입자별 보간 Ũ_l^{s-1} → 강제항 F_l^s → relaxed update
        residual = max_l ‖U_d − Ũ_l‖
        residual 증가 시 ω ← max(ω·0.5, ω_min)
        residual < tol 도달 + min_iter 충족 시 종료
      enddo

    단일 입자 경로와 달리 `U_work` 를 입자 간 공유하여 coupled 보정
    """
    import numpy as _np

    ny, nx = state.ny, state.nx
    force_scale = float(getattr(state, "_ibm_force_scale", 1.0))
    if getattr(state, "_mdf_omega", None) is None:
        state._mdf_omega = estimate_optimal_omega_from_type(
            Lx=state.Lx,
            Ly=state.Ly,
            delta_type=cfg.delta_type,
            Larea=float(state.particles[0].Larea),
            dx=float(state.dx),
        )

    omega_opt = state._mdf_omega
    omega_use = min(omega_opt * 0.8, MDF_COUPLED_OMEGA_MAX)
    omega_min = omega_use * 0.125
    guard_trips = 0
    iter_count = 0
    last_residual = float("inf")
    diverged = False
    converged = False

    Ro_2d = state.ro.reshape(ny, nx)
    Eux_work = np.ascontiguousarray(state.U[:, 0].reshape(ny, nx))
    Euy_work = np.ascontiguousarray(state.U[:, 1].reshape(ny, nx))
    velocity_scale = None if cfg.incompressible_lbgk else state.dt / (2.0 * Ro_2d)
    fib_total = np.zeros((state.nodenums, 2))
    particle_fib = [np.zeros((state.nodenums, 2)) for _ in state.particles]
    particle_inputs = [_particle_marker_inputs(p) for p in state.particles] if _use_gpu else None

    n_iter = cfg.mdf_iterations
    min_iter = max(1, min(int(cfg.mdf_min_iterations), int(n_iter)))
    tol = cfg.mdf_tolerance
    prev_residual = float("inf")

    if _use_gpu:
        from ..gpu_kernels import ibm_direct_forcing_fields_gpu

        fib_total_x = np.zeros((ny, nx))
        fib_total_y = np.zeros((ny, nx))
        particle_fib_x = [np.zeros((ny, nx)) for _ in state.particles]
        particle_fib_y = [np.zeros((ny, nx)) for _ in state.particles]

        for it in range(n_iter):
            fib_this_x = np.zeros((ny, nx))
            fib_this_y = np.zeros((ny, nx))
            fib_per_particle = []
            residual_max = 0.0

            for p_idx, particle in enumerate(state.particles):
                Lx_use, Ly_use, dv_use = particle_inputs[p_idx]
                Efx, Efy, Lux, Luy = ibm_direct_forcing_fields_gpu(
                    Lx_use,
                    Ly_use,
                    dv_use,
                    Eux_work,
                    Euy_work,
                    Ro_2d,
                    state.dx,
                    state.dy,
                    state.dt,
                    float(particle.Larea),
                    ny,
                    nx,
                    delta_type=cfg.delta_type,
                    incompressible=cfg.incompressible_lbgk,
                )

                vdx = dv_use[:, 0] - Lux
                vdy = dv_use[:, 1] - Luy
                res_p = np.max(np.sqrt(vdx * vdx + vdy * vdy))
                residual_max = max(residual_max, float(res_p))

                Efx *= omega_use
                Efy *= omega_use
                fib_this_x += Efx
                fib_this_y += Efy
                fib_per_particle.append((Efx, Efy))

            if it > 0 and residual_max > prev_residual * 1.2:
                guard_trips += 1
                omega_use = max(omega_use * 0.5, omega_min)
                if omega_use <= omega_min:
                    diverged = True
                    break
                # A rejected iteration applies no force, so the next iteration sees the
                # same residual; refresh the baseline so the halved relaxation factor
                # actually gets a chance to act.
                prev_residual = residual_max
                continue

            fib_total_x += fib_this_x
            fib_total_y += fib_this_y
            for p_idx, (Efx, Efy) in enumerate(fib_per_particle):
                particle_fib_x[p_idx] += Efx
                particle_fib_y[p_idx] += Efy
            if cfg.incompressible_lbgk:
                Eux_work += 0.5 * fib_this_x * state.dt
                Euy_work += 0.5 * fib_this_y * state.dt
            else:
                Eux_work += fib_this_x * velocity_scale
                Euy_work += fib_this_y * velocity_scale
            prev_residual = residual_max
            iter_count += 1
            last_residual = float(residual_max)

            if (it + 1) >= min_iter and residual_max < tol:
                converged = True
                break

        if force_scale != 1.0:
            fib_total_x = force_scale * fib_total_x
            fib_total_y = force_scale * fib_total_y
        state.fib = np.empty((state.nodenums, 2))
        state.fib[:, 0] = fib_total_x.ravel()
        state.fib[:, 1] = fib_total_y.ravel()

        for p_idx, particle in enumerate(state.particles):
            fib_x = particle_fib_x[p_idx]
            fib_y = particle_fib_y[p_idx]
            if force_scale != 1.0:
                fib_x = force_scale * fib_x
                fib_y = force_scale * fib_y
            particle._fib_force = _np.array([
                -float(fib_x.sum()),
                -float(fib_y.sum()),
            ])
            if cfg.enable_rotation:
                cx_lat = float(particle.pos[0]) / state.dx
                cy_lat = float(particle.pos[1]) / state.dy
                particle._torque_new = -float(np.sum(
                    (state._torque_II - cx_lat) * fib_y - (state._torque_JJ - cy_lat) * fib_x
                ))
        state._mdf_iter_stats = {
            "iter_count": iter_count,
            "last_residual": last_residual,
            "diverged": diverged,
            "converged": converged,
            "guard_trips": guard_trips,
            "omega_used": float(omega_use),
        }
        return

    for it in range(n_iter):
        fib_this_iter = np.zeros((state.nodenums, 2))
        fib_per_particle = [np.zeros((state.nodenums, 2)) for _ in state.particles]
        residual_max = 0.0

        for p_idx, particle in enumerate(state.particles):
            if _use_gpu:
                Lx_use, Ly_use, dv_use = particle_inputs[p_idx]
            else:
                Lx_use = particle.Lx
                Ly_use = particle.Ly
                dv_use = particle.desired_velocity

            fib_p, Lux, Luy, _, _, _ = ibm_direct_forcing(
                Lx_use,
                Ly_use,
                dv_use,
                Eux_work,
                Euy_work,
                Ro_2d,
                state.dx,
                state.dy,
                state.dt,
                float(particle.Larea),
                ny,
                nx,
                delta_type=cfg.delta_type,
                incompressible=cfg.incompressible_lbgk,
            )

            if Lux is not None and Luy is not None:
                vdx = dv_use[:, 0] - Lux
                vdy = dv_use[:, 1] - Luy
                res_p = np.max(np.sqrt(vdx * vdx + vdy * vdy))
                residual_max = max(residual_max, float(res_p))

            fib_p *= omega_use
            fib_this_iter += fib_p
            fib_per_particle[p_idx] = fib_p

        if it > 0 and residual_max > prev_residual * 1.2:
            guard_trips += 1
            omega_use = max(omega_use * 0.5, omega_min)
            if omega_use <= omega_min:
                diverged = True
                break
            # A rejected iteration applies no force, so the next iteration sees the
            # same residual; refresh the baseline so the halved relaxation factor
            # actually gets a chance to act.
            prev_residual = residual_max
            continue

        fib_total += fib_this_iter
        for p_idx in range(len(state.particles)):
            particle_fib[p_idx] += fib_per_particle[p_idx]
        if cfg.incompressible_lbgk:
            Eux_work += 0.5 * fib_this_iter[:, 0].reshape(ny, nx) * state.dt
            Euy_work += 0.5 * fib_this_iter[:, 1].reshape(ny, nx) * state.dt
        else:
            Eux_work += fib_this_iter[:, 0].reshape(ny, nx) * velocity_scale
            Euy_work += fib_this_iter[:, 1].reshape(ny, nx) * velocity_scale
        prev_residual = residual_max
        iter_count += 1
        last_residual = float(residual_max)

        if (it + 1) >= min_iter and residual_max < tol:
            converged = True
            break

    if force_scale != 1.0:
        fib_total = force_scale * fib_total
    state.fib = fib_total

    for p_idx, particle in enumerate(state.particles):
        fib_p = particle_fib[p_idx]
        if force_scale != 1.0:
            fib_p = force_scale * fib_p
        particle._fib_force = _np.array([
            -float(fib_p[:, 0].sum()),
            -float(fib_p[:, 1].sum()),
        ])
        if cfg.enable_rotation:
            fib_x = fib_p[:, 0].reshape(ny, nx)
            fib_y = fib_p[:, 1].reshape(ny, nx)
            cx_lat = float(particle.pos[0]) / state.dx
            cy_lat = float(particle.pos[1]) / state.dy
            particle._torque_new = -float(np.sum(
                (state._torque_II - cx_lat) * fib_y - (state._torque_JJ - cy_lat) * fib_x
            ))

    state._mdf_iter_stats = {
        "iter_count": iter_count,
        "last_residual": last_residual,
        "diverged": diverged,
        "converged": converged,
        "guard_trips": guard_trips,
        "omega_used": float(omega_use),
    }


def _particle_marker_inputs(particle):
    """입자 marker 입력 3종 (Lx, Ly, desired_velocity) — GPU 전용 cupy 배열 변환."""
    if _use_gpu:
        import cupy as cp

        return (
            cp.asarray(particle.Lx),
            cp.asarray(particle.Ly),
            cp.asarray(particle.desired_velocity),
        )
    return particle.Lx, particle.Ly, particle.desired_velocity


def _particle_torque_from_fib(state, particle, fib_p) -> float:
    """Eulerian fib_p 로부터 입자 토크 `T_h = −Σ (r_x f_y − r_y f_x)` 계산.

    `state._torque_II`, `state._torque_JJ` 는 격자 인덱스 기반 상대좌표 precomputed buffer
    """
    ny, nx = state.ny, state.nx
    fib_x = fib_p[:, 0].reshape(ny, nx)
    fib_y = fib_p[:, 1].reshape(ny, nx)
    cx_lat = float(particle.pos[0]) / state.dx
    cy_lat = float(particle.pos[1]) / state.dy
    return -float(np.sum(
        (state._torque_II - cx_lat) * fib_y - (state._torque_JJ - cy_lat) * fib_x
    ))


def multi_particle_df(state, cfg):
    """다입자 DF 분배 — 입자별 `ibm_df_particle_lagrangian_closure` 합산.

    두 경로
      - `cfg.df_coupled` and Np > 1   : `U_work` 공유 coupled 보정
      - 그 외                         : 입자 독립 스윕 후 `state.fib` 에 누적

    각 입자 `particle._fib_force = F_h`, 회전 시 `particle._torque_new = T_h`
    """
    import numpy as _np

    force_scale = float(getattr(state, "_ibm_force_scale", 1.0))
    df_coupled = getattr(cfg, "df_coupled", False)

    if df_coupled and len(state.particles) > 1:
        U_work = state.U.copy()
        fib_total = np.zeros((state.nodenums, 2))

        for particle in state.particles:
            Lx_use, Ly_use, dv_use = _particle_marker_inputs(particle)

            fib_p, fib_force, torque_new = ibm_df_particle_lagrangian_closure(
                state,
                cfg,
                Lx_use,
                Ly_use,
                dv_use,
                float(particle.Larea),
                float(particle.pos[0]),
                float(particle.pos[1]),
            )
            if force_scale != 1.0:
                fib_p = force_scale * fib_p
                fib_force = force_scale * fib_force
                if torque_new is not None:
                    torque_new = force_scale * torque_new
            fib_total += fib_p
            particle._fib_force = fib_force
            if cfg.enable_rotation:
                particle._torque_new = torque_new

            if cfg.incompressible_lbgk:
                U_work = U_work + 0.5 * fib_p * state.dt
            else:
                U_work = U_work + fib_p * state.dt / (2.0 * state.ro[:, None])

        state.fib = fib_total
        return

    fib_total = np.zeros((state.nodenums, 2))
    for particle in state.particles:
        Lx_use, Ly_use, dv_use = _particle_marker_inputs(particle)
        fib_p, fib_force, torque_new = ibm_df_particle_lagrangian_closure(
            state,
            cfg,
            Lx_use,
            Ly_use,
            dv_use,
            float(particle.Larea),
            float(particle.pos[0]),
            float(particle.pos[1]),
        )
        if force_scale != 1.0:
            fib_p = force_scale * fib_p
            fib_force = force_scale * fib_force
            if torque_new is not None:
                torque_new = force_scale * torque_new

        fib_total += fib_p
        particle._fib_force = fib_force

        if cfg.enable_rotation:
            particle._torque_new = torque_new

    state.fib = fib_total


def multi_particle_mdf(state, cfg):
    """MDF 분배 — 단일 입자는 Wang 2008 고정 반복, 다입자는 coupled 경로 위임.

    단일 입자 경로: `ibm_mdf_particle` 호출 후 `F_h = −Σ f Δs` 집계
    다입자 경로: `multi_particle_mdf_coupled` 로 위임 (coupled 반복 + Zhang 2020 ω)
    """
    import numpy as _np

    if len(state.particles) > 1:
        multi_particle_mdf_coupled(state, cfg)
        return

    force_scale = float(getattr(state, "_ibm_force_scale", 1.0))
    fib_total = np.zeros((state.nodenums, 2))
    for particle in state.particles:
        Lx_use, Ly_use, dv_use = _particle_marker_inputs(particle)
        fib_p, mdf_stats = ibm_mdf_particle(state, cfg, Lx_use, Ly_use, dv_use, particle.Larea)
        state._mdf_iter_stats = mdf_stats
        fib_force = _np.array([
            -float(fib_p[:, 0].sum()),
            -float(fib_p[:, 1].sum()),
        ])
        if force_scale != 1.0:
            fib_p = force_scale * fib_p
            fib_force = force_scale * fib_force

        fib_total += fib_p
        particle._fib_force = fib_force
        if cfg.enable_rotation:
            particle._torque_new = _particle_torque_from_fib(state, particle, fib_p)

    state.fib = fib_total
