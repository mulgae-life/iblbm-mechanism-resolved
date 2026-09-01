"""Full-volume 내부 유체 관성 보정 (preliminary velocity + volume integral).

핵심 식 (preliminary velocity update)
  U_p^* = (1 − ρ_f/ρ_s) · U_p^n + (ρ_f/ρ_s) · V_p⁻¹ · ∫_{Ω_p} u^* dV
  ω_p^* = (1 − ρ_f/ρ_s) · ω_p^n + (ρ_f/ρ_s) · J_p⁻¹ · ∫_{Ω_p} (r × u^*) dV

내부 체적 적분 개념 도식
        ┌────────────────────┐
        │ Eulerian mesh      │
        │  ┌──────────────┐  │
        │  │   Ω_p        │  │    ● : internal sample point (|x − x_c| ≤ R)
        │  │    ● ● ●     │  │    0 : external mesh node
        │  │  ● ● ● ● ●   │  │
        │  │    ● ● ●     │  │
        │  │              │  │    binary mask `(XX − cx)² + (YY − cy)² ≤ r²`
        │  └──────────────┘  │    ∫_{Ω_p} (·) dV  ≈  Σ_{mask} (·) · ΔA
        └────────────────────┘

계보
  - Suzuki & Inamuro (2011) scheme (C) — Lagrangian points approximation (내부 점 집합)
  - García-Villalba et al. (2023) Eq. (28)-(29) — preliminary velocity + interior Lagrangian sum
  - Breugem (2012) §4.3 — `ρ_f` internal fluid inertia 직접 처리 (rigid-body assumption 제거)

본 모듈의 지위 (중요)
  - 특정 문헌을 그대로 구현한 것이 아니라 본 프로젝트 고유의 혼합 방식
  - `FULL_VOLUME_FORCE_SCALE = 0.58` — empirical force scale, IBM volume forcing 보정 계수
  - 현재 surface IBM (`DF`/`MDF`/`DFC`) + Eulerian binary-mask internal 적분을 결합
  - `IBMStrategy` 경유로 DF/MDF/DFC에 결합되며, 다입자 DFC는 동일 `Larea` particle set 한정
"""

from __future__ import annotations

import numpy as np

from ...backend import xp as _xp, _use_gpu
from .base import register_inertia_model
from .none import NoneInertia

# 프로젝트 empirical force scale — IBM volume forcing 보정 계수
# surface IBM + internal mask 혼합 방식에서 튜닝된 값이며, 특정 문헌의 수치가 아님
FULL_VOLUME_FORCE_SCALE = 0.58

_FULL_VOLUME_STATE_FIELDS = (
    "_fv_XX_flat",
    "_fv_YY_flat",
    "_fv_Lx_c",
    "_fv_Ly_c",
    "_fv_Lb",
    "_fv_dV",
    "_fv_dV_lattice",
    "_fv_cx_init",
    "_fv_cy_init",
    "_fv_pos_old",
    "_fv_vel_old",
    "_fv_angle_old",
    "_fv_omega_old",
    "_fv_U_prelim",
    "_fv_ro_prelim",
    "_fv_Lx",
    "_fv_Ly",
    "_fv_desired_velocity",
    "_fv_force",
    "_fv_torque",
    "_fv_base_Lx_c",
    "_fv_base_Ly_c",
    "_fv_base_cx_init",
    "_fv_base_cy_init",
    "_full_volume_zero_fib",
)


def preliminary_velocity_update(vel_prev, omega_prev, U_prelim, dx, dy, cx, cy, r, rho_ratio, g_lattice, dt, gravity_direction="down", enable_rotation=False, XX_flat=None, YY_flat=None):
    """preliminary velocity/angular velocity 업데이트 (Eulerian mask 적분).

    식
      α = 1 − ρ_f/ρ_s,    β = ρ_f/ρ_s = 1/rho_ratio
      U_p^* = α · U_p^n + β · V_p⁻¹ · ∫_{Ω_p} u^* dV  ( + gravity Δt·α·g )
      ω_p^* = α · ω_p^n + β · J_p⁻¹ · ∫_{Ω_p} (r × u^*) dV

    적분 방식
      binary mask (XX − cx)² + (YY − cy)² ≤ r² → `np.sum(... [mask]) · dA`
      V_p = n_inside · dA,    dA = dx · dy
      비어있는 경우 (`n_inside = 0`) `vel_prev` 그대로 반환
    """
    dA = dx * dy
    mask = (XX_flat - cx) ** 2 + (YY_flat - cy) ** 2 <= r ** 2
    n_inside = int(_xp.sum(mask))
    if n_inside == 0:
        return vel_prev.copy(), omega_prev

    V_p = n_inside * dA
    alpha = 1.0 - 1.0 / rho_ratio
    beta = 1.0 / rho_ratio

    u_int_x = float(_xp.sum(U_prelim[mask, 0])) * dA
    u_int_y = float(_xp.sum(U_prelim[mask, 1])) * dA

    vel_new = np.empty(2)
    vel_new[0] = alpha * vel_prev[0] + beta * u_int_x / V_p
    vel_new[1] = alpha * vel_prev[1] + beta * u_int_y / V_p

    if gravity_direction == "right":
        vel_new[0] += dt * alpha * g_lattice
    else:
        vel_new[1] -= dt * alpha * g_lattice

    omega_new = omega_prev
    if enable_rotation:
        rx_lat = (XX_flat[mask] - cx) / dx
        ry_lat = (YY_flat[mask] - cy) / dy
        L_int = float(_xp.sum(rx_lat * U_prelim[mask, 1] - ry_lat * U_prelim[mask, 0])) * dA
        J_p = float(_xp.sum(rx_lat**2 + ry_lat**2)) * dA
        if J_p > 0:
            omega_new = alpha * omega_prev + beta * L_int / J_p

    return vel_new, omega_new


def full_volume_point_set(XX_flat, YY_flat, cx, cy, r, dx, dy):
    """내부 마스크로부터 등가중 internal Lagrangian point set 생성.

    식
      mask: (XX − cx)² + (YY − cy)² ≤ r²
      L_b  = Σ mask                            내부 점 개수
      ΔV   = π R² / L_b                         등가중 cell volume
      ΔV_lattice = ΔV / (dx · dy)               격자 단위 환산

    프로젝트 구현 주의
      - 내부 체적 적분 근사이며 특정 문헌의 수치가 아님
    """
    mask = (XX_flat - cx) ** 2 + (YY_flat - cy) ** 2 <= r ** 2
    Lb = int(_xp.sum(mask))
    if Lb == 0:
        return XX_flat[mask], YY_flat[mask], 0, 0.0, 0.0
    dV = float(np.pi * r ** 2 / Lb)
    dV_lattice = dV / (dx * dy)
    return XX_flat[mask], YY_flat[mask], Lb, dV, dV_lattice


def ensure_single_reference_points(state):
    """단일 입자용 internal reference point set을 한 번만 생성하여 캐시.

    - 이미 `state._fv_Lx_c`가 있으면 재사용
    - 없으면 `full_volume_point_set` 호출 후 state 필드에 저장
    - `L_b = 0`이면 내부 점이 비었다는 의미이므로 오류
    """
    if getattr(state, "_fv_Lx_c", None) is not None:
        return
    Lx_c, Ly_c, Lb, dV, dV_lattice = full_volume_point_set(
        state._fv_XX_flat,
        state._fv_YY_flat,
        float(state.particle_pos[0]),
        float(state.particle_pos[1]),
        state.r,
        state.dx,
        state.dy,
    )
    if Lb == 0:
        raise ValueError("full_volume interior point set 비어있음")
    state._fv_Lx_c = Lx_c
    state._fv_Ly_c = Ly_c
    state._fv_Lb = Lb
    state._fv_dV = dV
    state._fv_dV_lattice = dV_lattice
    state._fv_cx_init = float(state.particle_pos[0])
    state._fv_cy_init = float(state.particle_pos[1])


def ensure_multi_reference_points(state):
    """다입자용 internal reference point set을 입자별로 캐시.

    각 `ParticleState` 에 `_fv_*` 필드 추가 (단일 경로와 동일 규약)
    """
    for particle in state.particles:
        if getattr(particle, "_fv_Lx_c", None) is not None:
            continue
        Lx_c, Ly_c, Lb, dV, dV_lattice = full_volume_point_set(
            state._fv_XX_flat,
            state._fv_YY_flat,
            float(particle.pos[0]),
            float(particle.pos[1]),
            particle.r_domain,
            state.dx,
            state.dy,
        )
        if Lb == 0:
            raise ValueError("full_volume interior point set 비어있음")
        particle._fv_Lx_c = Lx_c
        particle._fv_Ly_c = Ly_c
        particle._fv_Lb = Lb
        particle._fv_dV = dV
        particle._fv_dV_lattice = dV_lattice
        particle._fv_cx_init = float(particle.pos[0])
        particle._fv_cy_init = float(particle.pos[1])


def full_volume_preliminary_update(vel_prev, omega_prev, Lx, Ly, Lux, Luy, dV, dx, dy, cx, cy, rho_ratio, g_lattice, dt, gravity_direction, enable_rotation):
    """Lagrangian point-set 합으로 근사한 preliminary velocity/angular velocity 업데이트.

    식
      α = 1 − 1/rho_ratio,   β = 1/rho_ratio
      ∫_{Ω_p} (·) dV ≈ Σ_l (·) · ΔV

      V_p = L_b · ΔV
      U_p^* = α · U_p^n + β · (Σ Lux) · ΔV / V_p   ( + gravity Δt·α·g )
      ω_p^* = α · ω_p^n + β · L_int / J_p
        L_int = Σ (r_x · Lu_y − r_y · Lu_x) · ΔV
        J_p   = Σ (r_x² + r_y²)           · ΔV

    참고
      - `preliminary_velocity_update`는 Eulerian binary mask 버전
      - 본 함수는 동일 식을 Lagrangian interior point set 기반으로 근사 (García-Villalba 2023 Eq. (28)-(29) 계보)
    """
    V_p = len(Lx) * dV
    alpha = 1.0 - 1.0 / rho_ratio
    beta = 1.0 / rho_ratio

    u_int_x = float(_xp.sum(Lux)) * dV
    u_int_y = float(_xp.sum(Luy)) * dV

    vel_new = np.empty(2, dtype=float)
    vel_new[0] = alpha * float(vel_prev[0]) + beta * u_int_x / V_p
    vel_new[1] = alpha * float(vel_prev[1]) + beta * u_int_y / V_p

    if gravity_direction == "right":
        vel_new[0] += dt * alpha * g_lattice
    else:
        vel_new[1] -= dt * alpha * g_lattice

    omega_new = omega_prev
    if enable_rotation:
        rx_lat = (Lx - cx) / dx
        ry_lat = (Ly - cy) / dy
        L_int = float(_xp.sum(rx_lat * Luy - ry_lat * Lux)) * dV
        J_p = float(_xp.sum(rx_lat ** 2 + ry_lat ** 2)) * dV
        if J_p > 0.0:
            omega_new = alpha * omega_prev + beta * L_int / J_p

    return vel_new, omega_new


def _single_preliminary_step(state, cfg, U_prelim, ro_prelim):
    from ..markers import compute_desired_velocity, update_markers
    from ...ibm.forcing import interpolate_velocity_to_points

    state._fv_pos_old = state.particle_pos.copy()
    state._fv_vel_old = state.particle_vel.copy()
    state._fv_angle_old = state.particle_angle
    state._fv_omega_old = getattr(state, "particle_omega", 0.0)
    state._fv_U_prelim = U_prelim
    state._fv_ro_prelim = ro_prelim
    ensure_single_reference_points(state)

    fv_Lx, fv_Ly = update_markers(
        state._fv_Lx_c,
        state._fv_Ly_c,
        state._fv_cx_init,
        state._fv_cy_init,
        state.particle_pos,
        angle=state.particle_angle if cfg.enable_rotation else 0.0,
    )
    Lux, Luy = interpolate_velocity_to_points(state, cfg, fv_Lx, fv_Ly, U_prelim, ro_prelim)
    vel_new, _ = full_volume_preliminary_update(
        state.particle_vel,
        getattr(state, "particle_omega", 0.0),
        fv_Lx,
        fv_Ly,
        Lux,
        Luy,
        state._fv_dV,
        state.dx,
        state.dy,
        float(state.particle_pos[0]),
        float(state.particle_pos[1]),
        cfg.rho_ratio,
        state.gravity_lattice,
        state.dt,
        cfg.gravity_direction,
        False,
    )
    state._fv_Lx = fv_Lx
    state._fv_Ly = fv_Ly
    state.particle_vel_prev = state.particle_vel.copy()
    state.particle_vel = vel_new
    vel_ib = 0.5 * (state._fv_vel_old + vel_new)
    if cfg.enable_rotation:
        state._fv_desired_velocity = compute_desired_velocity(
            vel_ib,
            state._fv_Lb,
            omega=state.particle_omega,
            Lx=fv_Lx,
            Ly=fv_Ly,
            cx=float(state.particle_pos[0]),
            cy=float(state.particle_pos[1]),
            dx=state.dx,
        )
    else:
        state._fv_desired_velocity = compute_desired_velocity(vel_ib, state._fv_Lb)
    if _use_gpu:
        import cupy as cp

        state._fv_desired_velocity = cp.asarray(state._fv_desired_velocity)


def _restore_fields(target, fields) -> None:
    for name, value in fields.items():
        setattr(target, name, value)


def _restore_force_scale(state, original_scale) -> None:
    if original_scale is None:
        if hasattr(state, "_ibm_force_scale"):
            delattr(state, "_ibm_force_scale")
        return
    state._ibm_force_scale = original_scale


def _single_ibm_step(state, cfg):
    from ...ibm import get_ibm
    from ...ibm.dispatch import apply_velocity_correction
    saved_state = {
        "Lx": state.Lx,
        "Ly": state.Ly,
        "desired_velocity": state.desired_velocity,
        "Larea": state.Larea,
    }
    prev_force_scale = getattr(state, "_ibm_force_scale", None)
    strat = get_ibm(cfg.ibm_method)
    state.ro = state._fv_ro_prelim.copy()
    state.U = state._fv_U_prelim.copy()
    try:
        state.Lx = state._fv_Lx
        state.Ly = state._fv_Ly
        state.desired_velocity = state._fv_desired_velocity
        state.Larea = state._fv_dV_lattice
        state._ibm_force_scale = FULL_VOLUME_FORCE_SCALE
        strat.prepare(state, cfg)
        strat.apply_single(state, cfg, 0)
        if strat.requires_velocity_correction():
            apply_velocity_correction(state, cfg)
        state._fv_force = strat.extract_force(state, cfg)
        if cfg.enable_rotation:
            state._fv_torque = strat.extract_torque(
                state,
                cfg,
                float(state.particle_pos[0]),
                float(state.particle_pos[1]),
            )
    finally:
        _restore_fields(state, saved_state)
        _restore_force_scale(state, prev_force_scale)


def _single_post_step(state, cfg):
    from ..markers import compute_desired_velocity, update_markers

    state.particle_force = state._fv_force
    if cfg.enable_rotation:
        state.particle_torque = state._fv_torque
        state.particle_omega = state._fv_omega_old + state.dt * state._fv_torque / state.particle_I
        state.particle_angle = state._fv_angle_old + 0.5 * state.dt * (state._fv_omega_old + state.particle_omega)
    state.particle_pos = state._fv_pos_old + 0.5 * state.dt * (state._fv_vel_old + state.particle_vel) * state.dx
    state.Lx, state.Ly = update_markers(
        state._fv_base_Lx_c,
        state._fv_base_Ly_c,
        state._fv_base_cx_init,
        state._fv_base_cy_init,
        state.particle_pos,
        angle=state.particle_angle if cfg.enable_rotation else 0.0,
    )
    if cfg.enable_rotation:
        state.desired_velocity = compute_desired_velocity(
            state.particle_vel,
            state.Lb,
            omega=state.particle_omega,
            Lx=state.Lx,
            Ly=state.Ly,
            cx=float(state.particle_pos[0]),
            cy=float(state.particle_pos[1]),
            dx=state.dx,
        )
    else:
        state.desired_velocity = compute_desired_velocity(state.particle_vel, state.Lb)


def _multi_preliminary_step(state, cfg, U_prelim, ro_prelim):
    from ..markers import compute_desired_velocity, update_markers
    from ...ibm.forcing import interpolate_velocity_to_points

    state._fv_U_prelim = U_prelim
    state._fv_ro_prelim = ro_prelim
    ensure_multi_reference_points(state)
    for particle in state.particles:
        particle._fv_pos_old = particle.pos.copy()
        particle._fv_vel_old = particle.vel.copy()
        particle._fv_angle_old = particle.angle
        particle._fv_omega_old = particle.omega
        fv_Lx, fv_Ly = update_markers(
            particle._fv_Lx_c,
            particle._fv_Ly_c,
            particle._fv_cx_init,
            particle._fv_cy_init,
            particle.pos,
            angle=particle.angle if cfg.enable_rotation else 0.0,
        )
        Lux, Luy = interpolate_velocity_to_points(state, cfg, fv_Lx, fv_Ly, U_prelim, ro_prelim)
        vel_new, _ = full_volume_preliminary_update(
            particle.vel,
            particle.omega,
            fv_Lx,
            fv_Ly,
            Lux,
            Luy,
            particle._fv_dV,
            state.dx,
            state.dy,
            float(particle.pos[0]),
            float(particle.pos[1]),
            particle.rho_ratio,
            state.gravity_lattice,
            state.dt,
            cfg.gravity_direction,
            False,
        )
        particle._fv_Lx = fv_Lx
        particle._fv_Ly = fv_Ly
        particle.vel_prev = particle.vel.copy()
        particle.vel = vel_new
        vel_ib = 0.5 * (particle._fv_vel_old + vel_new)
        if cfg.enable_rotation:
            particle._fv_desired_velocity = compute_desired_velocity(
                vel_ib,
                particle._fv_Lb,
                omega=particle.omega,
                Lx=fv_Lx,
                Ly=fv_Ly,
                cx=float(particle.pos[0]),
                cy=float(particle.pos[1]),
                dx=state.dx,
            )
        else:
            particle._fv_desired_velocity = compute_desired_velocity(vel_ib, particle._fv_Lb)
        if _use_gpu:
            import cupy as cp

            particle._fv_desired_velocity = cp.asarray(particle._fv_desired_velocity)


def _multi_ibm_step(state, cfg):
    from ...ibm import get_ibm
    from ...ibm.dispatch import apply_velocity_correction
    saved_state = {
        "Lx": state.Lx,
        "Ly": state.Ly,
        "Larea": state.Larea,
    }
    saved_particles = []
    prev_force_scale = getattr(state, "_ibm_force_scale", None)
    strat = get_ibm(cfg.ibm_method)
    state.ro = state._fv_ro_prelim.copy()
    state.U = state._fv_U_prelim.copy()
    try:
        for particle in state.particles:
            saved_particles.append((
                particle,
                {
                    "Lx": particle.Lx,
                    "Ly": particle.Ly,
                    "desired_velocity": particle.desired_velocity,
                    "Larea": particle.Larea,
                },
            ))
            particle.Lx = particle._fv_Lx
            particle.Ly = particle._fv_Ly
            particle.desired_velocity = particle._fv_desired_velocity
            particle.Larea = particle._fv_dV_lattice
        if state.particles:
            lead = state.particles[0]
            state.Lx = lead._fv_Lx
            state.Ly = lead._fv_Ly
            state.Larea = lead._fv_dV_lattice
        state._ibm_force_scale = FULL_VOLUME_FORCE_SCALE
        strat.prepare(state, cfg)
        strat.apply_multi(state, cfg, 0)
        if strat.requires_velocity_correction():
            apply_velocity_correction(state, cfg)
    finally:
        _restore_fields(state, saved_state)
        for particle, fields in saved_particles:
            _restore_fields(particle, fields)
        _restore_force_scale(state, prev_force_scale)


def _multi_post_step(state, cfg):
    from ..markers import compute_desired_velocity, update_markers

    for particle in state.particles:
        particle.force = particle._fib_force
        if cfg.enable_rotation:
            particle.torque = particle._torque_new
            particle.omega = particle._fv_omega_old + state.dt * particle._torque_new / particle.I_particle
            particle.angle = particle._fv_angle_old + 0.5 * state.dt * (particle._fv_omega_old + particle.omega)
        particle.pos = particle._fv_pos_old + 0.5 * state.dt * (particle._fv_vel_old + particle.vel) * state.dx
        particle.Lx, particle.Ly = update_markers(
            particle.Lx_c,
            particle.Ly_c,
            particle.cx_init,
            particle.cy_init,
            particle.pos,
            angle=particle.angle if cfg.enable_rotation else 0.0,
        )
        if cfg.enable_rotation:
            particle.desired_velocity = compute_desired_velocity(
                particle.vel,
                particle.Lb,
                omega=particle.omega,
                Lx=particle.Lx,
                Ly=particle.Ly,
                cx=float(particle.pos[0]),
                cy=float(particle.pos[1]),
                dx=state.dx,
            )
        else:
            particle.desired_velocity = compute_desired_velocity(particle.vel, particle.Lb)


class FullVolumeInertia(NoneInertia):
    """full-volume 경로용 ExtendedInertiaHook.

    step sequence (solver 관점)
      1. `attach_state`    — `_fv_base_*` 초기 마커 snapshot 저장
      2. `preliminary_step` — forcing-free preliminary u^*, ρ^* 계산 후 interior 적분
      3. `ibm_step`         — IBM force 평가 + velocity correction
      4. `post_step`        — 최종 pos/vel/omega/angle 갱신 + 마커 재생성
      5. `detach_state`     — `_fv_*` 캐시 해제

    `uses_preliminary_update = True` → solver가 preliminary-state 경로를 활성화
    """

    name = "full_volume"
    uses_preliminary_update = True

    def attach_state(self, state) -> None:
        if getattr(state, "_fv_XX_flat", None) is None or getattr(state, "_fv_YY_flat", None) is None:
            raise ValueError("full_volume은 sedimentation state 초기화 이후에만 사용 가능")
        if getattr(state, "particles", None) is None:
            from ...runtime.device import to_cpu

            state._fv_base_Lx_c = to_cpu(state.Lx.copy())
            state._fv_base_Ly_c = to_cpu(state.Ly.copy())
            state._fv_base_cx_init = float(state.particle_pos[0])
            state._fv_base_cy_init = float(state.particle_pos[1])

    def detach_state(self, state) -> None:
        for name in _FULL_VOLUME_STATE_FIELDS:
            if hasattr(state, name):
                setattr(state, name, None)
        if getattr(state, "particles", None) is not None:
            for particle in state.particles:
                for name in _FULL_VOLUME_STATE_FIELDS:
                    if hasattr(particle, name):
                        setattr(particle, name, None)

    def preliminary_step(self, state, cfg, snapshot) -> None:
        if snapshot is None:
            raise ValueError("full_volume preliminary_step requires snapshot")

        from ...runtime import step as step_mod

        fstar_prev, feq_prev, U_prev, ro_prev = snapshot
        fstar_prelim, ro_prelim, U_prelim = step_mod.forcing_free_preliminary_state(
            state,
            cfg,
            0,
            fstar_prev=fstar_prev,
            feq_prev=feq_prev,
            U_prev=U_prev,
            ro_prev=ro_prev,
        )
        state.fstar = fstar_prelim
        if getattr(state, "particles", None) is None:
            _single_preliminary_step(state, cfg, U_prelim, ro_prelim)
        else:
            _multi_preliminary_step(state, cfg, U_prelim, ro_prelim)

    def ibm_step(self, state, cfg) -> None:
        if getattr(state, "particles", None) is None:
            _single_ibm_step(state, cfg)
        else:
            _multi_ibm_step(state, cfg)

    def post_step(self, state, cfg) -> None:
        if getattr(state, "particles", None) is None:
            _single_post_step(state, cfg)
        else:
            _multi_post_step(state, cfg)

    def preliminary_velocity_update(self, *args, **kwargs):
        return preliminary_velocity_update(*args, **kwargs)


register_inertia_model(FullVolumeInertia())
