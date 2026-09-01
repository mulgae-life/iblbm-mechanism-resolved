"""단일 입자 침강 runtime (Wang 2008 / Majumder 2023 benchmark anchor).

Wang 2008 §3.1 (Glowinski 2001 parameters)
  - 도메인 0.2 × 0.6 cm, D_p = 0.25 cm, 초기 중심 (1, 4) cm
  - ρ_p ∈ {1.25, 1.5} g/cm³, ρ_f = 1.0 g/cm³
  - μ ∈ {0.1, 0.01} g/(cm·s)
  - H/D = 16 (장방형 채널, Uhlmann anchor 와 공통 frame)

Majumder 2023 §4.4 (2입자 사례지만 침강 기준값 비교에 사용)
  - 비차원: Re_max = |u_c|·D/ν (particle velocity 기반)
  - Majumder 2-particle Heavy Re_max ≈ 276.61, Light ≈ 231.41

hook 구성
  - `initialize`   IBM prepare + inertia attach_state (explicit_history 는 hook 객체)
  - `pre_step`     prescribed_velocity / verlet half / euler_explicit 분기
                   입자 위치/desired_velocity 갱신 (rotation_coupling 반영)
                   GPU 모드에서 host→device 전송 (Lx, Ly, desired_velocity)
  - `pre_ibm`      ExtendedInertiaHook 이면 preliminary_step 수행
  - `apply_ibm`    inertia hook 이거나 표준 IBM 경로
  - `post_ibm`     Verlet full / Euler full 시간 적분, rotation coupling,
                   hydrodynamic force 추출 후 particle_* 필드 갱신
  - `diagnostics`  y*, v*, internal residual/momentum/angular momentum 기록
  - `should_stop`  nan / contact / offset / domain_bounds 순 판정
"""

from __future__ import annotations

import math

from ...backend import _use_gpu
from ...diagnostics import (
    compute_inside_residual,
    compute_internal_angular_momentum,
    compute_internal_momentum,
    compute_l_int_z,
    record_sedimentation_state,
)
from ...ibm import get_ibm
from ...physics.inertia import ExtendedInertiaHook, get_inertia_model
from ...physics.sedimentation import (
    compute_desired_velocity,
    euler_explicit_rotation,
    euler_explicit_step_with_inertia_model,
    rotation_full_step,
    rotation_half_step,
    update_markers,
    verlet_full_step_with_inertia_model,
    verlet_half_step_with_inertia_model,
)
from .. import step as step_mod
from ..device import to_cpu
from .base import register_scenario
from .sedimentation_common import sedimentation_bound_warning, sedimentation_stop_reason


class SedimentationSingleRuntime:
    """단일 입자 침강 시나리오 runtime handler (`cfg.particles_config is None`)."""

    name = "sedimentation_single"

    def matches(self, cfg) -> bool:
        return (
            cfg.motion_type == "sedimentation"
            and cfg.particles_config is None
            and cfg.scenario_type is None
        )

    def initialize(self, state, cfg) -> dict:
        ibm = get_ibm(cfg.ibm_method)
        inertia = get_inertia_model(cfg.settling_inertia_model)
        ibm.prepare(state, cfg)
        if isinstance(inertia, ExtendedInertiaHook):
            inertia.attach_state(state)
        return {
            "Lx_c": to_cpu(state.Lx.copy()),
            "Ly_c": to_cpu(state.Ly.copy()),
            "cx_init": cfg.cylinder_center[0],
            "cy_init": cfg.cylinder_center[1],
            "y0_init": cfg.cylinder_center[1],
            "d_lattice": 2.0 * state.lattice_r,
            "history": [],
            "vel_half_cache": None,
            "omega_half_cache": 0.0,
            "last_record": None,
            "ibm": ibm,
            "inertia": inertia,
            "termination_reason": "max_steps",
            "extended_inertia_snapshot": None,
        }

    def pre_step(self, state, cfg, cache, ttt: int) -> None:
        if cfg.prescribed_velocity is not None:
            import numpy as _np
            pv = _np.array(cfg.prescribed_velocity, dtype=float)
            state.particle_pos = state.particle_pos + state.dt * pv * state.dx
            state.Lx, state.Ly = update_markers(cache["Lx_c"], cache["Ly_c"], cache["cx_init"], cache["cy_init"], state.particle_pos)
            state.desired_velocity = compute_desired_velocity(pv, state.Lb)
            cache["vel_half_cache"] = pv
        elif cfg.time_integrator == "euler_explicit":
            pass
        elif isinstance(cache["inertia"], ExtendedInertiaHook):
            pass
        else:
            pos_new, vel_half = verlet_half_step_with_inertia_model(
                state.particle_pos,
                state.particle_vel,
                state.particle_vel_prev,
                state.particle_force,
                state.particle_mass,
                cfg.rho_ratio,
                state.lattice_r,
                state.gravity_lattice,
                state.dt,
                state.dx,
                cfg.settling_inertia_model,
                gravity_direction=cfg.gravity_direction,
            )
            state.particle_pos = pos_new
            if cfg.enable_rotation:
                omega_half, angle_inc = rotation_half_step(
                    state.particle_omega,
                    state.particle_torque,
                    state.particle_I,
                    state.dt,
                )
                state.particle_angle += angle_inc
                cache["omega_half_cache"] = omega_half
            else:
                omega_half = 0.0
            state.Lx, state.Ly = update_markers(
                cache["Lx_c"],
                cache["Ly_c"],
                cache["cx_init"],
                cache["cy_init"],
                pos_new,
                angle=state.particle_angle if cfg.enable_rotation else 0.0,
            )
            if cfg.rotation_coupling == "semi_implicit" and cfg.enable_rotation:
                state.desired_velocity = compute_desired_velocity(
                    vel_half,
                    state.Lb,
                    omega=omega_half,
                    Lx=state.Lx,
                    Ly=state.Ly,
                    cx=float(state.particle_pos[0]),
                    cy=float(state.particle_pos[1]),
                    dx=state.dx,
                )
            else:
                state.desired_velocity = compute_desired_velocity(vel_half, state.Lb)
            cache["vel_half_cache"] = vel_half
            if cfg.rotation_coupling == "iterative" and cfg.enable_rotation:
                state._vel_half_cache = vel_half
                state._omega_half_cache = omega_half

        if _use_gpu:
            import cupy as cp
            state.Lx = cp.asarray(state.Lx)
            state.Ly = cp.asarray(state.Ly)
            state.desired_velocity = cp.asarray(state.desired_velocity)

    def pre_ibm(self, state, cfg, cache, ttt: int) -> None:
        inertia = cache["inertia"]
        if not isinstance(inertia, ExtendedInertiaHook):
            return
        inertia.preliminary_step(state, cfg, cache.get("extended_inertia_snapshot"))

    def apply_ibm(self, state, cfg, cache, ttt: int) -> None:
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            cache["inertia"].ibm_step(state, cfg)
        else:
            step_mod.apply_standard_ibm(state, cfg, ttt)

    def post_ibm(self, state, cfg, cache, ttt: int) -> None:
        ibm = cache["ibm"]
        if cfg.prescribed_velocity is not None:
            F_hydro = ibm.extract_force(state, cfg)
            import numpy as _np
            state.particle_vel = _np.array(cfg.prescribed_velocity, dtype=float)
            state.particle_force = F_hydro
            return
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            cache["inertia"].post_step(state, cfg)
            return
        F_hydro = ibm.extract_force(state, cfg)

        if cfg.time_integrator == "euler_explicit":
            pos_old = state.particle_pos.copy()
            omega_new = 0.0
            if cfg.enable_rotation:
                T = ibm.extract_torque(
                    state,
                    cfg,
                    float(state.particle_pos[0]),
                    float(state.particle_pos[1]),
                )
                l_int_now = compute_l_int_z(
                    state.U,
                    state.ro,
                    state.nx,
                    state.ny,
                    state.dx,
                    float(pos_old[0]),
                    float(pos_old[1]),
                    state.r,
                )
                dl_int = l_int_now - state.particle_l_int_prev
                state.particle_l_int_prev = l_int_now
                omega_new, angle_inc = euler_explicit_rotation(
                    state.particle_omega,
                    T,
                    state.particle_I,
                    state.dt,
                    dl_int=dl_int,
                    angle_update=cfg.sedimentation_euler_update_scheme,
                )
                state.particle_angle += angle_inc
                state.particle_torque = T

            pos_new, vel_new = euler_explicit_step_with_inertia_model(
                state.particle_pos,
                state.particle_vel,
                state.particle_vel_prev,
                F_hydro,
                state.particle_mass,
                cfg.rho_ratio,
                state.lattice_r,
                state.gravity_lattice,
                state.dt,
                state.dx,
                cfg.settling_inertia_model,
                position_update=cfg.sedimentation_euler_update_scheme,
                gravity_direction=cfg.gravity_direction,
            )
            vel_new = cache["inertia"].post_correction(vel_new, state.particle_vel, state.particle_vel_prev, cfg.rho_ratio)
            if cfg.enable_rotation:
                state.particle_omega = omega_new
            state.particle_vel_prev = state.particle_vel.copy()
            state.particle_pos = pos_new
            state.particle_vel = vel_new
            state.particle_force = F_hydro
            state.Lx, state.Ly = update_markers(
                cache["Lx_c"],
                cache["Ly_c"],
                cache["cx_init"],
                cache["cy_init"],
                pos_new,
                angle=state.particle_angle if cfg.enable_rotation else 0.0,
            )
            if cfg.enable_rotation:
                state.desired_velocity = compute_desired_velocity(
                    vel_new,
                    state.Lb,
                    omega=omega_new,
                    Lx=state.Lx,
                    Ly=state.Ly,
                    cx=float(pos_new[0]),
                    cy=float(pos_new[1]),
                    dx=state.dx,
                )
            else:
                state.desired_velocity = compute_desired_velocity(vel_new, state.Lb)
            return

        vel_new = verlet_full_step_with_inertia_model(
            cache["vel_half_cache"],
            state.particle_vel,
            state.particle_vel_prev,
            F_hydro,
            state.particle_mass,
            cfg.rho_ratio,
            state.lattice_r,
            state.gravity_lattice,
            state.dt,
            cfg.settling_inertia_model,
            gravity_direction=cfg.gravity_direction,
        )
        omega_new = 0.0
        if cfg.enable_rotation:
            T = ibm.extract_torque(
                state,
                cfg,
                float(state.particle_pos[0]),
                float(state.particle_pos[1]),
            )
            omega_new = rotation_full_step(cache["omega_half_cache"], T, state.particle_I, state.dt)
            state.particle_torque = T
        vel_new = cache["inertia"].post_correction(vel_new, state.particle_vel, state.particle_vel_prev, cfg.rho_ratio)
        if cfg.enable_rotation:
            state.particle_omega = omega_new
        state.particle_vel_prev = state.particle_vel.copy()
        state.particle_vel = vel_new
        state.particle_force = F_hydro

    def diagnostics(self, state, cfg, cache, ttt: int) -> dict:
        record = record_sedimentation_state(
            state.particle_pos,
            state.particle_vel,
            ttt,
            cache["d_lattice"],
            state.gravity_lattice,
            cfg.rho_ratio,
            cache["y0_init"],
            dx=state.dx,
        )
        record["f_hydro_x"] = float(state.particle_force[0])
        record["f_hydro_y"] = float(state.particle_force[1])
        if cfg.diagnostics_interval > 0 and ttt % cfg.diagnostics_interval == 0:
            _dx = state.dx
            cx, cy = float(state.particle_pos[0]), float(state.particle_pos[1])
            r_domain = state.lattice_r * _dx
            D_domain = cache["d_lattice"] * _dx
            Eux = to_cpu(state.U[:, 0]).reshape(state.ny, state.nx)
            Euy = to_cpu(state.U[:, 1]).reshape(state.ny, state.nx)
            ro_2d = to_cpu(state.ro).reshape(state.ny, state.nx)
            u_g = math.sqrt(abs(cfg.rho_ratio - 1.0) * state.gravity_lattice * cache["d_lattice"])
            res = compute_inside_residual(Eux, Euy, _dx, _dx, cx, cy, D_domain, kappa=2.0, u_ref=max(float(u_g), 1e-12))
            record["inside_residual_mean"] = res["mean"]
            record["inside_residual_max"] = res["max"]
            record["inside_residual_mean_norm"] = res["mean_normalized"]
            px, py = compute_internal_momentum(Eux, Euy, ro_2d, _dx, _dx, cx, cy, r_domain)
            record["p_int_x"] = px
            record["p_int_y"] = py
            lz = compute_internal_angular_momentum(Eux, Euy, ro_2d, _dx, _dx, cx, cy, r_domain)
            record["l_int_z"] = lz
            state.inside_residual_mean = res["mean"]
            state.inside_residual_max = res["max"]
            import numpy as _np
            state.p_int = _np.array([px, py])
            state.l_int = lz
        cache["last_record"] = record
        cache["history"].append(record)
        return {
            "Cd": 0.0,
            "Cl": 0.0,
            "log_line": f"step {ttt:>7d} | y*={record['y_star']:.4f} vy*={record['vy_star']:.6f} vx*={record['vx_star']:.6f}",
        }

    def should_stop(self, state, cfg, cache, ttt: int) -> str | None:
        record = cache["last_record"]
        if record is None:
            return None
        if not math.isfinite(record["vy_star"]):
            return "nan"
        stop_reason = sedimentation_stop_reason(state.particle_pos, state.r, cfg)
        if stop_reason is not None:
            return stop_reason
        bound_warn = sedimentation_bound_warning(state.particle_pos, state.r, cfg, state.dx, safety=2.0)
        if bound_warn:
            return "domain_bounds"
        return None

    def finalize(self, state, cfg, cache, result: dict) -> dict:
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            cache["inertia"].detach_state(state)
        result["sedimentation_history"] = cache["history"]
        result["termination_reason"] = cache["termination_reason"]
        return result


register_scenario(SedimentationSingleRuntime())
