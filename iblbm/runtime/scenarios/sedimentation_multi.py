"""다입자 침강 runtime (Uhlmann 2005 / Majumder 2023 2-particle anchor).

Uhlmann 2005 §5.2.2 (pure wake interaction, benchmark 기본 틀)
  - 도메인 [0, 10] × [−1, 1]
  - 입자 2개, 반지름 동일, 수직/수평 offset 존재
  - "trailing" 입자가 더 무거움 → 앞 입자 추월 후 wake 상호작용
  - 입자-입자 접촉이 없어 collision model 불필요 (순수 유체 상호작용)

Majumder 2023 §4.4 (drafting/kissing 없는, 밀도가 다른 두 입자의 침강)
  - 도메인 [0, 10] × [−1, 1] (Uhlmann 과 동일)
  - 초기 위치 (0.8, −0.13), (1.2, 0.13)
  - ρ_1/ρ_f = 1.5 (heavy), ρ_2/ρ_f = 1.25 (light)
  - ν_f = 8 × 10⁻⁴ m²/s, 격자 4000 × 800, τ = 0.59
  - 비교 지표 Re_max(heavy) ≈ 276.61, Re_max(light) ≈ 231.41

hook 구성
  - `_multi_particle_half_step`     Verlet 반스텝 (위치/desired_velocity 갱신)
  - `_multi_particle_full_step`     Verlet 완료스텝 (속도/각속도 finalize)
  - `_multi_particle_euler_step`    explicit Euler 1차 적분
  - `pre_step`                      inertia hook 이 아니면 Verlet half 수행
  - `apply_ibm`                     hook / standard_ibm 분기
  - `post_ibm`                      full / euler 선택 후 particle[0] 필드로 동기화
  - `diagnostics`                   입자별 dist*/vx*/vy* (수평 중력) 또는 y*/v* (수직)
  - `should_stop`                   nan / contact / offset / domain_bounds
"""

from __future__ import annotations

import math

from ...diagnostics import compute_l_int_z, record_sedimentation_state
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
from .base import register_scenario
from .sedimentation_common import sedimentation_bound_warning, sedimentation_stop_reason


def _multi_particle_half_step(state, cfg):
    """Verlet 반스텝 — 각 입자 위치/desired_velocity 갱신 (rotation_coupling 반영)."""
    for particle in state.particles:
        pos_new, vel_half = verlet_half_step_with_inertia_model(
            particle.pos,
            particle.vel,
            particle.vel_prev,
            particle.force,
            particle.mass,
            particle.rho_ratio,
            particle.lattice_r,
            state.gravity_lattice,
            state.dt,
            state.dx,
            cfg.settling_inertia_model,
            gravity_direction=cfg.gravity_direction,
        )
        particle.pos = pos_new
        if cfg.enable_rotation:
            omega_half, angle_inc = rotation_half_step(particle.omega, particle.torque, particle.I_particle, state.dt)
            particle.angle += angle_inc
        else:
            omega_half = 0.0
        particle.Lx, particle.Ly = update_markers(
            particle.Lx_c,
            particle.Ly_c,
            particle.cx_init,
            particle.cy_init,
            pos_new,
            angle=particle.angle if cfg.enable_rotation else 0.0,
        )
        if cfg.rotation_coupling == "semi_implicit" and cfg.enable_rotation:
            particle.desired_velocity = compute_desired_velocity(
                vel_half,
                particle.Lb,
                omega=particle.omega,
                Lx=particle.Lx,
                Ly=particle.Ly,
                cx=float(particle.pos[0]),
                cy=float(particle.pos[1]),
                dx=state.dx,
            )
        else:
            particle.desired_velocity = compute_desired_velocity(vel_half, particle.Lb)
        particle._vel_half_cache = vel_half
        particle._omega_half_cache = omega_half



def _multi_particle_full_step(state, cfg, inertia):
    """IBM 이후 Verlet 완료스텝 — 각 입자 속도/각속도 finalize + inertia post_correction."""
    for particle in state.particles:

        vel_new = verlet_full_step_with_inertia_model(
            particle._vel_half_cache,
            particle.vel,
            particle.vel_prev,
            particle._fib_force,
            particle.mass,
            particle.rho_ratio,
            particle.lattice_r,
            state.gravity_lattice,
            state.dt,
            cfg.settling_inertia_model,
            gravity_direction=cfg.gravity_direction,
        )
        omega_new = 0.0
        if cfg.enable_rotation:
            omega_new = rotation_full_step(particle._omega_half_cache, particle._torque_new, particle.I_particle, state.dt)
            particle.torque = particle._torque_new
        vel_new = inertia.post_correction(vel_new, particle.vel, particle.vel_prev, particle.rho_ratio)
        if cfg.enable_rotation:
            particle.omega = omega_new
        particle.vel_prev = particle.vel.copy()
        particle.vel = vel_new
        particle.force = particle._fib_force


def _multi_particle_euler_step(state, cfg, inertia):
    """IBM 이후 explicit Euler 1차 적분 — 위치/속도/마커/desired_velocity 일괄 갱신."""
    for particle in state.particles:
        pos_new, vel_new = euler_explicit_step_with_inertia_model(
            particle.pos,
            particle.vel,
            particle.vel_prev,
            particle._fib_force,
            particle.mass,
            particle.rho_ratio,
            particle.lattice_r,
            state.gravity_lattice,
            state.dt,
            state.dx,
            cfg.settling_inertia_model,
            position_update=cfg.sedimentation_euler_update_scheme,
            gravity_direction=cfg.gravity_direction,
        )
        omega_new = 0.0
        if cfg.enable_rotation:
            l_int_now = compute_l_int_z(
                state.U,
                state.ro,
                state.nx,
                state.ny,
                state.dx,
                float(particle.pos[0]),
                float(particle.pos[1]),
                particle.r_domain,
            )
            dl_int = l_int_now - particle.l_int_prev
            particle.l_int_prev = l_int_now
            omega_new, angle_inc = euler_explicit_rotation(
                particle.omega,
                particle._torque_new,
                particle.I_particle,
                state.dt,
                dl_int=dl_int,
                angle_update=cfg.sedimentation_euler_update_scheme,
            )
            particle.angle += angle_inc
            particle.torque = particle._torque_new
        vel_new = inertia.post_correction(vel_new, particle.vel, particle.vel_prev, particle.rho_ratio)
        if cfg.enable_rotation:
            particle.omega = omega_new
        particle.vel_prev = particle.vel.copy()
        particle.vel = vel_new
        particle.pos = pos_new
        particle.force = particle._fib_force
        particle.Lx, particle.Ly = update_markers(
            particle.Lx_c,
            particle.Ly_c,
            particle.cx_init,
            particle.cy_init,
            pos_new,
            angle=particle.angle if cfg.enable_rotation else 0.0,
        )
        if cfg.enable_rotation:
            particle.desired_velocity = compute_desired_velocity(
                vel_new,
                particle.Lb,
                omega=omega_new,
                Lx=particle.Lx,
                Ly=particle.Ly,
                cx=float(pos_new[0]),
                cy=float(pos_new[1]),
                dx=state.dx,
            )
        else:
            particle.desired_velocity = compute_desired_velocity(vel_new, particle.Lb)


class SedimentationMultiRuntime:
    """다입자 침강 시나리오 runtime handler (`cfg.particles_config is not None`)."""

    name = "sedimentation_multi"

    def matches(self, cfg) -> bool:
        return cfg.motion_type == "sedimentation" and cfg.particles_config is not None

    def initialize(self, state, cfg) -> dict:
        ibm = get_ibm(cfg.ibm_method)
        inertia = get_inertia_model(cfg.settling_inertia_model)
        ibm.prepare(state, cfg)
        if isinstance(inertia, ExtendedInertiaHook):
            inertia.attach_state(state)
        return {
            "d_lattice": 2.0 * state.lattice_r,
            "history": [],
            "last_record": None,
            "ibm": ibm,
            "inertia": inertia,
            "termination_reason": "max_steps",
            "extended_inertia_snapshot": None,
        }

    def pre_step(self, state, cfg, cache, ttt: int) -> None:
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            return
        if cfg.time_integrator != "euler_explicit":
            _multi_particle_half_step(state, cfg)

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
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            cache["inertia"].post_step(state, cfg)
        elif cfg.time_integrator == "euler_explicit":
            _multi_particle_euler_step(state, cfg, cache["inertia"])
        else:
            _multi_particle_full_step(state, cfg, cache["inertia"])
        p0 = state.particles[0]
        state.particle_pos = p0.pos
        state.particle_vel = p0.vel
        state.particle_force = p0.force

    def diagnostics(self, state, cfg, cache, ttt: int) -> dict:
        record = {"step": ttt, "particles": []}
        for pi, particle in enumerate(state.particles):
            if cfg.gravity_direction == "right":
                u_g = float(math.sqrt(abs(particle.rho_ratio - 1.0) * state.gravity_lattice * cache["d_lattice"]))
                x0_init = particle.cx_init
                dist_star = float((particle.pos[0] - x0_init) / (cache["d_lattice"] * state.dx)) if u_g > 1e-15 else 0.0
                vx_star = float(particle.vel[0] / u_g) if u_g > 1e-15 else 0.0
                vy_star = float(particle.vel[1] / u_g) if u_g > 1e-15 else 0.0
                t_star = float(ttt * u_g / cache["d_lattice"]) if u_g > 1e-15 else 0.0
                prec = {
                    "id": pi,
                    "rho_ratio": particle.rho_ratio,
                    "t_star": t_star,
                    "dist_star": dist_star,
                    "vx_star": vx_star,
                    "vy_star": vy_star,
                    "x": float(particle.pos[0]),
                    "y": float(particle.pos[1]),
                    "vx": float(particle.vel[0]),
                    "vy": float(particle.vel[1]),
                }
            else:
                prec = record_sedimentation_state(
                    particle.pos,
                    particle.vel,
                    ttt,
                    cache["d_lattice"],
                    state.gravity_lattice,
                    particle.rho_ratio,
                    particle.cy_init,
                    dx=state.dx,
                )
                prec["id"] = pi
                prec["rho_ratio"] = particle.rho_ratio
            record["particles"].append(prec)
        cache["last_record"] = record
        cache["history"].append(record)
        p0 = record["particles"][0]
        if cfg.gravity_direction == "right":
            log_line = f"step {ttt:>7d} | dist*={p0['dist_star']:.4f} vx*={p0['vx_star']:.6f}"
        else:
            log_line = f"step {ttt:>7d} | y*={p0['y_star']:.4f} vy*={p0['vy_star']:.6f}"
        return {"Cd": 0.0, "Cl": 0.0, "log_line": log_line}

    def should_stop(self, state, cfg, cache, ttt: int) -> str | None:
        record = cache["last_record"]
        if record is None:
            return None
        p0 = state.particles[0]
        nan_check = float(p0.vel[0]) if cfg.gravity_direction == "right" else float(p0.vel[1])
        if not math.isfinite(nan_check):
            return "nan"
        stop_reason = sedimentation_stop_reason(p0.pos, p0.r_domain, cfg)
        if stop_reason is not None:
            return stop_reason
        for particle in state.particles:
            bw = sedimentation_bound_warning(particle.pos, particle.r_domain, cfg, state.dx, safety=2.0)
            if bw:
                return "domain_bounds"
        return None

    def finalize(self, state, cfg, cache, result: dict) -> dict:
        if isinstance(cache["inertia"], ExtendedInertiaHook):
            cache["inertia"].detach_state(state)
        result["sedimentation_history"] = cache["history"]
        result["termination_reason"] = cache["termination_reason"]
        return result


register_scenario(SedimentationMultiRuntime())
