"""Feng 2009 explicit-history 관성 모델 (history 항 분리 적용).

핵심 식
  ΔU_hist = (m_f / m_s) · (U^n − U^{n−1})
    - `m_f = ρ_f · π r²` (2D, ρ_f = 1 격자 단위) — internal fluid mass
    - `m_s = ρ_s · π r²` — particle mass

배경 (Feng & Michaelides 2009)
  - Eq. (13) Ladd 1994 형태의 `m_eff = m_s + m_f` 분모 치환은
    `ρ_s/ρ_f > 1 + 10/r` (r = 격자 단위 반경) 형태의 안정성 제약 → 실질 `ρ_s/ρ_f > 2` 제한
  - Eq. (18)-(20)은 내부 유체 질량 항을 입자 실질 질량과 분리해 이전 스텝 값으로 명시적으로 평가
  - Eq. (20) 원형:
        U_s^{n+1} = (1 + ρ_f/ρ_s) U_s^n − (ρ_f/ρ_s) U_s^{n−1}
                   + (Σ_i f_i^n ΔV_i + (m_s − m_f) g) · Δt / m_s

원전 대비 구현 범위
  - Eq. (20) 전체 속도 update를 그대로 쓰지 않고, **history 항 `ΔU_hist`만 분리**하여
    현재 적분기(Velocity Verlet 또는 explicit Euler)에 추가 kick 형태로 합산

적분기별 history 배분
  Velocity Verlet (half / full) : 각 half-kick에 `½ · ΔU_hist` 추가
       u^{n+1/2} = u^n + ½·Δt·F^n/m + ½·ΔU_hist
       u^{n+1}   = u^{n+1/2} + ½·Δt·F^{n+1}/m + ½·ΔU_hist
  Explicit Euler               : 1-step 합산
       u^{n+1} = u^n + Δt·F^n/m + ΔU_hist

참조
  - Feng & Michaelides (2009) Eq. (20) — history 항 계보
  - Majumder et al. (2023) — added-mass benchmark 해석 축
"""

from __future__ import annotations

import numpy as np

from .base import register_inertia_model
from ..gravity import compute_net_gravity
from ..rigid_body import euler_explicit_position_update


def _explicit_history_velocity_increment(vel, vel_prev, mass: float, r_lattice: float, m_fluid_override: float | None = None):
    """history 항 증분 `ΔU_hist = (m_f/m_s)·(U^n − U^{n−1})`.

    `m_f = π r²` (ρ_f = 1 격자 단위), `m_s = mass` (particle mass)
    """
    if m_fluid_override is None:
        m_fluid = np.pi * r_lattice**2
    else:
        m_fluid = m_fluid_override
    return (m_fluid / mass) * (vel - vel_prev)


class ExplicitHistoryInertia:
    name = "explicit_history"
    uses_preliminary_update = False

    def velocity_verlet_half(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, gravity_direction="down", displaced_area_lattice=None, m_fluid_override=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total = force_hydro + F_gravity
        history_dv = _explicit_history_velocity_increment(
            vel,
            vel_prev,
            mass,
            r_lattice,
            m_fluid_override=m_fluid_override,
        )
        vel_half = vel + 0.5 * dt * F_total / mass + 0.5 * history_dv
        pos_new = pos + dt * vel_half * dx
        return pos_new, vel_half

    def velocity_verlet_full(self, vel_half, vel, vel_prev, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt, gravity_direction="down", displaced_area_lattice=None, m_fluid_override=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total_new = force_hydro_new + F_gravity
        history_dv = _explicit_history_velocity_increment(
            vel,
            vel_prev,
            mass,
            r_lattice,
            m_fluid_override=m_fluid_override,
        )
        return vel_half + 0.5 * dt * F_total_new / mass + 0.5 * history_dv

    def euler_explicit(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, position_update="new_velocity", gravity_direction="down", displaced_area_lattice=None, m_fluid_override=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total = force_hydro + F_gravity
        history_dv = _explicit_history_velocity_increment(
            vel,
            vel_prev,
            mass,
            r_lattice,
            m_fluid_override=m_fluid_override,
        )
        vel_new = vel + dt * F_total / mass + history_dv
        pos_new = euler_explicit_position_update(pos, vel, vel_new, dt, dx, position_update)
        return pos_new, vel_new

    def post_correction(self, vel_new, vel_current, vel_prev, rho_ratio):
        return vel_new


register_inertia_model(ExplicitHistoryInertia())
