"""Baseline 관성 모델 (내부 유체 관성 보정 없음).

식
  m_s · dU_p/dt = F_h + F_g
    - `F_h` IBM 유체력, `F_g` 순중력
    - Suzuki 2011 scheme (A) `F_in(t) ≡ 0` 계보

활용
  - 벤치마크 대비 Re 불일치 진단 시 참조 기준선
  - 다른 모델의 `velocity_verlet_*` / `euler_explicit` / `post_correction` 기반 클래스
"""

from __future__ import annotations

from .base import register_inertia_model
from ..gravity import compute_net_gravity
from ..rigid_body import euler_explicit_position_update


class NoneInertia:
    name = "none"
    uses_preliminary_update = False

    def velocity_verlet_half(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, gravity_direction="down", displaced_area_lattice=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total = force_hydro + F_gravity
        vel_half = vel + 0.5 * dt * F_total / mass
        pos_new = pos + dt * vel_half * dx
        return pos_new, vel_half

    def velocity_verlet_full(self, vel_half, vel, vel_prev, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt, gravity_direction="down", displaced_area_lattice=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total_new = force_hydro_new + F_gravity
        return vel_half + 0.5 * dt * F_total_new / mass

    def euler_explicit(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, position_update="new_velocity", gravity_direction="down", displaced_area_lattice=None, **kwargs):
        F_gravity = compute_net_gravity(
            rho_ratio,
            r_lattice,
            g_lattice,
            gravity_direction,
            displaced_area_lattice=displaced_area_lattice,
        )
        F_total = force_hydro + F_gravity
        vel_new = vel + dt * F_total / mass
        pos_new = euler_explicit_position_update(pos, vel, vel_new, dt, dx, position_update)
        return pos_new, vel_new

    def post_correction(self, vel_new, vel_current, vel_prev, rho_ratio):
        return vel_new


register_inertia_model(NoneInertia())
