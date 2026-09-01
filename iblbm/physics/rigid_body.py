"""강체 Newton-Euler 적분 helper (병진 + 회전).

Velocity Verlet 3-step 흐름
     [n]                                              [n+1]
     u, x ──► u^{n+1/2} = u^n + ½·Δt·F^n / m
              x^{n+1}   = x^n + Δt · u^{n+1/2} · Δx
     F^{n+1} 평가 (IBM force 재계산)
              u^{n+1}   = u^{n+1/2} + ½·Δt·F^{n+1} / m

회전도 동일 구조 (T ↔ F,  I ↔ m)
     ω^{n+1/2} = ω^n + ½·Δt·T^n / I
     ω^{n+1}   = ω^{n+1/2} + ½·Δt·T^{n+1} / I

Explicit Euler 회전 (full-volume 경로 전용)
     ω^{n+1}   = ω^n + (Δt·T + ΔL_int) / I
     - `ΔL_int` 는 `euler_explicit_rotation()`에서만 쓰는 internal-fluid 각운동량 보정
     - 그 외 경로에서는 기본값 `ΔL_int = 0`

계보
  - Uhlmann (2005) §4 Eq. (13)-(14) : 강체 Newton-Euler 반-explicit RK 계보
  - Breugem (2012) §4.3            : `ρ_f` internal-fluid inertia 직접 처리
  - Majumder et al. (2023)         : added-mass benchmark 해석 관점
"""
from __future__ import annotations


VALID_EULER_UPDATE_SCHEMES = {
    "new_velocity",
    "trapezoidal",
}


def verlet_half_step(pos, vel, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, gravity_direction="down"):
    """Velocity Verlet half-kick + drift.

    `u^{n+1/2} = u^n + ½·Δt·F^n/m`
    `x^{n+1}   = x^n + Δt · u^{n+1/2} · Δx`
        - `F^n = F_h^n + F_g` — 유체력 + 순중력
        - baseline (internal mass 보정 없음)
    """
    from .gravity import compute_net_gravity

    F_gravity = compute_net_gravity(rho_ratio, r_lattice, g_lattice, gravity_direction)
    F_total = force_hydro + F_gravity
    vel_half = vel + 0.5 * dt * F_total / mass
    pos_new = pos + dt * vel_half * dx
    return pos_new, vel_half


def verlet_full_step(vel_half, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt, gravity_direction="down"):
    """Velocity Verlet second half-kick.

    `u^{n+1} = u^{n+1/2} + ½·Δt·F^{n+1}/m`
        - `F^{n+1}` — 갱신된 IBM 유체력 + 순중력
        - `x^{n+1}`은 `verlet_half_step`에서 이미 업데이트됨
    """
    from .gravity import compute_net_gravity

    F_gravity = compute_net_gravity(rho_ratio, r_lattice, g_lattice, gravity_direction)
    F_total_new = force_hydro_new + F_gravity
    vel_new = vel_half + 0.5 * dt * F_total_new / mass
    return vel_new


def rotation_half_step(omega: float, torque: float, I_particle: float, dt: float):
    """회전 Velocity Verlet half-kick.

    `ω^{n+1/2} = ω^n + ½·Δt·T^n / I`
    `Δθ        = Δt · ω^{n+1/2}`
        - 병진 Verlet과 동일 구조 (T ↔ F, I ↔ m)
    """
    omega_half = omega + 0.5 * dt * torque / I_particle
    angle_inc = dt * omega_half
    return omega_half, angle_inc


def rotation_full_step(omega_half: float, torque_new: float, I_particle: float, dt: float) -> float:
    """회전 Velocity Verlet second half-kick.

    `ω^{n+1} = ω^{n+1/2} + ½·Δt·T^{n+1} / I`
    """
    return omega_half + 0.5 * dt * torque_new / I_particle


def euler_explicit_position_update(pos, vel_current, vel_new, dt, dx, scheme: str = "new_velocity"):
    """Explicit Euler 위치 업데이트.

    `x^{n+1} = x^n + Δt · u · Δx`
        - `scheme = "new_velocity"` → u = u^{n+1}
        - `scheme = "trapezoidal"`  → u = ½·(u^n + u^{n+1})
    """
    if scheme == "new_velocity":
        return pos + dt * vel_new * dx
    if scheme == "trapezoidal":
        return pos + 0.5 * dt * (vel_current + vel_new) * dx
    raise ValueError(
        f"position_update='{scheme}' 미지원. 허용값: {sorted(VALID_EULER_UPDATE_SCHEMES)}"
    )


def euler_explicit_rotation(omega: float, torque: float, I_particle: float, dt: float, dl_int: float = 0.0, angle_update: str = "new_velocity"):
    """Explicit Euler 회전 업데이트 (+ internal-fluid 각운동량 보정).

    `ω^{n+1} = ω^n + (ΔL_int + Δt · T) / I`
        - `ΔL_int` : full-volume 경로에서만 쓰는 내부 유체 각운동량 증분
        - 그 외 경로에서는 기본값 `ΔL_int = 0`

    각도 증분
        - `angle_update = "new_velocity"` → Δθ = Δt · ω^{n+1}
        - `angle_update = "trapezoidal"`  → Δθ = ½·Δt·(ω^n + ω^{n+1})
    """
    omega_new = omega + (dl_int + dt * torque) / I_particle
    if angle_update == "new_velocity":
        angle_inc = dt * omega_new
    elif angle_update == "trapezoidal":
        angle_inc = 0.5 * dt * (omega + omega_new)
    else:
        raise ValueError(
            f"angle_update='{angle_update}' 미지원. 허용값: {sorted(VALID_EULER_UPDATE_SCHEMES)}"
        )
    return omega_new, angle_inc
