"""침강 시나리오용 public facade.

수록 요소
  - 상태 초기화 helper (`init_sedimentation_state`, `init_multi_particle_state`)
  - 도메인 bounds 경고 (`check_domain_bounds`)
  - 하위 모듈 (`gravity` / `markers` / `rigid_body` / `inertia/*`) 공개 re-export

본체 위치
  - 순중력          → `gravity.py`
  - 마커 기하·운동   → `markers.py`, `motion.py`
  - Newton-Euler    → `rigid_body.py`
  - 내부 질량/유체 관성 → `inertia/*.py`
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend import xp as _xp
from ..config import SimConfig
from .gravity import compute_net_gravity
from .markers import compute_desired_velocity, create_circle_markers, update_markers
from .rigid_body import (
    VALID_EULER_UPDATE_SCHEMES,
    euler_explicit_rotation,
    rotation_full_step,
    rotation_half_step,
    verlet_full_step,
    verlet_half_step,
)
from .inertia import (
    VALID_SETTLING_INERTIA_MODELS,
    apply_imc_correction,
    euler_explicit_step_with_inertia_model,
    preliminary_velocity_update,
    verlet_full_step_with_inertia_model,
    verlet_half_step_with_inertia_model,
)


@dataclass
class ParticleState:
    """다입자 침강 경로의 입자 1개 상태 묶음.

    필드 계열
      - 위치·속도·힘  : `pos`, `vel`, `vel_prev`, `force`
      - 질량·관성     : `mass`, `I_particle`, `rho_ratio`, `lattice_r`, `r_domain`
      - 회전          : `omega`, `torque`, `angle`
      - 마커          : `Lx_c`, `Ly_c`, `Lx`, `Ly`, `Lb`, `Larea`, `desired_velocity`
      - 초기 기준     : `cx_init`, `cy_init`
      - 관성 보정     : `l_int_prev` — full-volume 경로 각운동량 이력
    """

    pos: np.ndarray
    vel: np.ndarray
    vel_prev: np.ndarray
    force: np.ndarray
    mass: float
    rho_ratio: float
    lattice_r: float
    r_domain: float
    omega: float = 0.0
    torque: float = 0.0
    angle: float = 0.0
    I_particle: float = 0.0
    Lx_c: np.ndarray = field(default_factory=lambda: np.array([]))
    Ly_c: np.ndarray = field(default_factory=lambda: np.array([]))
    Lx: np.ndarray = field(default_factory=lambda: np.array([]))
    Ly: np.ndarray = field(default_factory=lambda: np.array([]))
    Lb: int = 0
    Larea: float = 0.0
    desired_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    cx_init: float = 0.0
    cy_init: float = 0.0
    l_int_prev: float = 0.0

def check_domain_bounds(pos: np.ndarray, r: float, xmax: float, ymax: float, safety: float = 2.0, dx: float = 1.0) -> str | None:
    """입자가 벽 safety margin 안쪽에 들어오면 경고 문자열 반환.

      margin = safety · Δx
      경고 조건: `y − r < margin`, `y + r > ymax − margin`, x축 동일
    """
    margin = safety * dx
    x, y = pos
    if y - r < margin:
        return f"입자 하단이 하벽에 근접 (y-r={y-r:.4f} < margin={margin:.4f})"
    if y + r > ymax - margin:
        return f"입자 상단이 상벽에 근접 (y+r={y+r:.4f} > ymax-margin={ymax-margin:.4f})"
    if x - r < margin:
        return "입자 좌측이 좌벽에 근접"
    if x + r > xmax - margin:
        return "입자 우측이 우벽에 근접"
    return None


def init_sedimentation_state(s, cfg: SimConfig, lattice_r: float) -> None:
    """단일/다입자 침강 공용 상태 초기화.

    초기화 내용
      - 입자 위치·속도·힘·질량·관성 모멘트 (`m = ρ_s π r²`, `I = ½ m r²`)
      - full-volume 경로용 좌표 그리드 `_fv_XX_flat`, `_fv_YY_flat`
      - torque 계산용 index grid `_torque_II`, `_torque_JJ`

    검증
      - `gravity > 0`, `rho_ratio > 1` (침강 요건)
      - `particles_config`가 있으면 `init_multi_particle_state`로 확장
    """
    cx, cy = cfg.cylinder_center
    s.particle_pos = np.array([cx, cy], dtype=float)
    s.particle_vel = np.array([0.0, 0.0], dtype=float)
    s.particle_vel_prev = np.array([0.0, 0.0], dtype=float)
    s.particle_force = np.array([0.0, 0.0], dtype=float)
    s.gravity_lattice = cfg.gravity

    if cfg.gravity <= 0.0:
        raise ValueError(
            f"침강에는 gravity > 0 필요 (현재: {cfg.gravity}). 시나리오에서 물리 단위 변환 후 전달 필요"
        )
    if cfg.rho_ratio <= 1.0:
        raise ValueError(f"침강에는 rho_ratio > 1.0 필요 (현재: {cfg.rho_ratio}).")

    s.particle_omega = 0.0
    s.particle_torque = 0.0
    s.particle_l_int_prev = 0.0
    s.particle_mass = cfg.rho_ratio * np.pi * lattice_r**2
    s.particle_I = 0.5 * s.particle_mass * lattice_r**2
    s.particle_angle = 0.0

    _II, _JJ = _xp.meshgrid(_xp.arange(s.nx, dtype=float), _xp.arange(s.ny, dtype=float), indexing='xy')
    s._torque_II = _II
    s._torque_JJ = _JJ

    _x_lin = _xp.arange(s.nx, dtype=float) * s.dx
    _y_lin = _xp.arange(s.ny, dtype=float) * s.dy
    _YY_g, _XX_g = _xp.meshgrid(_y_lin, _x_lin, indexing='ij')
    s._fv_XX_flat = _XX_g.ravel()
    s._fv_YY_flat = _YY_g.ravel()

    if cfg.particles_config is not None:
        init_multi_particle_state(s, cfg)


def init_multi_particle_state(s, cfg: SimConfig) -> None:
    """`cfg.particles_config` → `ParticleState` 목록 생성.

    입자별로
      - 질량·관성      m = ρ_p π r²,    I = ½ m r²
      - 마커 set       `create_circle_markers` 호출
      - `desired_velocity` `(L_b, 2)` zero 초기화
    """
    particles = []
    dx = s.dx

    for pc in cfg.particles_config:
        cx_p, cy_p = pc["center"]
        rho_p = pc["rho_ratio"]
        lr = s.lattice_r
        r_dom = s.r

        mass_p = rho_p * np.pi * lr**2
        I_p = 0.5 * mass_p * lr**2

        Lx, Ly, Lb, Larea = create_circle_markers(
            cx_p,
            cy_p,
            r_dom,
            lr,
            dx,
            cfg.marker_spacing_factor,
            cfg.retraction_dx,
        )

        particle = ParticleState(
            pos=np.array([cx_p, cy_p], dtype=float),
            vel=np.array([0.0, 0.0], dtype=float),
            vel_prev=np.array([0.0, 0.0], dtype=float),
            force=np.array([0.0, 0.0], dtype=float),
            mass=mass_p,
            rho_ratio=rho_p,
            lattice_r=lr,
            r_domain=r_dom,
            I_particle=I_p,
            Lx_c=Lx.copy(),
            Ly_c=Ly.copy(),
            Lx=Lx,
            Ly=Ly,
            Lb=Lb,
            Larea=Larea,
            desired_velocity=np.zeros((Lb, 2)),
            cx_init=cx_p,
            cy_init=cy_p,
        )
        particles.append(particle)

    s.particles = particles


__all__ = [
    "ParticleState",
    "VALID_SETTLING_INERTIA_MODELS",
    "VALID_EULER_UPDATE_SCHEMES",
    "compute_net_gravity",
    "verlet_half_step",
    "verlet_full_step",
    "verlet_half_step_with_inertia_model",
    "verlet_full_step_with_inertia_model",
    "euler_explicit_step_with_inertia_model",
    "euler_explicit_rotation",
    "rotation_half_step",
    "rotation_full_step",
    "apply_imc_correction",
    "preliminary_velocity_update",
    "update_markers",
    "compute_desired_velocity",
    "check_domain_bounds",
    "init_sedimentation_state",
    "init_multi_particle_state",
]
