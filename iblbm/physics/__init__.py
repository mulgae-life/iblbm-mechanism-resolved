"""실험 상황별 물리 패키지 facade.

수록 계열
  - 강체 Newton-Euler 적분 (`rigid_body`): Velocity Verlet + explicit Euler
  - 마커 기하 (`markers`) + 시나리오별 운동 (`motion`)
  - 침강 공통 상태 (`sedimentation`)
  - 내부 유체 관성 계열 (`inertia/*`) — `none` / `explicit_history` / `feng_b2` / `full_volume`

수식 본체는 하위 모듈 docstring 참조
"""
from .gravity import compute_net_gravity
from .markers import compute_desired_velocity, create_circle_markers, update_markers
from .motion import update_oscillating
from .rigid_body import (
    VALID_EULER_UPDATE_SCHEMES,
    euler_explicit_rotation,
    rotation_full_step,
    rotation_half_step,
    verlet_full_step,
    verlet_half_step,
)
from .sedimentation import (
    ParticleState,
    check_domain_bounds,
    init_multi_particle_state,
    init_sedimentation_state,
)
from .inertia import (
    ExtendedInertiaHook,
    InertiaModel,
    VALID_SETTLING_INERTIA_MODELS,
    apply_imc_correction,
    euler_explicit_step_with_inertia_model,
    get_inertia_model,
    list_inertia_models,
    preliminary_velocity_update,
    register_inertia_model,
    verlet_full_step_with_inertia_model,
    verlet_half_step_with_inertia_model,
)

__all__ = [
    "update_oscillating",
    "compute_net_gravity",
    "compute_desired_velocity",
    "create_circle_markers",
    "update_markers",
    "VALID_EULER_UPDATE_SCHEMES",
    "verlet_half_step",
    "verlet_full_step",
    "rotation_half_step",
    "rotation_full_step",
    "euler_explicit_rotation",
    "ParticleState",
    "check_domain_bounds",
    "init_sedimentation_state",
    "init_multi_particle_state",
    "ExtendedInertiaHook",
    "InertiaModel",
    "VALID_SETTLING_INERTIA_MODELS",
    "register_inertia_model",
    "get_inertia_model",
    "list_inertia_models",
    "verlet_half_step_with_inertia_model",
    "verlet_full_step_with_inertia_model",
    "euler_explicit_step_with_inertia_model",
    "apply_imc_correction",
    "preliminary_velocity_update",
]
