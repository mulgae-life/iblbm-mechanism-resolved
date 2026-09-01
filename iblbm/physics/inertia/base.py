"""관성 모델 Protocol + 이름 기반 레지스트리.

계약
  - `InertiaModel`     — velocity/euler update + post-correction hook
  - `ExtendedInertiaHook` — full-volume 계열에서 쓰는 preliminary/ibm/post-step hook
  - 등록명: `none | explicit_history | feng_b2 | full_volume`

dispatch helper
  - `verlet_half_step_with_inertia_model`  : half-kick + drift
  - `verlet_full_step_with_inertia_model`  : second half-kick
  - `euler_explicit_step_with_inertia_model`: explicit Euler + position update
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

VALID_SETTLING_INERTIA_MODELS = {
    "none",
    "explicit_history",
    "feng_b2",
    "full_volume",
}


@runtime_checkable
class InertiaModel(Protocol):
    name: str
    uses_preliminary_update: bool

    def velocity_verlet_half(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, gravity_direction="down", **kwargs):
        ...

    def velocity_verlet_full(self, vel_half, vel, vel_prev, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt, gravity_direction="down", **kwargs):
        ...

    def euler_explicit(self, pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, position_update="new_velocity", gravity_direction="down", **kwargs):
        ...

    def post_correction(self, vel_new, vel_current, vel_prev, rho_ratio):
        ...


@runtime_checkable
class ExtendedInertiaHook(Protocol):
    def attach_state(self, state) -> None:
        ...

    def detach_state(self, state) -> None:
        ...

    def preliminary_step(self, state, cfg, snapshot) -> None:
        ...

    def ibm_step(self, state, cfg) -> None:
        ...

    def post_step(self, state, cfg) -> None:
        ...


_REGISTRY: dict[str, InertiaModel] = {}


def register_inertia_model(instance: InertiaModel) -> InertiaModel:
    if instance.name in _REGISTRY:
        raise ValueError(f"settling_inertia_model='{instance.name}' already registered")
    _REGISTRY[instance.name] = instance
    return instance


def get_inertia_model(name: str) -> InertiaModel:
    if name not in _REGISTRY:
        raise KeyError(
            f"등록되지 않은 settling_inertia_model='{name}'. 등록 목록: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_inertia_models() -> list[str]:
    return sorted(_REGISTRY)


def verlet_half_step_with_inertia_model(pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, settling_inertia_model: str, gravity_direction: str = "down", **kwargs):
    return get_inertia_model(settling_inertia_model).velocity_verlet_half(
        pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx,
        gravity_direction=gravity_direction,
        **kwargs,
    )


def verlet_full_step_with_inertia_model(vel_half, vel, vel_prev, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt, settling_inertia_model: str, gravity_direction: str = "down", **kwargs):
    return get_inertia_model(settling_inertia_model).velocity_verlet_full(
        vel_half, vel, vel_prev, force_hydro_new, mass, rho_ratio, r_lattice, g_lattice, dt,
        gravity_direction=gravity_direction,
        **kwargs,
    )


def euler_explicit_step_with_inertia_model(pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx, settling_inertia_model: str, position_update: str = "new_velocity", gravity_direction: str = "down", **kwargs):
    return get_inertia_model(settling_inertia_model).euler_explicit(
        pos, vel, vel_prev, force_hydro, mass, rho_ratio, r_lattice, g_lattice, dt, dx,
        position_update=position_update,
        gravity_direction=gravity_direction,
        **kwargs,
    )
