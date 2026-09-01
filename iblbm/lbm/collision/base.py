"""충돌 Strategy Protocol 및 이름 기반 Registry.

Protocol 요구 메서드
  - `init_state(cfg, state)`          사전 계산 상태(τ⁻, MRT 행렬 등) 준비
  - `extra_kwargs(state)`             외부에서 필요한 추가 인자 반환
  - `step_cpu(...)`, `step_gpu(...)`  백엔드별 충돌 한 스텝 수행
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CollisionStrategy(Protocol):
    name: str

    def init_state(self, cfg, state) -> None:
        ...

    def extra_kwargs(self, state) -> dict:
        ...

    def step_cpu(self, fstar, feq, U, fib, tau, dt, lattice, state) -> Any:
        ...

    def step_gpu(self, fstar, feq, U, fib, tau, dt, lattice, state) -> Any:
        ...


_REGISTRY: dict[str, CollisionStrategy] = {}


def register_collision(instance: CollisionStrategy) -> CollisionStrategy:
    if instance.name in _REGISTRY:
        raise ValueError(f"collision '{instance.name}' already registered")
    _REGISTRY[instance.name] = instance
    return instance


def get_collision(name: str) -> CollisionStrategy:
    if name not in _REGISTRY:
        raise KeyError(
            f"등록되지 않은 collision_model='{name}'. 등록 목록: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_collisions() -> list[str]:
    return sorted(_REGISTRY)
