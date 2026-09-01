"""IBM Strategy Protocol 및 이름 기반 Registry.

Protocol 요구 메서드
  - `prepare(state, cfg)`                      one-shot 사전 계산(ω 추정, 캐시 슬롯 등)
  - `apply_single(state, cfg, ttt)`            단일 particle IBM 단계 수행 (`state.fib` 또는 `state.fstar` 갱신)
  - `apply_multi(state, cfg, ttt)`             다입자 particle set 처리
  - `extract_force(state, cfg)`                Lagrangian 힘 합 → 유체 반작용 2-벡터
  - `extract_torque(state, cfg, cx, cy)`       입자 중심 `(cx, cy)` 기준 토크 스칼라
  - `uses_force_level_current_step_scheduler`  force-level Δt 스케줄러(현재 스텝) 적용 여부
  - `requires_velocity_correction`             속도 재보정 path 필요 여부(DF/MDF True, DFC False)
  - `compute_cd_cl(state, cfg, u_ref)`         Cd/Cl 집계 (DFC는 자체 보정력 경로)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IBMStrategy(Protocol):
    name: str

    def prepare(self, state, cfg) -> None:
        ...

    def apply_single(self, state, cfg, ttt: int) -> None:
        ...

    def apply_multi(self, state, cfg, ttt: int) -> None:
        ...

    def extract_force(self, state, cfg):
        ...

    def extract_torque(self, state, cfg, cx: float, cy: float) -> float:
        ...

    def uses_force_level_current_step_scheduler(self) -> bool:
        ...

    def requires_velocity_correction(self) -> bool:
        ...

    def compute_cd_cl(self, state, cfg, u_ref: float):
        ...


_REGISTRY: dict[str, IBMStrategy] = {}


def register_ibm(instance: IBMStrategy) -> IBMStrategy:
    if instance.name in _REGISTRY:
        raise ValueError(f"ibm_method='{instance.name}' already registered")
    _REGISTRY[instance.name] = instance
    return instance


def get_ibm(name: str) -> IBMStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"등록되지 않은 ibm_method='{name}'. 등록 목록: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ibm() -> list[str]:
    return sorted(_REGISTRY)
