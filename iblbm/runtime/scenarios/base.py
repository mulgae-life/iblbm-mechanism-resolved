"""Scenario runtime Protocol + 순서형 레지스트리.

loop 에서 호출되는 hook 순서 (기본)
  `initialize → pre_step → (collide) → (stream/BC/macro) → pre_ibm →
   apply_ibm → post_ibm → update_feq → (collide if force-level) →
   [opt] post_collision → diagnostics → should_stop → finalize`

Protocol 메서드 역할
  - `matches(cfg)`           현재 cfg 가 해당 runtime 대상인지
  - `initialize(state, cfg)` 1회 setup, 반환한 dict 가 loop 의 `cache`
  - `pre_step`               시간 적분 half-step, 마커 갱신 등
  - `pre_ibm`                IBM 호출 직전 훅 (inertia preliminary 등)
  - `apply_ibm`              실제 IBM forcing 경로 (standard / extended)
  - `post_ibm`               IBM 후 입자 속도 final update (Verlet full 등)
  - `post_collision` (opt)   force-level collision 이후 마커 갱신
  - `diagnostics`            Cd/Cl, y*/v*, log line 기록
  - `should_stop`            조기 종료 사유 문자열 (None → 계속)
  - `finalize`               result dict 보강 (history 등)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScenarioRuntime(Protocol):
    """시나리오별 runtime handler 계약 (Protocol)."""

    name: str

    def matches(self, cfg) -> bool:
        ...

    def initialize(self, state, cfg) -> dict:
        ...

    def pre_step(self, state, cfg, cache, ttt: int) -> None:
        ...

    def pre_ibm(self, state, cfg, cache, ttt: int) -> None:
        ...

    def apply_ibm(self, state, cfg, cache, ttt: int) -> None:
        ...

    def post_ibm(self, state, cfg, cache, ttt: int) -> None:
        ...

    def diagnostics(self, state, cfg, cache, ttt: int) -> dict:
        ...

    def should_stop(self, state, cfg, cache, ttt: int) -> str | None:
        ...

    def finalize(self, state, cfg, cache, result: dict) -> dict:
        ...


_REGISTRY: list[ScenarioRuntime] = []


def register_scenario(instance: ScenarioRuntime) -> ScenarioRuntime:
    """Scenario runtime 인스턴스를 레지스트리에 추가 (append 순서 유지)."""
    _REGISTRY.append(instance)
    return instance


def list_scenarios() -> list[str]:
    """등록된 scenario runtime 이름 목록."""
    return [runtime.name for runtime in _REGISTRY]


def get_scenario_runtime(cfg) -> ScenarioRuntime:
    """현재 cfg 에 `matches(cfg)` 가 True 인 첫 번째 runtime 선택.

    매칭 실패 시 ValueError (scenario_type/motion_type 포함 메시지).
    """
    for runtime in _REGISTRY:
        if runtime.matches(cfg):
            return runtime
    raise ValueError(
        f"No ScenarioRuntime matches cfg (scenario_type={cfg.scenario_type}, motion_type={cfg.motion_type})"
    )
