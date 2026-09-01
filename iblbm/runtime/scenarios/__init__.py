"""시나리오 runtime 레지스트리 bootstrap.

등록 대상 (side-effect import)
  - `fixed_cylinder`        고정/진동/병진/회전 실린더
  - `taylor_green`          감쇠 Taylor-Green 와류
  - `sedimentation_single`  단일 입자 침강
  - `sedimentation_multi`   다입자 침강

공개 심볼
  - `ScenarioRuntime`       Protocol
  - `get_scenario_runtime`  cfg → 매칭 handler
  - `list_scenarios`, `register_scenario`
"""
from __future__ import annotations

from .base import ScenarioRuntime, get_scenario_runtime, list_scenarios, register_scenario
from . import fixed_cylinder, taylor_green, sedimentation_single, sedimentation_multi  # noqa: F401

__all__ = [
    "ScenarioRuntime",
    "get_scenario_runtime",
    "list_scenarios",
    "register_scenario",
]
