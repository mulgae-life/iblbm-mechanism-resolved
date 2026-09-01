"""IB-LBM solver: Immersed Boundary + Lattice Boltzmann Method 공개 API.

공개 심볼
  - `SimConfig`   시나리오/수치/물리 파라미터 dataclass
  - `run`         `runtime.loop.run` 재노출 (메인 시간 루프 facade)

대표 시나리오
  - 고정/진동 실린더 (`motion_type ∈ {None, oscillating}`)
  - 침강 단일 입자 / 다입자 (`motion_type="sedimentation"`)
  - Taylor-Green 감쇠 와류 (`scenario_type="taylor_green"`)
"""
from __future__ import annotations

from .config import SimConfig
from .solver import run

__all__ = ["SimConfig", "run"]
