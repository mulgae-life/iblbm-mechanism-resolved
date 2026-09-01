"""Runtime facade — 메인 시간 루프 (`loop.run`) 재노출.

외부 import 경로
  - `iblbm.run` → `iblbm.solver.run` → `iblbm.runtime.run` → `iblbm.runtime.loop.run`
"""
from __future__ import annotations

from .loop import run

__all__ = ["run"]
