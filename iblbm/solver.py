"""Solver facade — `runtime.run` 재노출.

- 패키지 최상위 (`iblbm.run`) 에서 호출될 때의 단일 진입점
- 실제 시간 루프 구현체는 `iblbm.runtime.loop.run`
"""
from __future__ import annotations

from .runtime import run

__all__ = ["run"]
