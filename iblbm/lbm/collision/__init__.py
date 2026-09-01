"""충돌 모델 디스패치 파사드.

  - Strategy 등록: BGK / TRT / CM_MRT (모듈 import 시 `register_collision` 자동 호출)
  - Public API: `collision_step(...)`, `guo_forcing(...)`
  - Registry 헬퍼: `register_collision`, `get_collision`, `list_collisions`
"""
from __future__ import annotations

from ...backend import _use_gpu
from .base import (
    CollisionStrategy,
    get_collision,
    list_collisions,
    register_collision,
)
from . import bgk, trt, cm_mrt  # noqa: F401
from .guo_forcing import guo_forcing

__all__ = [
    "CollisionStrategy",
    "register_collision",
    "get_collision",
    "list_collisions",
    "collision_step",
    "guo_forcing",
]


def collision_step(
    fstar,
    feq,
    U,
    fib,
    tau,
    dt,
    lattice,
    collision_model="BGK",
    state=None,
    **kwargs,
):
    """`collision_model` 이름으로 Strategy 조회 → GPU/CPU 분기 호출.

    `state` 미지정 시 `kwargs` 를 속성으로 갖는 임시 객체로 대체.
    """
    strat = get_collision(collision_model)
    if state is None:
        class _TempState:
            pass
        state = _TempState()
        for key, value in kwargs.items():
            setattr(state, key, value)
    if _use_gpu:
        return strat.step_gpu(fstar, feq, U, fib, tau, dt, lattice, state)
    return strat.step_cpu(fstar, feq, U, fib, tau, dt, lattice, state)
