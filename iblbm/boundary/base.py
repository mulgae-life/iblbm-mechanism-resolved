"""Boundary Strategy Protocol + BoundaryIndices.

역할
  - `BoundaryIndices`  face별 노드 1D 인덱스 컨테이너 (left/right/top/bottom + nx, ny)
  - `BoundaryStrategy` `apply(fstar, ro, U, state, cfg, ttt)` Protocol
  - 레지스트리        name → strategy 매핑, `cfg.bc_type`으로 디스패치
  - `scenario_type == "taylor_green"` 특례: `tg_analytical` 강제 선택

참조
  - face closure 계보: Zou & He (1997)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..backend import xp as np


@dataclass
class BoundaryIndices:
    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    nx: int
    ny: int


@runtime_checkable
class BoundaryStrategy(Protocol):
    name: str

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        ...


_REGISTRY: dict[str, BoundaryStrategy] = {}


def register_boundary(instance: BoundaryStrategy) -> BoundaryStrategy:
    _REGISTRY[instance.name] = instance
    return instance


def register_boundary_as(instance: BoundaryStrategy, key: str) -> BoundaryStrategy:
    _REGISTRY[key] = instance
    return instance


def get_boundary(cfg) -> BoundaryStrategy:
    if cfg.scenario_type == "taylor_green":
        return _REGISTRY["tg_analytical"]
    return _REGISTRY[cfg.bc_type]


def list_boundaries() -> list[str]:
    return sorted(_REGISTRY)
