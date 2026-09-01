"""IBM 서브패키지 public re-export.

  - Strategy 등록: DF / MDF / DFC (`strategies/*` 모듈 import 시 `register_ibm` 자동 호출)
  - Public API: `apply_ibm_step(_multi)`, `apply_velocity_correction`
  - Kernel 함수: `ibm_direct_forcing`, `ibm_multi_direct_forcing`, `apply_dfc_correction`
  - Registry 헬퍼: `IBMStrategy`, `register_ibm`, `get_ibm`, `list_ibm`
  - Delta 커널: `delta_hat`, `delta_peskin4pt`, `delta_function`, `get_delta`
"""
from __future__ import annotations

from .common import (
    delta_hat,
    delta_peskin4pt,
    delta_function,
    get_delta,
    _DELTA_REGISTRY,
)
from .df import ibm_direct_forcing
from .mdf import ibm_multi_direct_forcing
from .dfc import apply_dfc_correction
from .base import IBMStrategy, get_ibm, list_ibm, register_ibm
from .dispatch import apply_ibm_step, apply_ibm_step_multi, apply_velocity_correction
from .forcing import interpolate_velocity_to_points, ibm_df_particle_lagrangian_closure
from . import strategies  # noqa: F401

__all__ = [
    "delta_hat",
    "delta_peskin4pt",
    "delta_function",
    "get_delta",
    "_DELTA_REGISTRY",
    "ibm_direct_forcing",
    "ibm_multi_direct_forcing",
    "apply_dfc_correction",
    "IBMStrategy",
    "get_ibm",
    "list_ibm",
    "register_ibm",
    "apply_ibm_step",
    "apply_ibm_step_multi",
    "apply_velocity_correction",
    "interpolate_velocity_to_points",
    "ibm_df_particle_lagrangian_closure",
]
