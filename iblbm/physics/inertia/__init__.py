"""내부 유체 관성(inertia) 모델 레지스트리와 호환 helper.

등록 모델
  - `none`            baseline, 내부 유체 관성 보정 없음
  - `explicit_history` Feng 2009 Eq. (20) history 항 분리 적용 (현재 메인 경로)
  - `feng_b2`          Feng 2009 Eq. (20) + Suzuki 2011 (B-2) post-correction
  - `full_volume`      Suzuki 2011 (C) + García-Villalba 2023 preliminary velocity 계보

선택 지점
  - `config.SimConfig.settling_inertia_model` (`"none" | "explicit_history" | "feng_b2" | "full_volume"`)
"""
from __future__ import annotations

from .base import (
    ExtendedInertiaHook,
    InertiaModel,
    VALID_SETTLING_INERTIA_MODELS,
    euler_explicit_step_with_inertia_model,
    get_inertia_model,
    list_inertia_models,
    register_inertia_model,
    verlet_full_step_with_inertia_model,
    verlet_half_step_with_inertia_model,
)
from .feng_b2 import apply_imc_correction
from .full_volume import preliminary_velocity_update
from . import none, explicit_history, feng_b2, full_volume  # noqa: F401

__all__ = [
    "ExtendedInertiaHook",
    "InertiaModel",
    "VALID_SETTLING_INERTIA_MODELS",
    "register_inertia_model",
    "get_inertia_model",
    "list_inertia_models",
    "verlet_half_step_with_inertia_model",
    "verlet_full_step_with_inertia_model",
    "euler_explicit_step_with_inertia_model",
    "apply_imc_correction",
    "preliminary_velocity_update",
]
