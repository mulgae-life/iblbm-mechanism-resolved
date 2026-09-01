"""진단 모듈 public facade.

구성
  - `core`           C_D / C_L / St, tail window 통계, L₂ 오차, Taylor-Green 해석해
  - `ibm_quality`    IBM 경계 충실도 진단 (slip error ε_slip, leakage Φ_leak)
  - `internal_flow`  입자 내부 유체 모멘트 (P_int, L_int, 침강 무차원 기록)
  - `surface`        D2Q9 Chapman-Enskog 응력 → 표면 적분 유체력

공용 상수
  - `STEADY_TAIL_FRAC = 0.3`   후반부 tail window 비율 (수렴 검정 + 진폭 집계 공용)

용도
  - benchmark 후처리 (Cd/Cl/St, L_r, slip/leak)
  - runtime logging (L_int_z, 수렴 판정)
"""
from __future__ import annotations

from .core import (
    STEADY_TAIL_FRAC,
    check_convergence,
    compute_cd_cl,
    compute_cd_cl_dfc,
    compute_cl_amplitude,
    compute_l2_error,
    compute_recirculation_length,
    compute_strouhal,
    tail_mean,
    tail_peak_to_peak_amp,
    tail_start_index,
    tail_values,
    tg_analytical_velocity_field,
)
from .ibm_quality import compute_leakage_flux, compute_slip_error
from .internal_flow import (
    compute_inside_residual,
    compute_internal_angular_momentum,
    compute_internal_momentum,
    compute_l_int_z,
    record_sedimentation_state,
)
from .surface import compute_stress_tensor, integrate_surface_force

__all__ = [
    "STEADY_TAIL_FRAC",
    "tail_start_index",
    "tail_values",
    "tail_mean",
    "tail_peak_to_peak_amp",
    "compute_cd_cl",
    "compute_cd_cl_dfc",
    "check_convergence",
    "compute_strouhal",
    "tg_analytical_velocity_field",
    "compute_l2_error",
    "compute_recirculation_length",
    "compute_slip_error",
    "compute_leakage_flux",
    "compute_inside_residual",
    "compute_internal_momentum",
    "compute_internal_angular_momentum",
    "compute_l_int_z",
    "compute_cl_amplitude",
    "record_sedimentation_state",
    "compute_stress_tensor",
    "integrate_surface_force",
]
