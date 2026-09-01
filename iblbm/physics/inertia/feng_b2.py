"""Feng 2009 Eq. (20) + Suzuki 2011 (B-2) post-correction.

핵심 식
  ΔU_B2 = (ρ_f / ρ_s) · (U^n − U^{n−1})
    - 2D 원판에서는 `(m_f / m_s) = ρ_f/ρ_s`와 동일 형태
    - `explicit_history`와 수학적 증분 자체는 동등하지만, **적용 지점이 다름**

적용 지점 차이
       ┌──────────── explicit_history ────────────┐
       │  history 항을 적분기 내부에 kick으로 합산 │
       └──────────────────────────────────────────┘
       ┌──────────────── feng_b2 ─────────────────┐
       │  적분기 완료 후 post_correction으로 분리  │
       │  `vel_new ← vel_new + ΔU_B2`              │
       └──────────────────────────────────────────┘

계보
  - Feng & Michaelides (2009) Eq. (20) — history 항 계보
  - Suzuki & Inamuro (2011) scheme (B-2) — internal mass 영향 보정 중 explicit discretization 계열
    (B-1은 `M_eff = M − m_f`로 분모 치환, B-2는 이전 스텝 내부 모멘텀을 explicit 평가)

본 모듈의 지위
  - Eq. (20) 전체 update가 아니라 history 항만 **post-correction**으로 분리
  - `NoneInertia`의 `velocity_verlet_*` / `euler_explicit` 결과에 `ΔU_B2`를 덧씀
"""

from __future__ import annotations

from .base import register_inertia_model
from .none import NoneInertia


def apply_imc_correction(vel_new, vel_current, vel_prev, rho_ratio: float):
    """post-correction `vel_new ← vel_new + ΔU_B2`.

    `ΔU_B2 = (ρ_f / ρ_s) · (U^n − U^{n−1})`
        - `rho_ratio = ρ_s / ρ_f`
        - baseline 적분 결과(`vel_new`)에 history 증분을 덧씀
    """
    return vel_new + (1.0 / rho_ratio) * (vel_current - vel_prev)


class FengB2Inertia(NoneInertia):
    name = "feng_b2"
    uses_preliminary_update = False

    def post_correction(self, vel_new, vel_current, vel_prev, rho_ratio):
        return apply_imc_correction(vel_new, vel_current, vel_prev, rho_ratio)


register_inertia_model(FengB2Inertia())
