"""Force-level (current-step) collision 스케줄 판정 헬퍼.

판정 규칙
  1. IBM Strategy 가 force-level 스케줄러 미지원 → False
  2. 침강 시나리오 외 → True (고정/진동/회전은 IBM flag 그대로 반영)
  3. 침강 시나리오
       - preliminary-update 를 쓰는 inertia model (예: full_volume) → False
       - 그 외 (none / explicit_history / feng_b2) → True

용도
  - `loop.run` 에서 collision → stream 순서 분기
  - force-level 스케줄은 매 스텝 시작 시 미리 collision 을 수행해 둠
"""
from __future__ import annotations

from ..config import SimConfig
from ..ibm import get_ibm


def uses_force_level_current_step_scheduler(cfg: SimConfig) -> bool:
    """현재 IBM + inertia 조합이 current-step collision 스케줄러 사용 여부."""
    if not get_ibm(cfg.ibm_method).uses_force_level_current_step_scheduler():
        return False
    if cfg.motion_type != "sedimentation":
        return True
    from ..physics.inertia import get_inertia_model
    return not get_inertia_model(cfg.settling_inertia_model).uses_preliminary_update
