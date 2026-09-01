"""경계 조건 dispatch + 하위 호환 re-export.

구성
  - `BoundaryStrategy`           face별 Zou-He closure + corner refill 인터페이스
  - `build_boundary_indices`     4-face (left/right/top/bottom) 노드 인덱스 사전 계산
  - face별 BC 구현
      · velocity_inlet            속도 inlet + 압력 outlet
      · open_boundary             4-face 압력형 개방 경계
      · settling_channel          4-wall no-slip 침강 채널
      · dirichlet_neumann         Kang-Hassan benchmark용 Dirichlet+Neumann 혼합
      · tg_analytical             Taylor-Green 해석해 face closure

디스패치 규칙
  - `cfg.scenario_type == "taylor_green"` → `tg_analytical`
  - 그 외 → `cfg.bc_type` 키로 레지스트리 조회

계보
  - 직선 face Zou-He closure: Zou & He (1997)
      · Eq. (17) 경계 밀도 ρ
      · Eq. (18) 벽면 unknown population {f_2, f_5, f_6} (bottom 예)
      · Eq. (22) 압력 inlet 법선 속도 u_x
      · Eq. (23) inlet unknown population {f_1, f_5, f_8}
  - Corner refill / 접선 외삽 / 4-face 조합: 프로젝트 구현
"""
from __future__ import annotations

from .base import (
    BoundaryIndices,
    BoundaryStrategy,
    get_boundary,
    list_boundaries,
    register_boundary,
    register_boundary_as,
)
from .indices import build_boundary_indices
from .velocity_inlet import apply_bc_velocity_inlet, VelocityInletBC
from .open_boundary import apply_bc_open_boundary, OpenBoundaryBC
from .settling_channel import apply_bc_settling_channel, SettlingChannelBC
from .dirichlet_neumann import (
    DirichletNeumannBC,
    apply_bc_dirichlet_neumann,
)
from .tg_analytical import apply_bc_analytical, TGAnalyticalBC

__all__ = [
    "BoundaryIndices",
    "BoundaryStrategy",
    "build_boundary_indices",
    "get_boundary",
    "list_boundaries",
    "register_boundary",
    "register_boundary_as",
    "apply_boundary_step",
    "apply_bc_velocity_inlet",
    "apply_bc_open_boundary",
    "apply_bc_settling_channel",
    "apply_bc_dirichlet_neumann",
    "apply_bc_analytical",
]


def apply_boundary_step(fstar, ro, U, state, cfg, ttt: int) -> None:
    get_boundary(cfg).apply(fstar, ro, U, state, cfg, ttt)
