"""고정/진동 실린더 runtime (Kang & Hassan 2011 벤치마크 anchor).

대상 시나리오 (motion_type)
  - `None`          고정 실린더 (steady Re 20/40, unsteady Re 100/150)
  - `oscillating`   진동 실린더 (Dütsch 1998 실험 anchor, KC 파라미터)

Kang-Hassan 2011 §3.2 setup
  - 도메인          40D × 40D
  - 경계조건        left/top/bottom = Dirichlet (u_inlet),
                    right           = homogeneous Neumann (outlet)
  - Re 정의         Re = u∞·D / ν

hook 구성
  - `initialize`       IBM prepare + 초기 마커 스냅샷 cache 저장
  - `apply_ibm`        `apply_standard_ibm` (단일 입자 경로)
  - `post_collision`   `oscillating` 에서 마커 위치 갱신
  - `diagnostics`      IBM Strategy 의 `compute_cd_cl(state, cfg, u_ref)`
      u_ref: oscillating → `cfg.lattice_u`, 그 외 → `state.in_u`
"""

from __future__ import annotations

from ...ibm import get_ibm
from ...physics.motion import update_oscillating
from .. import step as step_mod
from .base import register_scenario


class FixedCylinderRuntime:
    """고정/진동 실린더 공용 runtime handler."""

    name = "fixed_cylinder"

    def matches(self, cfg) -> bool:
        excluded = {"taylor_green"}
        return cfg.scenario_type not in excluded and cfg.motion_type in {None, "oscillating"}

    def initialize(self, state, cfg) -> dict:
        ibm = get_ibm(cfg.ibm_method)
        ibm.prepare(state, cfg)
        return {"Lx_c": state.Lx.copy(), "Ly_c": state.Ly.copy(), "ibm": ibm}

    def pre_step(self, state, cfg, cache, ttt: int) -> None:
        return None

    def pre_ibm(self, state, cfg, cache, ttt: int) -> None:
        return None

    def apply_ibm(self, state, cfg, cache, ttt: int) -> None:
        step_mod.apply_standard_ibm(state, cfg, ttt)

    def post_ibm(self, state, cfg, cache, ttt: int) -> None:
        return None

    def post_collision(self, state, cfg, cache, ttt: int) -> None:
        if cfg.motion_type == "oscillating":
            state.Lx, state.Ly, state.desired_velocity = update_oscillating(
                ttt,
                cache["Lx_c"],
                cache["Ly_c"],
                state.desired_velocity,
                cfg.lattice_u,
                cfg.KC,
                state.r,
                state.lattice_r,
            )

    def diagnostics(self, state, cfg, cache, ttt: int) -> dict:
        u_ref = cfg.lattice_u if cfg.motion_type == "oscillating" else state.in_u
        Cd, Cl = cache["ibm"].compute_cd_cl(state, cfg, u_ref)
        return {"Cd": Cd, "Cl": Cl}

    def should_stop(self, state, cfg, cache, ttt: int) -> str | None:
        return None

    def finalize(self, state, cfg, cache, result: dict) -> dict:
        return result


register_scenario(FixedCylinderRuntime())
