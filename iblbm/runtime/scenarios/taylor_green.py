"""Taylor-Green 감쇠 와류 벤치마크 runtime.

대상 (scenario_type="taylor_green")
  - 표준 2D Taylor-Green vortex — 주기 정사각 도메인 [−L, L]²
  - 해석해 u(x, y, t) = u₀ · exp(−2·ν·(π/L)²·t) · 조화함수 성분
  - 용도: 공간/시간 정확도 2차 수렴성 검증, IBM 포함 시 경계 오차 측정

옵션
  - `cfg.tg_with_ibm=False`   순수 LBM 수렴 테스트 (IBM 불호출)
  - `cfg.tg_with_ibm=True`    IBM 포함, 마커 desired velocity 를 매 스텝 해석해로 갱신
"""

from __future__ import annotations

from ...ibm import get_ibm
from ...diagnostics import tg_analytical_velocity_field
from .. import step as step_mod
from .base import register_scenario


def _update_tg_desired_velocity(state, cfg, ttt: int) -> None:
    """현재 시간 t 의 해석해를 각 마커의 desired_velocity 로 주입."""
    t_physical = ttt * state.dt
    L_lat = 0.5 * (cfg.NN - 1)
    lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
    nu = cfg.lattice_u * lattice_D / cfg.Re
    Lx_centered = state.Lx / state.dx - (state.nx - 1) / 2.0
    Ly_centered = state.Ly / state.dy - (state.ny - 1) / 2.0
    ux_ana, uy_ana = tg_analytical_velocity_field(Lx_centered, Ly_centered, t_physical, cfg.tg_u0, L_lat, nu)
    state.desired_velocity[:, 0] = ux_ana
    state.desired_velocity[:, 1] = uy_ana


class TaylorGreenRuntime:
    """Taylor-Green 시나리오 runtime handler."""

    name = "taylor_green"

    def matches(self, cfg) -> bool:
        return cfg.scenario_type == "taylor_green"

    def initialize(self, state, cfg) -> dict:
        ibm = get_ibm(cfg.ibm_method)
        ibm.prepare(state, cfg)
        return {"ibm": ibm}

    def pre_step(self, state, cfg, cache, ttt: int) -> None:
        if cfg.tg_with_ibm:
            _update_tg_desired_velocity(state, cfg, ttt)

    def pre_ibm(self, state, cfg, cache, ttt: int) -> None:
        return None

    def apply_ibm(self, state, cfg, cache, ttt: int) -> None:
        step_mod.apply_standard_ibm(state, cfg, ttt)

    def post_ibm(self, state, cfg, cache, ttt: int) -> None:
        return None

    def diagnostics(self, state, cfg, cache, ttt: int) -> dict:
        if not cfg.tg_with_ibm:
            return {"Cd": 0.0, "Cl": 0.0}
        Cd, Cl = cache["ibm"].compute_cd_cl(state, cfg, state.in_u)
        return {"Cd": Cd, "Cl": Cl}

    def should_stop(self, state, cfg, cache, ttt: int) -> str | None:
        return None

    def finalize(self, state, cfg, cache, result: dict) -> dict:
        return result


register_scenario(TaylorGreenRuntime())
