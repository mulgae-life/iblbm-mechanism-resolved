"""IBM Strategy dispatch facade.

역할
  - `cfg.ibm_method` ("DF" | "MDF" | "DFC") → `get_ibm()` 으로 IBMStrategy 인스턴스 획득
  - `prepare → apply_single / apply_multi → (opt) velocity correction` 순서로 호출
  - Strategy Protocol 내부는 각 `iblbm/ibm/*.py` 가 구현

velocity correction (incompressible vs compressible 분기만)
  incompressible LBGK :  u ← u + 0.5 · f_ib · Δt
  compressible LBGK   :  u ← u + f_ib · Δt / (2 ρ)
"""
from __future__ import annotations

from ..backend import _use_gpu
from .base import get_ibm


def apply_velocity_correction(state, cfg) -> None:
    """Guo forcing 후 half-step velocity correction — incompressible/compressible 분기."""
    if _use_gpu:
        from ..gpu_kernels import velocity_correction_gpu
        velocity_correction_gpu(
            state.U,
            state.fib,
            state.ro,
            state.dt,
            incompressible=cfg.incompressible_lbgk,
        )
        return
    if cfg.incompressible_lbgk:
        state.U = state.U + 0.5 * state.fib * state.dt
    else:
        state.U = state.U + state.fib * state.dt / (2.0 * state.ro[:, None])


def apply_ibm_step(state, cfg, ttt: int) -> None:
    """단일 입자 IBM 1-step — Strategy 선택 후 `prepare → apply_single → (opt) correction`."""
    if cfg.scenario_type == "taylor_green" and not cfg.tg_with_ibm:
        return
    strat = get_ibm(cfg.ibm_method)
    strat.prepare(state, cfg)
    strat.apply_single(state, cfg, ttt)
    if strat.requires_velocity_correction():
        apply_velocity_correction(state, cfg)


def apply_ibm_step_multi(state, cfg, ttt: int) -> None:
    """다입자 IBM 1-step — Strategy 선택 후 `prepare → apply_multi → (opt) correction`."""
    strat = get_ibm(cfg.ibm_method)
    strat.prepare(state, cfg)
    strat.apply_multi(state, cfg, ttt)
    if strat.requires_velocity_correction():
        apply_velocity_correction(state, cfg)
