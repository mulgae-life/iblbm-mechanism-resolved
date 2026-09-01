"""IB-LBM 메인 시간 루프 (`run(cfg)`).

한 스텝 알고리즘 흐름

    ┌───────────────────────────────────────────────────────────────┐
    │  runtime.pre_step         (시나리오별 입자 위치/desired_vel)  │
    │  collision                f = f* − (f* − f_eq)/τ + Guo 체적력 │
    │  streaming                f* ← stream(f)                     │
    │  boundary                 Zou-He / Kang-Hassan closure       │
    │  macro                    ρ = Σf*, u = Σf*·e / ρ             │
    │  runtime.pre_ibm          (inertia preliminary step 등)       │
    │  runtime.apply_ibm        DF / MDF / DFC → fib, u 재보정      │
    │  runtime.post_ibm         (Verlet full, Euler step 등)        │
    │  update_feq               f_eq(ρ, u)                          │
    │  (force-level)  collision 다시 수행                           │
    │  runtime.post_collision   (진동 실린더 marker 갱신)            │
    │  diagnostics              Cd/Cl, y*/v*, 수렴 error            │
    │  should_stop              contact/offset/nan/bound            │
    └───────────────────────────────────────────────────────────────┘

collision 타이밍 분기
  - `stream_first=True` 또는 force-level 스케줄러 → 루프 진입 전 1회 + 매 스텝 말미에 수행
  - 기본 경로                                    → 매 스텝 상단에서 수행

반환 결과
  - `Cd_history`, `Cl_history`, `state`, `converged`, `final_step`
  - 침강 시 `sedimentation_history`, `termination_reason` 추가
"""
from __future__ import annotations

import json
import time

from ..config import SimConfig
from ..diagnostics import check_convergence
from ..init import initialize
from .device import to_cpu
from .scheduler import uses_force_level_current_step_scheduler
from .scenarios.base import get_scenario_runtime
from . import scenarios  # noqa: F401
from . import step as step_mod


def run(cfg: SimConfig, verbose: bool = True, callback=None, save_dir=None) -> dict:
    state = initialize(cfg)
    runtime = get_scenario_runtime(cfg)
    cache = runtime.initialize(state, cfg)

    Cd_history = []
    Cl_history = []
    converged = False
    error = 1.0
    pre_Eux = None
    pre_Euy = None
    t_start = time.time()
    ttt = 1
    termination_reason = "max_steps"
    f = None
    force_level_current_step = uses_force_level_current_step_scheduler(cfg)
    if cfg.stream_first or force_level_current_step:
        f = step_mod.collide(state, cfg)

    while True:
        # use_convergence(velocity) 또는 cd_convergence_tol(Cd-기반) 어느 쪽이든 조기 종료 허용
        if converged and (cfg.use_convergence or cfg.cd_convergence_tol is not None):
            if termination_reason == "max_steps":
                termination_reason = "converged"
            break
        if ttt > cfg.max_steps:
            break

        if cache.get("inertia") is not None and hasattr(cache["inertia"], "preliminary_step"):
            cache["extended_inertia_snapshot"] = (state.fstar, state.feq, state.U, state.ro)

        runtime.pre_step(state, cfg, cache, ttt)

        if not cfg.stream_first and not force_level_current_step:
            f = step_mod.collide(state, cfg)

        step_mod.stream_boundary_macro(state, cfg, f, ttt)
        runtime.pre_ibm(state, cfg, cache, ttt)
        runtime.apply_ibm(state, cfg, cache, ttt)
        runtime.post_ibm(state, cfg, cache, ttt)
        step_mod.update_feq(state, cfg)

        if cfg.stream_first or force_level_current_step:
            f = step_mod.collide(state, cfg)
        if hasattr(runtime, "post_collision"):
            runtime.post_collision(state, cfg, cache, ttt)

        if ttt % cfg.check_interval == 0:
            diag = runtime.diagnostics(state, cfg, cache, ttt)
            Cd = float(diag.get("Cd", 0.0))
            Cl = float(diag.get("Cl", 0.0))
            Cd_history.append(Cd)
            Cl_history.append(Cl)
            if verbose:
                elapsed = time.time() - t_start
                if diag.get("log_line") is not None:
                    print(f"{diag['log_line']} | err={error:.2e} | {elapsed:.1f}s")
                else:
                    print(f"step {ttt:>7d} | Cd={Cd:.6f} Cl={Cl:.6f} | err={error:.2e} | {elapsed:.1f}s")
            if callback is not None:
                callback(step=ttt, Cd=Cd, Cl=Cl, error=error, state=state, elapsed=time.time() - t_start, converged=converged)
            reason = runtime.should_stop(state, cfg, cache, ttt)
            if reason is not None:
                termination_reason = reason
                break
            if save_dir and cfg.motion_type == "sedimentation" and ttt % 5000 == 0:
                with open(save_dir + "/sedimentation_history.json", "w") as handle:
                    json.dump(cache.get("history", []), handle)

            Eux_now = to_cpu(state.U[:, 0]).reshape(state.ny, state.nx)
            Euy_now = to_cpu(state.U[:, 1]).reshape(state.ny, state.nx)
            if cfg.use_convergence:
                if ttt == cfg.convergence_start:
                    pre_Eux = Eux_now.copy()
                    pre_Euy = Euy_now.copy()
                elif ttt > cfg.convergence_start and pre_Eux is not None:
                    error = check_convergence(Eux_now, Euy_now, pre_Eux, pre_Euy)
                    pre_Eux = Eux_now.copy()
                    pre_Euy = Euy_now.copy()
                    if error < cfg.convergence_threshold:
                        converged = True

            # Cd-기반 조기 종료 (Δ-window spread; transient 이후 정상상태 빠른 종료)
            if (
                cfg.cd_convergence_tol is not None
                and ttt > cfg.cd_convergence_start_step
                and len(Cd_history) >= cfg.cd_convergence_window
            ):
                recent = Cd_history[-cfg.cd_convergence_window:]
                cd_spread = max(recent) - min(recent)
                if cd_spread < cfg.cd_convergence_tol:
                    converged = True
                    termination_reason = "cd_converged"

        ttt += 1

    result = {
        "Cd_history": __import__("numpy").array(Cd_history),
        "Cl_history": __import__("numpy").array(Cl_history),
        "state": state,
        "converged": converged,
        "final_step": ttt - 1,
        "termination_reason": termination_reason,
    }
    if cfg.motion_type == "sedimentation":
        cache["termination_reason"] = termination_reason
    result = runtime.finalize(state, cfg, cache, result)
    return result
