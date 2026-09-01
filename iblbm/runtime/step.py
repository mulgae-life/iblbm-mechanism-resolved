"""LBM 1-step 공통 러너 (loop 에서 호출되는 얇은 wrapper 집합).

구성
  - `collision_kwargs`              충돌 모델별 추가 파라미터 추출
  - `collide`                       f = f* − (f* − f_eq)/τ + Guo forcing
  - `stream_boundary_macro`         streaming → BC → macro (ρ, u)
  - `update_feq`                    f_eq(ρ, u) 재계산
  - `apply_standard_ibm`            단일/다입자 IBM 분기
  - `forcing_free_preliminary_state`
        full-volume preliminary 경로 — fib = 0 으로 collision/stream/BC/macro 수행
"""
from __future__ import annotations

from ..backend import xp as _xp
from ..boundary import apply_boundary_step
from ..ibm import apply_ibm_step, apply_ibm_step_multi
from ..lbm import collision_step, compute_feq, macroscopic, streaming_step
from ..lbm.collision import get_collision


def collision_kwargs(state, cfg) -> dict:
    """충돌 모델(`BGK`/`TRT`/`CM_MRT`) 의 `extra_kwargs(state)` 위임."""
    return get_collision(cfg.collision_model).extra_kwargs(state)


def collide(state, cfg, *, fstar=None, feq=None, U=None, fib=None):
    """1회 충돌 수행 — f = f* − (f* − f_eq)/τ + Guo forcing.

    - 기본 입력은 `state.*` 필드이고, 명시 인자로 override 가능 (preliminary 경로용)
    """
    fstar_in = state.fstar if fstar is None else fstar
    feq_in = state.feq if feq is None else feq
    U_in = state.U if U is None else U
    fib_in = state.fib if fib is None else fib
    return collision_step(
        fstar_in,
        feq_in,
        U_in,
        fib_in,
        state.tau,
        state.dt,
        state.lattice,
        collision_model=cfg.collision_model,
        state=state,
        **collision_kwargs(state, cfg),
    )


def stream_boundary_macro(state, cfg, f_post_collision, ttt: int) -> None:
    """post-collision f → streaming → BC → macro (ρ, u) 체인.

    부작용: `state.fstar`, `state.ro`, `state.U` 갱신
    """
    state.fstar = streaming_step(state.fstar, f_post_collision, state.nx, state.ny)
    apply_boundary_step(state.fstar, state.ro, state.U, state, cfg, ttt)
    state.ro, state.U = macroscopic(
        state.fstar,
        state.lattice,
        incompressible=cfg.incompressible_lbgk,
    )


def update_feq(state, cfg) -> None:
    """현재 (ρ, u) 기반 f_eq 재계산 (`state.feq` in-place 갱신)."""
    state.feq = compute_feq(
        state.ro,
        state.U,
        state.lattice,
        incompressible=cfg.incompressible_lbgk,
    )


def apply_standard_ibm(state, cfg, ttt: int) -> None:
    """다입자 존재 + 침강 → multi path, 그 외 → single path."""
    if cfg.motion_type == "sedimentation" and state.particles is not None:
        apply_ibm_step_multi(state, cfg, ttt)
        return
    apply_ibm_step(state, cfg, ttt)


def forcing_free_preliminary_state(state, cfg, ttt, fstar_prev, feq_prev, U_prev, ro_prev):
    """full-volume IMC preliminary 경로 — fib = 0 (forcing-free) collision/stream/BC/macro.

    - 반환: `(fstar_pre, ro_pre, U_pre)` — 현재 state 는 변경하지 않음
    - García-Villalba 2023 preliminary velocity 계산에 사용됨
    """
    if getattr(state, "_full_volume_zero_fib", None) is None:
        state._full_volume_zero_fib = _xp.zeros_like(state.fib)

    f_pre = collision_step(
        fstar_prev,
        feq_prev,
        U_prev,
        state._full_volume_zero_fib,
        state.tau,
        state.dt,
        state.lattice,
        collision_model=cfg.collision_model,
        state=state,
        **collision_kwargs(state, cfg),
    )
    fstar_pre = streaming_step(fstar_prev, f_pre, state.nx, state.ny)
    apply_boundary_step(fstar_pre, ro_prev, U_prev, state, cfg, ttt)
    ro_pre, U_pre = macroscopic(
        fstar_pre,
        state.lattice,
        incompressible=cfg.incompressible_lbgk,
    )
    return fstar_pre, ro_pre, U_pre
