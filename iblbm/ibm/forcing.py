"""IBM 공통 forcing helper: Lagrangian ↔ Eulerian 매핑 + 유체력 집계.

Regularized delta function δ_h 기반 보간/분산

        Eulerian grid (x)                Lagrangian markers (X_l)
        ┌─┬─┬─┬─┬─┬─┐                   •  X_l-1
        ├─┼─┼─┼─┼─┼─┤    δ_h(x-X_l)     •  X_l     ← body surface
        ├─┼─●─┼─┼─┼─┤  ─────────────▶   •  X_l+1
        ├─┼─┼─┼─┼─┼─┤    interp (9a)
        └─┴─┴─┴─┴─┴─┘
              │                              │
              │  spread (9b)                 │
              ▼                              ▼
        f(x) = Σ_l F_l δ_h(x-X_l) ΔV_l   F_l = (U_d - Ũ_l) / Δt

핵심 식
  Ũ(X_l) = Σ_x ũ(x) δ_h(x − X_l) h³        Uhlmann (2005) Eq. (9a)
  f(x)   = Σ_l F(X_l) δ_h(x − X_l) ΔV_l    Uhlmann (2005) Eq. (9b)
  δ_h 수송 보존 (Uhlmann 2005 Eq. (11a, 11b))
    Σ_x f(x) h³          = Σ_l F_l ΔV_l
    Σ_x x × f(x) h³      = Σ_l X_l × F_l ΔV_l

유체력·토크 집계 (프로젝트 구현)
  F_h = −Σ_l F_l Δs
  T_h = −Σ_l (X_l − X_c) × F_l Δs
    - Uhlmann (2005) Eq. (13a, 13b) 의 `−ρ_f Σ_l F_l ΔV_l` 항 (ρ_f = 1, ΔV_l = Δs·h)
    - 내부 유체 관성 `ρ_f d/dt ∫_V_p u dV` 는 별도 모듈(`physics/inertia/`)에서 처리
    - 내부 유체 토크 관성 `ρ_f d/dt ∫_S (x−x_c)×u dx` 도 분리

참조 계보
  - Peskin, The Immersed Boundary Method (2002)                          regularized δ_h 기초
  - Uhlmann, An Immersed Boundary Method with Direct Forcing (2005)      식 (9), (11), (13)
  - Breugem, A Second-Order Accurate Immersed Boundary Method (2012)     식 (7), (8) 외력 집계 관점
"""
from __future__ import annotations

from ..backend import xp as np, _use_gpu
from .df import ibm_direct_forcing


def interpolate_velocity_to_points(state, cfg, Lx, Ly, U_src, ro_src):
    """Eulerian → Lagrangian 속도 보간 (Uhlmann 2005 Eq. (9a)).

    수식
      Ũ(X_l) = Σ_x u(x) δ_h(x − X_l) h³

    입력·용도
      - `desired_zero` 는 force 계산을 생략하고 순수 보간만 하기 위한 더미 desired velocity
      - GPU: `_ibm_interp_force_kern` 커널의 interp 부분 재활용
      - CPU: `ibm_direct_forcing` 의 첫 패스 (Larea=1.0)

    Returns
      (Lux, Luy) — Lagrangian 점 위치의 보간된 속도 성분
    """
    if _use_gpu:
        import cupy as cp
        from ..gpu_kernels import (
            _BLOCK,
            _DELTA_STENCIL_A,
            _DELTA_TYPE_IDS,
            _grid,
            _ibm_interp_force_kern,
        )

        Eux = U_src[:, 0].reshape(state.ny, state.nx)
        Euy = U_src[:, 1].reshape(state.ny, state.nx)
        Ro = ro_src.reshape(state.ny, state.nx)
        Lx = cp.ascontiguousarray(cp.asarray(Lx))
        Ly = cp.ascontiguousarray(cp.asarray(Ly))
        desired_zero = cp.zeros((len(Lx), 2), dtype=cp.float64)
        delta_type_id = _DELTA_TYPE_IDS[cfg.delta_type]
        stencil_a = _DELTA_STENCIL_A[cfg.delta_type]

        Lfx = cp.empty(len(Lx), dtype=cp.float64)
        Lfy = cp.empty(len(Lx), dtype=cp.float64)
        Lux = cp.empty(len(Lx), dtype=cp.float64)
        Luy = cp.empty(len(Lx), dtype=cp.float64)
        _ibm_interp_force_kern(
            _grid(len(Lx)),
            (_BLOCK,),
            (
                Lx,
                Ly,
                desired_zero,
                cp.ascontiguousarray(Eux),
                cp.ascontiguousarray(Euy),
                cp.ascontiguousarray(Ro),
                cp.float64(state.dx),
                cp.float64(state.dy),
                cp.float64(state.dt),
                cp.int32(state.nx),
                cp.int32(state.ny),
                cp.int32(len(Lx)),
                cp.int32(delta_type_id),
                cp.int32(stencil_a),
                cp.int32(int(cfg.incompressible_lbgk)),
                Lfx,
                Lfy,
                Lux,
                Luy,
            ),
        )
        return Lux, Luy

    desired_zero = np.zeros((len(Lx), 2))
    _, Lux, Luy, _, _, _ = ibm_direct_forcing(
        Lx,
        Ly,
        desired_zero,
        U_src[:, 0].reshape(state.ny, state.nx),
        U_src[:, 1].reshape(state.ny, state.nx),
        ro_src.reshape(state.ny, state.nx),
        state.dx,
        state.dy,
        state.dt,
        1.0,
        state.ny,
        state.nx,
        delta_type=cfg.delta_type,
        incompressible=cfg.incompressible_lbgk,
    )
    return Lux, Luy


def ibm_df_particle_lagrangian_closure(
    state,
    cfg,
    Lx,
    Ly,
    desired_vel,
    Larea,
    cx,
    cy,
    U_src=None,
    ro_src=None,
):
    """입자 1개 DF closure: 보간 → 강제항 → 분산 → 유체력/토크 집계.

    단계별 식
      1. 보간    Ũ(X_l) = Σ_x ũ(x) δ_h(x − X_l) h³              Uhlmann (2005) Eq. (9a)
      2. 강제항   F(X_l) = (U_d(X_l) − Ũ(X_l)) / Δt              Uhlmann (2005) Eq. (12c)
      3. 분산    f(x)   = Σ_l F(X_l) δ_h(x − X_l) ΔV_l          Uhlmann (2005) Eq. (9b)
      4. 집계    F_h    = −Σ_l F_l Δs                            프로젝트 집계
                T_h    = −Σ_l (X_l − X_c) × F_l Δs

    Returns
      (fib_p, force, torque) — Eulerian body force 분포, 총 유체력, 토크
    """
    if U_src is None:
        U_src = state.U
    if ro_src is None:
        ro_src = state.ro

    if _use_gpu:
        import cupy as cp
        from ..runtime.device import to_cpu
        from ..gpu_kernels import (
            _BLOCK,
            _DELTA_STENCIL_A,
            _DELTA_TYPE_IDS,
            _grid,
            _ibm_interp_force_kern,
            _ibm_spread_kern,
        )

        Eux = U_src[:, 0].reshape(state.ny, state.nx)
        Euy = U_src[:, 1].reshape(state.ny, state.nx)
        Ro = ro_src.reshape(state.ny, state.nx)
        delta_type_id = _DELTA_TYPE_IDS[cfg.delta_type]
        stencil_a = _DELTA_STENCIL_A[cfg.delta_type]
        nodenums = state.ny * state.nx
        Lb = len(Lx)

        Lx = cp.ascontiguousarray(cp.asarray(Lx))
        Ly = cp.ascontiguousarray(cp.asarray(Ly))
        desired_vel = cp.ascontiguousarray(cp.asarray(desired_vel))
        Eux = cp.ascontiguousarray(Eux)
        Euy = cp.ascontiguousarray(Euy)
        Ro = cp.ascontiguousarray(Ro)

        Lfx = cp.empty(Lb, dtype=cp.float64)
        Lfy = cp.empty(Lb, dtype=cp.float64)
        Lux = cp.empty(Lb, dtype=cp.float64)
        Luy = cp.empty(Lb, dtype=cp.float64)
        _ibm_interp_force_kern(
            _grid(Lb),
            (_BLOCK,),
            (
                Lx,
                Ly,
                desired_vel,
                Eux,
                Euy,
                Ro,
                cp.float64(state.dx),
                cp.float64(state.dy),
                cp.float64(state.dt),
                cp.int32(state.nx),
                cp.int32(state.ny),
                cp.int32(Lb),
                cp.int32(delta_type_id),
                cp.int32(stencil_a),
                cp.int32(int(cfg.incompressible_lbgk)),
                Lfx,
                Lfy,
                Lux,
                Luy,
            ),
        )

        Efx = cp.zeros((state.ny, state.nx), dtype=cp.float64)
        Efy = cp.zeros((state.ny, state.nx), dtype=cp.float64)
        _ibm_spread_kern(
            _grid(Lb),
            (_BLOCK,),
            (
                Lfx,
                Lfy,
                Lx,
                Ly,
                cp.float64(state.dx),
                cp.float64(state.dy),
                cp.float64(Larea),
                cp.int32(state.nx),
                cp.int32(state.ny),
                cp.int32(Lb),
                cp.int32(delta_type_id),
                cp.int32(stencil_a),
                Efx,
                Efy,
            ),
        )

        fib_p = cp.empty((nodenums, 2), dtype=cp.float64)
        fib_p[:, 0] = Efx.ravel()
        fib_p[:, 1] = Efy.ravel()

        force = to_cpu(cp.asarray([
            -cp.sum(Lfx * Larea),
            -cp.sum(Lfy * Larea),
        ]))
        rx = (Lx - cx) / state.dx
        ry = (Ly - cy) / state.dy
        torque = float(to_cpu(-cp.sum((rx * Lfy - ry * Lfx) * Larea)))
        return fib_p, force, torque

    import numpy as _np

    Eux = U_src[:, 0].reshape(state.ny, state.nx)
    Euy = U_src[:, 1].reshape(state.ny, state.nx)
    Ro = ro_src.reshape(state.ny, state.nx)
    fib_p, _, _, Lfx, Lfy, _ = ibm_direct_forcing(
        Lx,
        Ly,
        desired_vel,
        Eux,
        Euy,
        Ro,
        state.dx,
        state.dy,
        state.dt,
        Larea,
        state.ny,
        state.nx,
        delta_type=cfg.delta_type,
        incompressible=cfg.incompressible_lbgk,
    )
    force = _np.array([
        -float(_np.sum(Lfx * Larea)),
        -float(_np.sum(Lfy * Larea)),
    ])
    rx = (Lx - cx) / state.dx
    ry = (Ly - cy) / state.dy
    torque = -float(_np.sum((rx * Lfy - ry * Lfx) * Larea))
    return fib_p, force, torque
