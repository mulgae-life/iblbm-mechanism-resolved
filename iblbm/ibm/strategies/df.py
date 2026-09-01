"""DF (Direct Forcing) Strategy.

  - `prepare`         추가 상태 없음
  - `apply_single`    Eulerian 속도(U, ρ) reshape → `ibm_direct_forcing` 1-pass → `state.fib`
  - `apply_multi`     `multi_particle_df` 위임
  - `extract_*`       Lagrangian 힘의 음수 합(유체→입자 반작용), 토크는 `(r × fib)` 적분
  - 플래그            `requires_velocity_correction=True`, `uses_force_level_current_step_scheduler=True`
"""
from __future__ import annotations

import numpy as _np

from ...diagnostics import compute_cd_cl
from ...backend import xp as _xp
from ...runtime.device import to_cpu
from ..base import register_ibm
from ..df import ibm_direct_forcing
from ..multi_particle import multi_particle_df


class DFStrategy:
    name = "DF"

    def prepare(self, state, cfg) -> None:
        return None

    def apply_single(self, state, cfg, ttt: int) -> None:
        Eux = state.U[:, 0].reshape(state.ny, state.nx)
        Euy = state.U[:, 1].reshape(state.ny, state.nx)
        Ro = state.ro.reshape(state.ny, state.nx)
        state.fib, _, _, _, _, _ = ibm_direct_forcing(
            state.Lx,
            state.Ly,
            state.desired_velocity,
            Eux,
            Euy,
            Ro,
            state.dx,
            state.dy,
            state.dt,
            state.Larea,
            state.ny,
            state.nx,
            delta_type=cfg.delta_type,
            incompressible=cfg.incompressible_lbgk,
        )
        force_scale = float(getattr(state, "_ibm_force_scale", 1.0))
        if force_scale != 1.0:
            state.fib = force_scale * state.fib

    def apply_multi(self, state, cfg, ttt: int) -> None:
        multi_particle_df(state, cfg)

    def extract_force(self, state, cfg):
        return -_np.array([
            float(state.fib[:, 0].sum()),
            float(state.fib[:, 1].sum()),
        ])

    def extract_torque(self, state, cfg, cx: float, cy: float) -> float:
        fib_x = state.fib[:, 0].reshape(state.ny, state.nx)
        fib_y = state.fib[:, 1].reshape(state.ny, state.nx)
        cx_lat = cx / state.dx
        cy_lat = cy / state.dy
        return -float(_xp.sum(
            (state._torque_II - cx_lat) * fib_y - (state._torque_JJ - cy_lat) * fib_x
        ))

    def uses_force_level_current_step_scheduler(self) -> bool:
        return True

    def requires_velocity_correction(self) -> bool:
        return True

    def compute_cd_cl(self, state, cfg, u_ref: float):
        return compute_cd_cl(
            to_cpu(state.fib),
            to_cpu(state.ro),
            u_ref,
            state.lattice_r,
        )


register_ibm(DFStrategy())
