"""MDF (Multi-Direct Forcing) Strategy.

  - `prepare`         δ-type 기반 최적 ω 1회 추정 → `state._mdf_omega` 캐시
  - `apply_single`    iterative `ibm_multi_direct_forcing` (n_iter, tol, ω) + rotation_ctx 조립
  - `apply_multi`     `multi_particle_mdf` 위임
  - `extract_*`       DF Strategy와 동일(부호·적분 규약 공유)
  - 플래그            `requires_velocity_correction=True`, `uses_force_level_current_step_scheduler=True`
"""
from __future__ import annotations

import numpy as _np

from ...backend import xp as _xp
from ...diagnostics import compute_cd_cl
from ...physics.inertia import get_inertia_model
from ...runtime.device import to_cpu
from ...physics.sedimentation import compute_desired_velocity
from ..a_norm import estimate_optimal_omega_from_type
from ..base import register_ibm
from ..mdf import ibm_multi_direct_forcing
from ..multi_particle import multi_particle_mdf


MOVING_ROTATION_OMEGA_MAX = 1.0


class MDFStrategy:
    name = "MDF"

    def prepare(self, state, cfg) -> None:
        if getattr(state, "_mdf_omega", None) is None:
            omega = estimate_optimal_omega_from_type(
                Lx=state.Lx,
                Ly=state.Ly,
                delta_type=cfg.delta_type,
                Larea=float(state.Larea),
                dx=float(state.dx),
            )
            if cfg.motion_type == "sedimentation" and cfg.enable_rotation:
                omega = min(float(omega), MOVING_ROTATION_OMEGA_MAX)
            state._mdf_omega = omega

    def apply_single(self, state, cfg, ttt: int) -> None:
        rotation_ctx = None
        if (
            cfg.rotation_coupling == "iterative"
            and cfg.enable_rotation
            and cfg.motion_type == "sedimentation"
            and not get_inertia_model(cfg.settling_inertia_model).uses_preliminary_update
        ):
            rotation_ctx = {
                "vel_half": getattr(state, "_vel_half_cache", None),
                "omega_half": getattr(state, "_omega_half_cache", 0.0),
                "I_particle": state.particle_I,
                "cx": float(state.particle_pos[0]),
                "cy": float(state.particle_pos[1]),
                "II": state._torque_II,
                "JJ": state._torque_JJ,
                "compute_desired_velocity": compute_desired_velocity,
            }

        state.fib, state._mdf_iter_stats = ibm_multi_direct_forcing(
            state.Lx,
            state.Ly,
            state.desired_velocity,
            state.U,
            state.ro,
            state.dx,
            state.dy,
            state.dt,
            state.Larea,
            state.ny,
            state.nx,
            n_iter=cfg.mdf_iterations,
            min_iter=cfg.mdf_min_iterations,
            delta_type=cfg.delta_type,
            omega=state._mdf_omega,
            tol=cfg.mdf_tolerance,
            rotation_ctx=rotation_ctx,
        )
        force_scale = float(getattr(state, "_ibm_force_scale", 1.0))
        if force_scale != 1.0:
            state.fib = force_scale * state.fib

    def apply_multi(self, state, cfg, ttt: int) -> None:
        multi_particle_mdf(state, cfg)

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


register_ibm(MDFStrategy())
