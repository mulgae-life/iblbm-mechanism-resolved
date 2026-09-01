"""DFC (Distribution Function Correction) Strategy.

  - `prepare`         `lambda_cache`, `dfc_force_lagr` 슬롯 초기화
  - `apply_single`    `apply_dfc_correction` → `fstar` 직접 보정 + λ 캐시 갱신(고정 입자만) → macroscopic 재계산
  - `apply_multi`     particle set 병합 → 단일 호출 처리 후 per-particle force/torque 분배
  - `extract_*`       자체 보정력 `dfc_force_lagr` 사용 (DF/MDF의 `fib`와 경로 분리)
  - `compute_cd_cl`   `compute_cd_cl_dfc` 전용 (부호·정규화 DF/MDF와 상이)
  - 플래그            `requires_velocity_correction=False`, `uses_force_level_current_step_scheduler=False`
"""
from __future__ import annotations

import numpy as _np

from ...backend import _use_gpu, xp as _xp
from ...diagnostics import compute_cd_cl_dfc
from ...lbm import macroscopic
from ...runtime.device import to_cpu
from ..base import register_ibm
from ..dfc import apply_dfc_correction


class DFCStrategy:
    name = "DFC"

    def prepare(self, state, cfg) -> None:
        if not hasattr(state, "lambda_cache"):
            state.lambda_cache = None
        if not hasattr(state, "dfc_force_lagr"):
            state.dfc_force_lagr = None

    def apply_single(self, state, cfg, ttt: int) -> None:
        cache = state.lambda_cache if cfg.motion_type is None else None
        state.fstar, state.dfc_force_lagr, lambda_new = apply_dfc_correction(
            state.Lx,
            state.Ly,
            state.desired_velocity,
            state.fstar,
            state.dx,
            state.dy,
            state.Larea,
            state.ny,
            state.nx,
            delta_type=cfg.delta_type,
            lattice=state.lattice,
            lambda_cache=cache,
            correction_scale=float(getattr(state, "_ibm_force_scale", 1.0)),
        )
        if cfg.motion_type is None:
            state.lambda_cache = lambda_new
        state.ro, state.U = macroscopic(
            state.fstar,
            state.lattice,
            incompressible=cfg.incompressible_lbgk,
        )

    def apply_multi(self, state, cfg, ttt: int) -> None:
        if not state.particles:
            raise ValueError("DFC multi-particle path에 state.particles 필요")

        starts: list[tuple[object, int, int]] = []
        Lx_parts = []
        Ly_parts = []
        desired_parts = []
        cursor = 0
        Larea_ref = float(state.particles[0].Larea)
        for particle in state.particles:
            Lx_cpu = _np.asarray(to_cpu(particle.Lx))
            Ly_cpu = _np.asarray(to_cpu(particle.Ly))
            desired_cpu = _np.asarray(to_cpu(particle.desired_velocity))
            point_count = int(len(Lx_cpu))
            if point_count <= 0:
                raise ValueError("DFC multi-particle path에서 빈 particle marker set 발견")
            if abs(float(particle.Larea) - Larea_ref) > 1e-12:
                raise ValueError("DFC multi-particle path는 현재 동일 Larea particle set만 지원")
            if len(Ly_cpu) != point_count or len(desired_cpu) != point_count:
                raise ValueError("DFC multi-particle path에서 particle point-set 길이 불일치")
            starts.append((particle, cursor, cursor + point_count))
            Lx_parts.append(Lx_cpu)
            Ly_parts.append(Ly_cpu)
            desired_parts.append(desired_cpu)
            cursor += point_count

        Lx_all = _np.concatenate(Lx_parts, axis=0)
        Ly_all = _np.concatenate(Ly_parts, axis=0)
        desired_all = _np.concatenate(desired_parts, axis=0)
        if _use_gpu:
            import cupy as cp

            Lx_all = cp.asarray(Lx_all)
            Ly_all = cp.asarray(Ly_all)
            desired_all = cp.asarray(desired_all)

        state.fstar, state.dfc_force_lagr, _ = apply_dfc_correction(
            Lx_all,
            Ly_all,
            desired_all,
            state.fstar,
            state.dx,
            state.dy,
            Larea_ref,
            state.ny,
            state.nx,
            delta_type=cfg.delta_type,
            lattice=state.lattice,
            lambda_cache=None,
            correction_scale=float(getattr(state, "_ibm_force_scale", 1.0)),
        )
        state.ro, state.U = macroscopic(
            state.fstar,
            state.lattice,
            incompressible=cfg.incompressible_lbgk,
        )

        for particle, start, end in starts:
            force_lagr = to_cpu(state.dfc_force_lagr[start:end])
            Lx_cpu = _np.asarray(to_cpu(particle.Lx))
            Ly_cpu = _np.asarray(to_cpu(particle.Ly))
            particle._fib_force = _np.array([
                float(force_lagr[:, 0].sum()),
                float(force_lagr[:, 1].sum()),
            ])
            if cfg.enable_rotation:
                rx_lat = (Lx_cpu - float(particle.pos[0])) / state.dx
                ry_lat = (Ly_cpu - float(particle.pos[1])) / state.dy
                particle._torque_new = float(_np.sum(
                    rx_lat * force_lagr[:, 1] - ry_lat * force_lagr[:, 0]
                ))

    def extract_force(self, state, cfg):
        if state.dfc_force_lagr is None:
            raise ValueError("DFC에서 dfc_force_lagr가 None")
        return _np.array([
            float(state.dfc_force_lagr[:, 0].sum()),
            float(state.dfc_force_lagr[:, 1].sum()),
        ])

    def extract_torque(self, state, cfg, cx: float, cy: float) -> float:
        rx_lat = (state.Lx - cx) / state.dx
        ry_lat = (state.Ly - cy) / state.dy
        return float(_xp.sum(
            rx_lat * state.dfc_force_lagr[:, 1] - ry_lat * state.dfc_force_lagr[:, 0]
        ))

    def uses_force_level_current_step_scheduler(self) -> bool:
        return False

    def requires_velocity_correction(self) -> bool:
        return False

    def compute_cd_cl(self, state, cfg, u_ref: float):
        return compute_cd_cl_dfc(
            to_cpu(state.dfc_force_lagr),
            float(to_cpu(state.ro).mean()),
            u_ref,
            state.lattice_r,
        )


register_ibm(DFCStrategy())
