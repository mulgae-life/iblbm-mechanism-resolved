"""D2Q9 TRT 충돌 + Guo 강제항.

대칭/반대칭 분해 (e_i ↔ e_{opp(i)})

         f_i
          │     ┌─ f⁺ = ½(f_i + f_{opp(i)})     symmetric   (밀도·응력)
          │ ──▶ ┤
          │     └─ f⁻ = ½(f_i − f_{opp(i)})     anti-sym.   (운동량·열 flux)
      f_{opp(i)}

1-step update
    f_i^{n+1} = f_i*
              − s⁺ ( f⁺* − f⁺^eq )
              − s⁻ ( f⁻* − f⁻^eq )
              + Δt F_i

    - s⁺ = 1/τ            (viscosity 결정)
    - s⁻ = 1/τ⁻           (free parameter)
    - Λ_eo = (τ − ½)(τ⁻ − ½)  magic parameter (Ginzburg et al. 2008, Eq. (2.4))

강제항도 동일한 대칭 분해로 적용 (Guo et al. 2002)
    F_i⁺ = ½(F_i + F_{opp(i)})    with prefactor (1 − s⁺/2)
    F_i⁻ = ½(F_i − F_{opp(i)})    with prefactor (1 − s⁻/2)

참조
  - Ginzburg, Verhaeghe, d'Humières (2008) Eq. (2.4)
  - Guo, Zheng, Shi (2002) Eq. (20)
"""
from __future__ import annotations

from ...backend import _use_gpu
from .base import register_collision

if not _use_gpu:
    from numba import njit, prange
    import numpy as np

    @njit(parallel=True, cache=True)
    def _collision_trt_nb(fstar, feq, U, fib, s_plus, s_minus, dt, e, w, opp):
        N = fstar.shape[0]
        f = np.empty_like(fstar)
        guo_pref_plus = 1.0 - 0.5 * s_plus
        guo_pref_minus = 1.0 - 0.5 * s_minus

        for n in prange(N):
            bare = np.empty(9)
            for i in range(9):
                ax = e[i, 0] - U[n, 0]
                ay = e[i, 1] - U[n, 1]
                eU = e[i, 0] * U[n, 0] + e[i, 1] * U[n, 1]
                term = ((3.0 * ax + 9.0 * eU * e[i, 0]) * fib[n, 0]
                        + (3.0 * ay + 9.0 * eU * e[i, 1]) * fib[n, 1])
                bare[i] = w[i] * term

            for i in range(9):
                oi = opp[i]
                fstar_plus = 0.5 * (fstar[n, i] + fstar[n, oi])
                fstar_minus = 0.5 * (fstar[n, i] - fstar[n, oi])
                feq_plus = 0.5 * (feq[n, i] + feq[n, oi])
                feq_minus = 0.5 * (feq[n, i] - feq[n, oi])
                collision = (s_plus * (fstar_plus - feq_plus)
                             + s_minus * (fstar_minus - feq_minus))
                bare_plus = 0.5 * (bare[i] + bare[oi])
                bare_minus = 0.5 * (bare[i] - bare[oi])
                F_ni = guo_pref_plus * bare_plus + guo_pref_minus * bare_minus
                f[n, i] = fstar[n, i] - collision + F_ni * dt
        return f


class TRTStrategy:
    name = "TRT"

    def init_state(self, cfg, state) -> None:
        Lambda = cfg.trt_magic_param
        state.tau_minus = 0.5 + Lambda / (state.tau - 0.5)

    def extra_kwargs(self, state) -> dict:
        return {"tau_minus": state.tau_minus}

    def step_cpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        s_plus = 1.0 / tau
        s_minus = 1.0 / state.tau_minus
        return _collision_trt_nb(
            fstar, feq, U, fib, s_plus, s_minus, dt,
            lattice.e, lattice.w, lattice.opp,
        )

    def step_gpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        from ...gpu_kernels import collision_trt_gpu
        return collision_trt_gpu(fstar, feq, U, fib, tau, state.tau_minus, dt)


register_collision(TRTStrategy())
