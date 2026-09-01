"""D2Q9 BGK (SRT) 충돌 + Guo 강제항.

1-step update
    f_i^{n+1} = f_i* − (1/τ)(f_i* − f_i^eq) + Δt · F_i

강제항 F_i (Guo et al. 2002, Eq. (20))
    F_i = (1 − 1/(2τ)) w_i [ 3(e_i − u) + 9(e_i·u) e_i ] · f_ib

    - `f_ib` : IB body force per unit mass
    - `(1 − 1/(2τ))` 계수는 discrete lattice 보정 (Eq. (20))

흐름
    fstar, feq, U, fib  ──▶  (τ, Δt, w_i, e_i)  ──▶  f_i^{n+1}
                             └ 완화 항 ─┘   └ 강제 항 ─┘
"""
from __future__ import annotations

from ...backend import _use_gpu
from .base import register_collision

if not _use_gpu:
    from numba import njit, prange
    import numpy as np

    @njit(parallel=True, cache=True)
    def _collision_bgk_nb(fstar, feq, U, fib, tau, dt, e, w):
        N = fstar.shape[0]
        f = np.empty_like(fstar)
        inv_tau = 1.0 / tau
        guo_pref = 1.0 - 0.5 / tau
        for n in prange(N):
            for i in range(9):
                ax = e[i, 0] - U[n, 0]
                ay = e[i, 1] - U[n, 1]
                eU = e[i, 0] * U[n, 0] + e[i, 1] * U[n, 1]
                term = ((3.0 * ax + 9.0 * eU * e[i, 0]) * fib[n, 0]
                        + (3.0 * ay + 9.0 * eU * e[i, 1]) * fib[n, 1])
                F_ni = guo_pref * w[i] * term
                f[n, i] = fstar[n, i] - inv_tau * (fstar[n, i] - feq[n, i]) + F_ni * dt
        return f


class BGKStrategy:
    name = "BGK"

    def init_state(self, cfg, state) -> None:
        return None

    def extra_kwargs(self, state) -> dict:
        return {}

    def step_cpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        return _collision_bgk_nb(fstar, feq, U, fib, tau, dt, lattice.e, lattice.w)

    def step_gpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        from ...gpu_kernels import collision_gpu
        return collision_gpu(fstar, feq, U, fib, tau, dt)


register_collision(BGKStrategy())
