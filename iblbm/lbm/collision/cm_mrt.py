"""D2Q9 central-moment MRT 충돌 + Guo 강제항.

중심모멘트 정의 (c_i = e_i − u, frame co-moving with fluid)
    k_{pq} = Σ_i f_i · c_ix^p · c_iy^q

D2Q9 에서 사용하는 9개 모멘트 순서 (차수 (p, q))

      q↑
       2 │ k_{0,2}   k_{1,2}   k_{2,2}
       1 │ k_{0,1}   k_{1,1}   k_{2,1}
       0 │ k_{0,0}   k_{1,0}   k_{2,0}
         └────────────────────────────▶ p
               0         1         2

코드 인덱스 매핑 (cm0..cm8)
    cm0 = k_{0,0} = ρ
    cm1 = k_{1,0}                 cm2 = k_{0,1}
    cm3 = k_{2,0}                 cm4 = k_{0,2}
    cm5 = k_{1,1}
    cm6 = k_{2,1}                 cm7 = k_{1,2}
    cm8 = k_{2,2}

완화 (in CM space)
    k_{pq}^post = k_{pq} − s_{pq} (k_{pq} − k_{pq}^eq)
                + Δt · (1 − s_{pq}/2) · F_{pq}

    - s_{pq} 는 `S_diag[0..8]` 로 전달
    - 보존 모멘트(s0, s1, s2)=0, 점성 관련(s3, s4, s5)=1/τ

k^eq 의 Maxwell-Boltzmann 한계 (Galilean-invariant)
    k_{0,0}^eq = ρ,   k_{2,0}^eq = k_{0,2}^eq = ρ c_s²,   k_{2,2}^eq = ρ c_s⁴
    그 외 nonconserved k^eq = 0

CM → population 환원: raw moment 역변환 후 D2Q9 population 재구성
(코드 말단의 `f_out[n, 0..8]` 대수 블록)

참조
  - De Rosis (2017) *Alternative formulation ... central moments*
  - De Rosis, Huang, Coreixas (2019) *Universal formulation ...*
  - Lallemand & Luo (2000) moment ordering 및 MRT 구조
"""
from __future__ import annotations

from ...backend import _use_gpu
from ..lattice import make_d2q9_mrt_matrices
from .base import register_collision

if not _use_gpu:
    from numba import njit, prange
    import numpy as np

    @njit(parallel=True, cache=True)
    def _collision_cm_mrt_nb(fstar, feq, U, fib, S_diag, dt, e, w):
        N = fstar.shape[0]
        f_out = np.empty_like(fstar)

        for n in prange(N):
            ux = U[n, 0]
            uy = U[n, 1]
            cm0 = 0.0
            cm1 = 0.0
            cm2 = 0.0
            cm3 = 0.0
            cm4 = 0.0
            cm5 = 0.0
            cm6 = 0.0
            cm7 = 0.0
            cm8 = 0.0
            Fcm0 = 0.0
            Fcm1 = 0.0
            Fcm2 = 0.0
            Fcm3 = 0.0
            Fcm4 = 0.0
            Fcm5 = 0.0
            Fcm6 = 0.0
            Fcm7 = 0.0
            Fcm8 = 0.0

            for i in range(9):
                val = fstar[n, i]
                cix = e[i, 0] - ux
                ciy = e[i, 1] - uy
                cix2 = cix * cix
                ciy2 = ciy * ciy

                cm0 += val
                cm1 += val * cix
                cm2 += val * ciy
                cm3 += val * cix2
                cm4 += val * ciy2
                cm5 += val * cix * ciy
                cm6 += val * cix2 * ciy
                cm7 += val * cix * ciy2
                cm8 += val * cix2 * ciy2

                ax = e[i, 0] - ux
                ay = e[i, 1] - uy
                eU = e[i, 0] * ux + e[i, 1] * uy
                term = ((3.0 * ax + 9.0 * eU * e[i, 0]) * fib[n, 0]
                        + (3.0 * ay + 9.0 * eU * e[i, 1]) * fib[n, 1])
                Fb = w[i] * term

                Fcm0 += Fb
                Fcm1 += Fb * cix
                Fcm2 += Fb * ciy
                Fcm3 += Fb * cix2
                Fcm4 += Fb * ciy2
                Fcm5 += Fb * cix * ciy
                Fcm6 += Fb * cix2 * ciy
                Fcm7 += Fb * cix * ciy2
                Fcm8 += Fb * cix2 * ciy2

            rho = cm0
            cs2 = 1.0 / 3.0
            cs4 = 1.0 / 9.0

            s0 = S_diag[0]
            s1 = S_diag[1]
            s2 = S_diag[2]
            s3 = S_diag[3]
            s4 = S_diag[4]
            s5 = S_diag[5]
            s6 = S_diag[6]
            s7 = S_diag[7]
            s8 = S_diag[8]

            cp0 = cm0 - s0 * (cm0 - rho) + (1.0 - 0.5 * s0) * Fcm0 * dt
            cp1 = cm1 - s1 * (cm1 - 0.0) + (1.0 - 0.5 * s1) * Fcm1 * dt
            cp2 = cm2 - s2 * (cm2 - 0.0) + (1.0 - 0.5 * s2) * Fcm2 * dt
            cp3 = cm3 - s3 * (cm3 - rho * cs2) + (1.0 - 0.5 * s3) * Fcm3 * dt
            cp4 = cm4 - s4 * (cm4 - rho * cs2) + (1.0 - 0.5 * s4) * Fcm4 * dt
            cp5 = cm5 - s5 * (cm5 - 0.0) + (1.0 - 0.5 * s5) * Fcm5 * dt
            cp6 = cm6 - s6 * (cm6 - 0.0) + (1.0 - 0.5 * s6) * Fcm6 * dt
            cp7 = cm7 - s7 * (cm7 - 0.0) + (1.0 - 0.5 * s7) * Fcm7 * dt
            cp8 = cm8 - s8 * (cm8 - rho * cs4) + (1.0 - 0.5 * s8) * Fcm8 * dt

            ux2 = ux * ux
            uy2 = uy * uy

            k0 = cp0
            k1 = cp1 + ux * cp0
            k2 = cp2 + uy * cp0
            k3 = cp3 + 2.0 * ux * cp1 + ux2 * cp0
            k4 = cp4 + 2.0 * uy * cp2 + uy2 * cp0
            k5 = cp5 + uy * cp1 + ux * cp2 + ux * uy * cp0
            k6 = (cp6 + uy * cp3 + 2.0 * ux * cp5
                  + 2.0 * ux * uy * cp1 + ux2 * cp2 + ux2 * uy * cp0)
            k7 = (cp7 + 2.0 * uy * cp5 + ux * cp4
                  + 2.0 * ux * uy * cp2 + uy2 * cp1 + ux * uy2 * cp0)
            k8 = (cp8 + 2.0 * uy * cp6 + uy2 * cp3
                  + 2.0 * ux * cp7 + 4.0 * ux * uy * cp5
                  + 2.0 * ux * uy2 * cp1 + ux2 * cp4
                  + 2.0 * ux2 * uy * cp2 + ux2 * uy2 * cp0)

            f_out[n, 0] = k0 - k3 - k4 + k8
            f_out[n, 1] = 0.5 * (k1 + k3 - k7 - k8)
            f_out[n, 2] = 0.5 * (k2 + k4 - k6 - k8)
            f_out[n, 3] = 0.5 * (-k1 + k3 + k7 - k8)
            f_out[n, 4] = 0.5 * (-k2 + k4 + k6 - k8)
            f_out[n, 5] = 0.25 * (k5 + k6 + k7 + k8)
            f_out[n, 6] = 0.25 * (-k5 + k6 - k7 + k8)
            f_out[n, 7] = 0.25 * (k5 - k6 - k7 + k8)
            f_out[n, 8] = 0.25 * (-k5 - k6 + k7 + k8)

        return f_out


class CMMRTStrategy:
    name = "CM_MRT"

    def init_state(self, cfg, state) -> None:
        import numpy as _np

        state.M, state.M_inv = make_d2q9_mrt_matrices()
        s_eps = cfg.mrt_s_eps if cfg.mrt_s_eps is not None else 1.4
        s_q = cfg.mrt_s_q if cfg.mrt_s_q is not None else 1.2
        s_nu = 1.0 / state.tau
        state.S_diag = _np.array([
            0.0,
            0.0,
            0.0,
            s_nu,
            s_nu,
            s_nu,
            s_q,
            s_q,
            s_eps,
        ], dtype=_np.float64)

    def extra_kwargs(self, state) -> dict:
        return {"S_diag": state.S_diag}

    def step_cpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        return _collision_cm_mrt_nb(
            fstar, feq, U, fib, state.S_diag, dt, lattice.e, lattice.w,
        )

    def step_gpu(self, fstar, feq, U, fib, tau, dt, lattice, state):
        from ...gpu_kernels import collision_cm_mrt_gpu
        return collision_cm_mrt_gpu(fstar, feq, U, fib, state.S_diag, dt)


register_collision(CMMRTStrategy())
