"""GPU 커널 모음 (CuPy RawModule).

적용 조건
  - `_use_gpu=True` 경로에서만 import
  - 각 CUDA 커널은 `RawModule`로 1회 컴파일 후 JIT 캐시 재사용

CPU ↔ GPU 대응표 (수식·참조는 CPU 파일과 동일)

    ┌────────────────────────────┬─────────────────────────────────────────┐
    │ GPU 래퍼                    │ CPU 대응 파일 · 수식                      │
    ├────────────────────────────┼─────────────────────────────────────────┤
    │ collision_gpu              │ lbm/collision/bgk.py  (Guo Eq. (20))    │
    │ collision_trt_gpu          │ lbm/collision/trt.py  (Ginzburg 2008)   │
    │ collision_cm_mrt_gpu       │ lbm/collision/cm_mrt.py (De Rosis 2017) │
    │ streaming_gpu              │ lbm/streaming.py  (pull vs CPU push)    │
    │ macroscopic_gpu            │ lbm/macroscopic.py (ρ, u)               │
    │ feq_gpu                    │ lbm/equilibrium.py (f_i^eq)             │
    │ velocity_correction_gpu    │ solver 경로 (u ← u + f_ib·Δt/(2ρ))       │
    │ ibm_direct_forcing_gpu     │ ibm/df.py  (Uhlmann Eq. (9a)-(9b))      │
    │ compute_lambda_gpu         │ ibm/dfc.py  (Tao Eq. (22)-(23))         │
    │ dfc_correction_gpu         │ ibm/dfc.py  (Tao Eq. (15)-(25))         │
    └────────────────────────────┴─────────────────────────────────────────┘

수식 표기 규칙
  - 격자 단위 c ≡ Δx/Δt = 1,  c_s² = 1/3
  - Unicode: ρ, τ, Δ, Σ, ≈, ≤, ⁻¹ 등 (CPU 경로와 동일)

스트리밍 차이
  - CPU (`lbm/streaming.py`)  : push 방식, `dst[j+e_y, i+e_x] ← src[j, i]`
  - GPU (`streaming_gpu`)     : pull 방식, `dst[j, i] ← src[j−e_y, i−e_x]`
    수식 자체는 동일  `f_i(x + e_i·Δt, t + Δt) = f_i*(x, t)`
"""

import cupy as cp

# =============================================================================
# CUDA 소스 (단일 문자열, RawModule 컴파일 대상)
# =============================================================================

_CUDA_SOURCE = r"""
// === D2Q9 격자 상수 (lbm/lattice.py 와 동일 순서) ============================
//   i = 0        정지,        w₀ = 4/9
//   i ∈ {1..4}   축 e_i,      w_i = 1/9
//   i ∈ {5..8}   대각 e_i,    w_i = 1/36
__device__ const double D_E[9][2] = {
    {0,0}, {1,0}, {0,1}, {-1,0}, {0,-1},
    {1,1}, {-1,1}, {-1,-1}, {1,-1}
};
__device__ const double D_W[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// === Regularized delta kernel φ(r) (ibm/common.py 와 동일 수식) ==============
//   delta_type_id = 0 → Peskin 4-point (support 4·dx, stencil radius a = 2)
//   delta_type_id = 1 → hat            (support 2·dx, stencil radius a = 1)
__device__ double delta_func(double r, int delta_type_id) {
    double ar = fabs(r);
    if (delta_type_id == 0) {  // peskin4pt
        if (ar <= 1.0) {
            return 0.125 * (3.0 - 2.0*ar + sqrt(fmax(1.0 + 4.0*ar - 4.0*ar*ar, 0.0)));
        } else if (ar <= 2.0) {
            return 0.125 * (5.0 - 2.0*ar - sqrt(fmax(-7.0 + 12.0*ar - 4.0*ar*ar, 0.0)));
        }
        return 0.0;
    } else {  // hat
        return fmax(1.0 - ar, 0.0);
    }
}

// =========================================================================
// 1. BGK 충돌 + Guo forcing  —  lbm/collision/bgk.py 대응
//    1 thread per (n, i)  → 총 N·9 threads
//    f_i^{n+1} = f_i* − (1/τ)(f_i* − f_i^eq) + Δt · F_i
//    F_i       = (1 − 1/(2τ)) w_i [ 3(e_i − u) + 9(e_i·u) e_i ] · f_ib
//                                                  Guo et al. 2002 Eq. (20)
// =========================================================================
extern "C" __global__ void collision_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ feq,
    const double* __restrict__ U,
    const double* __restrict__ fib,
    double inv_tau, double guo_pref, double dt,
    int N,
    double* __restrict__ f_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 9) return;

    int n = tid / 9;
    int i = tid % 9;

    double ux = U[n*2], uy = U[n*2+1];
    double ex = D_E[i][0], ey = D_E[i][1];
    double eU = ex*ux + ey*uy;

    double term = (3.0*(ex - ux) + 9.0*eU*ex) * fib[n*2]
                + (3.0*(ey - uy) + 9.0*eU*ey) * fib[n*2+1];
    double F_ni = guo_pref * D_W[i] * term;

    f_out[tid] = fstar[tid] - inv_tau*(fstar[tid] - feq[tid]) + F_ni*dt;
}

// =========================================================================
// 2. D2Q9 스트리밍 (pull-based)  —  lbm/streaming.py 대응 (CPU는 push)
//    1 thread per node (총 nx·ny threads)
//
//    수식 (동일)       f_i(x + e_i·Δt, t + Δt) = f_i*(x, t)
//    pull 형태 접근    fstar_out[j, i, d] ← f[j − e_y[d], i − e_x[d], d]
//    비주기 경계       이웃이 격자 밖이면 `fstar_old` (이전 값) 보존
// =========================================================================
extern "C" __global__ void streaming_kernel(
    const double* __restrict__ f,
    const double* __restrict__ fstar_old,
    int nx, int ny,
    double* __restrict__ fstar_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int j = tid / nx;
    int i = tid % nx;

    // D2Q9 방향 벡터 (ex, ey), lbm/lattice.py 의 e_i 와 동일 순서
    const int ex[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int ey[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    // d = 0: 정지 항 — 자기 셀 복사
    fstar_out[tid*9] = f[tid*9];

    // d = 1..8: 이웃에서 자기 셀로 들어오는 f_i 를 pull
    for (int d = 1; d < 9; d++) {
        int sj = j - ey[d];
        int si = i - ex[d];
        if (sj >= 0 && sj < ny && si >= 0 && si < nx) {
            fstar_out[tid*9 + d] = f[(sj*nx + si)*9 + d];
        } else {
            fstar_out[tid*9 + d] = fstar_old[tid*9 + d];
        }
    }
}

// =========================================================================
// 3. 거시변수 복원 (ρ, u)  —  lbm/macroscopic.py 대응
//    1 thread per node
//    표준(compressible)     ρ = Σ_i f_i,   u = (Σ_i f_i e_i) / ρ
//    incompressible LBGK    ρ → p̂ 압력 surrogate (He & Luo 1997 계보)
//                           u = Σ_i f_i e_i (나눗셈 없이)
// =========================================================================
extern "C" __global__ void macroscopic_kernel(
    const double* __restrict__ fstar,
    int N,
    int incompressible,
    double* __restrict__ ro,
    double* __restrict__ U
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double s = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; i++) {
        double fi = fstar[n*9 + i];
        s += fi;
        ux += fi * D_E[i][0];
        uy += fi * D_E[i][1];
    }
    ro[n] = s;
    if (incompressible) {
        // incompressible LBGK: u = Σ f·e, ρ로 나누지 않음
        U[n*2]   = ux;
        U[n*2+1] = uy;
    } else {
        U[n*2]   = ux / s;
        U[n*2+1] = uy / s;
    }
}

// =========================================================================
// 4. 평형 분포 f_i^eq  —  lbm/equilibrium.py 대응
//    1 thread per (n, i)  → 총 N·9 threads
//    표준(compressible)   f_i^eq = w_i ρ [1 + 3(e_i·u) + 9/2 (e_i·u)² − 3/2 u²]
//    incompressible LBGK  f_i^eq = w_i [ρ + 3(e_i·u) + 9/2 (e_i·u)² − 3/2 u²]
//                                                 He & Luo (1997) 계보
// =========================================================================
extern "C" __global__ void feq_kernel(
    const double* __restrict__ ro,
    const double* __restrict__ U,
    int N,
    int incompressible,
    double* __restrict__ feq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 9) return;

    int n = tid / 9;
    int i = tid % 9;

    double ux = U[n*2], uy = U[n*2+1];
    double eU = D_E[i][0]*ux + D_E[i][1]*uy;
    double u2 = ux*ux + uy*uy;
    if (incompressible) {
        // incompressible LBGK form: feq = w·(ρ + 3(e·u) + 9/2(e·u)² − 3/2 u²)
        feq[tid] = D_W[i] * (ro[n] + 3.0*eU + 4.5*eU*eU - 1.5*u2);
    } else {
        feq[tid] = ro[n] * D_W[i] * (1.0 + 3.0*eU + 4.5*eU*eU - 1.5*u2);
    }
}

// =========================================================================
// 5. 속도 보정 (in-place, Guo forcing 2-moment 일관성)  —  solver 경로 대응
//    1 thread per node
//    표준(compressible)   u ← u + f_ib · Δt / (2ρ)
//    incompressible       u ← u + 0.5 · f_ib · Δt    (ρ로 나누지 않음)
// =========================================================================
extern "C" __global__ void velocity_correction_kernel(
    double* __restrict__ U,
    const double* __restrict__ fib,
    const double* __restrict__ ro,
    double dt, int N, int incompressible
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double factor = incompressible ? 0.5 * dt : dt / (2.0 * ro[n]);
    U[n*2]   += fib[n*2]   * factor;
    U[n*2+1] += fib[n*2+1] * factor;
}

// =========================================================================
// 6. IBM DF 보간 + 힘 계산  —  ibm/df.py 의 (a)+(b) 단계 대응
//    1 thread per Lagrangian marker (총 Lb threads)
//    (a) 보간 Ũ, ρ̃ = Σ_x {u, ρ}(x) δ_h(x − X_l)    Uhlmann 2005 Eq. (9a)
//    (b) 힘   F_l = 2 ρ̃ (U_d − Ũ) / Δt              Wang/Fan/Luo 2008 계열
//             F_l = 2    (U_d − Ũ) / Δt              Majumder 2023 Eq. (9)
// =========================================================================
extern "C" __global__ void ibm_interp_force_kernel(
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    const double* __restrict__ desired_vel,
    const double* __restrict__ Eux,
    const double* __restrict__ Euy,
    const double* __restrict__ Ro,
    double dx, double dy, double dt,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    int incompressible,
    double* __restrict__ Lfx,
    double* __restrict__ Lfy,
    double* __restrict__ Lux_out,
    double* __restrict__ Luy_out
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double lux = 0.0, luy = 0.0, r_interp = 0.0;

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            int idx = ej_c * nx + ei_c;
            r_interp += Ro[idx] * w;
            lux += Eux[idx] * w;
            luy += Euy[idx] * w;
        }
    }

    // incompressible: F = 2(U_d − Ũ)/Δt,  compressible: F = 2 ρ̃ (U_d − Ũ)/Δt
    double rw = incompressible ? 1.0 : r_interp;
    Lfx[k] = 2.0 * rw * (desired_vel[k*2]   - lux) / dt;
    Lfy[k] = 2.0 * rw * (desired_vel[k*2+1] - luy) / dt;
    // MDF velocity residual 수렴 판정을 위해 보간 속도 Ũ 반환
    Lux_out[k] = lux;
    Luy_out[k] = luy;
}

// =========================================================================
// 7. IBM DF 힘 분산 (atomicAdd)  —  ibm/df.py 의 (c) 단계 대응
//    1 thread per Lagrangian marker
//    f_ib(x) = Σ_l F_l δ_h(x − X_l) Δs                Uhlmann 2005 Eq. (9b)
//    중복 격자점 기여는 `atomicAdd` 로 race-free 누적
// =========================================================================
extern "C" __global__ void ibm_spread_kernel(
    const double* __restrict__ Lfx,
    const double* __restrict__ Lfy,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ Efx,
    double* __restrict__ Efy
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double lfx = Lfx[k], lfy = Lfy[k];

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((ei * dx - lx) / dx, delta_type_id);
            double wy = delta_func((ej * dy - ly) / dy, delta_type_id);
            double w = wx * wy * Larea;

            int idx = ej_c * nx + ei_c;
            atomicAdd(&Efx[idx], lfx * w);
            atomicAdd(&Efy[idx], lfy * w);
        }
    }
}

// =========================================================================
// D2Q9 반대 방향 인덱스 opp[i] (e_{opp(i)} = −e_i)
//   — DFC bounce-back (Eq. (16)) 및 TRT 대칭/반대칭 분해에서 공유
// =========================================================================
__device__ const int D_OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// =========================================================================
// 8. DFC f_i 보간  —  ibm/dfc.py::interpolate_f 대응
//    1 thread per Lagrangian marker
//    f_i*(X_k) = Σ_x f_i(x) W(x − X_k) dx²            Tao 2019 Eq. (15)
//    레이아웃  fstar (ny·nx, 9) → f_interp (Lb, 9)
// =========================================================================
extern "C" __global__ void dfc_interp_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ f_interp
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double fi[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            int idx = ej_c * nx + ei_c;
            for (int q = 0; q < 9; q++) {
                fi[q] += fstar[idx * 9 + q] * w;
            }
        }
    }

    for (int q = 0; q < 9; q++) {
        f_interp[k * 9 + q] = fi[q];
    }
}

// =========================================================================
// 9. DFC bounce-back + λ·편차 + 마커별 힘 집계 (융합 커널)
//    —  ibm/dfc.py::{bounce_back_fi, compute_dfc_fluid_force} 대응
//    1 thread per Lagrangian marker
//    f̄_i(X_k)     = f*_{opp(i)}(X_k) + 2 w_i ρ_f (e_i·u_w)/c_s²   Eq. (16)
//    Δf_i(X_k)    = λ(k) · [f̄_i − f_i*]                           Eq. (17)
//    q_k          = −Δs · Σ_i e_i · λ(k) · [f̄_i − f_i*]            Eq. (25) 분포부
// =========================================================================
extern "C" __global__ void dfc_bb_lambda_kernel(
    const double* __restrict__ f_interp,
    const double* __restrict__ desired_vel,
    const double* __restrict__ lambda_k,
    double rho_f, double Larea,
    int Lb,
    double* __restrict__ delta_f,
    double* __restrict__ force
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double ux_wall = desired_vel[k * 2];
    double uy_wall = desired_vel[k * 2 + 1];
    double lam = lambda_k[k];
    double cs2 = 1.0 / 3.0;

    double fx = 0.0, fy = 0.0;

    for (int q = 0; q < 9; q++) {
        double fi_star = f_interp[k * 9 + q];
        double fi_opp  = f_interp[k * 9 + D_OPP[q]];

        // Eq. (16): f̄_i = f*_{opp(i)} + 2 w_i ρ_f (e_i·u_w) / c_s²
        double e_dot_u = D_E[q][0] * ux_wall + D_E[q][1] * uy_wall;
        double f_bb = fi_opp + 2.0 * D_W[q] * rho_f * e_dot_u / cs2;

        // Eq. (17): Δf_i(X_k) = λ(k) · (f̄_i − f_i*)
        double dev = f_bb - fi_star;
        double df = lam * dev;
        delta_f[k * 9 + q] = df;

        // Eq. (25) 분포 기여부: q_k = −Δs · Σ_i e_i · λ · (f̄_i − f_i*)
        fx += -df * Larea * D_E[q][0];
        fy += -df * Larea * D_E[q][1];
    }

    force[k * 2]     = fx;
    force[k * 2 + 1] = fy;
}

// =========================================================================
// 10. DFC Δf_i 분산 (atomicAdd)  —  ibm/dfc.py::spread_delta_f 대응
//     1 thread per Lagrangian marker
//     Δf_i(x) = Σ_k Δf_i(X_k) W(x − X_k) Δs            Tao 2019 Eq. (18)
//     레이아웃  delta_f (Lb, 9) → Ef_corr (ny·nx, 9)
// =========================================================================
extern "C" __global__ void dfc_spread_kernel(
    const double* __restrict__ delta_f,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ Ef_corr
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double df[9];
    for (int q = 0; q < 9; q++) {
        df[q] = delta_f[k * 9 + q];
    }

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy * Larea;

            int idx = ej_c * nx + ei_c;
            for (int q = 0; q < 9; q++) {
                atomicAdd(&Ef_corr[idx * 9 + q], df[q] * w);
            }
        }
    }
}

// =========================================================================
// 11. DFC λ(k) — Step 1 : marker → 격자 weight 분산
//     —  ibm/dfc.py::compute_lambda 의 (a) spread 대응
//     1 thread per Lagrangian marker, atomicAdd 누적
//     W_total(x) = Σ_{k'} W(x − X_{k'})                Tao 2019 Eq. (22)-(23)
// =========================================================================
extern "C" __global__ void dfc_lambda_spread_kernel(
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ W_total
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            atomicAdd(&W_total[ej_c * nx + ei_c], w);
        }
    }
}

// =========================================================================
// 12. DFC λ(k) — Step 2+3 : 격자 → marker 보간 + λ 해석해
//     —  ibm/dfc.py::compute_lambda 의 (b) interpolate + λ 대응
//     1 thread per Lagrangian marker
//     W_sum(k)  = Σ_x W_total(x) · W(x − X_k)
//     λ(k)      = 1 / [ 2 ρ_f Δs · W_sum(k) ]          Tao 2019 Eq. (23)
// =========================================================================
extern "C" __global__ void dfc_lambda_interp_kernel(
    const double* __restrict__ W_total,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    double rho_f, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ lambda_k
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double W_sum = 0.0;

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            W_sum += W_total[ej_c * nx + ei_c] * w;
        }
    }

    lambda_k[k] = 1.0 / (2.0 * rho_f * Larea * W_sum);
}

// =========================================================================
// 13. TRT 충돌 + Guo forcing  —  lbm/collision/trt.py 대응
//     1 thread per node
//     대칭/반대칭 분해 (e_i ↔ e_{opp(i)})
//        f_i* = ½(f_i + f_{opp(i)})  symmetric   (밀도·응력)
//        f_i* = ½(f_i − f_{opp(i)})  anti-sym.   (운동량·열 flux)
//     f_i^{n+1} = f_i* − s⁺(f⁺* − f⁺^eq) − s⁻(f⁻* − f⁻^eq) + Δt F_i
//        s⁺ = 1/τ,  s⁻ = 1/τ⁻  (Ginzburg et al. 2008, Eq. (2.4))
//     강제항 F_i 도 동일 대칭 분해 + prefactor (Guo 2002 Eq. (20))
//     D_OPP 는 DFC 섹션에서 정의된 것을 공유
// =========================================================================
extern "C" __global__ void collision_trt_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ feq,
    const double* __restrict__ U,
    const double* __restrict__ fib,
    double s_plus, double s_minus, double dt,
    int N,
    double* __restrict__ f_out
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double ux = U[n*2], uy = U[n*2+1];
    double guo_pref_p = 1.0 - 0.5 * s_plus;
    double guo_pref_m = 1.0 - 0.5 * s_minus;

    // Guo bare term (prefactor 미포함): w_i [3(e_i−u) + 9(e_i·u) e_i] · f_ib
    double bare[9];
    for (int i = 0; i < 9; i++) {
        double ex = D_E[i][0], ey = D_E[i][1];
        double eU = ex*ux + ey*uy;
        double term = (3.0*(ex-ux) + 9.0*eU*ex) * fib[n*2]
                    + (3.0*(ey-uy) + 9.0*eU*ey) * fib[n*2+1];
        bare[i] = D_W[i] * term;
    }

    for (int i = 0; i < 9; i++) {
        int oi = D_OPP[i];
        double fs_p = 0.5 * (fstar[n*9+i] + fstar[n*9+oi]);
        double fs_m = 0.5 * (fstar[n*9+i] - fstar[n*9+oi]);
        double fe_p = 0.5 * (feq[n*9+i] + feq[n*9+oi]);
        double fe_m = 0.5 * (feq[n*9+i] - feq[n*9+oi]);

        double collision = s_plus*(fs_p - fe_p) + s_minus*(fs_m - fe_m);
        double bp = 0.5*(bare[i] + bare[oi]);
        double bm = 0.5*(bare[i] - bare[oi]);
        double F_ni = guo_pref_p*bp + guo_pref_m*bm;
        f_out[n*9+i] = fstar[n*9+i] - collision + F_ni*dt;
    }
}

// =========================================================================
// 14. Central-Moment MRT 충돌 + Guo forcing
//     —  lbm/collision/cm_mrt.py 대응 (De Rosis 2017/2019)
//     1 thread per node
//     자연 다항식 기저 central moment k_{pq} = Σ_i f_i · c_ix^p · c_iy^q
//     (c_i = e_i − u, frame co-moving with fluid)
//     4-단계 : 모멘트 산출 → k^eq 완화 → raw 모멘트 복원(shift) → f 복원
//     (역변환은 해석적, `f_out[n, 0..8]` 대수 블록)
// =========================================================================
extern "C" __global__ void collision_cm_mrt_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ feq,
    const double* __restrict__ U,
    const double* __restrict__ fib,
    const double* __restrict__ S_diag,
    double dt,
    int N,
    double* __restrict__ f_out
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double ux = U[n*2], uy = U[n*2+1];

    // --- Step 1: central moment k_{pq} + forcing CM F_{pq} 집계 ---
    //   c_ix = e_ix − u_x,  c_iy = e_iy − u_y
    //   cm0..cm8 인덱스 매핑은 lbm/collision/cm_mrt.py 와 동일
    double cm0=0, cm1=0, cm2=0, cm3=0, cm4=0, cm5=0, cm6=0, cm7=0, cm8=0;
    double Fc0=0, Fc1=0, Fc2=0, Fc3=0, Fc4=0, Fc5=0, Fc6=0, Fc7=0, Fc8=0;

    for (int i = 0; i < 9; i++) {
        double val = fstar[n*9+i];
        double cix = D_E[i][0] - ux;
        double ciy = D_E[i][1] - uy;
        double cix2 = cix*cix, ciy2 = ciy*ciy;

        cm0 += val;
        cm1 += val*cix;
        cm2 += val*ciy;
        cm3 += val*cix2;
        cm4 += val*ciy2;
        cm5 += val*cix*ciy;
        cm6 += val*cix2*ciy;
        cm7 += val*cix*ciy2;
        cm8 += val*cix2*ciy2;

        // Guo bare forcing (prefactor 미포함, Eq. (20))
        double ex = D_E[i][0], ey = D_E[i][1];
        double eU = ex*ux + ey*uy;
        double term = (3.0*(ex-ux)+9.0*eU*ex)*fib[n*2]
                    + (3.0*(ey-uy)+9.0*eU*ey)*fib[n*2+1];
        double Fb = D_W[i]*term;
        Fc0 += Fb;
        Fc1 += Fb*cix;
        Fc2 += Fb*ciy;
        Fc3 += Fb*cix2;
        Fc4 += Fb*ciy2;
        Fc5 += Fb*cix*ciy;
        Fc6 += Fb*cix2*ciy;
        Fc7 += Fb*cix*ciy2;
        Fc8 += Fb*cix2*ciy2;
    }

    // --- Step 2: k^eq 완화 (CM space)
    //   k_{pq}^post = k_{pq} − s_{pq}(k_{pq} − k_{pq}^eq) + Δt(1−s_{pq}/2) F_{pq}
    //   k^eq 비영(非零) 항만: k_{0,0}=ρ, k_{2,0}=k_{0,2}=ρ c_s², k_{2,2}=ρ c_s⁴
    double rho = cm0;
    double cs2 = 1.0/3.0, cs4 = 1.0/9.0;
    double s0=S_diag[0], s1=S_diag[1], s2=S_diag[2];
    double s3=S_diag[3], s4=S_diag[4], s5=S_diag[5];
    double s6=S_diag[6], s7=S_diag[7], s8=S_diag[8];

    double cp0 = cm0-s0*(cm0-rho)     + (1.0-0.5*s0)*Fc0*dt;
    double cp1 = cm1-s1*cm1           + (1.0-0.5*s1)*Fc1*dt;
    double cp2 = cm2-s2*cm2           + (1.0-0.5*s2)*Fc2*dt;
    double cp3 = cm3-s3*(cm3-rho*cs2) + (1.0-0.5*s3)*Fc3*dt;
    double cp4 = cm4-s4*(cm4-rho*cs2) + (1.0-0.5*s4)*Fc4*dt;
    double cp5 = cm5-s5*cm5           + (1.0-0.5*s5)*Fc5*dt;
    double cp6 = cm6-s6*cm6           + (1.0-0.5*s6)*Fc6*dt;
    double cp7 = cm7-s7*cm7           + (1.0-0.5*s7)*Fc7*dt;
    double cp8 = cm8-s8*(cm8-rho*cs4) + (1.0-0.5*s8)*Fc8*dt;

    // --- Step 3: central → raw polynomial (shift by u) ---
    double ux2 = ux*ux, uy2 = uy*uy;
    double k0 = cp0;
    double k1 = cp1 + ux*cp0;
    double k2 = cp2 + uy*cp0;
    double k3 = cp3 + 2.0*ux*cp1 + ux2*cp0;
    double k4 = cp4 + 2.0*uy*cp2 + uy2*cp0;
    double k5 = cp5 + uy*cp1 + ux*cp2 + ux*uy*cp0;
    double k6 = cp6 + uy*cp3 + 2.0*ux*cp5 + 2.0*ux*uy*cp1
                + ux2*cp2 + ux2*uy*cp0;
    double k7 = cp7 + 2.0*uy*cp5 + ux*cp4 + 2.0*ux*uy*cp2
                + uy2*cp1 + ux*uy2*cp0;
    double k8 = cp8 + 2.0*uy*cp6 + uy2*cp3 + 2.0*ux*cp7
                + 4.0*ux*uy*cp5 + 2.0*ux*uy2*cp1
                + ux2*cp4 + 2.0*ux2*uy*cp2 + ux2*uy2*cp0;

    // --- Step 4: raw moment → population (K⁻¹ 해석해, 축·대각 교차 결합) ---
    f_out[n*9+0] = k0 - k3 - k4 + k8;
    f_out[n*9+1] = 0.5*(k1 + k3 - k7 - k8);
    f_out[n*9+2] = 0.5*(k2 + k4 - k6 - k8);
    f_out[n*9+3] = 0.5*(-k1 + k3 + k7 - k8);
    f_out[n*9+4] = 0.5*(-k2 + k4 + k6 - k8);
    f_out[n*9+5] = 0.25*(k5 + k6 + k7 + k8);
    f_out[n*9+6] = 0.25*(-k5 + k6 - k7 + k8);
    f_out[n*9+7] = 0.25*(k5 - k6 - k7 + k8);
    f_out[n*9+8] = 0.25*(-k5 - k6 + k7 + k8);
}
"""

# =============================================================================
# 모듈 컴파일 + 커널 핸들 캐시
#   - `_module`  : NVRTC JIT 컴파일 결과 (프로세스 수명 동안 1회)
#   - `_*_kern`  : 커널별 device function 핸들 (launch 시 재사용)
# =============================================================================

_module = cp.RawModule(code=_CUDA_SOURCE)

_collision_kern = _module.get_function("collision_kernel")
_collision_trt_kern = _module.get_function("collision_trt_kernel")
_collision_cm_mrt_kern = _module.get_function("collision_cm_mrt_kernel")
_streaming_kern = _module.get_function("streaming_kernel")
_macroscopic_kern = _module.get_function("macroscopic_kernel")
_feq_kern = _module.get_function("feq_kernel")
_vel_correction_kern = _module.get_function("velocity_correction_kernel")
_ibm_interp_force_kern = _module.get_function("ibm_interp_force_kernel")
_ibm_spread_kern = _module.get_function("ibm_spread_kernel")
_dfc_interp_kern = _module.get_function("dfc_interp_kernel")
_dfc_bb_lambda_kern = _module.get_function("dfc_bb_lambda_kernel")
_dfc_spread_kern = _module.get_function("dfc_spread_kernel")
_dfc_lambda_spread_kern = _module.get_function("dfc_lambda_spread_kernel")
_dfc_lambda_interp_kern = _module.get_function("dfc_lambda_interp_kernel")

# 1D launch block 크기 (256은 대부분 GPU에서 occupancy·레지스터 균형에 무난)
_BLOCK = 256


def _grid(n):
    """n개 스레드를 커버하는 1D grid 크기  ⌈n / _BLOCK⌉."""
    return ((n + _BLOCK - 1) // _BLOCK,)


# =============================================================================
# Python 래퍼 — CPU `lbm/*`, `ibm/*` 경로와 1:1 시그니처 대응
# =============================================================================

def collision_gpu(fstar, feq, U, fib, tau, dt):
    """BGK 충돌 + Guo forcing GPU 래퍼.

    대응
        lbm/collision/bgk.py::_collision_bgk_nb

    수식
        f_i^{n+1} = f_i* − (1/τ)(f_i* − f_i^eq) + Δt · F_i
        F_i       = (1 − 1/(2τ)) w_i [3(e_i − u) + 9(e_i·u) e_i] · f_ib
                                                      Guo 2002 Eq. (20)

    Launch
        1 thread per (n, i)   →   _grid(N · 9)
    """
    N = fstar.shape[0]
    f = cp.empty_like(fstar)
    inv_tau = 1.0 / tau
    guo_pref = 1.0 - 0.5 * inv_tau
    _collision_kern(
        _grid(N * 9), (_BLOCK,),
        (fstar, feq, U, fib,
         cp.float64(inv_tau), cp.float64(guo_pref), cp.float64(dt),
         cp.int32(N), f),
    )
    return f


def collision_trt_gpu(fstar, feq, U, fib, tau, tau_minus, dt):
    """TRT 충돌 + Guo forcing GPU 래퍼.

    대응
        lbm/collision/trt.py::_collision_trt_nb

    수식
        s⁺ = 1/τ,  s⁻ = 1/τ⁻,  Λ_eo = (τ − ½)(τ⁻ − ½)
                                               Ginzburg et al. 2008 Eq. (2.4)
        f_i^{n+1} = f_i* − s⁺(f⁺* − f⁺^eq) − s⁻(f⁻* − f⁻^eq) + Δt F_i
        F_i 대칭 분해 prefactor (1 − s⁺/2), (1 − s⁻/2)  — Guo 2002 Eq. (20)

    Launch
        1 thread per node   →   _grid(N)
    """
    N = fstar.shape[0]
    f = cp.empty_like(fstar)
    s_plus = 1.0 / tau
    s_minus = 1.0 / tau_minus
    _collision_trt_kern(
        _grid(N), (_BLOCK,),
        (fstar, feq, U, fib,
         cp.float64(s_plus), cp.float64(s_minus), cp.float64(dt),
         cp.int32(N), f),
    )
    return f


def collision_cm_mrt_gpu(fstar, feq, U, fib, S_diag, dt):
    """Central-Moment MRT 충돌 + Guo forcing GPU 래퍼.

    대응
        lbm/collision/cm_mrt.py::_collision_cm_mrt_nb

    수식
        k_{pq}      = Σ_i f_i c_ix^p c_iy^q      (c_i = e_i − u)
        k_{pq}^post = k_{pq} − s_{pq}(k_{pq} − k_{pq}^eq)
                    + Δt (1 − s_{pq}/2) F_{pq}
        k^eq 비영 항  : k_{0,0}=ρ, k_{2,0}=k_{0,2}=ρ c_s², k_{2,2}=ρ c_s⁴
                                                      De Rosis 2017/2019

    Args
        S_diag : (9,) numpy 배열.  s_nu=1/τ 점성 완화율을 포함
                  커널 내부는 D_E/D_W 상수 참조 → e/w 인자 불필요

    Launch
        1 thread per node   →   _grid(N)
    """
    N = fstar.shape[0]
    f = cp.empty_like(fstar)
    S_diag_gpu = cp.asarray(S_diag, dtype=cp.float64)
    _collision_cm_mrt_kern(
        _grid(N), (_BLOCK,),
        (fstar, feq, U, fib, S_diag_gpu,
         cp.float64(dt), cp.int32(N), f),
    )
    return f


def streaming_gpu(fstar_old, f, nx, ny):
    """D2Q9 pull-based 스트리밍 GPU 래퍼.

    대응
        lbm/streaming.py::_streaming_step_nb (CPU는 push-based)

    수식 (공통)
        f_i(x + e_i·Δt, t + Δt) = f_i*(x, t)

    GPU(pull) vs CPU(push) 인덱스 시프트

        CPU push    src[j, i]           → dst[j + e_y, i + e_x]
        GPU pull    dst[j, i]           ← src[j − e_y, i − e_x]

        pull 방식에서는 각 셀이 이웃에서 자기 쪽으로 들어오는 f_i 를 가져옴
        → `fstar_out[j, i, d] ← f[j − e_y[d], i − e_x[d], d]`

    비주기 경계
        이웃이 격자 밖이면 `fstar_old` (이전 값) 보존. 이후 BC 모듈이 덮어씀

    Launch
        1 thread per node   →   _grid(nx · ny)
    """
    N = nx * ny
    fstar_out = cp.empty_like(fstar_old)
    _streaming_kern(
        _grid(N), (_BLOCK,),
        (f, fstar_old, cp.int32(nx), cp.int32(ny), fstar_out),
    )
    return fstar_out


def macroscopic_gpu(fstar, incompressible=False):
    """거시변수 복원 GPU 래퍼.

    대응
        lbm/macroscopic.py::_macroscopic_nb

    수식
        표준(compressible)   ρ = Σ_i f_i,   u = (Σ_i f_i e_i) / ρ
        incompressible       ρ → p̂ 압력 surrogate, u = Σ_i f_i e_i

    Launch
        1 thread per node   →   _grid(N)

    Returns
        (ro, U)
          - ro : (N,)    ρ 또는 p̂
          - U  : (N, 2)  u
    """
    N = fstar.shape[0]
    ro = cp.empty(N, dtype=cp.float64)
    U = cp.empty((N, 2), dtype=cp.float64)
    _macroscopic_kern(
        _grid(N), (_BLOCK,),
        (fstar, cp.int32(N), cp.int32(int(incompressible)), ro, U),
    )
    return ro, U


def feq_gpu(ro, U, incompressible=False):
    """평형 분포 f_i^eq GPU 래퍼.

    대응
        lbm/equilibrium.py::_compute_feq_nb

    수식
        표준(compressible)
            f_i^eq = w_i ρ [1 + 3(e_i·u) + 9/2(e_i·u)² − 3/2 u²]
        incompressible LBGK (He & Luo 1997 계보)
            f_i^eq = w_i [ρ + 3(e_i·u) + 9/2(e_i·u)² − 3/2 u²]

    Launch
        1 thread per (n, i)   →   _grid(N · 9)
    """
    N = ro.shape[0]
    feq = cp.empty((N, 9), dtype=cp.float64)
    _feq_kern(
        _grid(N * 9), (_BLOCK,),
        (ro, U, cp.int32(N), cp.int32(int(incompressible)), feq),
    )
    return feq


def velocity_correction_gpu(U, fib, ro, dt, incompressible=False):
    """Guo forcing 2-moment 일관성 속도 보정 GPU 래퍼 (in-place).

    수식
        표준(compressible)   u ← u + f_ib · Δt / (2ρ)
        incompressible       u ← u + 0.5 · f_ib · Δt
        Guo 2002 2차 모멘트 보정 (solver 경로에서 collision 전에 호출)

    Launch
        1 thread per node   →   _grid(N)

    Side effect
        `U` in-place 업데이트 (반환값 없음)
    """
    N = U.shape[0]
    _vel_correction_kern(
        _grid(N), (_BLOCK,),
        (U, fib, ro, cp.float64(dt), cp.int32(N), cp.int32(int(incompressible))),
    )


# delta_type → (id, stencil_a) 매핑. ibm/common.py 와 동일 분류
_DELTA_TYPE_IDS = {"peskin4pt": 0, "hat": 1}
_DELTA_STENCIL_A = {"peskin4pt": 2, "hat": 1}


def ibm_direct_forcing_gpu(Lx, Ly, desired_vel, Eux, Euy, Ro,
                            dx, dy, dt, Larea, ny, nx,
                            delta_type="peskin4pt",
                            return_interp_vel: bool = False,
                            incompressible: bool = False):
    """IBM Direct Forcing (3-단계) GPU 래퍼.

    대응
        ibm/df.py::ibm_direct_forcing (CPU 경로)

    파이프라인  — Uhlmann 2005, Eq. (9a)-(9b)  /  중간 힘 Eq. (5)
      (a) 보간  Ũ, ρ̃ = Σ_x {u, ρ}(x) δ_h(x − X_l)
      (b) 힘    F_l = 2 ρ̃ (U_d − Ũ) / Δt              compressible
                  = 2     (U_d − Ũ) / Δt              incompressible
      (c) 분산  f_ib(x) = Σ_l F_l δ_h(x − X_l) Δs

    계보 / 분기
      - 밀도 가중          Wang, Fan, Luo (2008) Eq. (18) 계열
      - 비압축 분기        Majumder et al. (2023) Eq. (9)
      - LBM 힘 결합        Guo, Zheng, Shi (2002) Eq. (20)  (collision에서)

    CUDA 커널 구성  (2-launch)
        (a)+(b)  `_ibm_interp_force_kern`   →   Lfx, Lfy, Lux, Luy
        (c)      `_ibm_spread_kern`         →   Efx, Efy  (atomicAdd)

    Args
        Lx, Ly              : (Lb,)      Lagrangian marker 좌표
        desired_vel         : (Lb, 2)    목표 속도 U_d
        Eux, Euy, Ro        : (ny, nx)   Eulerian 속도·밀도 (C-contiguous 보정)
        dx, dy, dt          : 격자·시간 간격
        Larea               : marker arc length Δs
        ny, nx              : 격자 크기
        delta_type          : `"peskin4pt"` | `"hat"`
        return_interp_vel   : True → (fib, Lux, Luy) 반환. MDF velocity
                              residual 수렴 판정용
        incompressible      : True → Majumder Eq. (9) 분기 (ρ-가중 없음)

    Returns
        fib       : (nodenums, 2)  IB 체적력 f_ib
        Lux, Luy  : (Lb,)          보간 속도 Ũ — `return_interp_vel=True` 일 때만
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]

    # CUDA 커널은 C-contiguous 메모리 레이아웃 전제
    Eux = cp.ascontiguousarray(Eux)
    Euy = cp.ascontiguousarray(Euy)
    Ro = cp.ascontiguousarray(Ro)
    desired_vel = cp.ascontiguousarray(desired_vel)
    Lb = len(Lx)
    nodenums = ny * nx

    # --- (a)+(b) 보간 + 힘 계산 (1 kernel launch) ---
    Lfx = cp.empty(Lb, dtype=cp.float64)
    Lfy = cp.empty(Lb, dtype=cp.float64)
    Lux = cp.empty(Lb, dtype=cp.float64)
    Luy = cp.empty(Lb, dtype=cp.float64)
    _ibm_interp_force_kern(
        _grid(Lb), (_BLOCK,),
        (Lx, Ly, desired_vel, Eux, Euy, Ro,
         cp.float64(dx), cp.float64(dy), cp.float64(dt),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         cp.int32(int(incompressible)),
         Lfx, Lfy, Lux, Luy),
    )

    # --- (c) 분산 (1 kernel launch, atomicAdd 누적) ---
    Efx = cp.zeros((ny, nx), dtype=cp.float64)
    Efy = cp.zeros((ny, nx), dtype=cp.float64)
    _ibm_spread_kern(
        _grid(Lb), (_BLOCK,),
        (Lfx, Lfy, Lx, Ly,
         cp.float64(dx), cp.float64(dy), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Efx, Efy),
    )

    # fib (nodenums, 2) 패킹
    fib = cp.empty((nodenums, 2), dtype=cp.float64)
    fib[:, 0] = Efx.ravel()
    fib[:, 1] = Efy.ravel()

    if return_interp_vel:
        return fib, Lux, Luy
    return fib


def ibm_direct_forcing_fields_gpu(Lx, Ly, desired_vel, Eux, Euy, Ro,
                                  dx, dy, dt, Larea, ny, nx,
                                  delta_type="peskin4pt",
                                  incompressible: bool = False):
    """MDF 전용 GPU 경로: 패킹 전 Eulerian force field를 반환.

    `ibm_direct_forcing_gpu`와 같은 두 커널을 쓰되, `(nodenums, 2)` fib 패킹을
    생략한다. MDF는 반복 중 x/y force field를 다시 분해해서 쓰므로 마지막
    누적 결과에서만 패킹하는 편이 빠르다.
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]

    Eux = cp.ascontiguousarray(Eux)
    Euy = cp.ascontiguousarray(Euy)
    Ro = cp.ascontiguousarray(Ro)
    desired_vel = cp.ascontiguousarray(desired_vel)
    Lb = len(Lx)

    Lfx = cp.empty(Lb, dtype=cp.float64)
    Lfy = cp.empty(Lb, dtype=cp.float64)
    Lux = cp.empty(Lb, dtype=cp.float64)
    Luy = cp.empty(Lb, dtype=cp.float64)
    _ibm_interp_force_kern(
        _grid(Lb), (_BLOCK,),
        (Lx, Ly, desired_vel, Eux, Euy, Ro,
         cp.float64(dx), cp.float64(dy), cp.float64(dt),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         cp.int32(int(incompressible)),
         Lfx, Lfy, Lux, Luy),
    )

    Efx = cp.zeros((ny, nx), dtype=cp.float64)
    Efy = cp.zeros((ny, nx), dtype=cp.float64)
    _ibm_spread_kern(
        _grid(Lb), (_BLOCK,),
        (Lfx, Lfy, Lx, Ly,
         cp.float64(dx), cp.float64(dy), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Efx, Efy),
    )
    return Efx, Efy, Lux, Luy


def compute_lambda_gpu(Lx, Ly, rho_f, Larea, dx, dy, ny, nx, delta_type):
    """DFC adjustment parameter λ(k) GPU 래퍼.

    대응
        ibm/dfc.py::compute_lambda

    수식  — Tao 2019 Eq. (22)-(23)
        W_total(x) = Σ_{k'} W(x − X_{k'})                Step 1: spread
        W_sum(k)   = Σ_x W_total(x) · W(x − X_k)         Step 2: interpolate
        λ(k)       = 1 / [ 2 ρ_f Δs · W_sum(k) ]          Step 3: 해석해

    CUDA 커널 구성  (2-launch)
        Step 1   `_dfc_lambda_spread_kern`     atomicAdd 누적
        Step 2+3 `_dfc_lambda_interp_kern`     W_sum + λ 동시

    Args
        Lx, Ly      : (Lb,)  Lagrangian marker 좌표
        rho_f       : float  기준 밀도 (보통 1.0)
        Larea       : float  Δs (lattice units)
        dx, dy      : 격자 간격
        ny, nx      : 격자 크기
        delta_type  : `"peskin4pt"` | `"hat"`

    Returns
        lambda_k : (Lb,)   marker 별 λ(k)
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]
    Lb = len(Lx)

    # Step 1: spread — W_total(x) = Σ_{k'} W(x − X_{k'})
    W_total = cp.zeros((ny * nx,), dtype=cp.float64)
    _dfc_lambda_spread_kern(
        _grid(Lb), (_BLOCK,),
        (Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         W_total),
    )

    # Step 2+3: interpolate + λ 해석해
    lambda_k = cp.empty(Lb, dtype=cp.float64)
    _dfc_lambda_interp_kern(
        _grid(Lb), (_BLOCK,),
        (W_total, Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.float64(rho_f), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         lambda_k),
    )

    return lambda_k


def dfc_correction_gpu(Lx, Ly, desired_vel, fstar, lambda_k,
                        dx, dy, Larea, ny, nx,
                        delta_type, lattice):
    """DFC 분포함수 보정 GPU 래퍼 (non-iterative Tao 2019 경로).

    대응
        ibm/dfc.py::apply_dfc_correction 의 (a)+(b)+(d)+(e)+(g) 구간
        (c) λ(k) 계산은 `compute_lambda_gpu`, (f) 덧셈은 상위에서 수행

    수식  — Tao 2019 Eq. (15)-(25)
        f_i*(X_k)   = Σ_x f_i(x) W(x − X_k) dx²                 Eq. (15)
        f̄_i(X_k)    = f*_{opp(i)}(X_k) + 2 w_i ρ_f (e_i·u_w)/c_s²  Eq. (16)
        Δf_i(X_k)   = λ(k) · [f̄_i(X_k) − f_i*(X_k)]             Eq. (17)
        Δf_i(x)     = Σ_k Δf_i(X_k) W(x − X_k) Δs               Eq. (18)
        q_k         = −Δs · Σ_i e_i · λ(k) · [f̄_i − f_i*]        Eq. (25) 분포부

    CUDA 커널 구성  (3-launch)
        (a)       `_dfc_interp_kern`         f_interp
        (b)+(d)+(g) `_dfc_bb_lambda_kern`    BB + λ·편차 + 힘 집계 (융합)
        (e)       `_dfc_spread_kern`         atomicAdd 누적

    Args
        Lx, Ly       : (Lb,)       Lagrangian marker 좌표
        desired_vel  : (Lb, 2)     경계 desired velocity U_d
        fstar        : (ny·nx, 9)  post-streaming 분포함수 (C-contiguous 보정)
        lambda_k     : (Lb,)       `compute_lambda_gpu` 결과 (또는 cache)
        dx, dy       : 격자 간격
        Larea        : Δs (lattice units)
        ny, nx       : 격자 크기
        delta_type   : `"peskin4pt"` | `"hat"`
        lattice      : D2Q9 (시그니처 호환 인자. opp/e/w 는 CUDA 상수 사용)

    Returns
        Ef_corr    : (ny·nx, 9)  Eulerian 분포함수 보정 Δf_i(x)
        dfc_force  : (Lb, 2)     Eq. (25) 분포 기여부 마커별 집계 q_k
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]
    Lb = len(Lx)

    fstar = cp.ascontiguousarray(fstar)
    desired_vel = cp.ascontiguousarray(desired_vel)

    # --- (a) f_i 보간 (Eq. (15)) ---
    f_interp = cp.empty((Lb, 9), dtype=cp.float64)
    _dfc_interp_kern(
        _grid(Lb), (_BLOCK,),
        (fstar, Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         f_interp),
    )

    # --- (b)+(d)+(g) BB + λ·편차 + 마커별 힘 (Eq. (16), (17), (25) 분포부) ---
    delta_f = cp.empty((Lb, 9), dtype=cp.float64)
    dfc_force = cp.empty((Lb, 2), dtype=cp.float64)
    _dfc_bb_lambda_kern(
        _grid(Lb), (_BLOCK,),
        (f_interp, desired_vel, lambda_k,
         cp.float64(1.0), cp.float64(Larea),
         cp.int32(Lb),
         delta_f, dfc_force),
    )

    # --- (e) Δf_i 분산 (Eq. (18), atomicAdd) ---
    Ef_corr = cp.zeros((ny * nx, 9), dtype=cp.float64)
    _dfc_spread_kern(
        _grid(Lb), (_BLOCK,),
        (delta_f, Lx, Ly,
         cp.float64(dx), cp.float64(dy), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Ef_corr),
    )

    return Ef_corr, dfc_force
