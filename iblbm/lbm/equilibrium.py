"""D2Q9 평형 분포 f_i^eq 계산.

표준(compressible) 형태
    f_i^eq = w_i ρ [ 1 + (e_i·u)/c_s² + (e_i·u)²/(2c_s⁴) − u²/(2c_s²) ]

  c_s² = 1/3, c = 1 대입 결과
    f_i^eq = w_i ρ [ 1 + 3(e_i·u) + 9/2 (e_i·u)² − 3/2 u² ]

Incompressible LBGK 분기 (`incompressible=True`)
    f_i^eq = w_i [ ρ + 3(e_i·u) + 9/2 (e_i·u)² − 3/2 u² ]

  - 밀도 변동 O(M²) 항 제거, ρ → 압력 surrogate
  - He & Luo (1997) 계보
  - 이 경로에서 `ro`는 물리 밀도가 아니라 압력 surrogate 수치

백엔드 분기
  - GPU: `gpu_kernels.feq_gpu` 벡터화 구현
  - CPU: numba prange 커널 `_compute_feq_nb`
"""

from ..backend import xp as np, _use_gpu
from .lattice import D2Q9

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _compute_feq_nb(ro, U, e, w, incompressible):
        """D2Q9 f_i^eq numba 커널, prange 병렬."""
        N = ro.shape[0]
        feq = np.empty((N, 9))
        for n in prange(N):
            u2 = U[n, 0] ** 2 + U[n, 1] ** 2
            for i in range(9):
                eU = e[i, 0] * U[n, 0] + e[i, 1] * U[n, 1]
                if incompressible:
                    # He & Luo (1997) 계보: f_i^eq = w_i [ρ + 3(e_i·u) + 9/2(e_i·u)² − 3/2 u²]
                    feq[n, i] = w[i] * (ro[n] + 3.0 * eU + 4.5 * eU * eU - 1.5 * u2)
                else:
                    # 표준 D2Q9: f_i^eq = w_i ρ [1 + 3(e_i·u) + 9/2(e_i·u)² − 3/2 u²]
                    feq[n, i] = ro[n] * w[i] * (1.0 + 3.0 * eU + 4.5 * eU * eU - 1.5 * u2)
        return feq


def compute_feq(ro, U, lattice, incompressible=False):
    """f_i^eq 계산 디스패처. GPU 경로는 벡터화, CPU 경로는 numba 커널."""
    if _use_gpu:
        from ..gpu_kernels import feq_gpu
        return feq_gpu(ro, U, incompressible=incompressible)
    return _compute_feq_nb(ro, U, lattice.e, lattice.w, incompressible)
