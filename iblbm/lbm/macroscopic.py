"""D2Q9 거시 변수 복원: 0차·1차 속도 모멘트.

표준(compressible)
    ρ   = Σ_i f_i
    ρu  = Σ_i f_i e_i   →   u = (Σ_i f_i e_i) / ρ

Incompressible LBGK 분기 (`incompressible=True`)
    p̂  = Σ_i f_i                (압력 surrogate, `ro` 반환)
    u   = Σ_i f_i e_i            (나눗셈 없이 속도 그대로 반환)

  - He & Luo (1997) 계보의 비압축 변형
  - 이 경로에서 `ro` 는 물리 밀도가 아니라 압력 surrogate 수치

백엔드 분기
  - GPU: `gpu_kernels.macroscopic_gpu`
  - CPU: numba prange 커널 `_macroscopic_nb`
"""

from __future__ import annotations

from ..backend import xp as np, _use_gpu
from .lattice import D2Q9

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _macroscopic_nb(fstar, e, incompressible):
        """ρ = Σ_i f_i, u = (Σ_i f_i e_i)/ρ 계산. incompressible=True 시 나눗셈 생략."""

        N = fstar.shape[0]
        ro = np.empty(N)
        U = np.empty((N, 2))
        for n in prange(N):
            s = 0.0
            ux = 0.0
            uy = 0.0
            for i in range(9):
                s += fstar[n, i]
                ux += fstar[n, i] * e[i, 0]
                uy += fstar[n, i] * e[i, 1]
            ro[n] = s
            if incompressible:
                U[n, 0] = ux
                U[n, 1] = uy
            else:
                U[n, 0] = ux / s
                U[n, 1] = uy / s
        return ro, U


def macroscopic(fstar, lattice: D2Q9, incompressible: bool = False):
    """ρ, u 복원 디스패처. GPU/CPU 백엔드 자동 선택."""

    if _use_gpu:
        from ..gpu_kernels import macroscopic_gpu

        return macroscopic_gpu(fstar, incompressible=incompressible)
    return _macroscopic_nb(fstar, lattice.e, incompressible)
