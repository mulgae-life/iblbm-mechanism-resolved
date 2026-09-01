"""D2Q9 push-streaming, 인덱스 시프트 방식.

알고리즘
    f_i(x + e_i·Δt, t + Δt) = f_i*(x, t)        ∀ i = 0..8

방향별 시프트 (dst ← src)

        6   2   5                       (j+1)
         ╲  │  ╱
      3 ─── 0 ─── 1                      (j)
         ╱  │  ╲
        7   4   8                       (j−1)

  - i = 0 정지       dst[j, i]       ← src[j, i]
  - i = 1 우         dst[j, 1:]      ← src[j, :-1]
  - i = 2 상         dst[1:, i]      ← src[:-1, i]
  - i = 3 좌         dst[j, :-1]     ← src[j, 1:]
  - i = 4 하         dst[:-1, i]     ← src[1:, i]
  - i = 5 우상 (↗)   dst[1:, 1:]     ← src[:-1, :-1]
  - i = 6 좌상 (↖)   dst[1:, :-1]    ← src[:-1, 1:]
  - i = 7 좌하 (↙)   dst[:-1, :-1]   ← src[1:, 1:]
  - i = 8 우하 (↘)   dst[:-1, 1:]    ← src[1:, :-1]

경계 처리
  - 비주기(non-periodic). 경계 행/열은 이전 `fstar` 값 보존
  - 이후 BC 모듈이 필요한 i만 덮어씀

배열 레이아웃
    fstar shape (N, 9) with N = ny·nx
    fstar[:, i].reshape(ny, nx) → (row=y=j, col=x=i)
"""

from ..backend import _use_gpu

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _streaming_step_nb(fstar, f, nx, ny):
        """D2Q9 push-streaming numba 커널. 방향별 인덱스 산술로 prange 병렬화."""
        fstar_new = fstar.copy()
        # i=0: 정지 — fstar_new = f
        for n in prange(ny * nx):
            fstar_new[n, 0] = f[n, 0]
        # i=1: 우 — dst[:, 1:nx] = src[:, 0:nx-1]
        for j in prange(ny):
            for i in range(1, nx):
                fstar_new[j * nx + i, 1] = f[j * nx + (i - 1), 1]
        # i=2: 상 — dst[1:ny, :] = src[0:ny-1, :]
        for j in prange(1, ny):
            for i in range(nx):
                fstar_new[j * nx + i, 2] = f[(j - 1) * nx + i, 2]
        # i=3: 좌 — dst[:, 0:nx-1] = src[:, 1:nx]
        for j in prange(ny):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 3] = f[j * nx + (i + 1), 3]
        # i=4: 하 — dst[0:ny-1, :] = src[1:ny, :]
        for j in prange(ny - 1):
            for i in range(nx):
                fstar_new[j * nx + i, 4] = f[(j + 1) * nx + i, 4]
        # i=5: 우상 — dst[1:,1:] = src[:ny-1,:nx-1]
        for j in prange(1, ny):
            for i in range(1, nx):
                fstar_new[j * nx + i, 5] = f[(j - 1) * nx + (i - 1), 5]
        # i=6: 좌상 — dst[1:,:nx-1] = src[:ny-1,1:]
        for j in prange(1, ny):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 6] = f[(j - 1) * nx + (i + 1), 6]
        # i=7: 좌하 — dst[:ny-1,:nx-1] = src[1:,1:]
        for j in prange(ny - 1):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 7] = f[(j + 1) * nx + (i + 1), 7]
        # i=8: 우하 — dst[:ny-1,1:] = src[1:,:nx-1]
        for j in prange(ny - 1):
            for i in range(1, nx):
                fstar_new[j * nx + i, 8] = f[(j + 1) * nx + (i - 1), 8]
        return fstar_new


def streaming_step(fstar, f, nx, ny):
    """D2Q9 push-streaming 디스패처. GPU/CPU 백엔드 자동 선택."""
    if _use_gpu:
        from ..gpu_kernels import streaming_gpu
        return streaming_gpu(fstar, f, nx, ny)
    return _streaming_step_nb(fstar, f, nx, ny)
