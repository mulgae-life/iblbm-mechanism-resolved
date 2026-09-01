"""Face별 경계 노드 1D 인덱스 빌더.

배열 레이아웃
  - f, ro, U가 `(ny·nx, ...)` 평면 저장 (row-major, row=y, col=x)
  - 노드 n = j·nx + i         (j: y-row, i: x-col)

Face 인덱싱 규칙
  - `open_boundary / settling_channel / dirichlet_neumann`
      · corner(4개)를 별도 처리하는 BC → left/right는 corner 제외
          left    = {n | i = 0,       j ∈ [1, ny-2]}
          right   = {n | i = nx-1,    j ∈ [1, ny-2]}
  - 그 외 (velocity_inlet, tg_analytical 등)
      · left/right가 corner 포함
          left    = {n | i = 0,       j ∈ [0, ny-1]}
          right   = {n | i = nx-1,    j ∈ [0, ny-1]}
  - bottom, top은 공통
      bottom  = {n | j = 0,     i ∈ [1, nx-2]}
      top     = {n | j = ny-1,  i ∈ [1, nx-2]}

        (0, ny-1) ┌─── top ────┐ (nx-1, ny-1)
                  │            │
                  left        right
                  │            │
           (0, 0) └── bottom ──┘ (nx-1, 0)
"""

from __future__ import annotations

from .base import BoundaryIndices
from ..backend import xp as np


def build_boundary_indices(cfg, nx: int, ny: int) -> BoundaryIndices:
    """Face별 평면 인덱스 배열 생성 (cfg.bc_type에 따라 corner 포함 규칙 분기)."""
    nodenums = nx * ny
    if cfg.bc_type in {
        "open_boundary",
        "settling_channel",
        "dirichlet_neumann",
    }:
        left = np.arange(nx, nodenums - nx, nx)
        right = np.arange(2 * nx - 1, nodenums - nx, nx)
        bottom = np.arange(1, nx - 1)
        top = np.arange(nx * (ny - 1) + 1, nodenums - 1)
    else:
        left = np.arange(0, nodenums, nx)
        right = np.arange(nx - 1, nodenums, nx)
        bottom = np.arange(1, nx - 1)
        top = np.arange(nx * (ny - 1) + 1, nodenums - 1)
    return BoundaryIndices(left=left, right=right, top=top, bottom=bottom, nx=nx, ny=ny)
