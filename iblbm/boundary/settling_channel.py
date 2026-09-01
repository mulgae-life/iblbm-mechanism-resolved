"""4-wall no-slip 침강 채널 경계 (모든 face에서 u = 0).

Face 구성
  - 모든 face에서 prescribed u = 0
  - ρ는 known population 합으로 복원 (Zou-He Eq. (17)에서 u = 0 환원)
  - 미지 population은 bounceback + 대각항 차이로 복원

D2Q9 방향 + face별 미지 population (u = 0 대입)
  - left face
        unknown = {f_1, f_5, f_8}
        f_1 = f_3
        f_5 = f_7 − ½ (f_2 − f_4)
        f_8 = f_6 + ½ (f_2 − f_4)
  - right face
        unknown = {f_3, f_6, f_7}
        f_3 = f_1
        f_6 = f_8 − ½ (f_2 − f_4)
        f_7 = f_5 + ½ (f_2 − f_4)
  - top face
        unknown = {f_4, f_7, f_8}
        f_4 = f_2
        f_7 = f_5 + ½ (f_1 − f_3)
        f_8 = f_6 − ½ (f_1 − f_3)
  - bottom face
        unknown = {f_2, f_5, f_6}
        f_2 = f_4
        f_5 = f_7 − ½ (f_1 − f_3)
        f_6 = f_8 + ½ (f_1 − f_3)

계보
  - 직선 face no-slip closure: Zou & He (1997) §3 Eq. (17-18)에서 u = 0 환원
  - 4-wall 동시 조합 + corner equilibrium refill은 프로젝트 구현
"""

from __future__ import annotations

from ..backend import xp as np
from ..lbm import compute_feq
from .base import BoundaryIndices, register_boundary


def apply_bc_settling_channel(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    dens: float, idx: BoundaryIndices,
    lattice=None,
    incompressible: bool = False,
) -> None:
    """4-wall no-slip 경계 적용 (u = 0 전 face).

    단계
      1. face별 u ← 0
      2. ρ ← Σ_known (Zou-He Eq. (17)에서 u = 0 환원)
      3. 미지 population ← bounceback + 대각항 차이 (위 파일 헤더 도식)
      4. (lattice != None) corner 4개에 ρ = dens + u = 0 + `compute_feq` refill
            · `incompressible` 플래그는 feq의 incompressible LBGK 여부 결정

    참조
      - Zou & He (1997) §3 no-slip face
      - 4-wall 조합 + corner refill은 프로젝트 구현
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    U[left, 0] = 0.0
    U[left, 1] = 0.0
    ro[left] = (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 6] + fstar[left, 7])
    )
    fstar[left, 1] = fstar[left, 3]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
    )

    U[right, 0] = 0.0
    U[right, 1] = 0.0
    ro[right] = (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    )
    fstar[right, 3] = fstar[right, 1]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
    )

    U[top, 0] = 0.0
    U[top, 1] = 0.0
    ro[top] = (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
    )

    U[bottom, 0] = 0.0
    U[bottom, 1] = 0.0
    ro[bottom] = (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
    )

    if lattice is not None:
        ny = idx.ny
        corners = np.array([
            0,
            nx - 1,
            nx * (ny - 1),
            nx * ny - 1,
        ])
        ro[corners] = dens
        U[corners, 0] = 0.0
        U[corners, 1] = 0.0
        fstar[corners] = compute_feq(
            ro[corners], U[corners], lattice, incompressible=incompressible,
        )


class SettlingChannelBC:
    name = "settling_channel"

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        apply_bc_settling_channel(
            fstar, ro, U, state.dens, state.idx, state.lattice,
            incompressible=cfg.incompressible_lbgk,
        )


register_boundary(SettlingChannelBC())
