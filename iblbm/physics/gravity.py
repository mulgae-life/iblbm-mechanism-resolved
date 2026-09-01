"""침강 순중력(buoyancy-corrected gravity) helper.

핵심 식 (2D 원판, 격자 단위)
    F_g = (ρ_s − ρ_f) · π r² · g
        = (ρ_s/ρ_f − 1) · π r_lattice² · g_lattice    (격자 단위 환산)

방향 규약
    - `gravity_direction = "down"`  → (0, −|F_g|)    y⁻ 방향 (화면 아래)
    - `gravity_direction = "right"` → (+|F_g|, 0)    x⁺ 방향
"""
from __future__ import annotations

import numpy as np


def compute_net_gravity(
    rho_ratio: float,
    r_lattice: float,
    g_lattice: float,
    gravity_direction: str = "down",
    displaced_area_lattice: float | None = None,
) -> np.ndarray:
    """순중력 벡터 `F_g = (ρ_s − ρ_f) π r² g` 계산.

    Returns
      `np.ndarray` shape (2,) — 방향 규약은 모듈 헤더 참조
    """
    area_lattice = displaced_area_lattice
    if area_lattice is None:
        area_lattice = np.pi * r_lattice**2
    mag = (rho_ratio - 1.0) * area_lattice * g_lattice
    if gravity_direction == "right":
        return np.array([mag, 0.0])
    return np.array([0.0, -mag])
