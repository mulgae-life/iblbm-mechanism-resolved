"""시나리오별 실린더 운동 업데이트 helper.

진동 (`update_oscillating`) : `x(t) = A sin(2π f₀ t)`, `u_b = x(t)` (누적 displacement convention)

parameter
  - KC (Keulegan-Carpenter) : 진동 진폭 스케일링  A = KC · D / (2π)
  - f₀                       : 진동 주파수         f₀ = U / (KC · D)
"""

from __future__ import annotations

from ..backend import xp as np


def update_oscillating(
    t: int, Lx_c: np.ndarray, Ly_c: np.ndarray,
    desired_vel: np.ndarray,
    lattice_u: float, KC: float, r: float, lattice_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """진동 실린더 위치/속도 업데이트.

    식
      x(t) = A sin(2π f₀ t)           변위 (격자 단위)
      u_b  = x(t)                      benchmark convention: 누적 변위 기반 desired velocity
      A    = KC · D / (2π)
      f₀   = U / (KC · D),    D = 2 r_lattice

    Args
      t            : 현재 시간 스텝
      Lx_c, Ly_c   : 초기 라그랑주 점 좌표, (L_b,)
      desired_vel  : 목표 속도, (L_b, 2) — in-place 수정
      lattice_u    : 격자 단위 U
      KC           : Keulegan-Carpenter 수
      r            : 실린더 반지름 (도메인 비율)
      lattice_r    : 실린더 반지름 (격자 단위)

    Returns
      `(Lx, Ly, desired_vel)`
    """
    f0 = lattice_u / (KC * 2.0 * lattice_r)
    A = KC * 2.0 * r / (2.0 * np.pi)

    xt = A * np.sin(f0 * 2.0 * np.pi * t)

    Lx = Lx_c - xt
    Ly = Ly_c.copy()
    # 진동 실린더 benchmark는 누적 displacement 기반 desired velocity convention을
    # 사용해 Cd_peak/phase profile reference와 정합 유지
    desired_vel[:, 0] = xt
    desired_vel[:, 1] = 0.0

    return Lx, Ly, desired_vel

