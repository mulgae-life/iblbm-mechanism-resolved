"""Guo 강제항 벡터화 헬퍼 (진단/검증용).

수식 (Guo et al. 2002, Eq. (20))
    F_i = (1 − 1/(2τ)) w_i [ 3(e_i − u) + 9(e_i·u) e_i ] · f_ib

  - prefactor `(1 − 1/(2τ))` : discrete lattice 보정
  - 실제 충돌 경로(BGK/TRT/CM-MRT)는 각 커널 내부에서 동일 수식을 직접 전개
  - 본 헬퍼는 강제항 만을 pure-numpy 로 계산하기 위한 우회 경로
"""
from __future__ import annotations

from ...backend import xp as np


def guo_forcing(U, fib, tau, lattice):
    """F_i = (1 − 1/(2τ)) w_i [3(e_i − u) + 9(e_i·u) e_i] · f_ib 의 (N, 9) 배열 반환."""
    e = lattice.e
    w = lattice.w
    a = e[None, :, :] - U[:, None, :]
    eU = np.sum(e[None, :, :] * U[:, None, :], axis=2)
    b = eU[:, :, None] * e[None, :, :]
    term = np.sum((3 * a + 9 * b) * fib[:, None, :], axis=2)
    return (1.0 - 0.5 / tau) * (w[None, :] * term)
