"""LBM 코어 서브패키지 re-export.

  - 격자: D2Q9, make_d2q9, make_d2q9_mrt_matrices
  - 단계: collision_step, streaming_step
  - 복원: macroscopic, compute_feq
  - 강제항: guo_forcing (진단용 헬퍼)
"""

from .lattice import D2Q9, make_d2q9, make_d2q9_mrt_matrices
from .collision import collision_step, guo_forcing
from .macroscopic import macroscopic
from .streaming import streaming_step
from .equilibrium import compute_feq

__all__ = [
    "D2Q9", "make_d2q9", "make_d2q9_mrt_matrices",
    "collision_step", "guo_forcing",
    "macroscopic",
    "streaming_step",
    "compute_feq",
]
