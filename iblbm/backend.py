"""GPU/CPU 배열 백엔드 선택.

분기 원칙
  - 기본값 GPU (CuPy). `IBLBM_GPU=0` 환경변수로만 CPU 모드 허용
  - GPU 요청 상태에서 CuPy 미설치/CUDA device 부재 → 즉시 fail-fast (조용한 CPU fallback 금지)
  - 다운스트림 모듈은 `from ..backend import xp as np` 형태로 획득

공개 심볼
  - `xp`         numpy 또는 cupy (선택된 배열 라이브러리)
  - `_use_gpu`   bool, True ↔ GPU path 활성
  - `add_at`     중복 인덱스 누적 래퍼 (`xp.add.at` 위임)
"""

import os

_gpu_requested = os.environ.get("IBLBM_GPU", "1") != "0"

if _gpu_requested:
    try:
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("사용 가능한 CUDA device 없음")
    except ImportError as exc:
        raise RuntimeError(
            "GPU 기본 모드에서 CuPy import 실패. "
            "CuPy 설치 또는 IBLBM_GPU=0 으로 CPU 모드 명시 필요"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "GPU 기본 모드 요청됐으나 CUDA runtime/device 사용 불가. "
            "GPU 환경 점검 또는 IBLBM_GPU=0 으로 CPU 모드 명시 필요"
        ) from exc
    _use_gpu = True
else:
    import numpy as xp
    _use_gpu = False


def add_at(target, indices, values):
    """중복 인덱스 누적 래퍼 (`np.add.at` / `cupy.add.at`).

    - NumPy/CuPy 공통 API: `xp.add.at(target, indices, values)`
    - CuPy 14+ 는 `cupy.add.at` 공식 지원 (`cupyx.scatter_add` deprecated)

    용도
      - `ibm.py` Lagrangian → Eulerian 힘 분산
      - 중복 인덱스에서 일반 `+=` 는 마지막 값만 반영, 본 래퍼는 모든 값 누적
    """
    xp.add.at(target, indices, values)
