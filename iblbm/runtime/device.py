"""GPU↔CPU 배열 전송 경계 유틸.

사용 원칙
  - 핫 루프 내부 호출 금지 (매 스텝 device↔host 전송은 성능 저해)
  - 진단/저장/로깅 직전에만 호출 (convergence check, history dump 등)
  - CPU 모드일 때는 no-op (원본 배열 그대로 반환)
"""

from __future__ import annotations

from ..backend import _use_gpu


def to_cpu(arr):
    """CuPy array → NumPy array. CPU 모드/None 은 그대로 반환."""
    if arr is None or not _use_gpu:
        return arr
    if hasattr(arr, "get"):
        return arr.get()
    return arr
