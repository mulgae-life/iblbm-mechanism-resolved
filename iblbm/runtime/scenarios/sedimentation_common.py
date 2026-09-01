"""침강 runtime 공통 helper (단일/다입자 공용).

제공 함수
  - `sedimentation_stop_reason`     contact/offset 기반 정지 판정
  - `sedimentation_bound_warning`   도메인 벽 근접 안전 마진 감시

gravity_direction 분기
  - `"down"`   기본 (-y 방향 가속). bottom wall 접촉 판정 `y ≤ r`
  - `"right"`  (+x 방향 가속, Uhlmann 2-particle 등). right wall 접촉 판정 `x ≥ xmax − r`
"""
from __future__ import annotations


def sedimentation_stop_reason(pos, r_domain, cfg):
    """접촉/offset 기반 종료 사유 문자열 반환 (없으면 None).

    - `sedimentation_stop_at_contact=True`       입자 접촉 (벽면 도달) 시 `"contact"`
    - `sedimentation_stop_offset_d > 0.0`        offset·D 거리 여유분에서 정지 → `"offset"`
    """
    if cfg.sedimentation_stop_at_contact:
        if cfg.gravity_direction == "right":
            if float(pos[0]) >= (cfg.xmax - r_domain):
                return "contact"
        elif float(pos[1]) <= r_domain:
            return "contact"
    if cfg.sedimentation_stop_offset_d > 0.0:
        d_domain = cfg.cylinder_D_ratio
        if cfg.gravity_direction == "right":
            if float(pos[0]) >= (cfg.xmax - cfg.sedimentation_stop_offset_d * d_domain):
                return "offset"
        elif float(pos[1]) <= (cfg.sedimentation_stop_offset_d * d_domain):
            return "offset"
    return None


def sedimentation_bound_warning(pos, r_domain, cfg, dx, safety: float = 2.0):
    """도메인 벽 `safety · dx` margin 침범 시 한국어 경고 문자열 반환.

    - 정지 조건(contact/offset) 쪽 벽은 감시 제외 (의도된 접촉)
    - 그 외 네 벽에서 근접 감지 시 경고 메시지, 근접이 없으면 None 반환
    """
    margin = safety * dx
    x = float(pos[0])
    y = float(pos[1])
    stop_on_terminal_wall = cfg.sedimentation_stop_at_contact or cfg.sedimentation_stop_offset_d > 0.0

    if cfg.gravity_direction == "right":
        if y - r_domain < margin:
            return f"입자 하단이 하벽에 근접 (y-r={y-r_domain:.4f} < margin={margin:.4f})"
        if y + r_domain > cfg.ymax - margin:
            return f"입자 상단이 상벽에 근접 (y+r={y+r_domain:.4f} > ymax-margin={cfg.ymax-margin:.4f})"
        if x - r_domain < margin:
            return "입자 좌측이 좌벽에 근접"
        if not stop_on_terminal_wall and x + r_domain > cfg.xmax - margin:
            return "입자 우측이 우벽에 근접"
        return None

    if not stop_on_terminal_wall and y - r_domain < margin:
        return f"입자 하단이 하벽에 근접 (y-r={y-r_domain:.4f} < margin={margin:.4f})"
    if y + r_domain > cfg.ymax - margin:
        return f"입자 상단이 상벽에 근접 (y+r={y+r_domain:.4f} > ymax-margin={cfg.ymax-margin:.4f})"
    if x - r_domain < margin:
        return "입자 좌측이 좌벽에 근접"
    if x + r_domain > cfg.xmax - margin:
        return "입자 우측이 우벽에 근접"
    return None
