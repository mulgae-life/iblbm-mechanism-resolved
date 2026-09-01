r"""A-1. Phase-resolved paired-difference 분해 + C.4 재계산 + S4.4 시계열 재생성
— locked confirmation.

A-1 규격의 공식 산출 스크립트다. 사건 시각과 C.4 비의 탐색 단계 값이 이미 공개된
상태이므로 blind 예측이 아니라 사전 확정 확인 재계산(locked confirmation)으로 분류하며,
실행 전 스크립트 해시로 고정한다.

규격 (사전 고정):
- 시간축: **heavy(ρ=1.5) 입자의 t\*** (per-particle t_star 중 particles[0]) — 탐색 계보와
  동일 규약. 사건의 물리 스텝 환산은 같은 보간에서 취한다.
- 최근접 시각: 중심 거리의 이산 최소 주변 **3점 포물선 보간**.
- 역할 교대 시각: 침강 방향 순서(x_light − x_heavy) 부호 교대의 **선형 보간**.
- |a\*| proxy (S4.4 규격): per-particle (vx\*, vy\*)의 **단측 나눗셈 차분**을 자기 t\* 축으로
  나눠 **구간 중점**에 배치, **무평활**. 각 입자 자신의 u_g²/D 정규화(축 자체가 별도).
- C.4 pair 내부 비: 각 구성의 자기 t\*_closest(물리 스텝 환산)로 pre/post 분할,
  각 위상의 |a\*| 피크로 light/heavy 비 산출, contrast = post/pre.
  * pre 최대는 step 0 레코드 부재의 **좌측 검열값**(저장된 최초 구간의 최대).
  * post 최대는 역할 교대보다 훨씬 뒤 — 피크 스텝을 함께 기록한다.
  * light/single 교차 비는 계산하지 않는다(교차 런 차분 기저 불일치).
- Δ(t) paired difference: 같은 입자의 vy\*를 eh − none으로 차분.
  * 위계: **event-aligned 정렬 primary**(각 팔 자신의 t\*_closest 기준 정렬),
    common-time secondary, global peak diagnostic.
  * 공통 스텝 격자(양팔 모두 500스텝 cadence)의 교집합에서 계산한다.
- S4.4: 공개 저장소용 |a\*(t)| 시계열을 동일 Verlet 런에서 재생성(pair 전용;
  single trace는 정성 참고선 한정이므로 여기서 생성하지 않음).

locked confirmation 대조 목표:
- 사건 시각(heavy t\* 축, 소수 6자리): Verlet eh 5.512074/5.912519 ·
  Verlet none 6.009970/6.517611 · Euler500 5.560135/5.975222 · Euler100 5.575521/5.980267
- C.4(소수 2자리): Verlet 정렬 1.11/2.81/2.54(주) · Euler500 1.11/2.89/2.60(병기) ·
  저장 간격 민감도 대역 [2.41, 2.60] (Euler100 contrast 2.41).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data/two_particle_sedimentation"
OUT_JSON = Path(__file__).with_name("a1_phase_result.json")
OUT_S44 = Path(__file__).with_name("a1_s44_astar_series.json")

RUNS = {
    "verlet_eh": DATA / "method_matrix/df_bgk_verlet_explicit_history",
    "verlet_none": DATA / "method_matrix/df_bgk_verlet_none",
    "euler500_eh": DATA / "baseline",
    "euler100_eh": DATA / "highlight_run",
}
LOCKED_EVENTS = {
    "verlet_eh": (5.512074, 5.912519),
    "verlet_none": (6.009970, 6.517611),
    "euler500_eh": (5.560135, 5.975222),
    "euler100_eh": (5.575521, 5.980267),
}
LOCKED_C4 = {  # (pre l/h, post l/h, contrast) — 소수 2자리
    "verlet_eh": (1.11, 2.81, 2.54),
    "euler500_eh": (1.11, 2.89, 2.60),
}
LOCKED_CADENCE_BAND = (2.41, 2.60)
EVENT_TOL = 1e-5


def load(run_dir: Path):
    with open(run_dir / "sedimentation_history.json") as f:
        return json.load(f)


def events(recs):
    """(t*_closest, t*_exchange, step_closest, step_exchange) — heavy t* 축."""
    steps = [r["step"] for r in recs]
    th = [r["particles"][0]["t_star"] for r in recs]
    d = [math.hypot(r["particles"][0]["x"] - r["particles"][1]["x"],
                    r["particles"][0]["y"] - r["particles"][1]["y"]) for r in recs]
    i = min(range(1, len(d) - 1), key=lambda k: d[k])
    denom = d[i - 1] - 2 * d[i] + d[i + 1]
    delta = 0.5 * (d[i - 1] - d[i + 1]) / denom
    t_closest = th[i] + delta * (th[i] - th[i - 1])
    s_closest = steps[i] + delta * (steps[i] - steps[i - 1])

    # 역할 교대: 침강 방향(+x) 순서 g = x_light − x_heavy 의 부호 교대(추월 시 앞뒤 교환).
    # 횡방향 y 순서는 전 구간 불변이므로 교대 신호는 x 교차뿐이다.
    g = [r["particles"][1]["x"] - r["particles"][0]["x"] for r in recs]
    j = next(k for k in range(1, len(g)) if g[k - 1] * g[k] <= 0 and g[k - 1] != g[k])
    frac = g[j - 1] / (g[j - 1] - g[j])
    t_exch = th[j - 1] + frac * (th[j] - th[j - 1])
    s_exch = steps[j - 1] + frac * (steps[j] - steps[j - 1])
    return t_closest, t_exch, s_closest, s_exch


def astar_series(recs, pi):
    """입자 pi의 |a*| 단측 차분(자기 t* 축), 구간 중점 배치. 반환 [(step_mid, t*_mid, a*)]."""
    out = []
    for k in range(1, len(recs)):
        p0, p1 = recs[k - 1]["particles"][pi], recs[k]["particles"][pi]
        dt = p1["t_star"] - p0["t_star"]
        a = math.hypot(p1["vx_star"] - p0["vx_star"], p1["vy_star"] - p0["vy_star"]) / dt
        out.append(((recs[k - 1]["step"] + recs[k]["step"]) / 2.0,
                    (p0["t_star"] + p1["t_star"]) / 2.0, a))
    return out


def phase_peaks(series, s_split):
    pre = [x for x in series if x[0] < s_split]
    post = [x for x in series if x[0] >= s_split]
    pk = lambda xs: max(xs, key=lambda x: x[2])
    return pk(pre), pk(post)


def main() -> int:
    mism = []
    histories = {name: load(p) for name, p in RUNS.items()}
    result = {"events": {}, "c4": {}, "paired_difference": {}}

    # 1) 사건 시각
    ev = {}
    for name, recs in histories.items():
        tc, te, sc, se = events(recs)
        ev[name] = {"t_closest": tc, "t_exchange": te, "step_closest": sc, "step_exchange": se}
        tgt = LOCKED_EVENTS[name]
        ok = abs(tc - tgt[0]) < EVENT_TOL and abs(te - tgt[1]) < EVENT_TOL
        if not ok:
            mism.append(("events", name, tgt, (tc, te)))
        print(f"[사건] {name}: 최근접 {tc:.6f} / 역할 교대 {te:.6f} "
              f"(목표 {tgt[0]}/{tgt[1]}) {'일치' if ok else '불일치'}")
    result["events"] = ev
    lag_c = ev["verlet_none"]["t_closest"] - ev["verlet_eh"]["t_closest"]
    lag_e = ev["verlet_none"]["t_exchange"] - ev["verlet_eh"]["t_exchange"]
    result["events"]["none_arm_lag"] = {"closest": lag_c, "exchange": lag_e}
    print(f"무보정 팔 사건 지연: 최근접 {lag_c:.3f} / 역할 교대 {lag_e:.3f}")

    # 2) C.4 pair 내부 비 (pre/post 분할 = 자기 t*_closest의 스텝 환산)
    contrasts = {}
    for name in ("verlet_eh", "euler500_eh", "euler100_eh"):
        recs = histories[name]
        s_split = ev[name]["step_closest"]
        (pre_h, post_h) = phase_peaks(astar_series(recs, 0), s_split)
        (pre_l, post_l) = phase_peaks(astar_series(recs, 1), s_split)
        pre_r, post_r = pre_l[2] / pre_h[2], post_l[2] / post_h[2]
        contrast = post_r / pre_r
        contrasts[name] = contrast
        result["c4"][name] = {
            "pre_light_over_heavy": pre_r, "post_light_over_heavy": post_r,
            "contrast": contrast,
            "pre_peak_steps_heavy_light": [pre_h[0], pre_l[0]],
            "post_peak_steps_heavy_light": [post_h[0], post_l[0]],
            "left_censored_pre": "pre 최대는 step 0 레코드 부재의 좌측 검열값",
            "post_after_exchange": {
                "step_exchange": ev[name]["step_exchange"],
                "post_peaks_well_after_exchange":
                    min(post_h[0], post_l[0]) > ev[name]["step_exchange"]},
        }
        if name in LOCKED_C4:
            tgt = LOCKED_C4[name]
            got = (round(pre_r, 2), round(post_r, 2), round(contrast, 2))
            ok = got == tgt
            if not ok:
                mism.append(("c4", name, tgt, got))
            print(f"[C.4] {name}: pre {pre_r:.4f} post {post_r:.4f} contrast {contrast:.4f} "
                  f"→ {got} (목표 {tgt}) {'일치' if ok else '불일치'}")
        else:
            print(f"[C.4] {name}: pre {pre_r:.4f} post {post_r:.4f} contrast {contrast:.4f}")
    band = (round(min(contrasts.values()), 2), round(max(contrasts.values()), 2))
    ok_band = band == LOCKED_CADENCE_BAND
    if not ok_band:
        mism.append(("cadence_band", "all", LOCKED_CADENCE_BAND, band))
    result["c4"]["cadence_sensitivity_band"] = {
        "band": band, "target": LOCKED_CADENCE_BAND, "match": ok_band,
        "note": "보조 운동학 진단의 저장 간격 민감도 — 인과 증거 승격 금지"}
    print(f"[C.4] 저장 간격 민감도 대역: {band} (목표 {LOCKED_CADENCE_BAND}) "
          f"{'일치' if ok_band else '불일치'}")

    # 3) Δ(t) = vy*_eh − vy*_none (같은 입자, Verlet쌍)
    eh, none = histories["verlet_eh"], histories["verlet_none"]
    eh_by_step = {r["step"]: r for r in eh}
    none_by_step = {r["step"]: r for r in none}
    common = sorted(set(eh_by_step) & set(none_by_step))
    for pi, pname in ((0, "heavy"), (1, "light")):
        series = []
        for s in common:
            th = eh_by_step[s]["particles"][0]["t_star"]
            dv = (eh_by_step[s]["particles"][pi]["vy_star"]
                  - none_by_step[s]["particles"][pi]["vy_star"])
            series.append({"step": s, "t_star_heavy_eh": th, "delta_vy_star": dv})
        # 위상 요약: eh 팔의 최근접/역할 교대(주 사건축) 기준 3구간
        sc, se = ev["verlet_eh"]["step_closest"], ev["verlet_eh"]["step_exchange"]
        seg = lambda lo, hi: [x["delta_vy_star"] for x in series if lo <= x["step"] < hi]
        pre, mid, post = seg(0, sc), seg(sc, se), seg(se, 10**9)
        agg = lambda xs: {"n": len(xs), "mean": sum(xs) / len(xs),
                          "max_abs": max(abs(v) for v in xs)} if xs else None
        result["paired_difference"][pname] = {
            "n_common_steps": len(common),
            "pre_closest": agg(pre), "closest_to_exchange": agg(mid),
            "post_exchange": agg(post),
            "global_peak_abs": max(abs(x["delta_vy_star"]) for x in series),
            "series": series,
        }
        print(f"[Δ(t)] {pname}: pre |Δ|max {agg(pre)['max_abs']:.5f} → "
              f"post |Δ|max {agg(post)['max_abs']:.5f} "
              f"(post/pre {agg(post)['max_abs']/agg(pre)['max_abs']:.2f}×)")
    result["paired_difference"]["hierarchy"] = (
        "event-aligned primary / common-time secondary / global peak diagnostic; "
        "event-aligned 축은 각 팔 자신의 t*_closest 기준 정렬(사건 지연 "
        f"{lag_c:.3f} 반영), 표·그림 조립 시 양 축 병기")

    # 3b) event-aligned Δ (primary 축): τ = step − step_closest(자기 팔),
    #     none 팔 vy*를 τ 축에서 선형 보간해 eh 팔 τ 격자와 차분.
    sc_eh, sc_none = ev["verlet_eh"]["step_closest"], ev["verlet_none"]["step_closest"]
    se_rel = ev["verlet_eh"]["step_exchange"] - sc_eh
    none_steps = sorted(none_by_step)
    for pi, pname in ((0, "heavy"), (1, "light")):
        tau_n = [s - sc_none for s in none_steps]
        vy_n = [none_by_step[s]["particles"][pi]["vy_star"] for s in none_steps]

        def interp(tau):
            if tau <= tau_n[0] or tau >= tau_n[-1]:
                return None
            k = next(i for i in range(1, len(tau_n)) if tau_n[i] >= tau)
            w = (tau - tau_n[k - 1]) / (tau_n[k] - tau_n[k - 1])
            return vy_n[k - 1] + w * (vy_n[k] - vy_n[k - 1])

        series = []
        for r in eh:
            tau = r["step"] - sc_eh
            v_none = interp(tau)
            if v_none is None:
                continue
            series.append({"tau_step": tau,
                           "delta_vy_star": r["particles"][pi]["vy_star"] - v_none})
        seg = lambda lo, hi: [x["delta_vy_star"] for x in series if lo <= x["tau_step"] < hi]
        pre, mid, post = seg(-10**9, 0), seg(0, se_rel), seg(se_rel, 10**9)
        agg = lambda xs: {"n": len(xs), "mean": sum(xs) / len(xs),
                          "max_abs": max(abs(v) for v in xs)} if xs else None
        result["paired_difference"][pname]["event_aligned"] = {
            "pre_closest": agg(pre), "closest_to_exchange": agg(mid),
            "post_exchange": agg(post), "series": series}
        print(f"[Δ(t) 정렬] {pname}: pre |Δ|max {agg(pre)['max_abs']:.5f} → "
              f"post |Δ|max {agg(post)['max_abs']:.5f} "
              f"(post/pre {agg(post)['max_abs']/agg(pre)['max_abs']:.2f}×)")

    # 4) S4.4 |a*(t)| 시계열 재생성 (Verlet eh — 공개 저장소용)
    s44 = {"run": "method_matrix/df_bgk_verlet_explicit_history",
           "convention": "단측 나눗셈 차분, 구간 중점, 무평활, per-particle u_g²/D 정규화",
           "events_heavy_tstar_axis": {k: ev["verlet_eh"][k] for k in
                                       ("t_closest", "t_exchange")},
           "series": {}}
    for pi, pname in ((0, "heavy"), (1, "light")):
        s44["series"][pname] = [
            {"step_mid": sm, "t_star_mid": tm, "a_star": a}
            for sm, tm, a in astar_series(histories["verlet_eh"], pi)]
    with open(OUT_S44, "w") as f:
        json.dump(s44, f, indent=2, ensure_ascii=False)

    result["all_locked_confirmations_match"] = not mism
    result["mismatches"] = [str(m) for m in mism]
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {OUT_JSON} / S4.4 시계열: {OUT_S44}")
    print("스크립트 SHA256:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return 0 if not mism else 1


if __name__ == "__main__":
    raise SystemExit(main())
