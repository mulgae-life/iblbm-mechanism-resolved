"""Peak-completion 4범주 분류기 — 정본 (신규 런 판정 전용, 기존 보고값 재라벨링에 소급 비적용).

신호: raw 무평활 속도 크기 기반 Re_f (피크 정규화라 단위 불변). 후보 피크 = admissible
window 내 global max(동률 시 earliest). N_post = 8 stored samples. ε = 0.8%.
판정 순서: resolved_plateau → resolved_transient_peak → right_censored → unresolved_nonmonotone.
- plateau: 최종 8점 range ≤2ε AND 전/후 4점 블록 median drift ≤ε AND Theil–Sen 총 drift ≤ε → 보고값 = 최종 8점 median
- transient_peak: 피크 후 ≥8샘플 AND robust decline ≥2ε (피크 대비 피크 후 8점 median 하락)
- right_censored: global max가 최종 8샘플 내 OR 꼬리 상승 ≥ε (최종 8점 Theil–Sen 총 상승)
- nonmonotone: 그 외
"""
import json, statistics, itertools

EPS = 0.8  # %

def theil_sen_total(vals):
    n = len(vals)
    slopes = [ (vals[j]-vals[i])/(j-i) for i,j in itertools.combinations(range(n),2) ]
    return statistics.median(slopes) * (n-1)

def classify(series_pct):  # 피크=100 정규화된 시계열(%)
    s = series_pct
    peak = max(s); ipk = s.index(peak)
    last8 = s[-8:]
    rng8 = max(last8) - min(last8)
    drift_blocks = abs(statistics.median(last8[4:]) - statistics.median(last8[:4]))
    ts8 = theil_sen_total(last8)
    if rng8 <= 2*EPS and drift_blocks <= EPS and abs(ts8) <= EPS:
        return 'resolved_plateau', statistics.median(last8)
    post = s[ipk+1:]
    if len(post) >= 8:
        decline = peak - statistics.median(post[:8])
        if decline >= 2*EPS:
            return 'resolved_transient_peak', peak
    if ipk >= len(s)-8 or ts8 >= EPS:
        return 'right_censored', peak
    return 'unresolved_nonmonotone', peak

if __name__ == '__main__':
    import pathlib
    BASE = pathlib.Path(__file__).resolve().parents[2] / 'data/two_particle_sedimentation/extended_60D'
    for name, pid in [('df_bgk_verlet_none_60D', 1), ('df_bgk_verlet_explicit_history_60D', 1),
                      ('df_bgk_verlet_none_60D', 0), ('df_bgk_verlet_explicit_history_60D', 0)]:
        h = json.load(open(f'{BASE}/{name}/sedimentation_history.json'))
        v = [ (r['particles'][pid]['vx']**2 + r['particles'][pid]['vy']**2) ** 0.5 for r in h ]
        pk = max(v); s = [x/pk*100 for x in v]
        cat, val = classify(s)
        arm = 'heavy' if pid==0 else 'light'
        print(f"{name:44s} {arm:6s} peak@{s.index(100.0)}/{len(s)-1}  →  {cat}")
