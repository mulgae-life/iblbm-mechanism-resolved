"""Dual-ε 합의 판정 래퍼.

동결 분류기(`peak_completion_classifier.py`, 파일 무수정)를 ε_low=0.8 / ε_high=0.9
두 패스로 실행(EPS 런타임 주입)하고 합의표로 결합한다. 축 B(observable_status)는
4조건(피크=100 정규화, ε_strict=0.8)으로 판정한다.

--selftest: completion_sensitive + admissible_interior_window_peak 조합이 결과
dict/JSON에 무손실 직렬화되는지 확인한다. synthetic 불일치 방향은
ε_low→transient / ε_high→plateau — plateau↔transient 상호 불일치 중에서 유일하게
실현 가능한 방향이다. 다른 범주쌍의 불일치(T→RC, T→U, RC→P, RC→U, U→P)는 별도로
실현 가능하다. 도달 가능 조합 10종의 완전성은 분류기 분기의 ε 단조성 유도가
증명하고, seeded randomized property-based 검증(400,000 synthetic traces, seed
20260725)이 실현 사례와 구현 일치를 보조 확인한다.

단조성상 불가능한 6개 (class_low, class_high) 조합은 정상 합의로
매핑하지 않고 C0_invariant_violation으로 기록한다. C0 행이 하나라도 나오면 판정
파이프라인을 중단하고 해당 런을 non-confirmatory software-diagnostic review로
회부한다(주장 미생성). 유효 trace의 분류 결과는 불변이다.

invariant_violation은 completion 상태가 아니라 별도 qa_status로
기록한다(C0 시 completion_status=None, claim_generation_allowed=False —
scientific taxonomy는 5값 유지). C0 증명의 전제(동일 trace·정규화·분류기·분기
순서, ε_low<ε_high)는 judge()가 단일 trace를 받아 내부에서 두 패스를 평가하므로
구조적으로 보장된다 — low/high를 외부에서 분리 평가해 병합하는 경로는 정본에서
금지. trajectory-level 단정형 "right-censored"는 두 문턱 모두 RC일 때만 적용한다.
"""
import hashlib
import json
import pathlib
import statistics
import sys

import peak_completion_classifier as pcc

EPS_LOW = 0.8   # % — lower completion-classification tolerance (peak 정규화 직접 캘리브레이션)
EPS_HIGH = 0.9  # % — upper completion-classification tolerance (tail 정규화 캘리브레이션 동기)
EPS_STRICT = 0.8  # % — 축 B tail 무상승 판정 (작은 ε가 더 엄격)
assert EPS_LOW < EPS_HIGH  # 단조성 유도·C0 전제
MANIFEST_REF = "manifest_v1 + amendments 001-011"
PAIRED_SHIFT_DEFINITION_ID = "ABL_NONE_MINUS_EH_OVER_EH_V1"
Q_DEFINITION_ID = "Q_FROZEN_PAIR_ANCHOR_V1"
Q_DENOMINATOR_FROZEN = 16.3859  # % — 사전 고정 앵커(2dp status 유래)

COMPLETION_VALUES = (
    "resolved_plateau", "resolved_transient_peak", "right_censored",
    "unresolved_nonmonotone", "completion_sensitive",
)
OBSERVABLE_VALUES = (
    "admissible_interior_window_peak", "right_edge_censored_window_maximum",
    "inadmissible",
)
QA_VALUES = ("ok", "invariant_violation")
RIGHT_CENSOR_SUPPORT_VALUES = (
    "epsilon_independent_edge_peak", "both_tolerances", "lower_only",
    "upper_only", "not_applicable",
)
_P, _T = "resolved_plateau", "resolved_transient_peak"
_RC, _U = "right_censored", "unresolved_nonmonotone"
# ε_low < ε_high 단조성상 도달 가능한 (class_low, class_high) 10종.
# 그 외 6종은 물리 결과가 아니라 분류기·직렬화 결함 신호 → C0 hard fail.
REACHABLE_PAIRS = {
    (_P, _P),
    (_T, _P), (_T, _T), (_T, _RC), (_T, _U),
    (_RC, _P), (_RC, _RC), (_RC, _U),
    (_U, _P), (_U, _U),
}


def classify_at(series_pct, eps):
    orig = pcc.EPS
    try:
        pcc.EPS = eps
        cat, val = pcc.classify(series_pct)
    finally:
        pcc.EPS = orig
    return cat, val


def consensus(cat_low, cat_high):
    return consensus_with_rule(cat_low, cat_high)[0]


def consensus_with_rule(cat_low, cat_high):
    """합의 결과와 적용 규칙 id를 함께 반환한다."""
    if (cat_low, cat_high) not in REACHABLE_PAIRS:
        return "invariant_violation", "C0_invariant_violation"
    if "right_censored" in (cat_low, cat_high):
        return "right_censored", "C1_right_censored_precedence"
    if cat_low == cat_high:
        return cat_low, "C2_threshold_agreement"
    return "completion_sensitive", "C3_threshold_disagreement"


def observable_status(series_pct, admissible=True):
    return observable_with_rule(series_pct, admissible)[0]


def observable_with_rule(series_pct, admissible=True):
    """축 B 상태와 적용 규칙 id를 함께 반환한다."""
    if not admissible:
        return "inadmissible", "B0_window_inadmissible"
    s = series_pct
    peak = max(s)
    k = s.index(peak)
    m = len(s) - 1
    if m - k < 8:
        return "right_edge_censored_window_maximum", "B1_post_peak_samples_lt8"
    last8 = s[-8:]
    ts8 = pcc.theil_sen_total(last8)
    block_growth = statistics.median(last8[4:]) - statistics.median(last8[:4])
    if ts8 > EPS_STRICT:
        return "right_edge_censored_window_maximum", "B2_tail_theil_sen_rise"
    if block_growth > EPS_STRICT:
        return "right_edge_censored_window_maximum", "B3_tail_block_median_rise"
    return "admissible_interior_window_peak", "B4_interior_peak_flat_tail"


def right_censor_support(cat_low, cat_high, obs_rule_id):
    """right-censoring 근거 provenance.

    B1(창 최대가 최종 8샘플 내)은 ε 독립 관측 축 증거라 최우선으로 기록한다.
    단 epsilon_independent_edge_peak는 observable-level 문장(창 최대의 우단
    위치)만 지원한다 — plateau 배제를 증명하지 않으므로 trajectory-level 단정형
    "right-censored"는 class_eps_low·class_eps_high가 모두 RC일 때만 허용한다.
    문장 생성 규칙 전체는 위 합의표가 정의한다.
    """
    if obs_rule_id == "B1_post_peak_samples_lt8":
        return "epsilon_independent_edge_peak"
    if cat_low == _RC and cat_high == _RC:
        return "both_tolerances"
    if cat_low == _RC:
        return "lower_only"
    if cat_high == _RC:
        return "upper_only"
    return "not_applicable"


def signed_ablation_shift_pct(peak_eh, peak_none):
    """S_abl = 100·(none−eh)/eh — signed paired shift 정본 규약.

    eh > none ⇔ S_abl < 0. anchor: (238.62, 199.52) → −16.3859%
    (사전 고정 효과 크기 앵커 −16.4%와 정합).
    """
    return 100.0 * (peak_none - peak_eh) / peak_eh


def ablation_magnitude_pct(peak_eh, peak_none):
    """A_abl = |S_abl| — signed shift의 크기."""
    return abs(signed_ablation_shift_pct(peak_eh, peak_none))


def effect_size_q_decision(s_ctrl_raw_pct):
    """q_decision = |S_ctrl_raw| / 16.3859 — 유일한 판정용 q.

    분모는 사전 고정된 상수다(raw 원칙의 유일한 예외).
    effect-size class는 이 값만으로 결정한다.
    """
    return abs(s_ctrl_raw_pct) / Q_DENOMINATOR_FROZEN


def effect_size_q_raw_reference(s_ctrl_raw_pct, s_pair_raw_pct):
    """정밀도 민감도 참고 전용 — 판정에 사용하지 않는다."""
    return abs(s_ctrl_raw_pct) / abs(s_pair_raw_pct)


def ablation_direction(signed_shift_pct):
    """signed direction 3값 완비 — full-precision exact zero.

    zero면 direction preservation·reversal 주장 금지. practical-zero tolerance는
    신설하지 않는다(필요 시 별도 사전 정의 선행).
    """
    if signed_shift_pct < 0:
        return "negative"
    if signed_shift_pct > 0:
        return "positive"
    return "zero"


def paired_claim_scope(arm_a, arm_b):
    """paired claim-scope 교집합. 인자는 judge() 결과 dict 2개.

    "나쁜 쪽 상속"의 정밀화 — completion taxonomy에 전순서가 없으므로 arm 상태를
    한 값으로 축약하지 않고 허용 가능한 claim의 교집합을 파생한다. arm-level
    상태는 호출자가 별도 보존한다.
    """
    arms = (arm_a, arm_b)
    if any(a["qa_status"] != "ok" for a in arms):
        return "no_claim"
    if any(a["observable_status"] == "inadmissible" for a in arms):
        return "no_claim"
    if any(a["completion_status"] == "right_censored" for a in arms):
        return "censoring_not_excluded"
    if any(a["completion_status"] in ("unresolved_nonmonotone",
                                      "completion_sensitive") for a in arms):
        return "finite_window_only"
    return "both_arms_resolved"


def paired_observable(arm_a, arm_b):
    """paired observable 집계 — 두 팔 중 보수적인 쪽을 취한다."""
    arms = (arm_a, arm_b)
    if any(a["observable_status"] == "inadmissible" for a in arms):
        return "inadmissible"
    if any(a["observable_status"] == "right_edge_censored_window_maximum"
           for a in arms):
        return "right_edge_limited"
    return "admissible_interior"


def classifier_sha256():
    """동결 분류기 파일 해시 — 판정 표에 분류기 판본을 각인한다."""
    path = pathlib.Path(pcc.__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judge(series_pct, admissible=True):
    cat_low, val_low = classify_at(series_pct, EPS_LOW)
    cat_high, val_high = classify_at(series_pct, EPS_HIGH)
    comp, comp_rule = consensus_with_rule(cat_low, cat_high)
    qa = "invariant_violation" if comp_rule == "C0_invariant_violation" else "ok"
    if qa != "ok":
        comp = None  # QA 위반은 completion 상태가 아니다
    obs, obs_rule = observable_with_rule(series_pct, admissible)
    rc_support = right_censor_support(cat_low, cat_high, obs_rule)
    result = {
        "eps_low_pct": EPS_LOW,
        "eps_high_pct": EPS_HIGH,
        "class_eps_low": cat_low,
        "class_eps_high": cat_high,
        "completion_status": comp,
        "observable_status": obs,
        "value_eps_low": val_low,
        "value_eps_high": val_high,
        "consensus_rule_id": comp_rule,
        "observable_rule_id": obs_rule,
        "right_censor_support": rc_support,
        "qa_status": qa,
        "claim_generation_allowed": qa == "ok",
        "classifier_sha256": classifier_sha256(),
        "manifest_ref": MANIFEST_REF,
    }
    assert result["qa_status"] in QA_VALUES
    if result["qa_status"] == "ok":
        assert result["completion_status"] in COMPLETION_VALUES
    else:
        assert result["completion_status"] is None
    assert result["observable_status"] in OBSERVABLE_VALUES
    assert result["right_censor_support"] in RIGHT_CENSOR_SUPPORT_VALUES
    return result


def _selftest():
    # 앞부분 상승 → 피크(idx 2) → 완만 감쇠 꼬리(range 1.7: 0.8 plateau 탈락,
    # 0.9 plateau 통과) + 피크 후 8점 median 하락 1.7 (≥2·0.8, <2·0.9)
    s = [90.0, 95.0, 100.0, 98.5, 98.4, 98.3, 98.3, 98.2, 98.3, 98.2,
         98.3, 98.4, 99.4, 97.7, 99.3, 97.8, 99.2, 97.9, 99.1, 98.0]
    r = judge(s)
    assert r["class_eps_low"] == "resolved_transient_peak", r
    assert r["class_eps_high"] == "resolved_plateau", r
    assert r["completion_status"] == "completion_sensitive", r
    assert r["observable_status"] == "admissible_interior_window_peak", r
    assert r["consensus_rule_id"] == "C3_threshold_disagreement", r
    assert r["observable_rule_id"] == "B4_interior_peak_flat_tail", r
    assert r["right_censor_support"] == "not_applicable", r
    assert r["qa_status"] == "ok" and r["claim_generation_allowed"] is True, r
    blob = json.dumps(r)
    back = json.loads(blob)
    for key in ("completion_status", "observable_status", "consensus_rule_id",
                "observable_rule_id", "right_censor_support", "qa_status",
                "claim_generation_allowed", "classifier_sha256", "manifest_ref"):
        assert back[key] == r[key], key
    assert back["completion_status"] == "completion_sensitive"
    assert back["observable_status"] == "admissible_interior_window_peak"
    # 합의표 전 조합 완비성: 도달 가능 10종은 C1/C2/C3, 불가능 6종은 C0 hard fail
    cats = [_P, _T, _RC, _U]
    assert sum((a, b) in REACHABLE_PAIRS for a in cats for b in cats) == 10
    for a in cats:
        for b in cats:
            val, rule = consensus_with_rule(a, b)
            if (a, b) in REACHABLE_PAIRS:
                assert val in COMPLETION_VALUES
                assert rule in ("C1_right_censored_precedence",
                                "C2_threshold_agreement",
                                "C3_threshold_disagreement")
            else:
                assert val == "invariant_violation", (a, b)
                assert rule == "C0_invariant_violation", (a, b)
    # right_censor_support 결정 규칙 (B1 최우선)
    assert right_censor_support(_RC, _RC, "B4_interior_peak_flat_tail") == "both_tolerances"
    assert right_censor_support(_RC, _P, "B4_interior_peak_flat_tail") == "lower_only"
    assert right_censor_support(_T, _RC, "B4_interior_peak_flat_tail") == "upper_only"
    assert right_censor_support(_RC, _P, "B1_post_peak_samples_lt8") == "epsilon_independent_edge_peak"
    assert right_censor_support(_T, _P, "B4_interior_peak_flat_tail") == "not_applicable"
    # paired 집계 truth table
    def _arm(comp, obs="admissible_interior_window_peak", qa="ok"):
        return {"qa_status": qa, "completion_status": comp,
                "observable_status": obs}
    ok_p, ok_t = _arm(_P), _arm(_T)
    assert paired_claim_scope(ok_p, ok_t) == "both_arms_resolved"
    assert paired_claim_scope(ok_p, _arm(_RC)) == "censoring_not_excluded"
    assert paired_claim_scope(ok_t, _arm("completion_sensitive")) == "finite_window_only"
    assert paired_claim_scope(ok_p, _arm(_U)) == "finite_window_only"
    assert paired_claim_scope(ok_p, _arm(_T, qa="invariant_violation")) == "no_claim"
    assert paired_claim_scope(ok_p, _arm(_T, obs="inadmissible")) == "no_claim"
    # RC가 unresolved보다 우선 (순서 3 > 4)
    assert paired_claim_scope(_arm(_RC), _arm(_U)) == "censoring_not_excluded"
    assert paired_observable(ok_p, ok_t) == "admissible_interior"
    assert paired_observable(ok_p, _arm(_T, obs="right_edge_censored_window_maximum")) == "right_edge_limited"
    assert paired_observable(_arm(_P, obs="inadmissible"),
                             _arm(_T, obs="right_edge_censored_window_maximum")) == "inadmissible"
    # signed shift 정본 규약 anchor
    assert round(signed_ablation_shift_pct(238.62, 199.52), 4) == -16.3859
    assert round(ablation_magnitude_pct(238.62, 199.52), 4) == 16.3859
    # 역방향은 반대칭 아님 — 분모가 항상 R_eh이므로 +19.5970 (단순 +16.3859 금지)
    assert round(signed_ablation_shift_pct(199.52, 238.62), 4) == 19.5970
    # 재구성 invariant: R_none = R_eh·(1 + S/100)
    s_fwd = signed_ablation_shift_pct(238.62, 199.52)
    assert abs(199.52 - 238.62 * (1 + s_fwd / 100)) < 1e-9
    # direction 3값
    assert ablation_direction(s_fwd) == "negative"
    assert ablation_direction(signed_ablation_shift_pct(199.52, 238.62)) == "positive"
    assert ablation_direction(0.0) == "zero"
    # q 예외 계약: frozen 분모 < |raw 앵커| ⇒ q_decision ≥ q_raw 항상
    s_pair_raw_prelim = -16.3864878605  # 50D 예비 실측 — 최종은 builder 동결 extractor 재산출
    assert abs(s_pair_raw_prelim) > Q_DENOMINATOR_FROZEN
    a_raw = abs(s_pair_raw_prelim)
    for c in (0.0, 0.25 * a_raw, 0.75 * a_raw, a_raw):
        assert effect_size_q_decision(c) >= effect_size_q_raw_reference(c, s_pair_raw_prelim)
    # C0 경로: 분류기 결함을 모사(mock)해 qa_status 분리·claim 차단 확인
    orig_classify = classify_at
    try:
        globals()["classify_at"] = lambda s, e: (
            (_P, 100.0) if e == EPS_LOW else (_T, 100.0))
        rr = judge([100.0] * 20)
        assert rr["qa_status"] == "invariant_violation", rr
        assert rr["completion_status"] is None, rr
        assert rr["consensus_rule_id"] == "C0_invariant_violation", rr
        assert rr["claim_generation_allowed"] is False, rr
        blob2 = json.dumps(rr)
        assert json.loads(blob2)["completion_status"] is None
    finally:
        globals()["classify_at"] = orig_classify
    print("SELFTEST PASS")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
