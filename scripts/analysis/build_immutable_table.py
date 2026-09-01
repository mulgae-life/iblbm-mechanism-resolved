"""N-계열 immutable table builder.

역할: 확인 런 완주 후 N-계열 paired 판정 표를 단일 결정론 경로로 생성한다.
플래그 없이 실행하면 synthetic 사례와 기존 공개 comparator만으로 selftest를
수행하고, confirmatory 경로는 명시 플래그로만 연결한다.

표가 이행하는 계약:
- paired schema·상태 3층·pair 5-assertion·masked equality·raw/display 분리
- q 예외 계약(frozen 분모 단독 판정), as-run resolved config 정본
- first-record 진단 강등, observable formula 동일성, keyset provenance
- native alignment 검증 전용, q comparability scope, canonical closure
  subobject, resolved effect 검증, 고정 순서
- schema별 엄격 closure decoder(암묵 default 금지), run-to-manifest 절대 계약
  (expected_run_contract_status — frozen 러너 registry materialized config 대조,
  N-7 light identity·N-6 viscosity 경로), N-4 exact rational tick contract
  (sample cardinality 포함, float mismatch는 진단 강등)
- contact rule 정본 사양·N-6 점성 사슬 3식·keyset 정본 판정
- 비교창 provenance 필드·호환 게이트·comparison_scope 5계급, N-4 S_801
  harmonized(legacy pair 동일 규격 재산출 + 3조건 안정 판정), N-1 50D
  harmonized context(비판정)·COMMON_EARLIEST_WINDOW 비판정 sensitivity,
  min surface gap/kernel-overlap 증거 필드
- P4 겹침 보수 기준(4√2 clearance — tensor-product support 기하), canonical
  analysis_window_definition_id(창 전체 교집합 ID)·window_end_reason,
  N-1 domain_specific scope, N-4 directness 등급 B(801 none 팔 미bridge 실측),
  arm full-history 무결성(termination_reason·history_len·등차)
- P4 겹침 판정 stored-sample scope 정본화(per-step online 기록 부재 실측 —
  continuous 주장 금지·peak 시점 gap 필드·marker 반경 구성 보장), termination
  어휘 전체 재실측(7개) + run별 frozen stop 계약 대조(어휘 유효 ≠ 계약 이행),
  N-4 외부 표시 문자열 분리, window_end 동시 trigger 보존
- 종료 상태 3층 분리(runtime 계약 / 창 완결 / metric eligibility — 혼합 명칭
  폐기, 차단 집합 불변), stop predicate 일치 QA(문자열-실상태 재계산 대조:
  offset·contact·max_steps, 동일 step 저장 실측)
- termination provenance를 reason별 발생 경로로 분리(should_stop/loop-bound/
  convergence — 단일 source ID 폐기), signed predicate margin + 저장 좌표
  round-trip 실측(무반올림 json.dump 체인), 예약 상태 출력 금지 전수 검사,
  3층×completion 통합 truth-table QA

판정 산술은 전부 dual_epsilon_consensus 함수 호출로만 한다 — 수동 산술 금지.
좌표 규약(실측): history 레코드의 x·y는 물리 좌표(채널 폭 1), 직경 D_domain =
cfg.cylinder_D_ratio. contact·y* 판정은 전부 물리 좌표 기준이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

import dual_epsilon_consensus as dec  # noqa: E402

DATA = ROOT / "data"

# ---------------------------------------------------------------- 상수·ID (전부 사전 고정)
SCHEMA_VERSION = "immutable_table_v1"
MANIFEST_REF = "manifest_v1 + amendments 001-020"
OBSERVABLE_FORMULA_ID = "PEAK_RE_F_SPEED_NORM_V1"          # |v| 노름·Re_f 규약
PAIRED_SHIFT_ID = dec.PAIRED_SHIFT_DEFINITION_ID            # ABL_NONE_MINUS_EH_OVER_EH_V1
Q_DEFINITION_ID = dec.Q_DEFINITION_ID                       # Q_FROZEN_PAIR_ANCHOR_V1
Q_DENOM_FROZEN = dec.Q_DENOMINATOR_FROZEN                   # 16.3859 (사전 고정 상수)
ABLATION_SENTINEL = "<ABLATION_ARM>"                        # ablation 팔 표식
FLOAT_TOL = 1e-9                                            # 검증 tolerance 관행

# contact rule 정본 사양 (단위·좌표·window 계약 명시. confirmatory 가드 대상)
CONTACT_CROSSING_RULE = "CENTER_DISTANCE_LE_D_V1"
CONTACT_RULE_SPEC = {
    "contact_rule_id": CONTACT_CROSSING_RULE,
    "coordinate_system_id": "PHYSICAL_DOMAIN_COORDINATES_V1",  # 레코드 x·y = 물리 좌표
    "diameter_source_id": "CYLINDER_D_RATIO_V1",               # D_domain = cfg.cylinder_D_ratio
    "distance_metric": "euclidean_center_distance",
    "contact_threshold": "R1+R2 (= D, 동일 지름)",
    "minimum_image": "not_applicable_nonperiodic_settling_channel",  # bc_type 실측
    "first_crossing_convention": "earliest_stored_index_with_d12_le_D",
    "contact_time_interpolation": "none",
    "paired_window_policy": "ARM_LOCAL_PRECONTACT_V1",  # 런 단위 규격 해석
    "contact_model_note": "활성 코드에 입자간 반발/윤활 모델 없음(실측) — geometric "
                          "surface contact 기준. IB kernel support 중첩은 D 이전 시작을 정직 기록",
}

# closure schema 판정의 정본 = 동결 keyset 해시 (presence 패턴은 2차 검사)
LEGACY_KEYSET_SHA256 = "9829faac3a63fd8b24ccea60d054c3dcec51da82553df90a503cdc6b0940d392"   # 50D 23키 실측
CURRENT_KEYSET_SHA256 = "97ca280728dc178b79c13ef7dd3be0eddf0cd32a17dec7ed8cc4d119ae9f8f12"  # 캠페인 21키 실측

# N-6 viscosity 사슬 (물리 상수 = 단일 입자 preset 계약: g=980 cm/s², d=0.25 cm)
G_PHYS_CM_S2 = 980.0
D_PHYS_CM = 0.25
COLLISION_VISCOSITY_SOURCE_ID = "CFG_RE_INIT_NU_V1"  # init.py: ν_lat = lattice_u·D_lat/Re → τ = 3ν+0.5
Q_COMPARABILITY_SCOPE = "cross_configuration_declared_window_peak_ratio"

# arm window rule ID 정본 상수 + comparison_scope (5계급)
WINDOW_RULE_TWO_ARM = "ARM_LOCAL_PRECONTACT_V1"
WINDOW_RULE_SINGLE_ARM = "SINGLE_ARM_YSTAR_LIMIT_V1"
# canonical analysis window ID (contact rule + y* cutoff + arm-local +
# stored-sample 규약의 교집합 전체에 부여. 내용 변경 없음 — ID 부여만)
WIN_STD50_15P5 = "STD50_ARMLOCAL_PRECONTACT_YSTAR15P5_STORED_SAMPLE_V1"
WIN_TALL60_51P5 = "TALL60_ARMLOCAL_PRECONTACT_YSTAR51P5_STORED_SAMPLE_V1"
WIN_STD24_SINGLE_15P5 = "STD24_SINGLE_YSTAR15P5_STORED_SAMPLE_V1"

# P4 겹침 보수 기준 (tensor-product support = 반폭 2h 사각형;
# 비겹침 충분조건 g > 4√2 h — Euclidean gap의 방향 무관 clearance)
KERNEL_OVERLAP_TEST_ID = "CONSERVATIVE_P4_EUCLIDEAN_CLEARANCE_V1"
KERNEL_SUPPORT_HALFWIDTH_CELLS = 2.0
KERNEL_CLEARANCE_THRESHOLD_DX = 4.0 * math.sqrt(2.0)

# N-4 directness (G-2 실측: eh 팔만 재현 bridge, none 팔 독립 재현 없음)
N4_DIRECTNESS_GRADE = "B_cross_version_bridged"
N4_EH_LINEAGE = "bridged_by_g2_gate"
N4_NONE_LINEAGE = "not_independently_rerun"

# 도달 가능 종료 어휘 전체 (loop.py + sedimentation runtime should_stop 실측)
REACHABLE_TERMINATION_REASONS = {
    "max_steps", "converged", "cd_converged", "nan", "contact", "offset", "domain_bounds"}

# run family(러너)별 frozen stop 계약 (러너 설정 실측: two 러너 전 케이스
# sedimentation_stop_offset_d=2.0, single 러너 전 케이스 stop_at_contact=True.
# 침강 프리셋 use_convergence=False·cd_convergence_tol=None → converged 계열 구조적 불가)
TERMINATION_CONTRACTS = {
    "two": {"termination_contract_id": "TWO_PARTICLE_OFFSET_STOP_V1",
            "expected_stop_predicate": "sedimentation_stop_offset_d=2.0",
            "allowed_primary": frozenset({"offset"})},
    "single": {"termination_contract_id": "SINGLE_CONTACT_STOP_V1",
               "expected_stop_predicate": "sedimentation_stop_at_contact=True",
               "allowed_primary": frozenset({"contact"})},
}
# termination provenance: reason별 발생 경로 분리. loop.py·runtime 실측 매핑.
TERMINATION_ORIGIN_IDS = {
    "offset": "RUNTIME_SHOULD_STOP_V1", "contact": "RUNTIME_SHOULD_STOP_V1",
    "nan": "RUNTIME_SHOULD_STOP_V1", "domain_bounds": "RUNTIME_SHOULD_STOP_V1",
    "max_steps": "LOOP_MAX_STEPS_EXHAUSTION_V1",
    "converged": "LOOP_CONVERGENCE_DECISION_V1",
    "cd_converged": "LOOP_CONVERGENCE_DECISION_V1",
}
# should_stop 재계산의 상태 소스 (실측: diagnostics 레코드 생성 직후 같은
# 스텝 should_stop 호출, 레코드 x·y = state.particle_pos 동일 소스)
PREDICATE_RECOMPUTE_SHOULD_STOP = "SAME_STEP_LAST_STORED_RECORD_V1"
PREDICATE_RECOMPUTE_MAX_STEPS = "TERMINAL_STEP_METADATA_V1"

# 저장 좌표 직렬화 실측: runtime float() 무반올림 → monitor 콜백 표준
# json.dump(indent=2, round() 비적용) → 파이썬 json float = repr 최단 round-trip
HISTORY_COORDINATE_ENCODING_ID = "PYTHON_JSON_FLOAT_REPR_V1"
HISTORY_COORDINATE_ROUNDTRIP_STATUS = "roundtrip_safe_by_construction"

# 예약 상태 (현 캠페인 출력 금지 — selftest 전수 검사)
RESERVED_NOT_EMITTED_IN_CURRENT_CAMPAIGN = (
    "complete_by_expected_runtime_stop", "eligible_with_censoring_limitation")

# gap 증거의 표본 범위 (per-step online minimum 기록이 코드에 없음을 실측 —
# 어떤 pair에서도 continuous 판정 불가, 고정 상수)
GAP_SAMPLING_SCOPE = "stored_samples_only"

# 같은 저장 표본에서 동시 trigger 시 선택 규칙 (기존 동작의 명시화)
WINDOW_END_PRECEDENCE = "CONTACT_BEFORE_YSTAR_V1"

# N-4 외부 표시 정본 (내부 enum direct_harmonized_window와 분리 —
# "direct"가 directness 등급 B와 충돌해 보이는 외부 노출 차단)
N4_EXTERNAL_DISPLAY_SCOPE = "harmonized-window comparison"
N4_EXTERNAL_DISPLAY_DIRECTNESS = "B — cross-version baseline, partially bridged"

# legacy 공개 pair harmonized 재산출 원천 (레지스트리 디렉터리 고정,
# expected_run_contract는 legacy 비적용 — 방어: keyset 해시 + closure tuple)
N4_REF_PAIR = {  # N-4의 S_801 (df 801 50D pair — ANCHOR_PAIR와 동일 원천)
    "eh_dir": "two_particle_sedimentation/method_matrix/df_bgk_verlet_explicit_history",
    "none_dir": "two_particle_sedimentation/method_matrix/df_bgk_verlet_none",
    "ystar_limit": 15.5, "label": "N-4-REF-NY801", "window_def": WIN_STD50_15P5,
}
N1_CONTEXT_PAIRS = {  # N-1의 50D harmonized context (비판정)
    "N-1-MDF-50D-CONTEXT": {
        "eh_dir": "two_particle_sedimentation/method_matrix/mdf_bgk_verlet_explicit_history",
        "none_dir": "two_particle_sedimentation/method_matrix/mdf_bgk_verlet_none",
        "ystar_limit": 15.5,
        "window_def": WIN_STD50_15P5,
    },
    "N-1-DFC-50D-CONTEXT": {
        "eh_dir": "two_particle_sedimentation/method_matrix/dfc_bgk_verlet_explicit_history",
        "none_dir": "two_particle_sedimentation/method_matrix/dfc_bgk_verlet_none",
        "ystar_limit": 15.5,
        "window_def": WIN_STD50_15P5,
    },
}

# N-4 안정 판정 상수 (사전 고정 — 수치 변경 금지)
N4_STABILITY_THRESHOLD_PP = 1.639  # 0.1·|S_pair| = 1.639%p
N4_TICK_RATIOS = {961: Fraction(36, 25), 1281: Fraction(64, 25)}  # 720/500, 1280/500
# sampling alignment 결속 (family 밖은 전부 null)
SAMPLING_ALIGNMENT_RULES = {
    "N-4": "NATIVE_COMMON_TSTAR_GRID_REFINEMENT_V1",
}

# 필수 물리 필드 존재 체크(퇴화 블록 차단)
REQUIRED_PHYSICS_FIELDS = (
    "NN", "gravity", "rho_ratio", "settling_inertia_model",
    "time_integrator", "sedimentation_stop_offset_d", "marker_spacing_factor",
)

# expected run 계약 대조 필드(config 블록 키 = SimConfig 속성명)
EXPECTED_CONTRACT_FIELDS = (
    "NN", "gravity", "rho_ratio", "max_steps", "check_interval",
    "time_integrator", "collision_model", "ibm_method", "delta_type",
    "settling_inertia_model", "marker_spacing_factor", "retraction_dx",
    "sedimentation_stop_offset_d",
)

# manifest §2 — 비교 레지스트리. runner_case_ids = frozen case registry의 case_id.
COMPARISONS = {
    "N-1-MDF": {
        "family": "N-1", "pairing_class": "matched_current_version", "runner": "two",
        "eh_dir": "two_particle_sedimentation/extended_60D/mdf_bgk_verlet_explicit_history_60D",
        "none_dir": "two_particle_sedimentation/extended_60D/mdf_bgk_verlet_none_60D",
        "eh_case": "TWO_PARTICLE_EXT60D_MDF_EXPLICIT_HISTORY", "none_case": "TWO_PARTICLE_EXT60D_MDF_NONE",
        "kind": "two_particle", "ystar_limit": 51.5, "particle": "light", "q_use": False,
        "window_def": WIN_TALL60_51P5,
    },
    "N-1-DFC": {
        "family": "N-1", "pairing_class": "matched_current_version", "runner": "two",
        "eh_dir": "two_particle_sedimentation/extended_60D/dfc_bgk_verlet_explicit_history_60D",
        "none_dir": "two_particle_sedimentation/extended_60D/dfc_bgk_verlet_none_60D",
        "eh_case": "TWO_PARTICLE_EXT60D_DFC_EXPLICIT_HISTORY", "none_case": "TWO_PARTICLE_EXT60D_DFC_NONE",
        "kind": "two_particle", "ystar_limit": 51.5, "particle": "light", "q_use": False,
        "window_def": WIN_TALL60_51P5,
    },
    "N-4-NY961": {
        "family": "N-4", "pairing_class": "matched_current_version", "runner": "two",
        "eh_dir": "two_particle_sedimentation/grid_sensitivity/df_bgk_verlet_explicit_history_ny961",
        "none_dir": "two_particle_sedimentation/grid_sensitivity/df_bgk_verlet_none_ny961",
        "eh_case": "TWO_PARTICLE_GRID_NY961_EXPLICIT_HISTORY", "none_case": "TWO_PARTICLE_GRID_NY961_NONE",
        "kind": "two_particle", "ystar_limit": 15.5, "particle": "light", "q_use": False,
        "grid_ref_nn": 801, "window_def": WIN_STD50_15P5,
    },
    "N-4-NY1281": {
        "family": "N-4", "pairing_class": "matched_current_version", "runner": "two",
        "eh_dir": "two_particle_sedimentation/grid_sensitivity/df_bgk_verlet_explicit_history_ny1281",
        "none_dir": "two_particle_sedimentation/grid_sensitivity/df_bgk_verlet_none_ny1281",
        "eh_case": "TWO_PARTICLE_GRID_NY1281_EXPLICIT_HISTORY", "none_case": "TWO_PARTICLE_GRID_NY1281_NONE",
        "kind": "two_particle", "ystar_limit": 15.5, "particle": "light", "q_use": False,
        "grid_ref_nn": 801, "window_def": WIN_STD50_15P5,
    },
    "N-6": {
        "family": "N-6", "pairing_class": "matched_current_version", "runner": "single",
        "eh_dir": "single_particle_sedimentation/method_matrix/rho125_nu001/df_bgk_verlet_explicit_history",
        "none_dir": "single_particle_sedimentation/method_matrix/rho125_nu001/df_bgk_verlet_none",
        "eh_case": "SINGLE_RHO125_NU001_DF_BGK_VERLET_EXPLICIT_HISTORY",
        "none_case": "SINGLE_RHO125_NU001_DF_BGK_VERLET_NONE",
        "kind": "single_particle", "ystar_limit": 15.5, "particle": "single", "q_use": True,
        "viscosity_contract": True, "window_def": WIN_STD24_SINGLE_15P5,
    },
    "N-7": {
        "family": "N-7", "pairing_class": "matched_current_version", "runner": "two",
        "eh_dir": "two_particle_sedimentation/isolated_light/df_bgk_verlet_light_explicit_history",
        "none_dir": "two_particle_sedimentation/isolated_light/df_bgk_verlet_light_none",
        "eh_case": "TWO_PARTICLE_ISOLATED_LIGHT_EXPLICIT_HISTORY", "none_case": "TWO_PARTICLE_ISOLATED_LIGHT_NONE",
        "kind": "two_particle", "ystar_limit": 15.5, "particle": "light", "q_use": True,
        "light_identity": {"count": 1, "rho_ratio": 1.25}, "window_def": WIN_STD50_15P5,
        # two 러너의 isolated 1입자 런은 single 레코드 스키마로 저장된다
        # (particles 리스트 아님 — y_star·x·y·vx·vy 직접 필드). 추출은 single 경로.
        "record_schema": "single",
    },
}

# q 분모 anchor의 원천 pair (기존 공개 데이터, 확인 런 아님. legacy 스키마)
ANCHOR_PAIR = {
    "eh_dir": "two_particle_sedimentation/method_matrix/df_bgk_verlet_explicit_history",
    "none_dir": "two_particle_sedimentation/method_matrix/df_bgk_verlet_none",
    "ystar_limit": 15.5,
}


# ---------------------------------------------------------------- frozen case registry 접근
_RUNNER_CACHE: dict = {}


def _runner_module(which: str):
    """frozen 러너 모듈 lazy import (case registry + materialized config builder)."""
    if which not in _RUNNER_CACHE:
        if which == "two":
            import run_two_particle_sedimentation_experiments as mod
        else:
            import run_single_particle_sedimentation_experiments as mod
        _RUNNER_CACHE[which] = mod
    return _RUNNER_CACHE[which]


def expected_materialized_config(which: str, case_id: str):
    """frozen case registry에서 case를 찾아 frozen builder로 materialize."""
    mod = _runner_module(which)
    registry = mod.TWO_PARTICLE_CASES if which == "two" else mod.SINGLE_PARTICLE_CASES
    cases = [c for c in registry if c["case_id"] == case_id]
    if len(cases) != 1:
        raise AssertionError(f"case_registry_lookup_failed:{case_id}")
    return cases[0], mod._build_config(cases[0])


def expected_run_contract(which: str, case_id: str, actual_cfg: dict) -> dict:
    """run-to-manifest 절대 계약: expected materialized vs actual resolved.

    pair equality와 독립 — 양팔이 같아도 둘 다 의도 설정이 아니면 차단.
    """
    case, exp = expected_materialized_config(which, case_id)
    mismatches = {}
    for field in EXPECTED_CONTRACT_FIELDS:
        if field not in actual_cfg:
            mismatches[field] = {"expected": getattr(exp, field, None), "actual": "<absent>"}
            continue
        want = getattr(exp, field)
        got = actual_cfg[field]
        same = (abs(want - got) <= FLOAT_TOL * max(1.0, abs(want))
                if isinstance(want, float) else want == got)
        if not same:
            mismatches[field] = {"expected": want, "actual": got}
    return {
        "expected_run_contract_status": "ok" if not mismatches else "contract_mismatch",
        "contract_mismatches": mismatches,
        "expected_config_case_id": case_id,
    }


def verify_viscosity_chain(exp_cfg, nu_phys_target: float, actual_gravity: float) -> dict:
    """N-6 ν override의 end-to-end 소비 검증 (세 경로 = cfg.Re 단일 소스).

    코드 사슬(init.initialize()): collision ν_lat = lattice_u·D_lat/cfg.Re,
    τ = 3ν+0.5 — collision이 소비하는 ν의 유일한 소스가 cfg.Re다. 따라서
    (i) Re_target(물리 상수·ν_phys_target에서 독립 계산) == cfg.Re full precision
        → mapping·collision 경로 검증
    (ii) lattice_u² == Δρ·gravity·D_lat → gravity mapping 자기 일관성
    (iii) extractor ν = 동일 materialized cfg에서 유도(단일 소스) — 경로 분리 없음
    """
    d_lat = float(exp_cfg.cylinder_D_ratio) * (exp_cfg.NN - 1)
    delta_rho = abs(exp_cfg.rho_ratio - 1.0)
    u_g_phys = math.sqrt(delta_rho * G_PHYS_CM_S2 * D_PHYS_CM)
    re_target = u_g_phys * D_PHYS_CM / nu_phys_target
    re_ok = abs(re_target - exp_cfg.Re) <= 1e-12 * re_target
    g_ok = abs(exp_cfg.lattice_u ** 2 - delta_rho * exp_cfg.gravity * d_lat) \
        <= 1e-12 * exp_cfg.lattice_u ** 2
    actual_g_ok = abs(actual_gravity - exp_cfg.gravity) <= 1e-12 * abs(exp_cfg.gravity)
    nu_lat = exp_cfg.lattice_u * d_lat / exp_cfg.Re
    status = "ok" if (re_ok and g_ok and actual_g_ok) else "collision_viscosity_mismatch"
    return {
        "viscosity_mapping_contract_status": status,
        "nu_phys_target": nu_phys_target,
        "nu_lattice_resolved": nu_lat,
        "collision_relaxation_parameter_resolved": 3.0 * nu_lat + 0.5,
        "collision_viscosity_source_id": COLLISION_VISCOSITY_SOURCE_ID,
        "gravity_mapping_id": "PRESET_AR_NU_MAPPING_V1",
        "g_lattice_resolved": exp_cfg.gravity,
        "reynolds_viscosity_source_id": COLLISION_VISCOSITY_SOURCE_ID,  # 동일 단일 소스
        # numpy.bool_ → JSON 직렬화 실패 방지 (값 불변 캐스팅)
        "re_target_check": bool(re_ok), "gravity_selfconsistency_check": bool(g_ok),
        "actual_gravity_match": bool(actual_g_ok),
    }


def expected_initial_particle_spec(which: str, case_id: str) -> dict:
    """N-7 — 동결 builder가 생성한 expected initial specification 해시.

    실제 t=0 상태 파일 해시가 아니라 선언 사양 해시임을 명명으로 구분한다.
    """
    _case, exp = expected_materialized_config(which, case_id)
    parts = getattr(exp, "particles_config", None)
    if parts:
        spec = [[i, dict(p) if isinstance(p, dict) else repr(p)] for i, p in enumerate(parts)]
    else:  # 단일 입자 계열 — rho·직경·초기 중심은 cfg 스칼라
        spec = [[0, {"rho_ratio": exp.rho_ratio, "cylinder_D_ratio": exp.cylinder_D_ratio}]]
    blob = json.dumps(spec, sort_keys=True, default=repr)
    return {
        "expected_initial_particle_spec_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "expected_particle_count": len(spec),
    }


# ---------------------------------------------------------------- config 계층
def load_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text())


def resolved_config(status: dict) -> dict:
    cfg = status.get("config")
    if not isinstance(cfg, dict) or not cfg:
        raise AssertionError("config_block_degenerate")
    missing = [k for k in REQUIRED_PHYSICS_FIELDS if k not in cfg]
    if missing:
        raise AssertionError(f"config_block_degenerate:missing={missing}")
    return dict(cfg)


# schema 판정 정본 = 동결 keyset 해시. presence 패턴은 2차 consistency
# 검사(손상 legacy가 current로 강등되는 경로 차단). 암묵 default(get(..., False)) 금지.
_LEGACY_SCHEMA_KEYS = ("use_added_mass", "imc_method")
# legacy 유효 tuple = (settling_inertia_model, use_added_mass, imc_method).
# imc_method는 legacy 코드의 semantic 필드(실측: feng_b2 런은 imc=sim=feng_b2 연동) —
# ablation pair decoder에서는 "none"만 유효, 존재만 확인하고 값을 방치하지 않는다.
_LEGACY_VALID = {
    ("explicit_history", True, "none"): "explicit_history",
    ("none", False, "none"): "none",
}
_CURRENT_VALID = ("explicit_history", "none")


def closure_schema(cfg: dict) -> str:
    ks = keyset_sha256(cfg)
    if ks == LEGACY_KEYSET_SHA256:
        schema = "legacy"
    elif ks == CURRENT_KEYSET_SHA256:
        schema = "current"
    else:
        raise AssertionError("closure_encoding_invalid")  # 미등재 keyset — 판정 확장 금지
    present = tuple(k in cfg for k in _LEGACY_SCHEMA_KEYS)
    expected_present = (True, True) if schema == "legacy" else (False, False)
    if present != expected_present:  # keyset과 presence 패턴의 상호 consistency
        raise AssertionError("closure_encoding_invalid")
    return schema


def canonical_closure_role(cfg: dict) -> str:
    if "settling_inertia_model" not in cfg:
        raise AssertionError("closure_encoding_invalid")
    sim = cfg["settling_inertia_model"]
    schema = closure_schema(cfg)
    if schema == "legacy":
        combo = (sim, cfg["use_added_mass"], cfg["imc_method"])
        if combo not in _LEGACY_VALID:
            raise AssertionError("closure_encoding_invalid")
        return _LEGACY_VALID[combo]
    if sim not in _CURRENT_VALID:
        raise AssertionError("closure_encoding_invalid")
    return sim


def canonicalize_config(cfg: dict) -> dict:
    """closure 관련 키를 전부 소비해 canonical subobject로 환원한다.

    decoder가 소비하는 키: settling_inertia_model + (legacy) use_added_mass.
    imc_method는 closure가 아니라 IBM 내부질량보정 방식 키 — 값 그대로 보존한다.
    """
    role = canonical_closure_role(cfg)
    consumed = {"settling_inertia_model"}
    if closure_schema(cfg) == "legacy":
        consumed.add("use_added_mass")
    out = {k: v for k, v in cfg.items() if k not in consumed}
    out["internal_fluid_closure"] = {"role": role}
    return out


def masked_config_sha256(cfg: dict) -> str:
    canon = canonicalize_config(cfg)
    canon["internal_fluid_closure"] = {"role": ABLATION_SENTINEL}
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def config_sha256(cfg: dict) -> str:
    blob = json.dumps(canonicalize_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def keyset_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(sorted(cfg.keys())).encode()).hexdigest()


# ---------------------------------------------------------------- pair 무결성
def assert_pair_integrity(arm_a: dict, arm_b: dict) -> dict:
    arms = [arm_a, arm_b]
    if arm_a["run_id"] == arm_b["run_id"]:
        raise AssertionError("duplicate_run_reference")
    roles = {a["role"] for a in arms}
    if roles != {"explicit_history", "none"}:
        raise AssertionError("duplicate_arm" if len(roles) == 1 else "missing_arm")
    for a in arms:
        if canonical_closure_role(a["config"]) != a["role"]:
            raise AssertionError("arm_label_config_mismatch")
    m_a, m_b = masked_config_sha256(arm_a["config"]), masked_config_sha256(arm_b["config"])
    if m_a != m_b:
        raise AssertionError("unexpected_config_difference")
    ks_a, ks_b = keyset_sha256(arm_a["config"]), keyset_sha256(arm_b["config"])
    if ks_a != ks_b:
        raise AssertionError("unexpected_config_difference")
    return {
        "pair_integrity_status": "ok",
        "masked_pair_config_sha256": m_a,
        "resolved_config_keyset_sha256": ks_a,
        "eh_resolved_config_sha256": config_sha256(
            arm_a["config"] if arm_a["role"] == "explicit_history" else arm_b["config"]),
        "none_resolved_config_sha256": config_sha256(
            arm_a["config"] if arm_a["role"] == "none" else arm_b["config"]),
    }


# ---------------------------------------------------------------- 동결 extractor
def load_history(run_dir: Path) -> list:
    recs = json.loads((run_dir / "sedimentation_history.json").read_text())
    if not recs:
        raise AssertionError("missing_samples")
    return recs


def _speed(vx: float, vy: float) -> float:
    return math.sqrt(vx * vx + vy * vy)


def series_two_particle(recs: list, particle_index: int, ystar_limit: float,
                        d_domain: float) -> dict:
    """다입자 raw trace → admissible window |v| 시계열. 좌표·직경 = 물리 단위.

    admissible 규격: 첫 contact crossing 전 + y*≤limit,
    유한값·단조 step. 절단은 위반이 아님. isolated(1입자) 런은 contact 자연 부재.
    """
    x0 = [p["x"] for p in recs[0]["particles"]]
    kept, t_star, steps, prev_step = [], [], [], None
    n_violation = 0
    contact_truncated = False
    window_end_reason = "run_end"  # 절단 분기에서 갱신
    window_end_trigger_set: list = []  # run_end면 빈 집합
    min_center_distance = None  # admissible 창 내 최소 중심거리 (물리)
    center_distances = []  # kept와 병렬 (peak 시점 gap 산출용, 2입자만)
    for rec in recs:
        ps = rec["particles"]
        vals = [ps[i][k] for i in range(len(ps)) for k in ("x", "y", "vx", "vy")]
        if not all(math.isfinite(v) for v in vals):
            n_violation += 1
            continue
        if prev_step is not None and rec["step"] <= prev_step:
            n_violation += 1
            continue
        dist = (math.hypot(ps[0]["x"] - ps[1]["x"], ps[0]["y"] - ps[1]["y"])
                if len(ps) >= 2 else None)
        ystar = (ps[particle_index]["x"] - x0[particle_index]) / d_domain
        # 종료 trigger를 같은 표본에서 전부 평가해 보존.
        # 선택·min 갱신·kept 채택 규칙은 기존과 동일(동작 보존 — 공개값 재현성)
        triggers = []
        if dist is not None and dist <= d_domain:  # contact crossing (물리 좌표)
            triggers.append("contact")
        if ystar > ystar_limit:
            triggers.append("y_star_limit")
        if "contact" in triggers:  # precedence: CONTACT_BEFORE_YSTAR_V1
            contact_truncated = True
            window_end_reason = "contact"
            window_end_trigger_set = triggers
            break
        if dist is not None and (min_center_distance is None or dist < min_center_distance):
            min_center_distance = dist
        if triggers:  # y_star_limit 단독
            window_end_reason = "y_star_limit"
            window_end_trigger_set = triggers
            break
        kept.append(_speed(ps[particle_index]["vx"], ps[particle_index]["vy"]))
        center_distances.append(dist)
        t_star.append(ps[particle_index]["t_star"])
        steps.append(int(rec["step"]))
        prev_step = rec["step"]
    if not kept:
        raise AssertionError("missing_samples")
    return {"speed": kept, "t_star": t_star, "steps": steps, "n_input": len(recs),
            "n_admissible": len(kept), "n_violation": n_violation,
            "contact_truncated": contact_truncated,
            "min_center_distance": min_center_distance,
            "center_distances": center_distances,
            "window_rule_id": WINDOW_RULE_TWO_ARM,
            "window_end_reason": window_end_reason,
            "window_end_trigger_set": window_end_trigger_set,
            "window_end_precedence_rule_id": WINDOW_END_PRECEDENCE,
            "window_end_step": steps[-1], "window_end_t_star": t_star[-1]}


def series_single_particle(recs: list, ystar_limit: float) -> dict:
    kept, t_star, steps, prev_step = [], [], [], None
    n_violation = 0
    window_end_reason = "run_end"
    window_end_trigger_set: list = []
    for rec in recs:
        vals = [rec[k] for k in ("x", "y", "vx", "vy", "y_star")]
        if not all(math.isfinite(v) for v in vals):
            n_violation += 1
            continue
        if prev_step is not None and rec["step"] <= prev_step:
            n_violation += 1
            continue
        if rec["y_star"] > ystar_limit:
            window_end_reason = "y_star_limit"
            window_end_trigger_set = ["y_star_limit"]
            break
        kept.append(_speed(rec["vx"], rec["vy"]))
        t_star.append(rec["t_star"])
        steps.append(int(rec["step"]))
        prev_step = rec["step"]
    if not kept:
        raise AssertionError("missing_samples")
    return {"speed": kept, "t_star": t_star, "steps": steps, "n_input": len(recs),
            "n_admissible": len(kept), "n_violation": n_violation,
            "contact_truncated": False, "min_center_distance": None,
            "center_distances": None,
            "window_rule_id": WINDOW_RULE_SINGLE_ARM,
            "window_end_reason": window_end_reason,
            "window_end_trigger_set": window_end_trigger_set,
            "window_end_precedence_rule_id": WINDOW_END_PRECEDENCE,
            "window_end_step": steps[-1], "window_end_t_star": t_star[-1]}


def declared_window_peak(series: dict) -> float:
    return max(series["speed"])


def normalized_pct(series: dict) -> list:
    pk = declared_window_peak(series)
    return [v / pk * 100.0 for v in series["speed"]]


def first_record_provenance(recs: list) -> dict:
    """diagnostic provenance 전용 (pair-integrity gate 사용 금지)."""
    r0 = recs[0]
    if "particles" in r0:
        payload = [[p.get("id", i), p.get("rho_ratio"), p["x"], p["y"], p["vx"], p["vy"]]
                   for i, p in enumerate(r0["particles"])]
        count = len(r0["particles"])
        rho = [p.get("rho_ratio") for p in r0["particles"]]
    else:
        payload = [[0, None, r0["x"], r0["y"], r0["vx"], r0["vy"]]]
        count, rho = 1, [None]
    blob = json.dumps([r0["step"], payload], sort_keys=True)
    return {
        "first_record_particle_state_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "first_record_step": int(r0["step"]),
        "particle_count": count,
        "particle_rho_ratios": rho,
        "angular_velocity_included": "omega" in r0,
    }


# ---------------------------------------------------------------- N-4 exact tick contract
def exact_tick_alignment(steps_coarse: list, steps_fine: list,
                         ratio: Fraction, expected_count: int,
                         tstar_coarse: list = None, tstar_fine: list = None) -> dict:
    """κ² 유리수 관계의 exact integer cross-product 검사 (ε_t 불요).

    관계: step_fine[i] · ratio.denominator == step_coarse[i] · ratio.numerator,
    ratio = (s_fine / s_coarse) = κ² (예: 720/500 = 36/25, 1280/500 = 64/25).
    검사: cardinality·순증가·결손 없음·전 표본 exact tick. float t* 최대 차는 진단.
    """
    out = {"interpolation_applied": False,
           "native_sample_count": [len(steps_coarse), len(steps_fine)],
           "expected_sample_count": expected_count}
    p, q = ratio.numerator, ratio.denominator

    def _fail(reason):
        out.update({"exact_tick_alignment_status": "cadence_alignment_failure",
                    "alignment_failure_reason": reason})
        return out

    if len(steps_coarse) != expected_count or len(steps_fine) != expected_count:
        return _fail("sample_count_mismatch")
    for seq in (steps_coarse, steps_fine):
        if any(seq[i + 1] <= seq[i] for i in range(len(seq) - 1)):
            return _fail("non_monotone_steps")
    for i in range(expected_count):
        if steps_fine[i] * q != steps_coarse[i] * p:
            return _fail(f"tick_mismatch_at_{i}")
    out["exact_tick_alignment_status"] = "ok"
    out["first_common_tick"] = [steps_coarse[0], steps_fine[0]]
    out["last_common_tick"] = [steps_coarse[-1], steps_fine[-1]]
    if tstar_coarse and tstar_fine:
        n = min(len(tstar_coarse), len(tstar_fine))
        out["max_tstar_mismatch_diagnostic"] = max(
            abs(tstar_coarse[i] - tstar_fine[i]) for i in range(n))
    return out


def pair_cadence_identity(steps_eh: list, steps_none: list) -> dict:
    """동일 NN pair의 cadence 계약 — 공통 prefix의 exact 동일 step 격자.

    arm-local 창(ARM_LOCAL_PRECONTACT_V1)은 팔별 종료 시점이 정당하게 다르다 —
    창 길이 차이는 위반이 아니며 window_end 필드가 기록한다. 검사 대상은 표집
    격자 동일성(공통 prefix 전 표본 exact 일치)이다.
    """
    n = min(len(steps_eh), len(steps_none))
    ok = n > 0 and steps_eh[:n] == steps_none[:n]
    return {"pair_cadence_status": "ok" if ok else "cadence_alignment_failure",
            "pair_common_prefix_count": n,
            "interpolation_applied": False}


# ---------------------------------------------------------------- run integrity
def run_integrity(status: dict, recs: list, expected_interval: int = None) -> str:
    """arm-level full raw history 무결성 (공통 prefix cadence와 별개 계층).

    조기 파일 절단·결손·중복을 raw 층에서 검출한다: completed +
    termination_reason 도달 가능 어휘(스키마 층 — 계약 이행 판정은 frozen stop
    evaluate_termination_contract가 별도 수행) + status.history_len == 실제
    레코드 수 + step 시퀀스 등차(= check_interval) + 최소 표본.
    """
    if not status.get("completed", False):
        return "abnormal_termination"
    if len(recs) < 16:
        return "missing_samples"
    if status.get("termination_reason") not in REACHABLE_TERMINATION_REASONS:
        return "invalid_termination_reason"
    hl = status.get("history_len")
    if hl is not None and int(hl) != len(recs):
        return "history_length_mismatch"  # 파일 절단/부분 기록 검출
    if expected_interval:
        steps = [int(r["step"]) for r in recs]
        if steps[0] != expected_interval or any(
                steps[i + 1] - steps[i] != expected_interval for i in range(len(steps) - 1)):
            return "raw_step_grid_violation"  # 결손·중복·간격 이상
    return "ok"


# ---------------------------------------------------------------- termination 계약
def evaluate_termination_contract(runner: str, status: dict, window_end_reason: str) -> dict:
    """run별 frozen stop 계약 — 3층 분리.

    A. runtime 층은 termination metadata + frozen 계약만 사용한다(max_steps는
       창 완결과 무관하게 safety-cap). B. 창 완결 층은 자동 추출된 창 종료
       실측의 매핑이다(metadata가 아님을 명시). C. eligibility 층은
       두 층의 곱이며 차단 집합은 종전 정의와 같다(정책 불변, 명명 분리).
    """
    c = TERMINATION_CONTRACTS[runner]
    actual = status.get("termination_reason")
    out = {
        "termination_contract_id": c["termination_contract_id"],
        "expected_stop_predicate": c["expected_stop_predicate"],
        "allowed_termination_reasons": sorted(c["allowed_primary"]) + [
            "max_steps(safety cap — eligibility는 창 완결로 별도 판정)"],
        "actual_termination_reason": actual,
        "actual_terminal_step": status.get("final_step"),
    }
    if actual in c["allowed_primary"]:
        rt = "expected_stop_satisfied"
    elif actual == "max_steps":
        rt = "safety_cap_reached_before_expected_stop"
    elif actual not in REACHABLE_TERMINATION_REASONS:
        rt = "invalid_termination_metadata"  # run_integrity와 중복 방어
    else:
        rt = "unexpected_early_termination"
    out["runtime_termination_contract_status"] = rt
    # 창 완결 층 — complete_by_expected_runtime_stop·inadmissible_or_insufficient는
    # 예약 상태(현 창 정의에서 도달 불가 / extractor 예외가 담당)
    wc = {"y_star_limit": "complete_by_y_star_limit",
          "contact": "complete_by_geometric_contact",
          "run_end": "run_end_limited"}.get(window_end_reason,
                                            "inadmissible_or_insufficient")
    out["analysis_window_completion_status"] = wc
    eligible = (rt in ("expected_stop_satisfied",
                       "safety_cap_reached_before_expected_stop")
                and wc in ("complete_by_y_star_limit",
                           "complete_by_geometric_contact"))
    out["metric_eligibility_status"] = (
        "eligible_declared_finite_window" if eligible else "blocked")
    return out


def verify_stop_predicate(runner: str, status: dict, recs: list, exp) -> dict:
    """termination_reason과 실상태의 일치 재계산 QA.

    provenance는 reason별 발생 경로로 분리한다: should_stop 계열은 같은
    스텝의 마지막 저장 레코드에서 signed margin으로 재계산하고(round-trip 보존
    실측), max_steps는 terminal step 메타데이터를 대조한다. 술어는
    sedimentation_stop_reason·loop.py 원식 그대로, 상수는 frozen materialized
    config에서 취득. gravity_direction이 실측 전제(two=right·single=down)와
    다르면 재계산 자동 일반화 금지.
    """
    actual = status.get("termination_reason")
    out = {"termination_reason_origin_id": TERMINATION_ORIGIN_IDS.get(
               actual, "unknown_reason"),
           "history_coordinate_encoding_id": HISTORY_COORDINATE_ENCODING_ID,
           "history_coordinate_roundtrip_status": HISTORY_COORDINATE_ROUNDTRIP_STATUS,
           "stop_predicate_recomputed": True,
           "stop_predicate_margin_raw": None}
    last = recs[-1]
    if actual == "max_steps":
        out["predicate_recompute_state_id"] = PREDICATE_RECOMPUTE_MAX_STEPS
        out["stop_evaluation_step"] = status.get("final_step")
        ok = int(status.get("final_step") or -1) == int(exp.max_steps)
    elif actual == "offset" and runner == "two":
        out["predicate_recompute_state_id"] = PREDICATE_RECOMPUTE_SHOULD_STOP
        if str(getattr(exp, "gravity_direction", "")) != "right":
            out.update({"stop_predicate_recomputed": False,
                        "stop_predicate_consistency_status": "predicate_form_unregistered"})
            return out
        out["stop_evaluation_step"] = int(last["step"])
        # isolated 1입자(two 러너)는 single 레코드 스키마: x 직접 필드
        p0_x = (float(last["particles"][0]["x"]) if "particles" in last
                else float(last["x"]))
        # margin ≥ 0 ⇔ 원식 x ≥ xmax − s·D
        margin = p0_x - (float(exp.xmax)
                         - float(exp.sedimentation_stop_offset_d)
                         * float(exp.cylinder_D_ratio))
        out["stop_predicate_margin_raw"] = margin
        ok = margin >= 0.0
    elif actual == "contact" and runner == "single":
        out["predicate_recompute_state_id"] = PREDICATE_RECOMPUTE_SHOULD_STOP
        if str(getattr(exp, "gravity_direction", "down")) != "down":
            out.update({"stop_predicate_recomputed": False,
                        "stop_predicate_consistency_status": "predicate_form_unregistered"})
            return out
        out["stop_evaluation_step"] = int(last["step"])
        # margin ≥ 0 ⇔ 원식 y ≤ r = D/2 (runtime.scenarios.sedimentation_common
        # 의 접촉 판정과 동일. r = D/2 는 init.initialize() 정의)
        margin = 0.5 * float(exp.cylinder_D_ratio) - float(last["y"])
        out["stop_predicate_margin_raw"] = margin
        ok = margin >= 0.0
    else:
        # 차단 사유(nan·domain_bounds·converged 계열)는 runtime 층이 이미 차단 —
        # 재계산 생략(발생 경로 provenance는 위 origin_id로 정확 기록)
        out.update({"predicate_recompute_state_id": None,
                    "stop_predicate_recomputed": False,
                    "stop_predicate_consistency_status": "not_applicable_blocked_reason"})
        return out
    out["stop_predicate_consistency_status"] = (
        "ok" if ok else "termination_predicate_mismatch")
    return out


# ---------------------------------------------------------------- paired 조립 (전 규칙 합류)
def build_pair_row(comparison_id: str, cdef: dict, *, base: Path = DATA) -> dict:
    eh_dir, none_dir = base / cdef["eh_dir"], base / cdef["none_dir"]
    st_eh, st_none = load_status(eh_dir), load_status(none_dir)
    cfg_eh, cfg_none = resolved_config(st_eh), resolved_config(st_none)

    row = {
        "comparison_id": comparison_id,
        "family": cdef["family"],
        "pairing_class": cdef["pairing_class"],
        "schema_version": SCHEMA_VERSION,
        "manifest_ref": MANIFEST_REF,
        "observable_formula_id": OBSERVABLE_FORMULA_ID,
        "paired_shift_definition_id": PAIRED_SHIFT_ID,
        "sampling_alignment_rule_id": SAMPLING_ALIGNMENT_RULES.get(cdef["family"]),
        "contact_crossing_rule_id": CONTACT_CROSSING_RULE if cdef["kind"] == "two_particle" else None,
        "coordinate_system_id": CONTACT_RULE_SPEC["coordinate_system_id"],
        "paired_window_policy": CONTACT_RULE_SPEC["paired_window_policy"],
        "admissibility_helper_sha256": "table_builder_sha256",  # canonical helper = 본 builder 단일
        "claim_generation_allowed": False,
    }

    row["analysis_window_definition_id"] = cdef["window_def"]  # canonical 창 ID

    # 상태 계층 순서: integrity → expected contract → admissibility → consensus → pair
    recs_eh, recs_none = load_history(eh_dir), load_history(none_dir)
    ri_eh = run_integrity(st_eh, recs_eh, int(cfg_eh.get("check_interval") or 0))
    ri_none = run_integrity(st_none, recs_none, int(cfg_none.get("check_interval") or 0))
    row["run_integrity_status_eh"], row["run_integrity_status_none"] = ri_eh, ri_none
    if ri_eh != "ok" or ri_none != "ok":
        row["signed_ablation_shift_pct"] = None
        return row

    # expected run 계약 — pair equality와 독립
    ec_eh = expected_run_contract(cdef["runner"], cdef["eh_case"], cfg_eh)
    ec_none = expected_run_contract(cdef["runner"], cdef["none_case"], cfg_none)
    row["expected_run_contract_eh"], row["expected_run_contract_none"] = ec_eh, ec_none
    if (ec_eh["expected_run_contract_status"] != "ok"
            or ec_none["expected_run_contract_status"] != "ok"):
        row["signed_ablation_shift_pct"] = None
        return row

    # N-6 viscosity end-to-end 계약 — run-level blocking assertion
    if cdef.get("viscosity_contract"):
        case_v, exp_v = expected_materialized_config(cdef["runner"], cdef["eh_case"])
        nu_target = case_v.get("nu_phys_override")
        if nu_target is None:
            row["expected_run_contract_eh"] = {"expected_run_contract_status": "contract_mismatch",
                                               "reason": "nu_phys_override_absent_in_registry"}
            row["signed_ablation_shift_pct"] = None
            return row
        for tag, cfg in (("eh", cfg_eh), ("none", cfg_none)):
            vc = verify_viscosity_chain(exp_v, float(nu_target), float(cfg["gravity"]))
            row[f"viscosity_contract_{tag}"] = vc
            if vc["viscosity_mapping_contract_status"] != "ok":
                row["signed_ablation_shift_pct"] = None
                return row

    # N-7 light identity: expected spec 해시 + 첫 레코드 증거 결합
    fr_eh, fr_none = first_record_provenance(recs_eh), first_record_provenance(recs_none)
    row["first_record_eh"], row["first_record_none"] = fr_eh, fr_none
    if "light_identity" in cdef:
        li = cdef["light_identity"]
        row.update(expected_initial_particle_spec(cdef["runner"], cdef["eh_case"]))
        for tag, fr, cfg in (("eh", fr_eh, cfg_eh), ("none", fr_none, cfg_none)):
            ok = (fr["particle_count"] == li["count"]
                  and all(abs(r - li["rho_ratio"]) <= FLOAT_TOL for r in fr["particle_rho_ratios"] if r is not None)
                  and abs(cfg["rho_ratio"] - li["rho_ratio"]) <= FLOAT_TOL)
            if not ok:
                row[f"expected_run_contract_{tag}"] = {
                    "expected_run_contract_status": "light_identity_mismatch"}
                row["signed_ablation_shift_pct"] = None
                return row
        row["light_identity_status"] = "ok"

    # pair 무결성 (shift 계산 전 최선행)
    try:
        prov = assert_pair_integrity(
            {"run_id": str(eh_dir), "role": "explicit_history", "config": cfg_eh},
            {"run_id": str(none_dir), "role": "none", "config": cfg_none},
        )
    except AssertionError as exc:
        row.update({"pair_integrity_status": str(exc), "signed_ablation_shift_pct": None})
        return row
    row.update(prov)

    # 동결 extractor — 물리 직경은 frozen builder materialized cfg에서 취득.
    # N-7(isolated 1입자, two 러너)은 single 레코드 스키마 → single 경로
    _case, exp = expected_materialized_config(cdef["runner"], cdef["eh_case"])
    if cdef["kind"] == "two_particle" and cdef.get("record_schema") != "single":
        d_domain = float(exp.cylinder_D_ratio)  # 물리 채널 폭 1 기준 직경
        pidx = 1 if (cdef["particle"] == "light" and len(recs_eh[0]["particles"]) > 1) else 0
        s_eh = series_two_particle(recs_eh, pidx, cdef["ystar_limit"], d_domain)
        s_none = series_two_particle(recs_none, pidx, cdef["ystar_limit"], d_domain)
    else:
        s_eh = series_single_particle(recs_eh, cdef["ystar_limit"])
        s_none = series_single_particle(recs_none, cdef["ystar_limit"])

    for tag, s in (("eh", s_eh), ("none", s_none)):
        row[f"admissible_{tag}"] = {k: s[k] for k in ("n_input", "n_admissible", "n_violation", "contact_truncated")}
        row[f"arm_{tag}_window_rule_id"] = s["window_rule_id"]
        row[f"arm_{tag}_window_end_step"] = s["window_end_step"]
        row[f"arm_{tag}_window_end_t_star"] = s["window_end_t_star"]
        row[f"arm_{tag}_window_end_reason"] = s["window_end_reason"]
        row[f"arm_{tag}_window_end_trigger_set"] = s["window_end_trigger_set"]
        row[f"arm_{tag}_contact_detected"] = s["contact_truncated"]
    row["window_end_precedence_rule_id"] = WINDOW_END_PRECEDENCE
    row["comparison_scope"] = "arm_local_finite_window"  # paired row 기본 지위

    # 종료 3층(runtime 계약/창 완결/eligibility) + stop
    # predicate 일치 QA. 창 완결 층이 series 실측에 의존하므로 admissibility
    # 계산 뒤에 위치한다. 차단 집합은 앞 계층과 동일하다(정책 불변).
    row["configured_max_steps"] = int(exp.max_steps)
    _term_blocked = False
    for tag, st, s, rc in (("eh", st_eh, s_eh, recs_eh),
                           ("none", st_none, s_none, recs_none)):
        tc = evaluate_termination_contract(cdef["runner"], st, s["window_end_reason"])
        tc.update(verify_stop_predicate(cdef["runner"], st, rc, exp))
        row[f"termination_contract_{tag}"] = tc
        if (tc["metric_eligibility_status"] != "eligible_declared_finite_window"
                or tc["stop_predicate_consistency_status"]
                in ("termination_predicate_mismatch", "predicate_form_unregistered")):
            _term_blocked = True
    if _term_blocked:
        row["signed_ablation_shift_pct"] = None
        return row

    # kernel-overlap 보수 판정, stored-sample scope 한정
    # (두 입자 실재 시에만; N-7 isolated는 해당 없음)
    if cdef["kind"] == "two_particle":
        gaps = [s["min_center_distance"] for s in (s_eh, s_none)
                if s["min_center_distance"] is not None]
        if gaps:
            # 도메인 격자 간격 정본 = 1/(NN−1) (init.initialize() 와 동일 —
            # SimConfig에 dx 속성 없음. S_801 공개값 42.0Δx = 0.0525/0.00125와 정합)
            dx_domain = 1.0 / (float(exp.NN) - 1.0)
            d_min = min(gaps)
            row["kernel_id"] = str(exp.delta_type)
            row["min_surface_gap_over_d"] = (d_min - d_domain) / d_domain
            row["min_surface_gap_dx"] = (d_min - d_domain) / dx_domain
            row["gap_sampling_scope"] = GAP_SAMPLING_SCOPE
            row["history_sampling_interval_steps"] = int(cfg_eh.get("check_interval") or 0)
            row["continuous_time_overlap_exclusion"] = "not_evaluated"  # per-step 기록 부재
            # 각 팔의 declared peak 표본에서의 surface gap (Δx 단위)
            for tag, s in (("eh", s_eh), ("none", s_none)):
                pk_i = max(range(len(s["speed"])), key=lambda i: s["speed"][i])
                d_pk = s["center_distances"][pk_i]
                row[f"surface_gap_at_{tag}_peak_dx"] = (
                    None if d_pk is None else (d_pk - d_domain) / dx_domain)
            # marker 반경 하한 전제 (create_circle_markers 구성 보장:
            # 마커 반경 = r_domain − retraction_dx·dx ≤ r_domain ⇔ retraction_dx ≥ 0)
            retr = float(getattr(exp, "retraction_dx", 0.0))
            row["marker_retraction_dx"] = retr
            row["marker_radius_bound_status"] = (
                "guaranteed_by_construction" if retr >= 0.0 else "not_guaranteed")
            if str(exp.delta_type) == "peskin4pt":
                row["kernel_overlap_test_id"] = KERNEL_OVERLAP_TEST_ID
                row["kernel_support_halfwidth_cells"] = KERNEL_SUPPORT_HALFWIDTH_CELLS
                row["kernel_clearance_threshold_dx"] = KERNEL_CLEARANCE_THRESHOLD_DX
                if row["marker_radius_bound_status"] != "guaranteed_by_construction":
                    row["kernel_overlap_excluded_at_stored_samples"] = None  # 전제 미보장
                else:
                    row["kernel_overlap_excluded_at_stored_samples"] = (
                        row["min_surface_gap_dx"] > KERNEL_CLEARANCE_THRESHOLD_DX)
            else:  # 미등재 kernel — 판정 금지
                row["kernel_overlap_test_id"] = "not_registered_for_kernel"
                row["kernel_overlap_excluded_at_stored_samples"] = None
    row.update(pair_cadence_identity(s_eh["steps"], s_none["steps"]))
    if row["pair_cadence_status"] != "ok":
        row["signed_ablation_shift_pct"] = None
        return row

    # completion·observable 판정 — dual ε consensus
    j_eh, j_none = dec.judge(normalized_pct(s_eh)), dec.judge(normalized_pct(s_none))
    row["arm_eh_judgment"], row["arm_none_judgment"] = j_eh, j_none
    row["paired_claim_scope"] = dec.paired_claim_scope(j_eh, j_none)
    row["paired_observable_status"] = dec.paired_observable(j_eh, j_none)

    # raw shift·direction·magnitude (정본 함수 호출만)
    pk_eh, pk_none = declared_window_peak(s_eh), declared_window_peak(s_none)
    s_raw = dec.signed_ablation_shift_pct(pk_eh, pk_none)
    row.update({
        "eh_peak_raw": pk_eh, "none_peak_raw": pk_none,
        "signed_shift_raw": s_raw,
        "ablation_direction": dec.ablation_direction(s_raw),
        "ablation_magnitude_pct": dec.ablation_magnitude_pct(pk_eh, pk_none),
        "signed_shift_display": round(s_raw, 2),
    })
    assert abs(pk_none - pk_eh * (1 + s_raw / 100.0)) <= FLOAT_TOL * max(1.0, abs(pk_none))

    # COMMON_EARLIEST_WINDOW 비판정 sensitivity (claim-action 미사용)
    t_end_common = min(s_eh["window_end_t_star"], s_none["window_end_t_star"])
    pk_eh_c = max(v for v, t in zip(s_eh["speed"], s_eh["t_star"]) if t <= t_end_common)
    pk_none_c = max(v for v, t in zip(s_none["speed"], s_none["t_star"]) if t <= t_end_common)
    row["non_decisional_common_window_sensitivity"] = {
        "window_rule_id": "COMMON_EARLIEST_WINDOW_V1",
        "window_end_t_star": t_end_common,
        "signed_shift_pct": dec.signed_ablation_shift_pct(pk_eh_c, pk_none_c),
        "role": "non_decisional_sensitivity",
    }

    row["q_use"] = cdef["q_use"]
    if cdef["family"] == "N-4":  # N-4 tick 게이트 입력 (JSON 출력 전 제거)
        row["_steps_eh"] = s_eh["steps"]
        row["_n_records_full"] = len(recs_eh)
    return row


def attach_q(row: dict, s_pair_raw_reference: float, anchor_window_rule_id: str,
             control_window_rule_id: str) -> dict:
    assert row["q_use"], "q_use=False comparison에 q 부착 금지"
    assert abs(s_pair_raw_reference) > Q_DENOM_FROZEN, "conservativeness precondition"
    s_ctrl = row["signed_shift_raw"]
    q_dec = dec.effect_size_q_decision(s_ctrl)
    q_raw = dec.effect_size_q_raw_reference(s_ctrl, s_pair_raw_reference)
    row.update({
        "q_decision": q_dec,
        "q_raw_reference": q_raw,
        "q_denominator_frozen_pct": Q_DENOM_FROZEN,
        "q_denominator_raw_pct": abs(s_pair_raw_reference),
        "q_definition_id": Q_DEFINITION_ID,
        "q_comparability_scope": Q_COMPARABILITY_SCOPE,
        "anchor_window_rule_id": anchor_window_rule_id,
        "control_window_rule_id": control_window_rule_id,
        "q_anchor_relation_check": {
            "A_raw_gt_A_frozen": abs(s_pair_raw_reference) > Q_DENOM_FROZEN,
            "q_decision_ge_q_raw_reference": q_dec >= q_raw,
        },
    })
    return row


# ---------------------------------------------------------------- anchor 재산출
ANCHOR_WINDOW_RULE_ID = "TWO_PARTICLE_50D_FULL_HORIZON_V1"


def _full_horizon_peak(recs: list, particle_index: int) -> float:
    """anchor 전용 extractor — 사전 고정 계약의 window(전 구간, monitor 동일).

    신규 admissible 규격(contact 절단 등)을 기존 보고값에 소급 적용하지 않는다
    (manifest 분류기 절 '기존 보고값 재라벨링에 소급 비적용'). 무결성 검사(유한값·
    단조 step)만 수행한다.
    """
    peak, prev_step = 0.0, None
    for rec in recs:
        p = rec["particles"][particle_index]
        if not all(math.isfinite(v) for v in (p["vx"], p["vy"])):
            continue
        if prev_step is not None and rec["step"] <= prev_step:
            continue
        peak = max(peak, _speed(p["vx"], p["vy"]))
        prev_step = rec["step"]
    if peak <= 0.0:
        raise AssertionError("missing_samples")
    return peak


def recompute_anchor_pair(base: Path = DATA) -> dict:
    """50D pair anchor의 raw 재산출 — 기존 공개 데이터(확인 런 아님)."""
    eh_dir, none_dir = base / ANCHOR_PAIR["eh_dir"], base / ANCHOR_PAIR["none_dir"]
    pk_eh = _full_horizon_peak(load_history(eh_dir), 1)
    pk_none = _full_horizon_peak(load_history(none_dir), 1)
    s_raw = dec.signed_ablation_shift_pct(pk_eh, pk_none)
    out = {"s_pair_raw_reference": s_raw,
           "anchor_window_rule_id": ANCHOR_WINDOW_RULE_ID,
           "anchor_peak_eh_raw": pk_eh, "anchor_peak_none_raw": pk_none}
    # 비판정 sensitivity diagnostic (값 불변): 창 종료 원인이
    # contact가 아니라 y_star_limit임을 실측(기존 18런 접촉 미발생) → 정본 명명은
    # harmonized admissible-window reference. 구명은 legacy_name으로 병기.
    try:
        _c, exp_a = expected_materialized_config("two", "TWO_PARTICLE_DF_BGK_VERLET_EXPLICIT_HISTORY")
        d_dom = float(exp_a.cylinder_D_ratio)
        s_eh_c = series_two_particle(load_history(eh_dir), 1, ANCHOR_PAIR["ystar_limit"], d_dom)
        s_no_c = series_two_particle(load_history(none_dir), 1, ANCHOR_PAIR["ystar_limit"], d_dom)
        out["harmonized_admissible_window_anchor_reference"] = dec.signed_ablation_shift_pct(
            declared_window_peak(s_eh_c), declared_window_peak(s_no_c))
        out["anchor_reference_window_definition_id"] = WIN_STD50_15P5
        out["anchor_reference_window_end_reasons"] = [
            s_eh_c["window_end_reason"], s_no_c["window_end_reason"]]
        out["anchor_reference_legacy_name"] = "contact_limited_anchor_shift_reference"
    except AssertionError:
        out["harmonized_admissible_window_anchor_reference"] = None
    return out


# ---------------------------------------------------------------- legacy harmonized 재산출
def compute_legacy_harmonized_pair(label: str, pdef: dict, role: str,
                                   *, base: Path = DATA) -> dict:
    """기존 공개 50D pair의 동일 규격(admissible) harmonized 재산출.

    legacy 런은 expected_run_contract 비적용 — 방어 계층: run_integrity +
    legacy keyset 정본 해시 + canonical closure tuple + 레지스트리 디렉터리 고정.
    기하 상수(d_domain·dx)는 legacy config에 cylinder_D_ratio가 없어(실측 23키)
    frozen 레지스트리의 df 참조 케이스 materialized config에서 취득한다(동일 기하).
    """
    eh_dir, none_dir = base / pdef["eh_dir"], base / pdef["none_dir"]
    st_eh, st_none = load_status(eh_dir), load_status(none_dir)
    cfg_eh, cfg_none = resolved_config(st_eh), resolved_config(st_none)
    out = {
        "comparison_id": label,
        "pairing_class": "legacy_reference_harmonized",
        "comparison_scope": "arm_local_finite_window",
        "role": role,  # "n4_reference_input" | "non_decisional_context"
        "contact_crossing_rule_id": CONTACT_CROSSING_RULE,
        "observable_formula_id": OBSERVABLE_FORMULA_ID,
        "paired_window_rule_id": WINDOW_RULE_TWO_ARM,
        "analysis_window_definition_id": pdef["window_def"],
        "claim_generation_allowed": False,
    }
    recs_eh, recs_none = load_history(eh_dir), load_history(none_dir)
    ri = (run_integrity(st_eh, recs_eh, int(cfg_eh.get("check_interval") or 0)),
          run_integrity(st_none, recs_none, int(cfg_none.get("check_interval") or 0)))
    out["run_integrity_status"] = list(ri)
    if ri != ("ok", "ok"):
        out["signed_shift_raw"] = None
        return out
    try:
        schemas = (closure_schema(cfg_eh), closure_schema(cfg_none))
        roles = (canonical_closure_role(cfg_eh), canonical_closure_role(cfg_none))
    except AssertionError:
        out.update({"closure_schema_status": "closure_encoding_invalid",
                    "signed_shift_raw": None})
        return out
    # 실측: df 50D pair = legacy 23키, mdf/dfc 50D pair = current 21키
    # (2키 동시 누락 퇴화와 외형 동일 — 방어는 레지스트리 디렉터리 고정 +
    # provenance. 두 팔 동일 스키마 + 유효 role tuple만 인정)
    if schemas[0] != schemas[1] or roles != ("explicit_history", "none"):
        out.update({"closure_schema_status": f"unexpected:{schemas}/{roles}",
                    "signed_shift_raw": None})
        return out
    out["closure_schema_status"] = "ok_same_schema_pair"
    out["closure_schemas"] = list(schemas)
    _case, exp = expected_materialized_config("two", "TWO_PARTICLE_DF_BGK_VERLET_EXPLICIT_HISTORY")
    d_dom = float(exp.cylinder_D_ratio)
    s_eh = series_two_particle(recs_eh, 1, pdef["ystar_limit"], d_dom)
    s_none = series_two_particle(recs_none, 1, pdef["ystar_limit"], d_dom)
    cad = pair_cadence_identity(s_eh["steps"], s_none["steps"])
    out.update(cad)
    if cad["pair_cadence_status"] != "ok":
        out["signed_shift_raw"] = None
        return out
    for tag, s in (("eh", s_eh), ("none", s_none)):
        out[f"arm_{tag}_window_rule_id"] = s["window_rule_id"]
        out[f"arm_{tag}_window_end_step"] = s["window_end_step"]
        out[f"arm_{tag}_window_end_reason"] = s["window_end_reason"]
        out[f"arm_{tag}_window_end_trigger_set"] = s["window_end_trigger_set"]
        out[f"arm_{tag}_contact_detected"] = s["contact_truncated"]
    # 계약은 confirmatory 런 전용 (방어 계층은 그대로 적용됨)
    out["runtime_termination_contract_status"] = "not_applicable_legacy_reference"
    pk_eh, pk_none = declared_window_peak(s_eh), declared_window_peak(s_none)
    out["signed_shift_raw"] = dec.signed_ablation_shift_pct(pk_eh, pk_none)
    out["_steps_eh"] = s_eh["steps"]  # N-4 tick 게이트용 (JSON 출력 전 제거)
    out["_n_records_full"] = len(recs_eh)
    return out


def n4_stability_block(s801_pack: dict, rows: list) -> dict:
    """N-4 3격자 안정 판정 (사전 고정 3조건).

    게이트: 5요소 동일성 — formula·extractor(단일 builder)·contact rule·
    window rule은 상수 동일이 구조 보장하므로 명시 기록하고, cadence 계약은
    full-history exact rational tick(801↔961=36/25, 801↔1281=64/25, cardinality
    120, 전 표본)으로 검사한다. 게이트 실패 → direct 판정 금지 + contextual 강등.
    """
    out = {
        "comparison_scope": "direct_harmonized_window",  # 내부 enum (표시 분리)
        "external_display_scope": N4_EXTERNAL_DISPLAY_SCOPE,
        "external_display_directness": N4_EXTERNAL_DISPLAY_DIRECTNESS,
        "grid_reference": s801_pack["comparison_id"],
        "stability_threshold_pp": N4_STABILITY_THRESHOLD_PP,
        "claim_generation_allowed": False,
        # solver lineage·directness (G-2 실측: eh 팔만 bridge)
        "solver_lineage_class": {"reference_801": "cross_version",
                                 "ny961": "current_version", "ny1281": "current_version"},
        "eh_lineage_bridge_status": N4_EH_LINEAGE,
        "none_lineage_bridge_status": N4_NONE_LINEAGE,
        "comparison_directness_grade": N4_DIRECTNESS_GRADE,
        "claim_scope_note": ("stability of the declared finite-window contrast under "
                             "the tested coupled space-time refinement; formal grid "
                             "convergence / code-version-independent convergence 금지"),
    }
    by_id = {r["comparison_id"]: r for r in rows}
    fine = {961: by_id.get("N-4-NY961"), 1281: by_id.get("N-4-NY1281")}
    win801 = s801_pack.get("analysis_window_definition_id")
    win_equal = all(
        fine[nn] is not None
        and fine[nn].get("analysis_window_definition_id") == win801
        for nn in fine) and win801 is not None
    compat = {
        "observable_formula_id_equal": True,   # 단일 상수 (구조 보장)
        "extractor_identical": True,           # canonical helper = 본 builder 단일 구현
        "contact_rule_id_equal": True,
        "analysis_window_definition_id_equal": win_equal,  # 창 전체 교집합 ID
    }
    s801 = s801_pack.get("signed_shift_raw")
    steps801 = s801_pack.get("_steps_eh")
    tick_ok = win_equal
    for nn, row in fine.items():
        if row is None or row.get("signed_shift_raw") is None or steps801 is None:
            tick_ok = False
            compat[f"tick_ny{nn}"] = {"exact_tick_alignment_status": "input_unavailable"}
            continue
        # full-history cardinality(120)는 별도 기록, tick은 admissible 공통 prefix 전 표본 검사
        steps_f = row.get("_steps_eh")
        if not steps_f:
            tick_ok = False
            compat[f"tick_ny{nn}"] = {"exact_tick_alignment_status": "fine_steps_unavailable"}
            continue
        n_common = min(len(steps801), len(steps_f))
        t = exact_tick_alignment(steps801[:n_common], steps_f[:n_common],
                                 N4_TICK_RATIOS[nn], n_common)
        t["checked_prefix_count"] = n_common
        t["native_full_record_count"] = [s801_pack.get("_n_records_full"),
                                         row.get("_n_records_full")]
        compat[f"tick_ny{nn}"] = t
        if t["exact_tick_alignment_status"] != "ok":
            tick_ok = False
    out["comparison_window_compatibility"] = compat
    allowed = tick_ok and s801 is not None and all(
        fine[nn] is not None and fine[nn].get("signed_shift_raw") is not None for nn in fine)
    out["direct_comparison_allowed"] = allowed
    out["metric_comparison_allowed"] = allowed  # directness 등급과 분리된 게이트 결과
    if not allowed:
        out["comparison_window_compatibility_status"] = "incompatible_or_incomplete"
        out["n4_stability_status"] = "contextual_only"  # 게이트 실패 시 801 비교는 맥락 참고로 강등
        return out
    out["comparison_window_compatibility_status"] = "ok"
    s961 = fine[961]["signed_shift_raw"]
    s1281 = fine[1281]["signed_shift_raw"]
    out["s_801_harmonized"] = s801
    out["s_961"] = s961
    out["s_1281"] = s1281
    sign_ok = (s801 < 0) == (s961 < 0) == (s1281 < 0) and s801 != 0 and s961 != 0 and s1281 != 0
    contraction_ok = abs(s1281 - s961) <= abs(s961 - s801)
    threshold_ok = abs(s1281 - s961) <= N4_STABILITY_THRESHOLD_PP
    out["n4_conditions"] = {"sign_consistent": sign_ok,
                           "contraction": contraction_ok,
                           "final_delta_within_threshold_pp": threshold_ok}
    out["n4_stability_status"] = ("stable_under_coupled_space_time_refinement"
                                  if (sign_ok and contraction_ok and threshold_ok)
                                  else "convergence_not_established")
    return out


# ---------------------------------------------------------------- selftest
# 실측 current 21키 전체(keyset 판정과 정합하도록 synthetic도 동일 키 집합 사용)
_CURRENT_KEYS_TEMPLATE = {
    "NN": 801, "check_interval": 500, "collision_model": "BGK",
    "delta_type": "peskin4pt", "enable_rotation": True,
    "gravity": 2.1559774464e-4, "gravity_direction": "right", "ibm_method": "DF",
    "incompressible_lbgk": False, "marker_spacing_factor": 0.83,
    "max_steps": 60000, "mdf_iterations": 1, "retraction_dx": 0.0,
    "rho_ratio": 1.5, "rotation_coupling": "semi_implicit",
    "sedimentation_euler_update_scheme": "new_velocity",
    "sedimentation_reference_basis": "standard",
    "sedimentation_stop_at_contact": False, "sedimentation_stop_offset_d": 2.0,
    "settling_inertia_model": "explicit_history", "time_integrator": "verlet",
}


def _mk_cfg(role: str, schema: str = "current", **over) -> dict:
    cfg = dict(_CURRENT_KEYS_TEMPLATE)
    cfg["settling_inertia_model"] = role
    if schema == "legacy":
        cfg["use_added_mass"] = (role == "explicit_history")
        cfg["imc_method"] = "none"
    cfg.update(over)
    return cfg


def _expect_invalid(cfg):
    try:
        canonical_closure_role(cfg)
    except AssertionError as e:
        return str(e)
    return None


def _selftest() -> None:
    # --- anchor 쌍·zero 순서
    assert round(dec.signed_ablation_shift_pct(238.62, 199.52), 4) == -16.3859
    assert round(dec.signed_ablation_shift_pct(199.52, 238.62), 4) == 19.5970
    assert dec.ablation_direction(dec.signed_ablation_shift_pct(200.0, 200.0)) == "zero"
    assert dec.ablation_direction(-1e-12) == "negative"

    # --- A. closure decoder (6사례 + schema 혼합)
    assert canonical_closure_role(_mk_cfg("explicit_history", "legacy")) == "explicit_history"
    assert canonical_closure_role(_mk_cfg("none", "legacy")) == "none"
    bad = _mk_cfg("explicit_history", "legacy"); bad["use_added_mass"] = False
    assert _expect_invalid(bad) == "closure_encoding_invalid"          # 모순 legacy 조합
    assert canonical_closure_role(_mk_cfg("explicit_history")) == "explicit_history"
    assert canonical_closure_role(_mk_cfg("none")) == "none"
    mixed = _mk_cfg("explicit_history"); mixed["use_added_mass"] = True  # 잡종 스키마
    assert _expect_invalid(mixed) == "closure_encoding_invalid"
    stray = _mk_cfg("explicit_history"); stray["imc_method"] = "none"    # legacy 키 단독 잔존
    assert _expect_invalid(stray) == "closure_encoding_invalid"
    assert _expect_invalid(_mk_cfg("feng_b2")) == "closure_encoding_invalid"  # unknown model
    nolabel = _mk_cfg("explicit_history"); del nolabel["settling_inertia_model"]
    assert _expect_invalid(nolabel) == "closure_encoding_invalid"
    # keyset 기반 판정 추가 사례
    imcbad = _mk_cfg("explicit_history", "legacy"); imcbad["imc_method"] = "feng_b2"
    assert _expect_invalid(imcbad) == "closure_encoding_invalid"   # imc 값 위반(존재만 확인 금지)
    onegone = _mk_cfg("explicit_history", "legacy"); del onegone["use_added_mass"]
    assert _expect_invalid(onegone) == "closure_encoding_invalid"  # 1키 누락 legacy → 미등재 keyset
    extra = _mk_cfg("explicit_history"); extra["Re"] = 247.6; extra["bc_type"] = "settling_channel"
    assert _expect_invalid(extra) == "closure_encoding_invalid"    # 25키 변형 → 미등재 keyset
    # 구조적 한계(정직 기록): legacy에서 2키가 동시 누락되면 keyset이 current와 동일 —
    # 정보상 구분 불가. 방어는 expected_run_contract(frozen registry 대조)와 비교 대상
    # 디렉터리 고정이 담당한다.
    twogone = _mk_cfg("explicit_history", "legacy")
    del twogone["use_added_mass"], twogone["imc_method"]
    assert canonical_closure_role(twogone) == "explicit_history"   # current로 해석됨(한계 문서화)

    # --- pair 5-assertion
    ok = assert_pair_integrity(
        {"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history")},
        {"run_id": "b", "role": "none", "config": _mk_cfg("none")})
    assert ok["pair_integrity_status"] == "ok"

    def _fail(a, b):
        try:
            assert_pair_integrity(a, b)
        except AssertionError as e:
            return str(e)
        return None
    assert _fail({"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history")},
                 {"run_id": "b", "role": "explicit_history", "config": _mk_cfg("explicit_history")}) == "duplicate_arm"
    assert _fail({"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history")},
                 {"run_id": "a", "role": "none", "config": _mk_cfg("none")}) == "duplicate_run_reference"
    assert _fail({"run_id": "a", "role": "explicit_history", "config": _mk_cfg("none")},
                 {"run_id": "b", "role": "none", "config": _mk_cfg("none")}) == "arm_label_config_mismatch"
    assert _fail({"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history")},
                 {"run_id": "b", "role": "none", "config": _mk_cfg("none", NN=961)}) == "unexpected_config_difference"
    assert _fail({"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history")},
                 {"run_id": "b", "role": "none", "config": _mk_cfg("none", "legacy")}) == "unexpected_config_difference"
    ok_leg = assert_pair_integrity(
        {"run_id": "a", "role": "explicit_history", "config": _mk_cfg("explicit_history", "legacy")},
        {"run_id": "b", "role": "none", "config": _mk_cfg("none", "legacy")})
    assert ok_leg["pair_integrity_status"] == "ok"

    # --- B. absolute run contract
    # B-1: N-7 frozen builder가 정확히 light 1개를 materialize
    case7, exp7 = expected_materialized_config("two", "TWO_PARTICLE_ISOLATED_LIGHT_EXPLICIT_HISTORY")
    parts7 = getattr(exp7, "particles_config", None)
    assert parts7 is not None and len(parts7) == 1, "N-7 expected particle count != 1"
    rho7 = parts7[0]["rho_ratio"] if isinstance(parts7[0], dict) else getattr(parts7[0], "rho_ratio")
    assert abs(rho7 - 1.25) <= FLOAT_TOL, "N-7 expected role is not light"
    spec7 = expected_initial_particle_spec("two", "TWO_PARTICLE_ISOLATED_LIGHT_EXPLICIT_HISTORY")
    assert spec7["expected_particle_count"] == 1
    # B-2: count=1이지만 heavy role synthetic → light_identity 실패 검출
    syn_fr = {"particle_count": 1, "particle_rho_ratios": [1.5]}
    li = {"count": 1, "rho_ratio": 1.25}
    ok_syn = (syn_fr["particle_count"] == li["count"]
              and all(abs(r - li["rho_ratio"]) <= FLOAT_TOL for r in syn_fr["particle_rho_ratios"]))
    assert not ok_syn, "heavy-as-light synthetic이 통과하면 안 됨"
    # B-3: N-6 override가 resolved mapping을 실제로 바꿈 (frozen builder end-to-end)
    case6, exp6 = expected_materialized_config("single", "SINGLE_RHO125_NU001_DF_BGK_VERLET_EXPLICIT_HISTORY")
    from scenarios.sedimentation import make_sedimentation_config
    exp6_no = make_sedimentation_config(rho_ratio=1.25)  # override 부재 기준
    assert abs(getattr(exp6, "gravity") - getattr(exp6_no, "gravity")) > 0, \
        "nu_phys_override가 g_lattice mapping을 바꾸지 않음 — end-to-end 소비 실패"
    # B-4: 보고 ν ≠ solver mapping synthetic → contract mismatch 검출
    syn_actual = {f: getattr(exp6, f) for f in EXPECTED_CONTRACT_FIELDS if hasattr(exp6, f)}
    syn_actual["gravity"] = syn_actual["gravity"] * 1.01  # mapping과 다른 g 보고
    ec = expected_run_contract("single", "SINGLE_RHO125_NU001_DF_BGK_VERLET_EXPLICIT_HISTORY", syn_actual)
    assert ec["expected_run_contract_status"] == "contract_mismatch"
    assert "gravity" in ec["contract_mismatches"]
    # B-5: viscosity 사슬 정상 — Re_target 독립 계산과 cfg.Re full-precision 일치
    nu_target = case6.get("nu_phys_override")
    assert nu_target == 0.01, f"registry nu_phys_override = {nu_target}"
    vc = verify_viscosity_chain(exp6, float(nu_target), float(exp6.gravity))
    assert vc["viscosity_mapping_contract_status"] == "ok", vc
    assert abs(vc["collision_relaxation_parameter_resolved"] - 0.5) > 0.0
    # B-6: collision 소스(cfg.Re)만 왜곡 → collision_viscosity_mismatch 검출
    import types
    twisted = types.SimpleNamespace(
        cylinder_D_ratio=exp6.cylinder_D_ratio, NN=exp6.NN, rho_ratio=exp6.rho_ratio,
        Re=exp6.Re * 1.01, lattice_u=exp6.lattice_u, gravity=exp6.gravity)
    vc_bad = verify_viscosity_chain(twisted, float(nu_target), float(exp6.gravity))
    assert vc_bad["viscosity_mapping_contract_status"] == "collision_viscosity_mismatch"

    # --- C. N-4 exact tick contract (5사례)
    r961 = Fraction(720, 500)    # 36/25 = κ²(6/5)
    ref = [500 * (i + 1) for i in range(120)]
    fine961 = [720 * (i + 1) for i in range(120)]
    fine1281 = [1280 * (i + 1) for i in range(120)]
    al = exact_tick_alignment(ref, fine961, r961, 120)
    assert al["exact_tick_alignment_status"] == "ok" and al["interpolation_applied"] is False
    al = exact_tick_alignment(ref, fine1281, Fraction(1280, 500), 120)
    assert al["exact_tick_alignment_status"] == "ok"
    al = exact_tick_alignment(ref, fine961[:-1] + [fine961[-1] + 720], r961, 120)
    assert al["exact_tick_alignment_status"] == "cadence_alignment_failure"  # off-by-one tick
    al = exact_tick_alignment(ref[:-1], fine961[:-1], r961, 120)
    assert al["alignment_failure_reason"] == "sample_count_mismatch"          # truncated tail
    missing = fine961[:50] + fine961[51:] + [fine961[-1] + 720]               # 결손 1개 보충 길이
    al = exact_tick_alignment(ref, missing, r961, 120)
    assert al["exact_tick_alignment_status"] == "cadence_alignment_failure"   # missing sample
    # interpolation 경로 부재: 출력 필드가 항상 False 고정
    assert exact_tick_alignment(ref, fine961, r961, 120)["interpolation_applied"] is False
    # pair cadence 자명 계약
    assert pair_cadence_identity([500, 1000], [500, 1000])["pair_cadence_status"] == "ok"
    assert pair_cadence_identity([500, 1000], [500, 1500])["pair_cadence_status"] == "cadence_alignment_failure"

    # --- q monotonicity 4점 + attach_q 계약
    A_raw_syn = 16.3864878605
    for c in (0.0, 0.25 * A_raw_syn, 0.75 * A_raw_syn, A_raw_syn):
        assert dec.effect_size_q_decision(-c) >= dec.effect_size_q_raw_reference(-c, -A_raw_syn)
    row = {"q_use": True, "signed_shift_raw": -4.0}
    attach_q(row, -A_raw_syn, "W_ANCHOR_SYN", "W_CTRL_SYN")
    assert row["q_decision"] == 4.0 / Q_DENOM_FROZEN
    assert row["q_comparability_scope"] == Q_COMPARABILITY_SCOPE
    assert row["q_anchor_relation_check"]["q_decision_ge_q_raw_reference"]

    # --- paired truth table 대표 사례 (정본 함수 위임 확인)
    j_p = {"qa_status": "ok", "observable_status": "admissible_interior_window_peak",
           "completion_status": "resolved_plateau"}
    j_rc = {"qa_status": "ok", "observable_status": "admissible_interior_window_peak",
            "completion_status": "right_censored"}
    assert dec.paired_claim_scope(j_p, j_p) == "both_arms_resolved"
    assert dec.paired_claim_scope(j_p, j_rc) == "censoring_not_excluded"

    # --- 기존 공개 comparator: anchor pair raw 재산출 + 보수성 전제
    anchor = recompute_anchor_pair()
    s_ref = anchor["s_pair_raw_reference"]
    assert abs(s_ref) > Q_DENOM_FROZEN, s_ref
    assert s_ref < 0, s_ref

    # --- S_801 harmonized 실데이터 재산출 == 참고값 경로 교차 검증
    s801 = compute_legacy_harmonized_pair("N-4-REF-NY801", N4_REF_PAIR, "n4_reference_input")
    assert s801["closure_schema_status"] == "ok_same_schema_pair", s801
    assert s801["closure_schemas"] == ["legacy", "legacy"], s801
    assert s801["signed_shift_raw"] is not None and s801["signed_shift_raw"] < 0
    _cl = anchor["harmonized_admissible_window_anchor_reference"]
    assert _cl is not None and abs(s801["signed_shift_raw"] - _cl) <= FLOAT_TOL * max(1.0, abs(_cl))
    # 창 종료 원인 실측 고정: 기존 18런은 접촉 미발생 → y_star_limit
    assert s801["arm_eh_window_end_reason"] == "y_star_limit", s801
    assert s801["arm_none_window_end_reason"] == "y_star_limit", s801
    assert anchor["anchor_reference_window_end_reasons"] == ["y_star_limit", "y_star_limit"]
    # 실데이터 trigger set: y_star_limit 단독 (동시 trigger 부재 실측)
    assert s801["arm_eh_window_end_trigger_set"] == ["y_star_limit"], s801
    assert s801["arm_none_window_end_trigger_set"] == ["y_star_limit"], s801
    # legacy pair는 계약 비적용 명시 (runtime 층 필드로 이관)
    assert s801["runtime_termination_contract_status"] == "not_applicable_legacy_reference"

    # --- 합성 records — min gap·창 종료·contact 절단
    def _rec2(step, x1, x2):
        return {"step": step, "particles": [
            {"x": x1, "y": 0.0, "vx": 0.0, "vy": -0.05, "t_star": 0.01 * step},
            {"x": x2, "y": 0.0, "vx": 0.0, "vy": -0.05, "t_star": 0.01 * step}]}
    recs_far = [_rec2(s, 0.01 * s, 0.01 * s + 0.30) for s in range(1, 30)]
    sw = series_two_particle(recs_far, 0, 15.5, 0.1)
    assert sw["contact_truncated"] is False
    assert abs(sw["min_center_distance"] - 0.30) <= 1e-12
    assert sw["window_rule_id"] == WINDOW_RULE_TWO_ARM
    assert sw["window_end_step"] == 29 and abs(sw["window_end_t_star"] - 0.29) <= 1e-12
    recs_touch = [_rec2(s, 0.01 * s, 0.01 * s + (0.30 if s < 10 else 0.09))
                  for s in range(1, 30)]
    st_c = series_two_particle(recs_touch, 0, 15.5, 0.1)
    assert st_c["contact_truncated"] is True and st_c["window_end_step"] == 9
    # window_end_reason 분기 검증
    assert sw["window_end_reason"] == "run_end" and st_c["window_end_reason"] == "contact"
    recs_fast = [_rec2(s, 0.2 * s, 0.2 * s + 0.30) for s in range(1, 30)]
    st_y = series_two_particle(recs_fast, 0, 15.5, 0.1)
    assert st_y["window_end_reason"] == "y_star_limit", st_y

    # --- trigger set 보존 — 단독·동시·run_end 3분기
    assert sw["window_end_trigger_set"] == [] and st_c["window_end_trigger_set"] == ["contact"]
    assert st_y["window_end_trigger_set"] == ["y_star_limit"]
    # 같은 표본(s=10)에서 contact(dist 0.09 ≤ 0.1)와 y*(16.2 > 15.5) 동시 첫 발생
    recs_both = [_rec2(s, 0.18 * s, 0.18 * s + (0.30 if s < 10 else 0.09))
                 for s in range(1, 30)]
    st_b = series_two_particle(recs_both, 0, 15.5, 0.1)
    assert st_b["window_end_trigger_set"] == ["contact", "y_star_limit"], st_b
    assert st_b["window_end_reason"] == "contact"  # precedence: CONTACT_BEFORE_YSTAR_V1
    assert st_b["window_end_precedence_rule_id"] == WINDOW_END_PRECEDENCE

    # --- peak 표본 gap — 합성 수동 대조 (peak index 2, dist=0.36)
    def _rec2v(step, x2_off, vy):
        return {"step": step, "particles": [
            {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": vy, "t_star": 0.01 * step},
            {"x": x2_off, "y": 0.0, "vx": 0.0, "vy": vy, "t_star": 0.01 * step}]}
    recs_pk = [_rec2v(s, 0.30 + 0.02 * s, -0.01 * (3 - abs(s - 3))) for s in range(1, 6)]
    s_pk = series_two_particle(recs_pk, 0, 15.5, 0.1)
    pk_i = max(range(len(s_pk["speed"])), key=lambda i: s_pk["speed"][i])
    assert pk_i == 2 and abs(s_pk["center_distances"][pk_i] - 0.36) <= 1e-12
    assert abs(s_pk["min_center_distance"] - 0.32) <= 1e-12  # min은 s=1 표본

    # --- 종료 3층 판정 + 도달 가능 어휘 실측 일치
    assert REACHABLE_TERMINATION_REASONS == {
        "max_steps", "converged", "cd_converged", "nan", "contact", "offset",
        "domain_bounds"}
    def _st(reason, step=60000):
        return {"completed": True, "termination_reason": reason, "final_step": step}
    def _3l(tc):
        return (tc["runtime_termination_contract_status"],
                tc["analysis_window_completion_status"],
                tc["metric_eligibility_status"])
    assert _3l(evaluate_termination_contract("two", _st("offset"), "y_star_limit")) == (
        "expected_stop_satisfied", "complete_by_y_star_limit",
        "eligible_declared_finite_window")
    assert _3l(evaluate_termination_contract("single", _st("contact"), "y_star_limit")) == (
        "expected_stop_satisfied", "complete_by_y_star_limit",
        "eligible_declared_finite_window")
    # safety cap + 창 완결 → runtime 층은 satisfied가 아니라 safety-cap
    assert _3l(evaluate_termination_contract("two", _st("max_steps"), "y_star_limit")) == (
        "safety_cap_reached_before_expected_stop", "complete_by_y_star_limit",
        "eligible_declared_finite_window")
    assert _3l(evaluate_termination_contract("two", _st("max_steps"), "contact")) == (
        "safety_cap_reached_before_expected_stop", "complete_by_geometric_contact",
        "eligible_declared_finite_window")
    assert _3l(evaluate_termination_contract("two", _st("max_steps"), "run_end")) == (
        "safety_cap_reached_before_expected_stop", "run_end_limited", "blocked")
    assert _3l(evaluate_termination_contract("two", _st("converged"), "y_star_limit")) == (
        "unexpected_early_termination", "complete_by_y_star_limit", "blocked")
    assert _3l(evaluate_termination_contract("single", _st("domain_bounds"), "y_star_limit"))[0] \
        == "unexpected_early_termination"
    # single 계약에서 two의 예정 stop(offset)은 예정 외 geometric stop → 차단
    assert _3l(evaluate_termination_contract("single", _st("offset"), "y_star_limit"))[2] == "blocked"
    # 어휘 밖 문자열 → invalid metadata (run_integrity 중복 방어)
    assert _3l(evaluate_termination_contract("two", _st("weird"), "y_star_limit"))[0] \
        == "invalid_termination_metadata"

    # --- stop predicate 일치 QA — 합성 재계산 6사례
    import types as _types
    _exp = _types.SimpleNamespace(max_steps=60000, xmax=1.0, cylinder_D_ratio=0.1,
                                  gravity_direction="right",
                                  sedimentation_stop_offset_d=2.0)
    _exp_s = _types.SimpleNamespace(max_steps=40000, xmax=1.0, cylinder_D_ratio=0.1,
                                    gravity_direction="down",
                                    sedimentation_stop_offset_d=0.0)
    def _rec_at(x):  # two — p0 x 좌표 종단 레코드
        return [{"step": 59500, "particles": [{"x": x, "y": 0.5}, {"x": x - 0.3, "y": 0.5}]}]
    # offset: x ≥ xmax − 2.0·D = 0.8
    v = verify_stop_predicate("two", _st("offset", 59500), _rec_at(0.85), _exp)
    assert v["stop_predicate_consistency_status"] == "ok", v
    v = verify_stop_predicate("two", _st("offset", 59500), _rec_at(0.55), _exp)
    assert v["stop_predicate_consistency_status"] == "termination_predicate_mismatch"
    # contact(single): y ≤ D/2 = 0.05
    v = verify_stop_predicate("single", _st("contact", 39500),
                              [{"step": 39500, "y": 0.049}], _exp_s)
    assert v["stop_predicate_consistency_status"] == "ok", v
    v = verify_stop_predicate("single", _st("contact", 39500),
                              [{"step": 39500, "y": 0.30}], _exp_s)
    assert v["stop_predicate_consistency_status"] == "termination_predicate_mismatch"
    # max_steps: final_step == configured
    assert verify_stop_predicate("two", _st("max_steps", 60000), _rec_at(0.5), _exp)[
        "stop_predicate_consistency_status"] == "ok"
    assert verify_stop_predicate("two", _st("max_steps", 59500), _rec_at(0.5), _exp)[
        "stop_predicate_consistency_status"] == "termination_predicate_mismatch"
    # isolated 1입자(single 스키마) offset 재계산 + N-7 스키마 선언
    assert COMPARISONS["N-7"].get("record_schema") == "single"
    v = verify_stop_predicate("two", _st("offset", 59500),
                              [{"step": 59500, "x": 0.85, "y": 0.5}], _exp)
    assert v["stop_predicate_consistency_status"] == "ok", v

    # --- 발생 경로 provenance — reason별 분리 (단일 source ID 폐기)
    v = verify_stop_predicate("two", _st("offset", 59500), _rec_at(0.85), _exp)
    assert v["termination_reason_origin_id"] == "RUNTIME_SHOULD_STOP_V1"
    assert v["predicate_recompute_state_id"] == PREDICATE_RECOMPUTE_SHOULD_STOP
    assert abs(v["stop_predicate_margin_raw"] - 0.05) <= 1e-12  # 0.85 − 0.8
    v = verify_stop_predicate("two", _st("max_steps", 60000), _rec_at(0.5), _exp)
    assert v["termination_reason_origin_id"] == "LOOP_MAX_STEPS_EXHAUSTION_V1"
    assert v["predicate_recompute_state_id"] == PREDICATE_RECOMPUTE_MAX_STEPS
    assert v["stop_predicate_margin_raw"] is None  # step 메타데이터 대조
    v = verify_stop_predicate("two", _st("cd_converged"), _rec_at(0.5), _exp)
    assert v["termination_reason_origin_id"] == "LOOP_CONVERGENCE_DECISION_V1"
    v = verify_stop_predicate("single", _st("contact", 39500),
                              [{"step": 39500, "y": 0.049}], _exp_s)
    assert abs(v["stop_predicate_margin_raw"] - 0.001) <= 1e-12  # 0.05 − 0.049
    v = verify_stop_predicate("two", _st("offset", 59500), _rec_at(0.55), _exp)
    assert v["stop_predicate_margin_raw"] < 0.0  # mismatch ⇔ 음수 margin

    # --- round-trip 보존 시연 (파이썬 json float = repr 최단 표현)
    for _x in (0.5252381907191696, 1.0 / 3.0, 0.1234567890123456789):
        assert json.loads(json.dumps(_x)) == _x

    # --- 예약 상태 출력 금지 — 도달 가능 입력 격자 전수
    _emitted = set()
    for _r in sorted(REACHABLE_TERMINATION_REASONS) + ["weird"]:
        for _w in ("y_star_limit", "contact", "run_end", "other"):
            for _rn in ("two", "single"):
                _tc = evaluate_termination_contract(_rn, _st(_r), _w)
                _emitted.add(_tc["analysis_window_completion_status"])
                _emitted.add(_tc["metric_eligibility_status"])
    for _res in RESERVED_NOT_EMITTED_IN_CURRENT_CAMPAIGN:
        assert _res not in _emitted, _res

    # --- 통합 truth-table — 3층 × completion 우선순위 (잠긴 계층 결합 확인)
    # ① eligibility blocked → judge 무관 차단 (build_pair_row 흐름: judge 이전 return)
    assert evaluate_termination_contract("two", _st("max_steps"), "run_end")[
        "metric_eligibility_status"] == "blocked"
    # ② safety-cap + 창 완결 + RC → metric 허용, censoring 한정은 기존 계층 유지
    _tc_rc = evaluate_termination_contract("two", _st("max_steps"), "y_star_limit")
    assert _tc_rc["metric_eligibility_status"] == "eligible_declared_finite_window"
    assert dec.paired_claim_scope(j_p, j_rc) == "censoring_not_excluded"
    # ③ expected stop + 창 완결 + plateau → 완전 허용 경로
    assert evaluate_termination_contract("two", _st("offset"), "y_star_limit")[
        "metric_eligibility_status"] == "eligible_declared_finite_window"
    assert dec.paired_claim_scope(j_p, j_p) == "both_arms_resolved"

    # 차단 사유는 재계산 생략 / 미등재 gravity 방향은 자동 일반화 금지
    assert verify_stop_predicate("two", _st("nan"), _rec_at(0.5), _exp)[
        "stop_predicate_consistency_status"] == "not_applicable_blocked_reason"
    _exp_rot = _types.SimpleNamespace(**{**_exp.__dict__, "gravity_direction": "down"})
    assert verify_stop_predicate("two", _st("offset", 59500), _rec_at(0.85), _exp_rot)[
        "stop_predicate_consistency_status"] == "predicate_form_unregistered"

    # --- dx 환산 정본 검증 (NN=801 → 1/800 = 0.00125; S_801 공개값 정합)
    assert abs(1.0 / (801.0 - 1.0) - 0.00125) <= 1e-15

    # --- P4 보수 clearance 상수·경계 (대각 반례: g=5h는 4h 초과지만 겹침 가능)
    assert abs(KERNEL_CLEARANCE_THRESHOLD_DX - 4.0 * math.sqrt(2.0)) <= 1e-12
    assert not (5.0 > KERNEL_CLEARANCE_THRESHOLD_DX)   # 단순 >4 기준은 g=5h 대각 겹침을 오판
    assert 6.0 > KERNEL_CLEARANCE_THRESHOLD_DX

    # --- N-4 안정 판정 synthetic (tick 게이트 + 3조건 + 창 ID 게이트)
    def _mk_n4(nn, shift, interval, win=WIN_STD50_15P5):
        return {"comparison_id": f"N-4-NY{nn}", "signed_shift_raw": shift,
                "analysis_window_definition_id": win,
                "_steps_eh": [interval * i for i in range(1, 13)], "_n_records_full": 120}
    s801_syn = {"comparison_id": "N-4-REF-NY801", "signed_shift_raw": -14.0,
                "analysis_window_definition_id": WIN_STD50_15P5,
                "_steps_eh": [500 * i for i in range(1, 13)], "_n_records_full": 120}
    blk = n4_stability_block(s801_syn, [_mk_n4(961, -13.5, 720), _mk_n4(1281, -13.2, 1280)])
    assert blk["direct_comparison_allowed"] is True, blk
    assert blk["n4_stability_status"] == "stable_under_coupled_space_time_refinement", blk
    blk = n4_stability_block(s801_syn, [_mk_n4(961, 1.0, 720), _mk_n4(1281, 1.2, 1280)])
    assert blk["n4_stability_status"] == "convergence_not_established", blk
    blk = n4_stability_block(s801_syn, [_mk_n4(961, -13.9, 720), _mk_n4(1281, -12.0, 1280)])
    assert blk["n4_stability_status"] == "convergence_not_established", blk
    _bad961 = _mk_n4(961, -13.5, 720)
    _bad961["_steps_eh"][3] += 1
    blk = n4_stability_block(s801_syn, [_bad961, _mk_n4(1281, -13.2, 1280)])
    assert blk["direct_comparison_allowed"] is False, blk
    assert blk["n4_stability_status"] == "contextual_only", blk
    # 창 ID 불일치 → direct 차단
    blk = n4_stability_block(s801_syn, [_mk_n4(961, -13.5, 720, win=WIN_TALL60_51P5),
                                        _mk_n4(1281, -13.2, 1280)])
    assert blk["direct_comparison_allowed"] is False, blk
    assert blk["comparison_window_compatibility"]["analysis_window_definition_id_equal"] is False
    # 게이트 통과와 directness 등급 분리 확인
    blk = n4_stability_block(s801_syn, [_mk_n4(961, -13.5, 720), _mk_n4(1281, -13.2, 1280)])
    assert blk["metric_comparison_allowed"] is True
    assert blk["comparison_directness_grade"] == N4_DIRECTNESS_GRADE

    # --- 단일 입자 y_star 저장값 정합(좌표 규약 검증 — 공개 데이터)
    sp = DATA / "single_particle_sedimentation/baseline/rho150"
    recs_sp = load_history(sp)
    r_last = recs_sp[len(recs_sp) // 2]
    _case_b, exp_b = None, None  # 좌표 규약은 y_star 필드 자체 저장으로 확인
    assert math.isfinite(r_last["y_star"]) and r_last["y_star"] >= 0.0

    # --- deprecated 패턴 자기 검사 — 패턴은 조합식으로 구성해
    # 검사 리스트 자체가 오탐되지 않게 한다
    src = Path(__file__).read_text()
    _pats = ["(eh-" + "none)/eh", "eh" + "−" + "none",
             "EH_" + "MINUS_NONE", "delta_" + "eh_none",
             "DECLARED_" + "WINDOW_V1",   # 비정본 window rule 리터럴 부재
             '"kernel_overlap_' + 'excluded"',      # 구 필드명(무scope) 부재
             "VALID_TERMINATION_" + "REASONS",      # 구 어휘 상수명 부재
             "satisfied_window_" + "complete_max_steps",  # 혼합 명칭 폐기
             "SHOULD_STOP_SAME_" + "STEP_LAST_STORED_RECORD_V1"]  # 단일 source ID 폐기
    for pat in _pats:
        assert pat not in src, pat  # 조합 구성이라 소스에 완성 리터럴 0건이어야 함

    print("BUILDER_SELFTEST_PASS")
    print(json.dumps({"anchor_recompute": anchor,
                      "classifier_sha256": dec.classifier_sha256()}, indent=1))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmatory", action="store_true",
                    help="확인 런 표 생성 (selftest·해시 고정 후에만)")
    ap.add_argument("--contact-rule-confirmed", default=None, metavar="RULE_ID",
                    help="contact 규칙 대조 확정 선언 — 확정된 rule ID를 값으로 전달"
                         "(Boolean 플래그가 아니라 ID 결속: 값이 정본 상수와 일치해야 개방)")
    ap.add_argument("--out", default=str(HERE / "immutable_table_n_series.json"))
    args = ap.parse_args()

    if not args.confirmatory:
        _selftest()
        return

    # confirmatory-result processing 가드: 런 실행이 아니라 결과 처리·열람의 차단이다
    if args.contact_rule_confirmed != CONTACT_CROSSING_RULE:
        raise SystemExit(
            "confirmatory-result processing 차단: contact 규칙의 원고 정본 대조 확정 후 "
            f"--contact-rule-confirmed {CONTACT_CROSSING_RULE} 로 rule ID를 결속 전달해야 "
            "합니다 (결과 열람 전 규칙 확정 강제)")

    anchor = recompute_anchor_pair()
    rows = []
    for cid, cdef in COMPARISONS.items():
        row = build_pair_row(cid, cdef)
        if row.get("q_use") and row.get("signed_shift_raw") is not None:
            arm_rule = (WINDOW_RULE_TWO_ARM if cdef["kind"] == "two_particle"
                        else WINDOW_RULE_SINGLE_ARM)
            attach_q(row, anchor["s_pair_raw_reference"],
                     anchor_window_rule_id=anchor["anchor_window_rule_id"],
                     control_window_rule_id=arm_rule)
        rows.append(row)

    # N-4 S_801 harmonized (기존 공개 df 801 pair) + 3격자 안정 판정
    s801_pack = compute_legacy_harmonized_pair("N-4-REF-NY801", N4_REF_PAIR,
                                               "n4_reference_input")
    n4_block = n4_stability_block(s801_pack, rows)

    # N-1 50D harmonized context (비판정)
    n1_context = [compute_legacy_harmonized_pair(cid, pdef, "non_decisional_context")
                  for cid, pdef in N1_CONTEXT_PAIRS.items()]

    def _strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    out = {
        "schema_version": SCHEMA_VERSION,
        "manifest_ref": MANIFEST_REF,
        "paired_schema_sha256": dec.classifier_sha256(),
        "table_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "anchor": anchor,
        "rows": [_strip(r) for r in rows],
        "n4_reference_ny801_harmonized": _strip(s801_pack),
        "n4_grid_stability": n4_block,
        "n1_context_50d_harmonized": [_strip(r) for r in n1_context],
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"WROTE {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
