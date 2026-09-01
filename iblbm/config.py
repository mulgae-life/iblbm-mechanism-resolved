"""시뮬레이션 설정 (`SimConfig` dataclass).

역할
  - 고정/진동/병진/회전 실린더, 단일/다입자 침강, Taylor-Green 와류 등
    시나리오별 물리·수치 파라미터를 단일 dataclass 로 수집
  - `__post_init__` 에서 상호배타 조합(IMC/added-mass 등) config-level 검증
  - `warn_if_unstable()` 로 τ 안정성 경고 제공

단위 주의
  - `lattice_u`, `inflow_u`, `gravity`              lattice units
  - `Re`, `KC`, `rho_ratio`, `cylinder_D_ratio`     비차원
  - `xmax`, `ymax`, `cylinder_center`               domain 비율 좌표
  - `phy_u`, `phy_l`, `phy_density`                 SI (m/s, m, kg/m³)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimConfig:
    """IB-LBM 시뮬레이션 설정.

    Attributes:
        Re: Reynolds 수 (실린더 직경 기준, 회전은 r²·ω/ν)
        NN: dx 해상도 파라미터, dx = 1/(NN-1). ny는 ymax/dx에서 유도
        xmax: x 방향 도메인 크기 [0, xmax]
        ymax: y 방향 도메인 크기 [0, ymax]
        cylinder_center: 실린더 중심 좌표 (x, y), 도메인 비율
        cylinder_D_ratio: 실린더 특성길이/물리 길이 비율 (원형은 직경 기준)
        lattice_u: 격자 단위 속도
        inflow_u: 유입 속도 (격자 단위)
        bc_type: 경계 조건 패턴 ('dirichlet_neumann' | 'velocity_inlet' | 'open_boundary'
                 | 'settling_channel' | 'tg_analytical')
        ibm_method: IBM 방법론 ('DF' | 'MDF' | 'DFC')
        mdf_iterations: MDF 최대 반복 횟수
        mdf_min_iterations: MDF 최소 반복 횟수
        mdf_tolerance: MDF no-slip residual 수렴 임계값
        max_steps: 최대 시간 스텝 수
        convergence_threshold: 수렴 판정 임계값
        convergence_start: 수렴 판정 시작 스텝 (기본값 100,000)
        check_interval: 수렴 체크 간격 (스텝)
        use_convergence: 수렴 판정 사용 여부
        motion_type: 운동 유형 (None | 'oscillating' | 'sedimentation')
        KC: Keulegan-Carpenter 수 (진동 시나리오)
        marker_spacing_factor: 라그랑주 마커 간격 = dx · factor
            정상류 0.5, 진동 2/3
        time_integrator: 시간 적분 방식
            'verlet': Velocity Verlet (2차 정확도, 기본)
            'euler_explicit': 1차 Euler Explicit (Majumder 2023 §2.3)
        nx_formula: 격자 nx 계산 방식
            'standard': nx = NN·xmax − (xmax−1) — 정상류/회전 (dx=dy 보장)
            'simple':   nx = NN·xmax           — 진동 ((nx−1)·dx ≠ xmax 이산화 부산물, ~0.08% 차이)
    """

    Re: float
    NN: int
    xmax: float = 1.0
    ymax: float = 1.0
    cylinder_center: tuple[float, float] = (0.4, 0.5)
    cylinder_D_ratio: float = 1 / 40
    lattice_u: float = 0.1
    inflow_u: float = 0.1
    bc_type: str = "velocity_inlet"
    delta_type: str = "peskin4pt"  # "peskin4pt" | "hat"
    ibm_method: str = "DF"   # "DF" | "MDF" | "DFC"
    df_coupled: bool = False  # DF 다입자: 입자 간 within-step velocity coupling
    stream_first: bool = False  # Feng 2004 순서: Stream→Macro→IBM→Collision (IBM이 충돌 전 분포 f를 사용)
    mdf_iterations: int = 20
    mdf_min_iterations: int = 5
    mdf_tolerance: float = 1e-4
    max_steps: int = 800_000
    convergence_threshold: float = 1e-5
    convergence_start: int = 100_000
    check_interval: int = 100
    use_convergence: bool = True
    # Cd-기반 조기 종료 (Re=20 등 정상상태 빠른 수렴 케이스용, 진동/침강 미적용)
    cd_convergence_tol: float | None = None       # None=비활성, 1e-4 권장
    cd_convergence_window: int = 5                # 최근 N개 Cd 측정 spread 기준
    cd_convergence_start_step: int = 20_000       # transient 회피용 최소 step
    motion_type: str | None = None
    KC: float = 5.0
    # --- 침강 (Sedimentation) ---
    rho_ratio: float = 1.0        # rho_s / rho_f (고체/유체 밀도비). 1.0 = 중성 부력
    gravity: float = 0.0           # 격자 단위 중력 가속도 (양수, -y 방향 적용)
    enable_rotation: bool = False  # 회전 자유도 + 토크 (침강)
    retraction_dx: float = 0.0  # marker 후퇴량 (Δx 단위). 0.3 = 0.3Δx 안쪽
    rotation_coupling: str = "indirect"  # "indirect" | "semi_implicit" | "iterative"
    prescribed_velocity: tuple[float, float] | None = None  # 입자 속도 고정 (vx, vy). None → Verlet 적분
    gravity_direction: str = "down"  # "down" (-y) | "right" (+x). Uhlmann 2입자는 "right"
    particles_config: list[dict] | None = None  # 다입자: [{center:(x,y), rho_ratio:ρ}, ...]. None→단일입자
    diagnostics_interval: int = 0  # 내부 진단 계산 주기 (0=비활성, >0: N스텝마다 P_int/inside_residual 계산)
    settling_inertia_model: str = "unset"  # "unset" | "none" | "explicit_history" | "feng_b2" | "full_volume"
    sedimentation_stop_at_contact: bool = False
    sedimentation_stop_offset_d: float = 0.0
    sedimentation_reference_basis: str = "standard"  # "standard" | "particle_basis"
    sedimentation_euler_update_scheme: str = "new_velocity"  # "new_velocity" | "trapezoidal"
    marker_spacing_factor: float = 0.5
    nx_formula: str = "standard"
    use_gpu: bool = False  # CuPy GPU 가속 사용 여부
    time_integrator: str = "verlet"  # "verlet" | "euler_explicit" (Majumder 2023: Euler)
    incompressible_lbgk: bool = False  # True → He & Luo (1997) incompressible LBGK

    # 충돌 모델
    collision_model: str = "BGK"      # "BGK" | "TRT" | "CM_MRT"
    trt_magic_param: float = 0.25     # Λ = 1/4 (Ginzburg 2008 권장)
    # CM_MRT 완화율 (Lallemand & Luo 2000 표준값, None → 자동)
    mrt_s_e: float | None = None      # 에너지 이완률 (기본 1.4)
    mrt_s_eps: float | None = None    # 에너지² 이완률 (기본 1.4)
    mrt_s_q: float | None = None      # 에너지 flux 이완률 (기본 1.2)

    # Taylor-Green 감쇠 와류
    scenario_type: str | None = None  # "taylor_green" 또는 None
    tg_L: float = 1.0      # 도메인 반폭 (도메인 = [-L, L]²)
    tg_u0: float = 0.1     # 초기 속도 진폭
    tg_T_end: float | None = None  # 측정 시점 (물리 시간)
    tg_with_ibm: bool = False      # True: IBM 포함 수렴 테스트


    # 물리 상수 (단위 변환용)
    phy_u: float = 1e-3       # m/s
    phy_l: float = 1e-3       # m
    phy_density: float = 1000  # kg/m^3

    def __post_init__(self) -> None:
        """Dataclass 생성 직후 config-level 상호 배제 규칙 검사."""
        valid_scenario_types = {None, "taylor_green"}
        if self.scenario_type not in valid_scenario_types:
            raise ValueError(
                f"scenario_type='{self.scenario_type}' 미지원. "
                f"허용값: {sorted(s for s in valid_scenario_types if s is not None)} 또는 None"
            )
        if self.mdf_iterations < 1:
            raise ValueError("mdf_iterations는 1 이상 값 필요")
        if self.mdf_min_iterations < 1:
            raise ValueError("mdf_min_iterations는 1 이상 값 필요")
        if self.mdf_tolerance <= 0.0:
            raise ValueError("mdf_tolerance는 양수 값 필요")

        if self.motion_type == "sedimentation":
            valid_models = {
                "unset", "none", "explicit_history", "feng_b2", "full_volume",
            }
            if self.settling_inertia_model not in valid_models:
                raise ValueError(
                    f"settling_inertia_model='{self.settling_inertia_model}' 미지원. "
                    f"허용값: {sorted(valid_models)}"
                )
            if self.settling_inertia_model == "unset":
                raise ValueError(
                    "Sedimentation benchmark requires explicit settling_inertia_model"
                )
            if self.sedimentation_stop_at_contact and self.sedimentation_stop_offset_d > 0.0:
                raise ValueError(
                    "contact and offset stop rules are mutually exclusive"
                )
            valid_reference_basis = {"standard", "particle_basis"}
            if self.sedimentation_reference_basis not in valid_reference_basis:
                raise ValueError(
                    f"sedimentation_reference_basis='{self.sedimentation_reference_basis}' "
                    f"미지원. 허용값: {sorted(valid_reference_basis)}"
                )
            valid_euler_schemes = {"new_velocity", "trapezoidal"}
            if self.sedimentation_euler_update_scheme not in valid_euler_schemes:
                raise ValueError(
                    f"sedimentation_euler_update_scheme="
                    f"'{self.sedimentation_euler_update_scheme}' 미지원. "
                    f"허용값: {sorted(valid_euler_schemes)}"
                )

    def warn_if_unstable(self) -> str | None:
        """τ 안정 한계 근접 여부 경고.

        - BGK/TRT: τ → 0.5 에서 불안정. 경험적 기준 τ ≥ 0.55
        - CM_MRT:  다중 이완율로 안정 마진 확장, τ ≥ 0.52 허용
        - 관계식:  τ = 3·ν + 0.5,  ν = lattice_u · lattice_D / Re
        """
        diameter_ratio = self.cylinder_D_ratio
        lattice_D = diameter_ratio * (self.NN - 1)
        lattice_nu = self.lattice_u * lattice_D / self.Re
        tau = 3.0 * lattice_nu + 0.5

        if self.collision_model == "CM_MRT":
            # CM_MRT는 다중 완화율로 안정성 마진이 넓음
            if tau < 0.52:
                return (f"tau={tau:.4f} < 0.52: CM_MRT도 불안정 가능. "
                        f"NN 증가 또는 Re 감소 권장 (현재 NN={self.NN}, Re={self.Re}).")
        else:
            # BGK, TRT
            if tau < 0.55:
                model = self.collision_model
                return (f"tau={tau:.4f} < 0.55: {model}-LBM 불안정 가능. "
                        f"NN 증가 또는 Re 감소 권장 (현재 NN={self.NN}, Re={self.Re}). "
                        f"NN ≥ {int(0.6 * self.Re / (3.0 * self.lattice_u * self.cylinder_D_ratio) + 1)} 권장.")
        return None
