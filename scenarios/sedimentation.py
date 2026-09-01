"""침강 시나리오 -- Glowinski et al. (2001) 표준 물리 구성.

Ma-constrained 격자 파라미터 결정:
  물리 파라미터(CGS)에서 Archimedes 수를 계산하고,
  Ma < Ma_max 제약으로 ν_lat → τ → g_lattice를 역산한다.
  NN이 부족하면 자동 상향 또는 에러 발생.

도메인 프리셋:
  "glowinski2001" — L=6cm, 초기 (1,4)cm. Glowinski 2001 / Wang 2008 canonical. W/D=8, H/D=24.
  "tall60D"       — L=15cm, 초기 (1,13)cm. saturation 검증 도메인. W/D=8, H/D=60.
  "tall80D"       — L=20cm, 초기 (1,18)cm. 장도메인 vortex 발달용. W/D=8, H/D=80.
"""

from iblbm.config import SimConfig
from iblbm.solver import run
import numpy as np


# 도메인 프리셋 (좌표는 W=2cm 기준 무차원)
DOMAIN_PRESETS = {
    "glowinski2001": {
        "ymax": 3.0,              # 3W = 6cm
        "cylinder_center": (0.5, 2.0),  # (1cm, 4cm) / W
    },
    "tall60D": {
        "ymax": 7.5,              # 7.5W = 15cm = 60D
        "cylinder_center": (0.5, 6.5),  # (1cm, 13cm) / W — 상단 1W 여유 (80D와 동일 컨벤션)
    },
    "tall80D": {
        "ymax": 10.0,             # 10W = 20cm = 80D
        "cylinder_center": (0.5, 9.0),  # (1cm, 18cm) / W — 상단 1W 여유
    },
}


def make_sedimentation_config(
    rho_ratio: float = 1.01,
    ibm_method: str = "DF",
    delta_type: str = "peskin4pt",
    collision_model: str = "BGK",
    NN: int | None = None,
    Ma_max: float = 0.08,
    tau_min: float = 0.52,
    verbose: bool = True,
    case_name: str = "glowinski2001",
    center_x_offset_dx: float = 0.0,
    diagnostics_interval: int = 0,
    retraction_dx: float = 0.0,
    nu_phys_override: float | None = None,
    enable_rotation: bool = False,
    rotation_coupling: str = "indirect",
    mdf_iterations: int = 20,
    time_integrator: str = "verlet",
    incompressible_lbgk: bool = False,
    settling_inertia_model: str = "none",
    sedimentation_stop_at_contact: bool = False,
    sedimentation_stop_offset_d: float = 0.0,
    sedimentation_reference_basis: str = "particle_basis",
) -> SimConfig:
    """침강 벤치마크 SimConfig 생성 — Ma-constrained 방식.

    Glowinski 2001 표준 물리 파라미터 (CGS):
        W = 2, d = 0.25, nu = 0.01 (rho=1.5) or 0.1 (rho=1.25), g = 980

    격자 파라미터는 Ma 제약으로 자동 결정:
        Ma = √(3·Ar) × ν_lat / D_lat < Ma_max
        ν_lat ≥ (τ_min - 0.5) / 3  (안정성)
        → NN이 부족하면 에러

    Args:
        rho_ratio: rho_s / rho_f
        ibm_method: "DF" | "MDF" | "DFC"
        delta_type: "peskin4pt" | "hat"
        NN: 격자 해상도 (None → 자동 결정)
        Ma_max: 최대 허용 Mach 수 (기본 0.08, 안전 여유 포함)
        tau_min: 최소 τ (기본 0.52, 안정성)
        case_name: 도메인 프리셋 ("glowinski2001" | "tall60D" | "tall80D")
        center_x_offset_dx: 입자 초기 x위치 오프셋 (격자 단위, 0=중심)

    Returns:
        SimConfig
    """
    # 물리 파라미터 (CGS) — Glowinski 2001 / Wang 2008 §3.1 표준
    # (Feng 1994는 무차원, 물리 단위 미사용)
    #
    # Wang 2008 §3.1 명시:
    #   rho_p=1.25: mu=0.1 g/(cm·s)  → Re_max ≈ 17  (low-Re case)
    #   rho_p=1.5 : mu=0.01 g/(cm·s) → Re_max ≈ 503 (high-Re case, 기본값)
    # ρ_f = 1.0 g/cm³ 이므로 ν = μ/ρ_f = μ 수치.
    # rho_ratio 기준 분기로 두 case 모두 원문에 정합시킨다.
    W_phys = 2.0       # cm
    d_phys = 0.25       # cm  → W/d = 8
    if abs(rho_ratio - 1.25) < 1e-6:
        nu_phys = 0.1   # cm²/s — Wang 2008 low-Re case
    else:
        nu_phys = 0.01  # cm²/s — Wang 2008 high-Re case (rho=1.5 및 scan 기본)
    if nu_phys_override is not None:
        # rho 분기 기본값 대신 명시 ν 지정 (N-6 high-Ga / Re-targeted 1.25 컨트롤: ν=0.01)
        nu_phys = float(nu_phys_override)
    g_phys = 980.0      # cm/s² (Wan & Turek 2006 명시)

    # 도메인 프리셋
    if case_name not in DOMAIN_PRESETS:
        raise ValueError(
            f"Unknown case_name '{case_name}'. "
            f"Available: {list(DOMAIN_PRESETS.keys())}"
        )
    preset = DOMAIN_PRESETS[case_name]
    xmax = 1.0           # W (채널 폭)
    ymax = preset["ymax"]
    D_ratio = d_phys / W_phys   # = 0.125

    # Archimedes 수 (무차원 — rho_ratio 의존)
    delta_rho = abs(rho_ratio - 1.0)
    Ar = delta_rho * g_phys * d_phys**3 / nu_phys**2
    nu_lat_min = (tau_min - 0.5) / 3.0

    # NN 결정: Ma 제약 + BGK τ 안전 여유.
    #
    # 유도: D_lat_min = SF × √(3·Ar) × ν_lat_min / Ma_max
    #       = (SF × ν_lat_min / Ma_max) × √(3·Ar)
    #
    # `safety_factor = (τ_target - 0.5) / (τ_min - 0.5)` 형태로 해석:
    #   tau_min=0.52 (ν_lat_min=0.00667), SF=4 ⇒ 실효 τ_target ≈ 0.58
    #   BGK 안전 한계 0.55 대비 3× 마진 (경험치).
    #
    # 기본 NN=1281 (rho=1.5) 에서 실제 τ≈0.58 달성. SF=1 이면
    # τ≈0.516 (BGK 불안정 영역)이라 SF>1 필요. 더 작은 SF(2-3)도 이론적
    # 으로 안정 가능하지만 shedding 진동에서 추가 margin 확보 차원.
    #
    # 참고: Wang 2008 §3.1 (FD) / Glowinski 2001 (FEM) 은 NS 직접 이산화 기반이라 LBM τ 기준
    # 을 제공하지 않음. SF=4는 우리 BGK 침강 경험치.
    if delta_rho > 1e-15 and Ar > 0:
        safety_factor = 4.0
        D_lat_min = safety_factor * np.sqrt(3.0 * Ar) * nu_lat_min / Ma_max
        NN_min = int(np.ceil(D_lat_min / D_ratio)) + 1
    else:
        NN_min = 81

    if NN is None:
        for candidate in [81, 161, 321, 641, 1281]:
            if candidate >= NN_min:
                NN = candidate
                break
        else:
            NN = NN_min
    elif NN < NN_min:
        import warnings
        warnings.warn(
            f"NN={NN} < NN_min={NN_min} (SF={safety_factor}). "
            f"안정성 미보장 — 격자 수렴 연구 등 의도적 사용만 권장.",
            stacklevel=2,
        )

    # 격자 파라미터
    dx = 1.0 / (NN - 1)
    lattice_D = D_ratio * (NN - 1)
    lattice_r = lattice_D / 2.0

    # ν_lat: Ma 제약의 80%로 설정 (안전 여유)
    if delta_rho > 1e-15 and Ar > 0:
        nu_lat_max = Ma_max * lattice_D / np.sqrt(3.0 * Ar)
        nu_lat = max(nu_lat_max * 0.8, nu_lat_min)
    else:
        nu_lat = 0.05

    tau = 3.0 * nu_lat + 0.5

    # g_lattice = Ar × ν_lat² / (Δρ × D_lat³)
    if delta_rho > 1e-15:
        g_lattice = Ar * nu_lat**2 / (delta_rho * lattice_D**3)
    else:
        g_lattice = 1e-6

    # 종단 속도 추정 + Ma 검증
    u_t_est = np.sqrt(delta_rho * g_lattice * lattice_D) if delta_rho > 0 else 0.0
    Ma_est = u_t_est * np.sqrt(3.0)

    if Ma_est > Ma_max:
        import warnings
        warnings.warn(
            f"Ma_est={Ma_est:.4f} > Ma_max={Ma_max}. "
            f"NN을 높이거나 Ma_max를 완화하세요.",
            stacklevel=2,
        )

    # lattice_u: 침강에서 참조 속도. u_t 추정값 사용 (config Re 계산용)
    lattice_u = max(u_t_est, 0.01)
    Re_input = lattice_u * lattice_D / nu_lat

    # max_steps: 채널 횡단 시간 × 2 (여유)
    channel_height_lattice = ymax / dx
    if u_t_est > 1e-10:
        max_steps = int(2.0 * channel_height_lattice / u_t_est)
    else:
        max_steps = 500_000

    # 초기 위치: 프리셋 + x 오프셋 (격자 단위)
    cx_base, cy = preset["cylinder_center"]
    cx = cx_base + center_x_offset_dx * dx
    cfg = SimConfig(
        Re=Re_input,
        NN=NN,
        xmax=xmax,
        ymax=ymax,
        cylinder_center=(cx, cy),
        cylinder_D_ratio=D_ratio,
        lattice_u=lattice_u,
        inflow_u=0.0,
        bc_type="settling_channel",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=mdf_iterations,
        max_steps=max_steps,
        use_convergence=False,
        check_interval=100,
        motion_type="sedimentation",
        rho_ratio=rho_ratio,
        gravity=g_lattice,
        enable_rotation=enable_rotation,
        retraction_dx=retraction_dx,
        rotation_coupling=rotation_coupling,
        # Δs/h 표준: Uhlmann 2005 Eq. 21, Wang 2008 §3.1, Breugem 2012 Eq. 9 만장일치 1.0
        marker_spacing_factor=1.0,
        nx_formula="standard",
        diagnostics_interval=diagnostics_interval,
        time_integrator=time_integrator,
        incompressible_lbgk=incompressible_lbgk,
        settling_inertia_model=settling_inertia_model,
        sedimentation_stop_at_contact=sedimentation_stop_at_contact,
        sedimentation_stop_offset_d=sedimentation_stop_offset_d,
        sedimentation_reference_basis=sedimentation_reference_basis,
    )

    if verbose:
        print(f"[Sedimentation] rho_ratio={rho_ratio}, Ar={Ar:.0f}, "
              f"case={case_name}")
        print(f"  NN={NN}, D_lat={lattice_D:.0f}, tau={tau:.4f}")
        print(f"  g_lat={g_lattice:.6e}, u_t_est={u_t_est:.4f}, "
              f"Ma_est={Ma_est:.4f}")
        print(f"  domain: xmax={xmax}, ymax={ymax}, "
              f"center=({cx:.6f}, {cy})")
        if center_x_offset_dx != 0.0:
            print(f"  x_offset: {center_x_offset_dx:.2f} dx "
                  f"= {center_x_offset_dx * dx:.6e} domain units")

    return cfg


def make_single_particle_config_reference_benchmark(
    rho_ratio: float = 1.5,
    ibm_method: str = "DF",
    collision_model: str = "BGK",
    delta_type: str = "peskin4pt",
    mdf_iterations: int = 20,
    time_integrator: str = "euler_explicit",
    marker_spacing_factor: float = 1.0,
    incompressible_lbgk: bool = False,
    settling_inertia_model: str = "none",
    sedimentation_stop_at_contact: bool = True,
    sedimentation_stop_offset_d: float = 0.0,
    sedimentation_reference_basis: str = "particle_basis",
    verbose: bool = True,
) -> SimConfig:
    """Wang 2008 IJMF §3.1 single-particle setup를 LBM 격자에 직접 사상한 전용 경로.

    Wang 2008 본문 §3.1 (pp. 288-292) 기준:
      - domain: (0, 2 cm) × (0, 6 cm)
      - particle diameter: 0.25 cm
      - initial center: (1 cm, 4 cm)
      - rho_p/rho_f = 1.25 또는 1.5
      - h = 1/256 cm (D/h = 64)
      - dt = 3.125e-6 s (rho=1.25), 3.125e-5 s (rho=1.5)

    기존 make_sedimentation_config()의 저-Mach 재스케일과 달리, 여기서는 Wang의
    물리 h, dt를 그대로 lattice로 옮겨 D_lat, nu_lat, g_lat를 고정한다.
    """
    if abs(rho_ratio - 1.25) < 1e-6:
        mu_phys = 0.1
        dt_phys = 3.125e-6
        max_steps = 300_000
    elif abs(rho_ratio - 1.5) < 1e-6:
        mu_phys = 0.01
        dt_phys = 3.125e-5
        max_steps = 40_000
    else:
        raise ValueError("make_single_particle_config_reference_benchmark는 rho_ratio 1.25 또는 1.5만 지원합니다.")

    W_phys = 2.0
    H_phys = 6.0
    d_phys = 0.25
    g_phys = 980.0
    rho_f = 1.0
    nu_phys = mu_phys / rho_f
    dx_phys = 1.0 / 256.0
    D_ratio = d_phys / W_phys
    delta_rho = rho_ratio - 1.0

    NN = int(round(W_phys / dx_phys)) + 1  # 513
    xmax = 1.0
    ymax = H_phys / W_phys  # 3.0
    lattice_D = D_ratio * (NN - 1)  # 64
    nu_lat = nu_phys * dt_phys / (dx_phys**2)
    tau_target = 3.0 * nu_lat + 0.5
    g_lattice = g_phys * dt_phys**2 / dx_phys
    u_ref = float(np.sqrt(delta_rho * g_lattice * lattice_D))
    Ma_est = u_ref * np.sqrt(3.0)
    Re_est = u_ref * lattice_D / nu_lat

    cfg = SimConfig(
        Re=Re_est,
        NN=NN,
        xmax=xmax,
        ymax=ymax,
        cylinder_center=(0.5, 2.0),
        cylinder_D_ratio=D_ratio,
        lattice_u=u_ref,
        inflow_u=0.0,
        bc_type="settling_channel",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=mdf_iterations,
        max_steps=max_steps,
        use_convergence=False,
        check_interval=500,
        motion_type="sedimentation",
        rho_ratio=rho_ratio,
        gravity=g_lattice,
        enable_rotation=True,
        rotation_coupling="indirect",
        marker_spacing_factor=marker_spacing_factor,
        nx_formula="standard",
        time_integrator=time_integrator,
        incompressible_lbgk=incompressible_lbgk,
        settling_inertia_model=settling_inertia_model,
        sedimentation_stop_at_contact=sedimentation_stop_at_contact,
        sedimentation_stop_offset_d=sedimentation_stop_offset_d,
        sedimentation_reference_basis=sedimentation_reference_basis,
        sedimentation_euler_update_scheme="trapezoidal",
    )

    if verbose:
        print(f"[SingleParticle-Wang2008] rho_ratio={rho_ratio}")
        print(f"  NN={NN}, D_lat={lattice_D:.0f}, tau={tau_target:.5f}")
        print(f"  nu_lat={nu_lat:.6f}, g_lat={g_lattice:.6e}, Ma_est={Ma_est:.4f}")
        print(f"  dt_phys={dt_phys:.6e} s, h_phys={dx_phys:.6e} cm")
        print(f"  Re_est={Re_est:.2f}, max_steps={max_steps}")

    return cfg


def make_two_particle_config(
    NN: int = 1281,
    ibm_method: str = "DF",
    collision_model: str = "BGK",
    delta_type: str = "peskin4pt",
    rotation_coupling: str = "semi_implicit",
    enable_rotation: bool = True,
    mdf_iterations: int = 20,
    time_integrator: str = "verlet",
    incompressible_lbgk: bool = False,
    settling_inertia_model: str = "none",
    sedimentation_stop_at_contact: bool = False,
    sedimentation_stop_offset_d: float = 2.0,
    sedimentation_reference_basis: str = "standard",
    verbose: bool = True,
) -> SimConfig:
    """Uhlmann 2005 §5.2.2 pure wake interaction 2입자 시나리오.

    도메인: [0, 10] × [-1, 1] (Uhlmann 원문 좌표)
    입자 직경 d=0.2, 채널 폭 W=2.0, W/d=10
    입자 1 (heavy): ρ=1.5, 초기 (0.8, center-0.13)
    입자 2 (light): ρ=1.25, 초기 (1.2, center+0.13)
    중력: +x 방향 (수평 침강)

    물리 파라미터 (Uhlmann 2005 §5.2.2 원문):
      ν = 8×10⁻⁴, g = 9.81 (준SI 단위계, §5.2.1 CGS와 다름)
    """
    W_phys = 2.0
    d_phys = 0.2   # W/d = 10 (Uhlmann 2005 §5.2.2)
    g_phys = 9.81   # Uhlmann 2005 §5.2.2 (준SI, §5.2.1의 CGS 980과 다름)
    D_ratio = d_phys / W_phys  # 0.1

    # Uhlmann 2005 §5.2.2 물리 파라미터
    rho_heavy = 1.5
    nu_phys = 0.0008  # Uhlmann 2005 §5.2.2 (§5.2.1의 0.01과 다름)
    delta_rho = rho_heavy - 1.0
    Ar = delta_rho * g_phys * d_phys**3 / nu_phys**2

    # 도메인: [0,10]×[-1,1] → 코드 좌표 [0,5]×[0,1] (W로 정규화)
    # x: 낙하 방향 (+x 중력), y: 횡방향 (채널 폭 W)
    xmax = 5.0   # 10d / (2*D_ratio) = 10*0.2 / (2*0.1*2) = 5
    ymax = 1.0   # W / W = 1 ([-1,1] → [0,1] shift)

    # 격자 파라미터 (Ma 제약)
    nu_lat_min = (0.52 - 0.5) / 3.0
    safety_factor = 4.0
    D_lat_min = safety_factor * np.sqrt(3.0 * Ar) * nu_lat_min / 0.08
    NN_min = int(np.ceil(D_lat_min / D_ratio)) + 1

    if NN < NN_min:
        import warnings
        warnings.warn(f"NN={NN} < NN_min={NN_min}. 안정성 미보장.", stacklevel=2)

    dx = 1.0 / (NN - 1)
    lattice_D = D_ratio * (NN - 1)
    lattice_r = lattice_D / 2.0

    # ν_lat: Ma 제약의 80%
    nu_lat_max = 0.08 * lattice_D / np.sqrt(3.0 * Ar)
    nu_lat = max(nu_lat_max * 0.8, nu_lat_min)
    tau = 3.0 * nu_lat + 0.5

    # g_lattice
    g_lattice = Ar * nu_lat**2 / (delta_rho * lattice_D**3)

    # 종단 속도 추정 (heavy 입자 기준)
    u_t_est = np.sqrt(delta_rho * g_lattice * lattice_D)
    Ma_est = u_t_est * np.sqrt(3.0)
    lattice_u = max(u_t_est, 0.01)
    Re_input = lattice_u * lattice_D / nu_lat

    # max_steps: 도메인 길이 / 종단속도 × 3 (여유)
    channel_length = xmax / dx
    if u_t_est > 1e-10:
        max_steps = int(3.0 * channel_length / u_t_est)
    else:
        max_steps = 1_000_000
    # 입자 초기 위치 (도메인 좌표 = 물리좌표/W)
    # Uhlmann 2005 §5.2.2 원문: x₁=(0.8,-0.13), x₂=(1.2,+0.13) in [0,10]×[-1,1]
    # 도메인 변환: x_code = x_phys/W, y_code = (y_phys + W/2)/W
    p1_cx = 0.8 / W_phys    # 0.4
    p1_cy = (-0.13 + W_phys / 2) / W_phys  # 0.435
    p2_cx = 1.2 / W_phys    # 0.6
    p2_cy = (0.13 + W_phys / 2) / W_phys   # 0.565

    # cylinder_center: 대표 입자 (heavy) 기준
    cfg = SimConfig(
        Re=Re_input,
        NN=NN,
        xmax=xmax,
        ymax=ymax,
        cylinder_center=(p1_cx, p1_cy),
        cylinder_D_ratio=D_ratio,
        lattice_u=lattice_u,
        inflow_u=0.0,
        bc_type="settling_channel",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=mdf_iterations,
        max_steps=max_steps,
        use_convergence=False,
        check_interval=100,
        motion_type="sedimentation",
        rho_ratio=rho_heavy,
        gravity=g_lattice,
        enable_rotation=enable_rotation,
        rotation_coupling=rotation_coupling,
        gravity_direction="right",
        time_integrator=time_integrator,
        incompressible_lbgk=incompressible_lbgk,
        settling_inertia_model=settling_inertia_model,
        sedimentation_stop_at_contact=sedimentation_stop_at_contact,
        sedimentation_stop_offset_d=sedimentation_stop_offset_d,
        sedimentation_reference_basis=sedimentation_reference_basis,
        particles_config=[
            {"center": (p1_cx, p1_cy), "rho_ratio": 1.5},
            {"center": (p2_cx, p2_cy), "rho_ratio": 1.25},
        ],
        # Δs/h 표준: Uhlmann 2005 Eq. 21, Wang 2008 §3.1, Breugem 2012 Eq. 9 만장일치 1.0
        marker_spacing_factor=1.0,
        nx_formula="standard",
    )

    if verbose:
        print(f"[TwoParticle] Uhlmann 2005 §5.2.2 pure wake interaction")
        print(f"  NN={NN}, D_lat={lattice_D:.0f}, tau={tau:.4f}")
        print(f"  g_lat={g_lattice:.6e}, Ma={Ma_est:.4f}")
        print(f"  P1 (heavy, ρ=1.5): ({p1_cx:.4f}, {p1_cy:.4f})")
        print(f"  P2 (light, ρ=1.25): ({p2_cx:.4f}, {p2_cy:.4f})")
        print(f"  domain: [{0},{xmax}]×[{0},{ymax}], gravity: +x")

    return cfg


def make_two_particle_config_reference_benchmark(
    ibm_method: str = "DF",
    collision_model: str = "BGK",
    delta_type: str = "peskin4pt",
    rotation_coupling: str = "semi_implicit",
    enable_rotation: bool = True,
    mdf_iterations: int = 20,
    time_integrator: str = "euler_explicit",
    marker_spacing_factor: float = 0.83,
    incompressible_lbgk: bool = True,
    settling_inertia_model: str = "explicit_history",
    sedimentation_stop_at_contact: bool = False,
    sedimentation_stop_offset_d: float = 2.0,
    sedimentation_reference_basis: str = "standard",
    streamwise_extent_factor: float = 1.0,
    particle_selection: str = "both",
    grid_refinement_factor: float = 1.0,
    verbose: bool = True,
) -> SimConfig:
    """Majumder 2023 CPM §4.4 / Uhlmann 2005 §5.2.2 원문 정합 재현.

    particle_selection: "both"(기본, 기존 경로 불변) 또는 "light_only" — heavy를 제거한
      matched isolated-light configuration. 1원소 particles_config로 다입자 런타임을 유지한다.
    grid_refinement_factor(κ): 확산 스케일링 격자 세분 — Δx→Δx/κ, Δt→Δt/κ²(τ=0.59 고정으로
      ν_lat 불변), g_lat·max_steps·check_interval(공통 t* cadence)이 함께 연동된다.

    Majumder 2023 (Computational Particle Mechanics 10:155-172) §4.4 "Sedimentation
    of two particles of different weights" — Uhlmann 2005 §5.2.2 pure wake
    interaction과 동일 setup을 LBM으로 구현, Re_max = 276.61 / 231.41 얻음
    (Uhlmann 280/230과 각각 -1.2%, +0.6% 차이).

    Majumder 원문 파라미터 (p16-17):
        domain: [0, 10] × [-1, 1] m
        D_p = 0.2 m, W = 2.0 m, D/W = 0.1
        ρ1/ρf = 1.5 (heavy), ρ2/ρf = 1.25 (light)
        ν_f = 8 × 10⁻⁴ m²/s
        g = 9.81 m/s²
        X1(t=0) = (0.8, -0.13), X2(t=0) = (1.2, +0.13)
        gravity direction: -y (standard sedimentation)

    Majumder LBM 이산화:
        Δx = Δy = 0.0025 m (4000 × 800 lattice units)
        τ = 0.59 → ν_lat = 0.03
        Δt = 2.344 × 10⁻⁴ s
        g_lat = 2.156 × 10⁻⁴ lattice-units
        Ma ≈ 0.16 (standard BGK 제약 0.08 초과 허용)

    우리 코드 좌표 변환:
        x: [0, 10] → [0, 5] (W=2로 정규화, +x 중력 방향)
        y: [-1, 1] → [0, 1]
        NN = 801 (y-direction lattice, W/Δx = 2/0.0025 + 1)
        nx = 4001 (x-direction lattice = nx_formula 자동)
        D_lat = 0.1 × 800 = 80 (D/h=80, Majumder와 일치)

    주의: Ma=0.16는 make_sedimentation_config의 기본 제약 0.08을 초과.
    BGK 수치 안정성은 τ=0.59 마진(0.09 vs BGK limit 0.5+ε) 덕분에 보장됨.
    Majumder도 동일 τ=0.59 사용.
    """
    W_phys = 2.0
    d_phys = 0.2
    g_phys = 9.81
    nu_phys = 0.0008
    rho_heavy = 1.5
    rho_light = 1.25
    delta_rho_heavy = rho_heavy - 1.0

    # Majumder 고정 격자/시간 파라미터 (κ=grid_refinement_factor 확산 스케일링)
    if grid_refinement_factor <= 0:
        raise ValueError(f"grid_refinement_factor > 0 필요: {grid_refinement_factor}")
    dx_phys = 0.0025 / grid_refinement_factor   # m
    dt_phys = 2.344e-4 / grid_refinement_factor**2  # s (τ=0.59 기반, ν_lat 불변)
    tau_target = 0.59
    if abs(2.0 / dx_phys - round(2.0 / dx_phys)) > 1e-9:
        raise ValueError(
            f"grid_refinement_factor={grid_refinement_factor}는 정수 격자를 만들지 않습니다"
        )
    nu_lat_target = (tau_target - 0.5) / 3.0  # = 0.03

    # 격자 해상도 (Majumder와 정확히 일치)
    NN = int(round(W_phys / dx_phys)) + 1  # 801
    D_ratio = d_phys / W_phys  # 0.1
    lattice_D = D_ratio * (NN - 1)  # 80
    lattice_r = lattice_D / 2.0

    # 도메인 (코드 좌표: W로 정규화)
    # streamwise_extent_factor=1.0 → Majumder default 50D (xmax=5.0)
    # streamwise_extent_factor=1.2 → 60D 연장 도메인 (xmax=6.0)
    xmax = (10.0 / W_phys) * streamwise_extent_factor
    ymax = 1.0  # 2 / 2

    # g_lat 검산: g_phys × Δt² / Δx = 9.81 × 5.494e-8 / 0.0025 = 2.156e-4
    g_lattice = g_phys * dt_phys**2 / dx_phys

    # u_t (heavy 종단 추정) — Ma 계산용
    u_t_est = float(np.sqrt(delta_rho_heavy * g_lattice * lattice_D))
    Ma_est = u_t_est * np.sqrt(3.0)
    Re_est = u_t_est * lattice_D / nu_lat_target

    # 입자 초기 위치 (도메인 좌표)
    # Majumder/Uhlmann: x1=(0.8, -0.13), x2=(1.2, 0.13)
    # 우리 좌표: x_code = x_phys/W, y_code = (y_phys + 1)/2
    p1_cx = 0.8 / W_phys   # 0.4
    p1_cy = (-0.13 + 1.0) / 2.0  # 0.435
    p2_cx = 1.2 / W_phys   # 0.6
    p2_cy = (0.13 + 1.0) / 2.0   # 0.565

    # N-7 matched isolated-light configuration — heavy 제거, 1원소 목록으로 다입자 런타임 유지
    if particle_selection == "both":
        particles_cfg = [
            {"center": (p1_cx, p1_cy), "rho_ratio": rho_heavy},
            {"center": (p2_cx, p2_cy), "rho_ratio": rho_light},
        ]
        legacy_center = (p1_cx, p1_cy)
        legacy_rho = rho_heavy
    elif particle_selection == "light_only":
        particles_cfg = [
            {"center": (p2_cx, p2_cy), "rho_ratio": rho_light},
        ]
        legacy_center = (p2_cx, p2_cy)
        legacy_rho = rho_light
    else:
        raise ValueError(f"particle_selection은 'both' 또는 'light_only'만 지원: {particle_selection}")

    # max_steps: Majumder 종료 조건 "heavier particle reaches within 2d from base"
    # 낙하 거리 ≈ 10 - 0.8 - 2×0.2 = 8.8 m (default), 60D 확장 시 비례 증가
    # 시간 ≈ 8.8 / u_t_phys ≈ 8.8 / 0.99 ≈ 8.9 s
    # step = 8.9 / 2.344e-4 ≈ 38000
    # 여유 1.5× = 57000 → 60000 baseline
    max_steps = int(round(60000 * streamwise_extent_factor * grid_refinement_factor**2))
    check_interval_scaled = int(round(500 * grid_refinement_factor**2))

    cfg = SimConfig(
        Re=Re_est,
        NN=NN,
        xmax=xmax,
        ymax=ymax,
        cylinder_center=legacy_center,
        cylinder_D_ratio=D_ratio,
        lattice_u=max(u_t_est, 0.01),
        inflow_u=0.0,
        bc_type="settling_channel",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=mdf_iterations,
        max_steps=max_steps,
        use_convergence=False,
        check_interval=check_interval_scaled,
        motion_type="sedimentation",
        rho_ratio=legacy_rho,
        gravity=g_lattice,
        enable_rotation=enable_rotation,
        rotation_coupling=rotation_coupling,
        gravity_direction="right",  # +x = 물리 -y (도메인 회전)
        particles_config=particles_cfg,
        marker_spacing_factor=marker_spacing_factor,
        nx_formula="standard",
        time_integrator=time_integrator,
        incompressible_lbgk=incompressible_lbgk,
        settling_inertia_model=settling_inertia_model,
        sedimentation_stop_at_contact=sedimentation_stop_at_contact,
        sedimentation_stop_offset_d=sedimentation_stop_offset_d,
        sedimentation_reference_basis=sedimentation_reference_basis,
    )

    if verbose:
        print(f"[TwoParticle-Majumder] Majumder 2023 §4.4 / Uhlmann 2005 §5.2.2")
        print(f"  NN={NN}, D_lat={lattice_D:.0f}, tau={tau_target:.4f}")
        print(f"  nu_lat={nu_lat_target:.6f}, g_lat={g_lattice:.6e}")
        print(f"  Ma_est={Ma_est:.4f} (> 0.08 BGK 보통 제약, τ 마진으로 허용)")
        print(f"  Re_est (heavy)={Re_est:.2f} (목표: Re_max ≈ 276)")
        print(f"  P1 (heavy, ρ=1.5): ({p1_cx:.4f}, {p1_cy:.4f})")
        print(f"  P2 (light, ρ=1.25): ({p2_cx:.4f}, {p2_cy:.4f})")
        print(f"  max_steps={max_steps}, check_interval=500")
        print(f"  time_integrator={time_integrator}, inertia={settling_inertia_model}")
        print(f"  marker_spacing_factor={marker_spacing_factor}")

    return cfg


# === CLI ===
if __name__ == "__main__":
    import sys

    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 1.01
    method = sys.argv[2] if len(sys.argv) > 2 else "DF"
    delta = sys.argv[3] if len(sys.argv) > 3 else "peskin4pt"
    coll = sys.argv[4] if len(sys.argv) > 4 else "BGK"

    cfg = make_sedimentation_config(
        rho_ratio=rho, ibm_method=method, delta_type=delta,
        collision_model=coll,
    )

    result = run(cfg, verbose=True)

    history = result.get("sedimentation_history", [])
    if history:
        last = history[-1]
        print(f"\nFinal: vy*={last['vy_star']:.6f}, y*={last['y_star']:.4f}")
