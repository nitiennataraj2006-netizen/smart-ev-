"""
SMART-EV 2.0: Battery Digital Twin and Dual-Loop Range Estimation
KPIT Sparkle Project

Inner Loop (20 ms / 50 Hz):
    Extended Kalman Filter (EKF) on a Thevenin Equivalent Circuit Model (ECM: R0 + R1||C1)
    to estimate State-of-Charge (SoC) and internal resistance (R_int).

Outer Loop (1 s / 1 Hz):
    Physics-constrained predictive range model integrating ambient temperature,
    C-rate history, State-of-Health (SoH), road grade, HVAC power, and aerodynamic drag.
    Strictly enforces energy conservation (no non-physical positive energy without external recharge).
"""

from typing import Tuple, Dict, Any, Optional
import math
import numpy as np


class BatteryECM:
    """
    1-RC Thevenin Equivalent Circuit Model for EV Li-ion battery pack.
    States: [SoC, V1]
    Parameters:
        Q_nom: Nominal pack capacity (Ah)
        R0: Ohmic internal resistance (Ohms)
        R1: Polarization resistance (Ohms)
        C1: Polarization capacitance (Farads)
        tau: Time constant R1 * C1 (s)
    """

    def __init__(
        self,
        capacity_ah: float = 150.0,
        r0_ref: float = 0.045,
        r1_ref: float = 0.025,
        c1_ref: float = 2400.0,
        t_ref_c: float = 25.0,
        nominal_voltage: float = 380.0,
    ) -> None:
        self.capacity_ah: float = capacity_ah
        self.capacity_coulombs: float = capacity_ah * 3600.0
        self.r0_ref: float = r0_ref
        self.r1_ref: float = r1_ref
        self.c1_ref: float = c1_ref
        self.t_ref_c: float = t_ref_c
        self.nominal_voltage: float = nominal_voltage

    def open_circuit_voltage(self, soc: float) -> float:
        """
        NMC/LFP open circuit voltage curve for 96s pack.
        Clamped to [0.01, 0.99] to prevent mathematical singularities.
        """
        s = float(np.clip(soc, 0.01, 0.99))
        # 96-cell series chemistry empirical polynomial + logarithmic tail model
        v_cell = (
            3.42
            + 0.65 * s
            + 0.12 * math.log(s + 0.005)
            - 0.08 * math.log(1.005 - s)
            - 0.15 * (s - 0.5) ** 3
        )
        return float(96.0 * v_cell)

    def docv_dsoc(self, soc: float) -> float:
        """Derivative dVoc/dSoC for EKF Jacobian linearization."""
        s = float(np.clip(soc, 0.01, 0.99))
        dv_cell = (
            0.65
            + 0.12 / (s + 0.005)
            + 0.08 / (1.005 - s)
            - 0.45 * (s - 0.5) ** 2
        )
        return float(96.0 * dv_cell)

    def temperature_adjusted_r0(self, temp_c: float) -> float:
        """
        Arrhenius relation for ohmic resistance R0 as a function of temperature.
        At cold temperatures (-10 C), internal resistance increases dramatically.
        """
        t_kelvin = temp_c + 273.15
        t_ref_k = self.t_ref_c + 273.15
        ea_over_r = 2100.0  # Activation energy / ideal gas constant (K)
        r0 = self.r0_ref * math.exp(ea_over_r * (1.0 / t_kelvin - 1.0 / t_ref_k))
        return float(np.clip(r0, 0.020, 0.250))

    def temperature_adjusted_capacity(self, temp_c: float, soh: float = 1.0) -> float:
        """
        Available usable capacity derating under cold/extreme conditions and SoH.
        At -10 C, usable electrochemical capacity drops by ~18-20%.
        """
        if temp_c >= 25.0:
            temp_derate = 1.0 - 0.0015 * min(temp_c - 25.0, 20.0)
        else:
            temp_derate = 1.0 - 0.0075 * (25.0 - temp_c)
        temp_derate = float(np.clip(temp_derate, 0.65, 1.02))
        return float(self.capacity_ah * soh * temp_derate)


class BatteryEKF:
    """
    Extended Kalman Filter for online SoC and internal resistance tracking.
    Runs at the inner loop rate (dt = 0.020 s = 20 ms).
    State vector: x = [SoC, V1]^T
    Parameter tracker: R0 (estimated via residual innovation)
    """

    def __init__(
        self,
        ecm: BatteryECM,
        initial_soc: float = 0.80,
        dt: float = 0.020,
    ) -> None:
        self.ecm: BatteryECM = ecm
        self.dt: float = dt

        # State vector [SoC, V1]
        self.x: np.ndarray = np.array([[initial_soc], [0.0]], dtype=float)

        # State covariance P
        self.P: np.ndarray = np.diag([1e-4, 1e-4]).astype(float)

        # Process noise covariance Q (low noise on SoC, higher on V1)
        self.Q: np.ndarray = np.diag([2e-7, 1e-5]).astype(float)

        # Measurement noise covariance R (voltage sensor accuracy ~10 mV)
        self.R: float = 0.015**2

        # Online estimated R0
        self.r0_est: float = ecm.r0_ref

        # Filter diagnostic history
        self.innovation: float = 0.0
        self.coulomb_integral_ah: float = 0.0

    def step(
        self,
        current_a: float,
        measured_voltage: float,
        temp_c: float = 25.0,
    ) -> Tuple[float, float, float]:
        """
        Execute one EKF iteration:
            current_a: positive for discharge, negative for regen charging
            measured_voltage: pack terminal voltage (V)
            temp_c: cell temperature (C)
        Returns:
            (estimated_soc, estimated_v1, estimated_r0)
        """
        r0 = self.ecm.temperature_adjusted_r0(temp_c)
        r1 = self.ecm.r1_ref
        c1 = self.ecm.c1_ref
        tau = r1 * c1
        cap_coulombs = self.ecm.temperature_adjusted_capacity(temp_c) * 3600.0

        # --- 1. PREDICT STEP ---
        # State transition: SoC_k = SoC_{k-1} - (dt / Q) * I
        # V1_k = exp(-dt/tau)*V1_{k-1} + (1 - exp(-dt/tau))*R1*I
        a_11 = 1.0
        a_12 = 0.0
        a_21 = 0.0
        a_22 = math.exp(-self.dt / tau)

        A = np.array([[a_11, a_12], [a_21, a_22]], dtype=float)

        soc_pred = float(self.x[0, 0] - (self.dt / cap_coulombs) * current_a)
        soc_pred = float(np.clip(soc_pred, 0.001, 0.999))
        v1_pred = float(a_22 * self.x[1, 0] + (1.0 - a_22) * r1 * current_a)

        x_pred = np.array([[soc_pred], [v1_pred]], dtype=float)
        P_pred = A @ self.P @ A.T + self.Q

        # --- 2. UPDATE STEP ---
        # Observation equation: V_t = Voc(SoC) - V1 - I * R0
        voc_pred = self.ecm.open_circuit_voltage(soc_pred)
        v_terminal_pred = voc_pred - v1_pred - current_a * self.r0_est

        # Jacobian H = [dVoc/dSoC, -1]
        h_soc = self.ecm.docv_dsoc(soc_pred)
        H = np.array([[h_soc, -1.0]], dtype=float)

        # Innovation (residual)
        y_tilde = measured_voltage - v_terminal_pred
        self.innovation = float(y_tilde)

        # Innovation covariance S
        s_val = float((H @ P_pred @ H.T)[0, 0] + self.R)
        self.S = s_val

        # Kalman gain K (2x1 column vector)
        K = (P_pred @ H.T) / s_val

        # State update
        self.x = x_pred + K * y_tilde
        self.x[0, 0] = float(np.clip(self.x[0, 0], 0.001, 0.999))

        # Covariance update (Joseph form for numerical stability)
        I_mat = np.eye(2, dtype=float)
        IKH = I_mat - K @ H
        self.P = IKH @ P_pred @ IKH.T + (K * self.R) @ K.T

        # --- 3. PARAMETER R0 TRACKING ---
        # Recursive least-squares adaptation of R0 when current is substantial
        if abs(current_a) > 15.0:
            lr = 0.0005
            r0_target = (voc_pred - self.x[1, 0] - measured_voltage) / (current_a + 1e-6)
            if 0.015 <= r0_target <= 0.30:
                self.r0_est += lr * (r0_target - self.r0_est)

        self.coulomb_integral_ah += (current_a * self.dt) / 3600.0

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.r0_est)


class PhysicsRangeEstimator:
    """
    Physics-Constrained Outer-Loop Range Model (1 s / 1 Hz).
    Strictly enforces energy conservation:
        Tractive Power: P_tract = (F_roll + F_aero + F_grade + F_accel) * v / eta_drivetrain
        HVAC Power: P_hvac(T_amb, T_set)
        Auxiliary Power: P_aux (ECU, radar, lighting)
        Ohmic Joule Loss: I^2 * R0
    """

    def __init__(
        self,
        mass_kg: float = 2080.0,
        cd: float = 0.280,
        frontal_area_m2: float = 2.42,
        crr: float = 0.0125,
        drivetrain_efficiency: float = 0.90,
        regen_efficiency: float = 0.78,
        aux_power_w: float = 450.0,
        pack_nominal_kwh: float = 60.0,
    ) -> None:
        self.mass_kg: float = mass_kg
        self.cd: float = cd
        self.frontal_area_m2: float = frontal_area_m2
        self.crr: float = crr
        self.drivetrain_eff: float = drivetrain_efficiency
        self.regen_eff: float = regen_efficiency
        self.aux_power_w: float = aux_power_w
        self.pack_nominal_kwh: float = pack_nominal_kwh
        self.g: float = 9.80665

    def air_density(self, temp_c: float, pressure_pa: float = 101325.0) -> float:
        """Air density as a function of temperature via ideal gas law."""
        r_spec = 287.058  # J/(kg*K)
        t_kelvin = temp_c + 273.15
        return float(pressure_pa / (r_spec * t_kelvin))

    def compute_instantaneous_power(
        self,
        speed_mps: float,
        grade_pct: float = 0.0,
        temp_c: float = 25.0,
        hvac_power_w: float = 0.0,
        accel_mps2: float = 0.0,
        cd_override: Optional[float] = None,
        soh: float = 1.0,
        r0_current: float = 0.045,
        is_city_cycle: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate physically conservative power demand at the battery pack.
        Energy conservation:
            Positive tractive power consumes energy through motor efficiency.
            Negative tractive power (regen) recaptures energy bounded by regen efficiency.
            Outputs are strictly non-perpetual (no negative power uphill without external pull).
        """
        v = max(0.1, speed_mps)
        cd_eff = cd_override if cd_override is not None else self.cd
        rho = self.air_density(temp_c)
        theta = math.atan(grade_pct / 100.0)

        # Cold temperature tire hardening increases rolling resistance
        crr_adj = self.crr * (1.0 + 0.006 * max(0.0, 25.0 - temp_c))

        # Forces in Newtons
        f_roll = crr_adj * self.mass_kg * self.g * math.cos(theta)
        f_aero = 0.5 * rho * cd_eff * self.frontal_area_m2 * (v**2)
        f_grade = self.mass_kg * self.g * math.sin(theta)
        f_accel = self.mass_kg * accel_mps2

        # Urban stop-and-go micro-acceleration energy dissipation
        f_city_inertia = (self.mass_kg * 0.22) if is_city_cycle else 0.0

        f_tractive = f_roll + f_aero + f_grade + f_accel + f_city_inertia

        # Mechanical power
        p_mech = f_tractive * v

        if p_mech >= 0.0:
            p_electrical = p_mech / self.drivetrain_eff
        else:
            # Regenerative braking: recapture limited by motor generator and battery acceptance
            p_electrical = p_mech * self.regen_eff
            # Physics guard: max 70 kW regen limit
            p_electrical = max(p_electrical, -70000.0)

        # Battery ohmic loss estimation (I^2 * R0)
        v_nom = 380.0
        i_est = p_electrical / v_nom
        p_ohmic = (i_est**2) * r0_current

        p_total_w = p_electrical + hvac_power_w + self.aux_power_w + p_ohmic

        return {
            "p_total_w": float(p_total_w),
            "p_mech_w": float(p_mech),
            "f_tractive_n": float(f_tractive),
            "f_roll_n": float(f_roll),
            "f_aero_n": float(f_aero),
            "f_grade_n": float(f_grade),
            "air_density": float(rho),
        }

    def estimate_remaining_range_km(
        self,
        soc: float,
        speed_kmh: float,
        temp_c: float = 25.0,
        grade_pct: float = 0.0,
        hvac_power_w: float = 0.0,
        cd_override: Optional[float] = None,
        soh: float = 1.0,
        r0_current: float = 0.045,
        is_city_cycle: bool = False,
    ) -> float:
        """
        Outer-loop physical range estimation (km).
        Computes realistic range based on remaining usable energy and specific consumption.
        Clamped to physically realistic bounds [0, 650] km.
        """
        # Usable pack capacity at given temperature and SoH
        if temp_c >= 25.0:
            temp_derate = 1.0 - 0.0022 * min(temp_c - 25.0, 25.0)
        else:
            temp_derate = 1.0 - 0.0085 * (25.0 - temp_c)
        temp_derate = float(np.clip(temp_derate, 0.65, 1.02))

        effective_pack_kwh = self.pack_nominal_kwh * soh * temp_derate
        remaining_energy_kwh = float(np.clip(soc, 0.0, 1.0)) * effective_pack_kwh

        speed_mps = speed_kmh / 3.6
        p_dict = self.compute_instantaneous_power(
            speed_mps=speed_mps,
            grade_pct=grade_pct,
            temp_c=temp_c,
            hvac_power_w=hvac_power_w,
            cd_override=cd_override,
            soh=soh,
            r0_current=r0_current,
            is_city_cycle=is_city_cycle,
        )

        p_total_w = p_dict["p_total_w"]

        # Specific consumption kWh/km
        # Physics guard: clamp to positive minimum to avoid infinite range on descent
        min_kwh_per_km = 0.165 if grade_pct < -2.0 else 0.080
        kwh_per_km = (p_total_w / 1000.0) / max(speed_kmh, 15.0)
        kwh_per_km_eff = max(min_kwh_per_km, kwh_per_km)

        range_km = remaining_energy_kwh / kwh_per_km_eff
        return float(np.clip(range_km, 0.0, 600.0))
