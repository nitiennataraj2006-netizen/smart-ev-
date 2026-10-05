"""
SMART-EV 2.0: Unit Test Suite
KPIT Sparkle Project

Tests:
1. Battery ECM and Extended Kalman Filter (EKF) state estimation
2. Split Conformal Prediction coverage (>=95%) and MAE (<4.5%)
3. Dempster-Shafer sensor trust fusion and Byzantine spoof gating
4. Fail-safe arbiter mode switching and dynamic fallback (>12% half-width)
5. Zero-trust V2V validator latency/loss rejection
6. Physical energy conservation constraints
"""

import math
import numpy as np
import pytest

from sim.battery import BatteryECM, BatteryEKF, PhysicsRangeEstimator
from sim.conformal import ConformalRangePredictor, generate_synthetic_drive_cycles
from sim.trust import MassFunction, MultiSensorTrustFusion, combine_dempster_shafer
from sim.arbiter import FailSafeArbiter, OperationalMode
from sim.v2v import BasicSafetyMessage, ZeroTrustValidator
from sim.thermal import CellThermalModel
from sim.platoon import CACCPlatoonModel


class TestBatteryEKF:
    """Unit tests for Battery ECM and Extended Kalman Filter."""

    def test_ekf_soc_tracking(self) -> None:
        ecm = BatteryECM(capacity_ah=150.0, r0_ref=0.045, r1_ref=0.025, c1_ref=2400.0)
        ekf = BatteryEKF(ecm=ecm, initial_soc=0.80, dt=0.020)

        current_discharge_a = 45.0  # Discharge current
        # Simulate 100 steps (2.0 seconds)
        for _ in range(100):
            # Simulated true terminal voltage
            v_oc = ecm.open_circuit_voltage(ekf.x[0, 0])
            v_meas = v_oc - current_discharge_a * ecm.r0_ref - 0.05
            soc_est, v1_est, r0_est = ekf.step(
                current_a=current_discharge_a,
                measured_voltage=v_meas,
                temp_c=25.0,
            )

        # SoC should have decreased during discharge
        assert 0.70 < soc_est < 0.80
        # Internal resistance estimate must be strictly positive and realistic
        assert 0.020 <= r0_est <= 0.150
        # Filter innovation magnitude must be bounded
        assert abs(ekf.innovation) < 2.0

    def test_ocv_derivatives_positive(self) -> None:
        ecm = BatteryECM()
        for soc in np.linspace(0.10, 0.90, 9):
            docv = ecm.docv_dsoc(soc)
            assert docv > 0.0, f"dVoc/dSoC must be strictly positive at SoC {soc}"


class TestConformalPrediction:
    """Unit tests for Split Conformal Predictor."""

    def test_coverage_and_mae_targets(self) -> None:
        predictor = ConformalRangePredictor(alpha=0.05, seed=42)
        _, y_true, y_pred, sigmas = generate_synthetic_drive_cycles(n_samples=1200, seed=42)

        # Calibrate on 800 samples
        q_hat = predictor.calibrate(y_true[:800], y_pred[:800], sigmas[:800])
        assert q_hat > 0.0

        # Evaluate on remaining 400 held-out samples
        metrics = predictor.evaluate_test_set(y_true[800:], y_pred[800:], sigmas[800:])

        # Core requirement: Target Coverage >= 95.0%
        assert metrics["coverage_pct"] >= 95.0, (
            f"Conformal coverage {metrics['coverage_pct']:.2f}% failed target >= 95.0%"
        )

        # Core requirement: Target MAE < 4.5%
        assert metrics["mae_pct"] < 4.50, (
            f"Conformal MAE {metrics['mae_pct']:.2f}% failed target < 4.5%"
        )

    def test_prediction_interval_validity(self) -> None:
        predictor = ConformalRangePredictor(alpha=0.05, seed=42)
        predictor.q_hat = 1.96
        predictor.is_calibrated = True

        pt, lo, hi = predictor.predict_interval(
            point_prediction_km=250.0,
            speed_kmh=80.0,
            temp_c=25.0,
            grade_pct=0.0,
            hvac_power_w=500.0,
        )

        assert lo >= 0.0
        assert lo < pt < hi
        assert (hi - lo) > 5.0


class TestDempsterShaferFusion:
    """Unit tests for Dempster-Shafer sensor trust fusion."""

    def test_nominal_fusion(self) -> None:
        fusion = MultiSensorTrustFusion()
        radar = MassFunction(m_safe=0.90, m_fault=0.02, m_theta=0.08)
        v2v = MassFunction(m_safe=0.92, m_fault=0.02, m_theta=0.06)
        imu = MassFunction(m_safe=0.95, m_fault=0.01, m_theta=0.04)

        res = fusion.fuse(radar, v2v, imu)
        assert res["decision"] == "NOMINAL_SAFE"
        assert res["trust_score"] >= 0.85
        assert res["conflict_k"] < 0.20
        assert not res["phantom_braking_blocked"]

    def test_byzantine_spoof_rejection(self) -> None:
        fusion = MultiSensorTrustFusion()
        # V2V reports high hazard / emergency brake, but Radar & IMU verify open road
        radar = MassFunction(m_safe=0.92, m_fault=0.02, m_theta=0.06)
        v2v = MassFunction(m_safe=0.02, m_fault=0.95, m_theta=0.03)  # Byzantine attack
        imu = MassFunction(m_safe=0.91, m_fault=0.02, m_theta=0.07)

        res = fusion.fuse(radar, v2v, imu)
        assert res["decision"] == "BYZANTINE_SPOOF_GATED"
        assert res["phantom_braking_blocked"] is True
        assert res["action"] == "DISCARD_V2V_USE_RADAR_IMU"
        assert res["conflict_k"] > 0.80


class TestFailSafeArbiter:
    """Unit tests for deterministic safety arbiter and mode switching."""

    def test_uncertainty_fallback_transition(self) -> None:
        arbiter = FailSafeArbiter(half_width_fallback_threshold=0.12)
        assert arbiter.current_mode == OperationalMode.AGGRESSIVE

        # Tight interval (half-width 10 km on 200 km = 5.0% < 12%) -> AGGRESSIVE
        res1 = arbiter.evaluate(
            timestamp_s=1.0,
            range_point_km=200.0,
            range_lo_km=190.0,
            range_hi_km=210.0,
        )
        assert res1["mode"] == OperationalMode.AGGRESSIVE.value

        # High uncertainty interval (half-width 30 km on 200 km = 15.0% > 12%) -> CONSERVATIVE
        res2 = arbiter.evaluate(
            timestamp_s=2.0,
            range_point_km=200.0,
            range_lo_km=170.0,
            range_hi_km=230.0,
        )
        assert res2["mode"] == OperationalMode.CONSERVATIVE.value
        assert len(arbiter.transition_log) == 1

    def test_hard_failsafe_on_thermal_precursor(self) -> None:
        arbiter = FailSafeArbiter()
        res = arbiter.evaluate(
            timestamp_s=5.0,
            range_point_km=250.0,
            range_lo_km=240.0,
            range_hi_km=260.0,
            thermal_precursor_active=True,
        )
        assert res["mode"] == OperationalMode.FAIL_SAFE.value
        # Fail-safe execution latency must be well under 50 ms budget
        assert res["fallback_latency_ms"] < 50.0

    def test_hard_failsafe_on_v2v_latency_loss(self) -> None:
        arbiter = FailSafeArbiter()
        # Excessive latency (120 ms > 100 ms)
        res_lat = arbiter.evaluate(
            timestamp_s=1.0,
            range_point_km=200.0,
            range_lo_km=195.0,
            range_hi_km=205.0,
            v2v_latency_ms=125.0,
        )
        assert res_lat["mode"] == OperationalMode.FAIL_SAFE.value

        # Excessive packet loss (8% > 5%)
        res_loss = arbiter.evaluate(
            timestamp_s=2.0,
            range_point_km=200.0,
            range_lo_km=195.0,
            range_hi_km=205.0,
            v2v_loss_rate=0.09,
        )
        assert res_loss["mode"] == OperationalMode.FAIL_SAFE.value


class TestEnergyConservation:
    """Unit tests for physical energy conservation constraints."""

    def test_energy_non_perpetual(self) -> None:
        estimator = PhysicsRangeEstimator(pack_nominal_kwh=60.0)

        # Uphill at 5% grade must strictly demand positive power
        p_uphill = estimator.compute_instantaneous_power(
            speed_mps=20.0,
            grade_pct=5.0,
            temp_c=25.0,
        )
        assert p_uphill["p_total_w"] > 0.0
        assert p_uphill["f_grade_n"] > 0.0

        # Range estimate cannot be infinite on steep mountain descent
        range_descent = estimator.estimate_remaining_range_km(
            soc=0.80,
            speed_kmh=65.0,
            grade_pct=-6.0,
        )
        assert 0.0 < range_descent <= 600.0
