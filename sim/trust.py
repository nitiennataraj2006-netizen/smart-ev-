"""
SMART-EV 2.0: Dempster-Shafer Sensor Trust Fusion
KPIT Sparkle Project

Implements Dempster-Shafer Evidence Theory for multi-sensor trust fusion
combining onboard Millimeter-Wave Radar, V2V Wireless, and Inertial Measurement Unit (IMU).

Frame of Discernment: Theta = {Safe, Fault}
Power set: {empty, {Safe}, {Fault}, Theta}
where Theta represents epistemic ignorance/uncertainty.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


class MassFunction:
    """
    Basic Probability Assignment (BPA) / Mass Function for frame Theta = {Safe, Fault}.
    m_safe: belief assigned specifically to {Safe}
    m_fault: belief assigned specifically to {Fault}
    m_theta: unassigned belief assigned to ignorance Theta = {Safe, Fault}
    Condition: m_safe + m_fault + m_theta == 1.0, all >= 0.
    """

    def __init__(self, m_safe: float, m_fault: float, m_theta: float) -> None:
        total = m_safe + m_fault + m_theta
        if abs(total - 1.0) > 1e-4:
            if total > 0:
                m_safe /= total
                m_fault /= total
                m_theta /= total
            else:
                m_safe, m_fault, m_theta = 0.0, 0.0, 1.0

        self.m_safe: float = float(np.clip(m_safe, 0.0, 1.0))
        self.m_fault: float = float(np.clip(m_fault, 0.0, 1.0))
        self.m_theta: float = float(np.clip(m_theta, 0.0, 1.0))

        # Re-normalize to exact 1.0
        s = self.m_safe + self.m_fault + self.m_theta
        self.m_safe /= s
        self.m_fault /= s
        self.m_theta /= s

    def belief(self) -> Dict[str, float]:
        """Belief (pessimistic credibility) of singletons."""
        return {
            "safe": self.m_safe,
            "fault": self.m_fault,
        }

    def plausibility(self) -> Dict[str, float]:
        """Plausibility (optimistic credibility) of singletons."""
        return {
            "safe": self.m_safe + self.m_theta,
            "fault": self.m_fault + self.m_theta,
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            "m_safe": round(self.m_safe, 4),
            "m_fault": round(self.m_fault, 4),
            "m_theta": round(self.m_theta, 4),
        }


def combine_dempster_shafer(m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
    """
    Orthogonal sum combination of two independent evidence sources m1 and m2.
    Returns:
        (fused_mass, conflict_k)
    """
    # Focal elements:
    # m1: Safe, Fault, Theta
    # m2: Safe, Fault, Theta

    # Combinations:
    # Safe & Safe -> Safe
    # Safe & Fault -> Empty (Conflict)
    # Safe & Theta -> Safe
    # Fault & Safe -> Empty (Conflict)
    # Fault & Fault -> Fault
    # Fault & Theta -> Fault
    # Theta & Safe -> Safe
    # Theta & Fault -> Fault
    # Theta & Theta -> Theta

    # Empty set intersection (Conflict measure K)
    k_conflict = m1.m_safe * m2.m_fault + m1.m_fault * m2.m_safe

    # If sources are in total conflict (K == 1), fusion is undefined
    if k_conflict >= 0.9999:
        # Extreme conflict: preserve ignorance
        return MassFunction(0.0, 0.0, 1.0), float(k_conflict)

    scale = 1.0 / (1.0 - k_conflict)

    safe_mass = scale * (
        m1.m_safe * m2.m_safe
        + m1.m_safe * m2.m_theta
        + m1.m_theta * m2.m_safe
    )

    fault_mass = scale * (
        m1.m_fault * m2.m_fault
        + m1.m_fault * m2.m_theta
        + m1.m_theta * m2.m_fault
    )

    theta_mass = scale * (m1.m_theta * m2.m_theta)

    fused = MassFunction(safe_mass, fault_mass, theta_mass)
    return fused, float(k_conflict)


class MultiSensorTrustFusion:
    """
    Dempster-Shafer Fusion Engine for Radar, V2V, and IMU.
    Gates malicious or degraded sensor signals and guards against phantom braking.
    """

    def __init__(self) -> None:
        self.last_conflict: float = 0.0
        self.last_decision: str = "SAFE_NOMINAL"

    def fuse(
        self,
        radar_mass: MassFunction,
        v2v_mass: MassFunction,
        imu_mass: MassFunction,
    ) -> Dict[str, Any]:
        """
        Sequentially combine Radar, V2V, and IMU evidence.
        """
        # Step 1: Combine Radar and V2V
        m12, k1 = combine_dempster_shafer(radar_mass, v2v_mass)

        # Step 2: Combine result with IMU
        m_fused, k2 = combine_dempster_shafer(m12, imu_mass)

        # Aggregate conflict
        total_conflict = float(np.clip(1.0 - (1.0 - k1) * (1.0 - k2), 0.0, 1.0))
        self.last_conflict = total_conflict

        # Decision logic
        # High Safe belief with low conflict -> Safe Nominal
        # High Fault belief -> Confirmed Hazard
        # Byzantine condition: V2V claims Fault/Emergency, but Radar and IMU assert Safe
        is_spoofed = (v2v_mass.m_fault > 0.70) and (radar_mass.m_safe > 0.70) and (imu_mass.m_safe > 0.70)

        if is_spoofed:
            decision = "BYZANTINE_SPOOF_GATED"
            phantom_braking_blocked = True
            action = "DISCARD_V2V_USE_RADAR_IMU"
        elif m_fused.m_safe >= 0.75 and total_conflict < 0.35:
            decision = "NOMINAL_SAFE"
            phantom_braking_blocked = False
            action = "EXECUTE_NORMAL_CACC"
        elif m_fused.m_fault >= 0.75:
            decision = "HAZARD_CONFIRMED"
            phantom_braking_blocked = False
            action = "TRIGGER_AEB"
        elif m_fused.m_theta >= 0.50 or total_conflict >= 0.45:
            decision = "UNCERTAIN_FALLBACK"
            phantom_braking_blocked = True
            action = "SWITCH_TO_CONSERVATIVE_LOCAL_RADAR"
        else:
            decision = "CONSERVATIVE_CONFIRMED"
            phantom_braking_blocked = False
            action = "MAINTAIN_ENHANCED_GAP"

        self.last_decision = decision

        # Overall trust score [0.0, 1.0]
        trust_score = float(m_fused.m_safe * (1.0 - 0.5 * total_conflict))

        return {
            "decision": decision,
            "action": action,
            "trust_score": round(trust_score, 4),
            "conflict_k": round(total_conflict, 4),
            "phantom_braking_blocked": phantom_braking_blocked,
            "fused_mass": m_fused.to_dict(),
            "radar_mass": radar_mass.to_dict(),
            "v2v_mass": v2v_mass.to_dict(),
            "imu_mass": imu_mass.to_dict(),
        }
