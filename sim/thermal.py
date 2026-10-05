"""
SMART-EV 2.0: Battery Cell Thermal Twin & dT/dt Precursor Detection
KPIT Sparkle Project

Implements:
1. Dynamic Lumped-Parameter Cell Thermal Model:
   m_cell * C_p * dT/dt = Q_joule + Q_entropy - Q_cooling
2. Real-time dT/dt Precursor Detection:
   Triggers proactive mitigation when dT/dt > 1.0 C/min (0.0167 C/s).
   Demonstrates mitigation at ~t = 40 s vs standard static BMS threshold (T > 60 C).
"""

from typing import Dict, List, Tuple, Any
import math
import numpy as np


class CellThermalModel:
    """
    Thermal dynamic twin for EV cylindrical/prismatic cell.
    Parameters:
        mass_kg: cell mass (~0.85 kg for large format pouch/blade cell)
        cp_j_per_kg_k: specific heat capacity (~950 J/(kg*K))
        h_cool_w_per_k: effective cooling convective conductance (~1.8 W/K)
        t_ambient_c: ambient/coolant baseline temperature
    """

    def __init__(
        self,
        mass_kg: float = 0.85,
        cp_j_per_kg_k: float = 950.0,
        h_cool_w_per_k: float = 1.6,
        initial_temp_c: float = 25.0,
        coolant_temp_c: float = 25.0,
    ) -> None:
        self.mass_kg: float = mass_kg
        self.cp: float = cp_j_per_kg_k
        self.h_cool: float = h_cool_w_per_k
        self.temp_c: float = initial_temp_c
        self.coolant_temp_c: float = coolant_temp_c

        # Internal resistance fault multiplier (simulates internal micro-short or localized dendrite)
        self.fault_active: bool = False
        self.fault_r_multiplier: float = 1.0

        # Moving average filter for dT/dt estimation (smoothing sensor noise)
        self.temp_history: List[Tuple[float, float]] = [(0.0, initial_temp_c)]
        self.dt_dt_c_per_min: float = 0.0

        # Mitigation state
        self.mitigation_triggered: bool = False
        self.mitigation_time_s: float = -1.0
        self.standard_bms_triggered: bool = False
        self.standard_bms_time_s: float = -1.0

    def step(
        self,
        time_s: float,
        dt_s: float,
        current_a: float,
        r_internal_nominal: float = 0.0018,  # Single cell resistance
        apply_smart_mitigation: bool = True,
    ) -> Dict[str, Any]:
        """
        Advance thermal state by dt_s seconds.
        """
        # If SMART-EV mitigation triggered, derate current by 60% and boost cooling by 2.5x
        effective_current = current_a
        effective_h_cool = self.h_cool

        if apply_smart_mitigation and self.mitigation_triggered:
            effective_current = current_a * 0.40  # Derate to 40%
            effective_h_cool = self.h_cool * 2.50  # Maximum coolant flow

        # Cell internal resistance with active thermal fault
        r_cell = r_internal_nominal * (self.fault_r_multiplier if self.fault_active else 1.0)

        # Thermal generation
        # 1. Joule heat: I^2 * R
        q_joule = (effective_current**2) * r_cell

        # 2. Entropic heat: I * T * (dVoc/dT), ~ -0.22 mV/K
        t_kelvin = self.temp_c + 273.15
        dvoc_dt = -0.00022  # V/K
        q_entropy = effective_current * t_kelvin * dvoc_dt

        q_gen = max(0.0, q_joule + q_entropy)

        # Heat dissipation to coolant loop
        q_cooling = effective_h_cool * (self.temp_c - self.coolant_temp_c)

        # Net thermal rate
        # m * Cp * dT/dt = q_gen - q_cooling
        thermal_inertia = self.mass_kg * self.cp
        dt_dt_c_per_sec = (q_gen - q_cooling) / thermal_inertia

        # Forward Euler integration
        self.temp_c += dt_dt_c_per_sec * dt_s

        # Update historical buffer for derivative calculation (window of ~3-5 seconds)
        self.temp_history.append((time_s, self.temp_c))
        while self.temp_history and (time_s - self.temp_history[0][0]) > 4.0:
            self.temp_history.pop(0)

        if len(self.temp_history) >= 2:
            t_span = self.temp_history[-1][0] - self.temp_history[0][0]
            temp_span = self.temp_history[-1][1] - self.temp_history[0][1]
            if t_span > 0.1:
                # Convert to degrees C per minute
                self.dt_dt_c_per_min = (temp_span / t_span) * 60.0

        # Precursor detection rule: dT/dt > 1.0 C/min
        if (
            not self.mitigation_triggered
            and self.dt_dt_c_per_min >= 1.00
            and time_s > 5.0
        ):
            self.mitigation_triggered = True
            self.mitigation_time_s = time_s

        # Standard static BMS rule: T > 60 C
        if not self.standard_bms_triggered and self.temp_c >= 60.0:
            self.standard_bms_triggered = True
            self.standard_bms_time_s = time_s

        return {
            "time_s": round(time_s, 2),
            "temp_c": round(self.temp_c, 3),
            "dt_dt_c_per_min": round(self.dt_dt_c_per_min, 3),
            "q_gen_w": round(q_gen, 2),
            "q_cooling_w": round(q_cooling, 2),
            "effective_current_a": round(effective_current, 2),
            "mitigation_triggered": self.mitigation_triggered,
            "mitigation_time_s": self.mitigation_time_s,
            "standard_bms_triggered": self.standard_bms_triggered,
            "standard_bms_time_s": self.standard_bms_time_s,
        }
