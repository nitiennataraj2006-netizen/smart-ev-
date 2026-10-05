# SMART-EV 2.0: Confidence-Calibrated, Dual-Loop Predictive Energy & Safety Architecture for EVs

[![KPIT Sparkle 2026](https://img.shields.io/badge/KPIT%20Sparkle-2026%20Project-0284c7.svg)](#)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#)
[![Node.js Express](https://img.shields.io/badge/Node.js-v24%20Express-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Status: SIMULATED](https://img.shields.io/badge/Data-SIMULATED%2C%20model--based-orange.svg)](#)

> **SMART-EV 2.0** is an automotive cyber-physical digital twin architecture combining **physics-constrained dual-loop range estimation**, **split conformal prediction uncertainty calibration**, **early thermal runaway precursor detection ($dT/dt > 1.0^\circ\text{C/min}$)**, and **cooperative V2V energy and safety coordination** backed by a **hard deterministic fail-safe arbiter** complying with ISO 26262 ASIL-D principles.

---

## 1. System Pipeline Overview

```
 [ SENSE ]       ==> Onboard Millimeter-Wave Radar (77 GHz), 6-DOF IMU, Battery Sensors (V, I, T)
    ||
 [ COMMUNICATE ] ==> 10 Hz V2V BSM Broadcast, Latency/Jitter/Loss Model, Zero-Trust Advisory Validator
    ||
 [ PREDICT ]     ==> Inner EKF Loop (20 ms) + Outer Physics Range Loop (1 s) + Split Conformal Calibration (95% CI)
    ||
 [ PROTECT ]     ==> Dempster-Shafer Evidence Fusion + Thermal dT/dt Precursor Detection + Fail-Safe Arbiter
    ||
 [ OPTIMIZE ]    ==> CACC Platooning Wake Drafting (~17% Range Boost) + Cooperative Anticipatory Regen (+16%)
```

---

## 2. Directory Structure

```
smart-ev2/
  sim/                  # Python 3.10+ Simulation Engine
    __init__.py         # Package declaration
    battery.py          # EKF on Thevenin ECM (R0 + R1||C1) & physics-constrained range engine
    conformal.py        # Split conformal prediction: calibration, residual quantiles, 95% CI
    v2v.py              # 10 Hz BSM broadcast, latency/loss model, zero-trust kinematic validator
    trust.py            # Dempster-Shafer multi-sensor evidence fusion & Byzantine spoof rejection
    thermal.py          # Cell lumped-parameter thermal twin & dT/dt precursor detector
    platoon.py          # CACC aerodynamic wake drafting & cooperative deceleration model
    arbiter.py          # Deterministic safety arbiter (AGGRESSIVE / CONSERVATIVE / FAIL-SAFE)
    scenarios.py        # Complete parameterized implementation of Scenarios 1 to 6
    run_all.py          # Simulation orchestrator, console reporting, JSON/CSV & Slide-7 exports
  server/
    server.js           # Express server: static hosting, REST APIs, Server-Sent Events (SSE)
  public/
    index.html          # Dark automotive digital twin cockpit UI
    style.css           # Glassmorphism dark cockpit design system & animations
    app.js              # Real-time SSE telemetry, 2D top-down canvas simulation, Chart.js hooks
    vendor/             # Vendored Chart.js (100% offline operational guarantee)
  tests/
    test_all.py         # Pytest verification suite (EKF, conformal coverage, fusion, arbiter)
  out/
    results.json        # Machine-readable scenario metrics & benchmark outputs
    slide7_data.csv     # Exported data table for Slide-7
    slide7_chart1.png   # 300 DPI high-resolution thermal stress range chart with 95% envelope
    slide7_chart2.png   # 300 DPI high-resolution predictive precursor vs standard 60°C BMS chart
  README.md             # Engineering documentation, formulas, assumptions, and limitations
  run.ps1               # Automated installation, test runner, benchmark generator & server launch
  package.json          # Node.js dependencies
```

---

## 3. Quick Start & Execution

### Prerequisites
- **Operating System:** Windows 11 (PowerShell)
- **Python:** Python 3.10+ with `numpy`, `scipy`, `matplotlib` (or `Pillow`), `pandas`, `pytest`
- **Node.js:** Node.js v18+ with `npm`

### One-Click Launch via PowerShell
Open PowerShell in the project root directory (`D:\smart-ev2` or `D:\kpit`) and execute:
```powershell
.\run.ps1
```
This script automatically:
1. Detects and configures the Python 3.11+ environment.
2. Installs Node.js dependencies (`express`, `chart.js`).
3. Executes the full `pytest` verification suite across all 10 unit tests.
4. Runs `sim/run_all.py`, printing the console tables and generating all `out/` artifacts.
5. Launches the live interactive Web Cockpit on **http://localhost:3000**.

---

## 4. Mathematical Formulations & Derivations

### 4.1 Inner EKF Loop (20 ms / 50 Hz): Thevenin ECM
The battery pack is modeled using a 1-RC Thevenin Equivalent Circuit Model:
- **State Vector:**
  $$\mathbf{x}_k = \begin{bmatrix} \text{SoC}_k \\ V_{1,k} \end{bmatrix}$$
- **State Transition Equations:**
  $$\text{SoC}_{k+1} = \text{SoC}_k - \frac{\eta_c \Delta t}{Q_{\text{usable}}(T)} I_k$$
  $$V_{1,k+1} = \exp\left(-\frac{\Delta t}{R_1 C_1}\right) V_{1,k} + R_1 \left(1 - \exp\left(-\frac{\Delta t}{R_1 C_1}\right)\right) I_k$$
- **Observation Model:**
  $$V_{t,k} = V_{OC}(\text{SoC}_k) - V_{1,k} - I_k R_0(T) + v_k$$
  where $V_{OC}(\text{SoC})$ is linearized via Jacobian $H_k = \left[\left.\frac{\partial V_{OC}}{\partial \text{SoC}}\right|_{\hat{\mathbf{x}}_k^-}, -1\right]$.
- **Arrhenius Temperature Correction for Internal Resistance:**
  $$R_0(T) = R_{0,\text{ref}} \exp\left(\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right)$$

### 4.2 Outer Loop (1 s / 1 Hz): Physics Range Model & Energy Conservation
The vehicle tractive force combines aerodynamic drag, rolling resistance, road gradient, and inertial work:
$$F_{\text{total}} = F_{\text{roll}} + F_{\text{aero}} + F_{\text{grade}} + F_{\text{accel}}$$
- **Rolling Resistance:** $F_{\text{roll}} = C_{rr}(T) \cdot m \cdot g \cdot \cos\theta$
- **Aerodynamic Drag:** $F_{\text{aero}} = \frac{1}{2} \rho(T) C_d A v^2$, with $\rho(T) = \frac{p_{\text{atm}}}{R_{\text{spec}} T}$
- **Road Incline:** $F_{\text{grade}} = m \cdot g \cdot \sin\theta$
- **Electrical Power Demand at Battery:**
  $$P_{\text{batt}} = \begin{cases} \frac{F_{\text{total}} v}{\eta_{\text{drivetrain}}} + P_{\text{HVAC}} + P_{\text{aux}} + I^2 R_0, & F_{\text{total}} \ge 0 \\ F_{\text{total}} v \eta_{\text{regen}} + P_{\text{HVAC}} + P_{\text{aux}} + I^2 R_0, & F_{\text{total}} < 0 \end{cases}$$
- **Energy Conservation Constraint:** Non-physical negative power (gaining energy without external force or gravity) is strictly clamped. On mountain descents, continuous regen energy is capped by battery charging acceptance ($P_{\text{regen, max}} \le 70\text{ kW}$).

### 4.3 Split Conformal Prediction (95% Calibrated Confidence)
Rather than a single point estimate, SMART-EV 2.0 outputs a confidence-calibrated interval:
$$\mathcal{C}(\mathbf{x}) = \left[ \hat{y}(\mathbf{x}) - \hat{q} \cdot \sigma(\mathbf{x}),\quad \hat{y}(\mathbf{x}) + \hat{q} \cdot \sigma(\mathbf{x}) \right]$$
- **Heteroscedastic Uncertainty Heuristic:**
  $$\sigma(\mathbf{x}) = \sigma_{\text{base}} \cdot f_T(\Delta T) \cdot f_G(|\text{grade}|) \cdot f_V(v) \cdot f_H(P_{\text{HVAC}})$$
- **Conformal Quantile:** Computed on held-out calibration cycles $\mathcal{D}_{\text{cal}}$:
  $$s_i = \frac{|y_i - \hat{y}_i|}{\sigma(\mathbf{x}_i)}, \quad \hat{q} = \text{Quantile}_{\lceil (n+1)(1-\alpha) \rceil / n}\left(\{s_i\}_{i=1}^n\right)$$
- **Guarantees:**
  $$\mathbb{P}\left(y_{\text{test}} \in \mathcal{C}(\mathbf{x}_{\text{test}})\right) \ge 1 - \alpha = 95.0\%$$
- **Benchmark Performance:** Empirical coverage = **96.00%**, MAE = **3.06%** (Target $< 4.5\%$).

### 4.4 Dynamic Fallback Condition
When the half-width exceeds 12% of the range estimate:
$$\frac{\text{hi} - \text{lo}}{2 \cdot \hat{y}} > 0.12$$
The arbiter automatically switches from `AGGRESSIVE` mode to `CONSERVATIVE` fail-safe mode, expanding time headway from 0.6 s to 1.2 s and capping peak acceleration.

### 4.5 Dempster-Shafer Sensor Trust Fusion
Frame of discernment: $\Theta = \{\text{Safe}, \text{Fault}\}$. Basic Probability Assignment (BPA) functions $m_1, m_2, m_3$ represent evidence from Radar, V2V, and IMU.
- **Orthogonal Sum Combination:**
  $$m_{1 \oplus 2}(A) = \frac{1}{1 - K} \sum_{B \cap C = A} m_1(B) m_2(C)$$
  where conflict metric $K = \sum_{B \cap C = \emptyset} m_1(B) m_2(C)$.
- **Byzantine Gating:** When an attacker injects a spoofed emergency brake packet ($m_{\text{V2V}}(\text{Fault}) = 0.94$), but Radar and IMU observe open road ($m(\text{Safe}) > 0.90$), conflict $K > 0.85$. The arbiter gates the packet, flags `BYZANTINE_SPOOF_GATED`, and completely blocks phantom braking.

### 4.6 Cell Thermal Precursor Detection ($dT/dt > 1.0^\circ\text{C/min}$)
Lumped-parameter cell heat balance:
$$m_{\text{cell}} C_p \frac{dT}{dt} = Q_{\text{gen}} - Q_{\text{cooling}}$$
where $Q_{\text{gen}} = I^2 R_{\text{int}} + I T \frac{\partial V_{OC}}{\partial T}$.
- Standard BMS checks static temperature ($T > 60^\circ\text{C}$), which triggers dangerously late.
- SMART-EV 2.0 monitors moving rate of change $\frac{dT}{dt}$. When $\frac{dT}{dt} > 1.0^\circ\text{C/min}$ (crossing at $t \approx 40\text{ s}$ when temperature is only ~28.8°C), proactive mitigation is engaged (current derating by 60%, chiller pump boosted by 250%), preventing thermal runaway propagation.

### 4.7 CACC Platooning & Cooperative Deceleration
- Solo EV: $C_d = 0.280$.
- In a 3-EV platoon with 0.6 s headway ($d \approx 18.3\text{ m}$ at 110 km/h), follower $C_d$ drops to $0.201$ (28.2% drag reduction).
- At 110 km/h over 100 km: consumption drops from $20.89\text{ kWh}/100\text{ km}$ to $17.61\text{ kWh}/100\text{ km}$, extending range by **18.59%** (meeting ~17% target).
- Cooperative deceleration utilizes 4 ms V2V sync to initiate anticipatory gentle regen curve ($50.8\text{ Wh}$ recaptured, +15.97% boost over reactive braking).

---

## 5. Scenario Benchmarks (All 6 Scenarios)

All scenarios are executed autonomously by `sim/run_all.py` with reproducible random seed (`seed = 42`):

| Scenario | Description | Key Physical Metrics | SMART-EV 2.0 Outcome |
|---|---|---|---|
| **S1** | Physics Range vs Static OEM (Base 80% SoC, 256 km) | City: 250.8 km<br>Cold: 144.8 km<br>Hill: 172.5 km<br>Descent: 290.4 km<br>Platoon: 247.2 km | Conformal Coverage: **96.00%** (Target &ge; 95%)<br>MAE: **3.06%** (Target &lt; 4.5%) |
| **S2** | Lead-Vehicle Emergency Braking (85 km/h, 40 m gap) | Lead $a = -7.2\text{ m/s}^2$ at $t = 1.0\text{ s}$<br>V2V arrives at $t = 1.01\text{ s}$<br>Radar detects at $t = 1.21\text{ s}$ | **+200.0 ms V2V Advantage**<br>Min Gap: 32.2 m (Zero collision)<br>Regen Recaptured: **50.8 Wh** (~49 Wh target) |
| **S3** | Dempster-Shafer Trust Gating | Nominal: Trust 0.97, $K = 0.059$<br>Mud: Trust 0.94, $K = 0.102$<br>Byzantine Spoof: Trust 0.47, $K = 0.936$<br>Tunnel Blackout: Trust 0.95 | Byzantine spoof packet gated.<br>**Phantom braking blocked**.<br>Smooth local radar + IMU dead reckoning. |
| **S4** | Thermal Runaway Precursor Detection | Fault starts at $t = 10\text{ s}$<br>Precursor: $\frac{dT}{dt} > 1.0^\circ\text{C/min}$ triggers at **$t = 43.5\text{ s}$** ($T = 28.8^\circ\text{C}$)<br>Standard BMS (60°C) does not trigger in 120 s | Precursor mitigation stabilizes cell at 28.0°C.<br>Unmitigated baseline climbs past 31.6°C towards runaway at $t > 165\text{ s}$. |
| **S5** | CACC Platooning (110 km/h, 100 km) | Solo $C_d = 0.280$ ($20.89\text{ kWh}/100\text{ km}$)<br>Follower $C_d = 0.201$ ($17.61\text{ kWh}/100\text{ km}$)<br>Drag reduction: 28.2% | **18.59% Range Extension** (~17% target)<br>Cooperative regen recapture: $90.25\text{ Wh}$ vs $77.82\text{ Wh}$ (**+15.97% boost**) |
| **S6** | 120 km/h Sudden Brake with Tailgater at 25 m | Traditional AEB: $-1.18\text{ G}$, Jerk $28.5\text{ m/s}^3$<br>SMART-EV 2.0: $-0.65\text{ G}$ S-curve, Jerk $7.2\text{ m/s}^3$, 4 ms V2V sync | Traditional: **Crash (Tailgater rear-ends host)**<br>SMART-EV 2.0: **Collision-Free** (Min gap 24.98 m)<br>Evasive lane shift verified feasible. |

---

## 6. Slide-7 Export Artifacts

The system generates publication-grade 300 DPI figures and structured tabular data in `out/`:
1. `out/results.json`: Full benchmark dictionary with all scenario metrics and conformal test set statistics.
2. `out/slide7_data.csv`: Tabular export containing distance traveled, ground-truth range, baseline uncalibrated range, conformal [lo, hi] bounds, time series, cell temperature, and $dT/dt$.
3. `out/slide7_chart1.png`: 300 DPI plot showing remaining range vs distance (0-120 km) with baseline uncalibrated curve, shaded 95% conformal envelope, and simulated ground truth.
4. `out/slide7_chart2.png`: 300 DPI dual-axis plot showing cell temperature (left Y) and thermal rate $dT/dt$ (right Y) over 0-120 s with vertical precursor alert line at $t \approx 40\text{ s}$ and dashed 60°C standard BMS threshold annotation.

---

## 7. Assumptions & Engineering Parameters

1. **Vehicle Platform:** Mid-size electric vehicle with curb mass + passenger/cargo = $2050\text{ kg}$, frontal area $A = 2.38\text{ m}^2$, base aerodynamic drag coefficient $C_d = 0.280$, rolling resistance $C_{rr} = 0.0112 - 0.0138$.
2. **Battery Pack:** 96-cell series Li-ion (NMC/LFP) pack, nominal pack voltage $380\text{ V}$, nominal usable capacity $60.0\text{ kWh}$ ($150\text{ Ah}$). Nominal static OEM range at 80% SoC is $256.0\text{ km}$ ($18.75\text{ kWh}/100\text{ km}$).
3. **Electrochemical Limits:** Usable capacity derates with temperature according to cold electrochemical constraints ($-0.82\%/^\circ\text{C}$ below 25°C). Ohmic resistance $R_0$ scales with Arrhenius activation energy $E_a/R = 2100\text{ K}$.
4. **V2V Communications:** Compliant with SAE J2735 Basic Safety Message (BSM) standard over DSRC / C-V2X PC5 direct interface. Base latency $\approx 4-6\text{ ms}$, packet loss threshold $5\%$, latency threshold $100\text{ ms}$.
5. **Radar Sensor:** 77 GHz FMCW millimeter-wave radar requiring Doppler Doppler integration filter (~160-200 ms detection delay).
6. **Data Disclaimer:** All outputs are strictly marked as **"SIMULATED, model-based"** to distinguish from physical dynamometer and test-track telemetry.

---

## 8. Limitations & Future Roadmap

1. **Cell Thermal Gradient:** The current thermal model employs a lumped-parameter single-state thermal capacity assumption ($m C_p \frac{dT}{dt}$). In physical vehicle packs, thermal gradients exist between core and skin; a 2D/3D thermal finite-element digital twin could improve localized dendrite hotspot prediction.
2. **Channel Multipath Fading:** The RF channel model simulates log-normal latency and Rayleigh packet loss. Dense urban canyon shadowing and electromagnetic interference from multi-path reflections could be augmented with ray-tracing models.
3. **Platoon Dynamics:** Lateral lane shifts currently assume ideal yaw torque vectoring without tire slip saturation under hydroplaning/icy asphalt conditions. Real-world implementation requires integration with ESC/ABS wheel-speed state estimators.

---

## 9. Verification & Unit Testing

Execute the comprehensive test suite at any time via:
```powershell
python -m pytest -v tests/test_all.py
```
Test categories include:
- `TestBatteryEKF`: Verifies monotonic SoC decay, positive internal resistance $R_0$, and bounded innovation error.
- `TestConformalPrediction`: Validates empirical coverage $\ge 95\%$ and MAE $< 4.5\%$ on 400 held-out drive cycle samples.
- `TestDempsterShaferFusion`: Validates nominal consensus and Byzantine attack rejection with phantom braking blocked.
- `TestFailSafeArbiter`: Validates mode switching when uncertainty $> 12\%$ and deterministic fallback within $< 50\text{ ms}$.
- `TestEnergyConservation`: Validates non-perpetual motion and strict bounds on uphill power and downhill regen.
