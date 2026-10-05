/**
 * SMART-EV 2.0: Backend Telemetry Server
 * KPIT Sparkle Project
 * 
 * Express HTTP + Server-Sent Events (SSE) server
 * Serves automotive web cockpit and live digital twin telemetry.
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const ROOT_DIR = path.resolve(__dirname, '..');

app.use(express.json());
app.use(express.static(path.join(ROOT_DIR, 'public')));
app.use('/out', express.static(path.join(ROOT_DIR, 'out')));

// Live simulated vehicle state
const vehicleState = {
  speedKmh: 92.5,
  targetSpeedKmh: 95.0,
  soc: 0.78,
  packVoltageV: 382.4,
  currentA: 42.0,
  internalResistanceR0: 0.046,
  cellTempC: 28.4,
  dtDtCPerMin: 0.24,
  rangePointKm: 246.5,
  rangeLoKm: 235.8,
  rangeHiKm: 257.2,
  mode: 'AGGRESSIVE', // AGGRESSIVE, CONSERVATIVE, FAIL-SAFE
  modeReason: 'System nominal. Calibrated uncertainty <= 12%.',
  radarTrust: 0.94,
  v2vTrust: 0.92,
  imuTrust: 0.95,
  fusionConflictK: 0.048,
  fusionDecision: 'NOMINAL_SAFE',
  v2vLatencyMs: 4.8,
  v2vPacketLossPct: 1.2,
  v2vTrusted: true,
  cooperativeBrakingActive: false,
  phantomBrakingBlocked: false,
  thermalPrecursorAlert: false,
  // Environment sliders
  ambientTempC: 25.0,
  roadGradePct: 0.0,
  hvacPowerW: 800.0,
  v2vPacketLossSlider: 1.5,
  byzantineAttackActive: false,
  leadVehicleBraking: false,
  leadGapM: 28.5,
  leadSpeedKmh: 95.0,
  tailgaterGapM: 22.0,
  tailgaterSpeedKmh: 95.0,
};

// Connected SSE clients
const sseClients = new Set();

// Telemetry physics tick (50 Hz / 20 ms internal tick, broadcast at 10 Hz / 100 ms)
let tickCount = 0;
setInterval(() => {
  tickCount++;

  // Environment impacts
  const tempDelta = Math.abs(vehicleState.ambientTempC - 25.0);
  const grade = vehicleState.roadGradePct;
  const hvac = vehicleState.hvacPowerW;

  // Slowly discharge battery
  vehicleState.soc = Math.max(0.05, vehicleState.soc - 0.000015 * (1.0 + hvac / 3000.0));

  // Physics range recalculation
  const effectiveCapacityKwh = 60.0 * (vehicleState.ambientTempC >= 25 ? (1.0 - 0.0018 * tempDelta) : (1.0 - 0.0082 * tempDelta));
  const remainingEnergy = vehicleState.soc * effectiveCapacityKwh;
  const gradePowerKw = (2050 * 9.81 * (grade / 100.0) * (vehicleState.speedKmh / 3.6)) / 1000.0;
  const aeroKw = (0.5 * 1.2 * 0.28 * 2.38 * Math.pow(vehicleState.speedKmh / 3.6, 3)) / 1000.0;
  const rollKw = (0.012 * 2050 * 9.81 * (vehicleState.speedKmh / 3.6)) / 1000.0;
  const totalPowerKw = Math.max(2.0, (aeroKw + rollKw + gradePowerKw) / 0.90 + (hvac + 400) / 1000.0);
  
  const consumptionKwhPerKm = totalPowerKw / Math.max(20.0, vehicleState.speedKmh);
  const rawRange = remainingEnergy / Math.max(0.05, consumptionKwhPerKm);

  vehicleState.rangePointKm = Math.round(rawRange * 10) / 10;
  
  // Conformal prediction bounds
  const sigma = Math.max(3.0, rawRange * 0.045) * (1.0 + 0.015 * tempDelta + 0.035 * Math.abs(grade) + 0.00006 * hvac);
  const halfWidth = 1.96 * sigma;
  vehicleState.rangeLoKm = Math.round(Math.max(0, rawRange - halfWidth) * 10) / 10;
  vehicleState.rangeHiKm = Math.round((rawRange + halfWidth) * 10) / 10;

  // Dynamic fallback check: half-width > 12%
  const halfWidthRatio = halfWidth / (rawRange + 1e-4);

  // Attack condition
  if (vehicleState.byzantineAttackActive) {
    vehicleState.v2vTrust = 0.08;
    vehicleState.fusionConflictK = 0.912;
    vehicleState.fusionDecision = 'BYZANTINE_SPOOF_GATED';
    vehicleState.phantomBrakingBlocked = true;
    vehicleState.v2vTrusted = false;
    vehicleState.mode = 'CONSERVATIVE';
    vehicleState.modeReason = 'ZERO-TRUST: Byzantine spoof detected. V2V gated; phantom braking prevented.';
  } else if (vehicleState.v2vPacketLossSlider > 5.0) {
    vehicleState.v2vPacketLossPct = vehicleState.v2vPacketLossSlider;
    vehicleState.v2vLatencyMs = 12.0 + vehicleState.v2vPacketLossSlider * 3.5;
    vehicleState.v2vTrust = Math.max(0.2, 1.0 - (vehicleState.v2vPacketLossSlider / 25.0));
    vehicleState.v2vTrusted = false;
    vehicleState.mode = 'FAIL-SAFE';
    vehicleState.modeReason = `V2V DROP: Packet loss ${vehicleState.v2vPacketLossSlider.toFixed(1)}% > 5.0%. Local radar safe mode active.`;
    vehicleState.fusionDecision = 'LOCAL_RADAR_FALLBACK';
  } else if (halfWidthRatio > 0.12) {
    vehicleState.mode = 'CONSERVATIVE';
    vehicleState.modeReason = `DYNAMIC FALLBACK: Uncertainty band ${(halfWidthRatio * 100).toFixed(1)}% > 12.0% of estimate.`;
    vehicleState.fusionDecision = 'CONSERVATIVE_GAP_EXPANSION';
    vehicleState.v2vTrust = 0.88;
    vehicleState.v2vTrusted = true;
  } else {
    vehicleState.mode = 'AGGRESSIVE';
    vehicleState.modeReason = 'System nominal: CACC tight platooning, full cooperative regen enabled.';
    vehicleState.fusionDecision = 'NOMINAL_SAFE';
    vehicleState.v2vTrust = 0.94;
    vehicleState.v2vTrusted = true;
    vehicleState.phantomBrakingBlocked = false;
  }

  // Thermal precursor simulation
  if (vehicleState.thermalPrecursorAlert) {
    vehicleState.cellTempC = Math.min(33.5, vehicleState.cellTempC + 0.02);
    vehicleState.dtDtCPerMin = Math.max(0.1, vehicleState.dtDtCPerMin - 0.01);
  } else {
    vehicleState.cellTempC = 27.5 + (vehicleState.ambientTempC - 25.0) * 0.15;
    vehicleState.dtDtCPerMin = 0.22;
  }

  // Broadcast to SSE clients every 100 ms (10 Hz)
  if (tickCount % 5 === 0 && sseClients.size > 0) {
    const payload = JSON.stringify({
      timestamp: Date.now(),
      ...vehicleState,
    });
    for (const res of sseClients) {
      res.write(`data: ${payload}\n\n`);
    }
  }
}, 20);

// API: Get computed scenario results
app.get('/api/results', (req, res) => {
  const resultsPath = path.join(ROOT_DIR, 'out', 'results.json');
  if (fs.existsSync(resultsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
      res.json(data);
    } catch (err) {
      res.status(500).json({ error: 'Failed to read results.json', details: err.message });
    }
  } else {
    res.status(404).json({ error: 'Results not generated yet. Please run sim/run_all.py' });
  }
});

// API: Trigger Python simulation
app.post('/api/run-simulation', (req, res) => {
  const pyCmd = process.platform === 'win32'
    ? 'C:\\Program Files\\KiCad\\10.0\\bin\\python.exe'
    : 'python3';

  const child = spawn(pyCmd, ['sim/run_all.py'], { cwd: ROOT_DIR });
  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (d) => { stdout += d.toString(); });
  child.stderr.on('data', (d) => { stderr += d.toString(); });

  child.on('close', (code) => {
    if (code === 0) {
      res.json({ success: true, stdout });
    } else {
      res.status(500).json({ success: false, error: stderr || stdout });
    }
  });
});

// API: Slider & Attack Control Input
app.post('/api/controls', (req, res) => {
  const { ambientTempC, roadGradePct, hvacPowerW, v2vPacketLossSlider, byzantineAttackActive, triggerBrakeWave } = req.body;

  if (ambientTempC !== undefined) vehicleState.ambientTempC = parseFloat(ambientTempC);
  if (roadGradePct !== undefined) vehicleState.roadGradePct = parseFloat(roadGradePct);
  if (hvacPowerW !== undefined) vehicleState.hvacPowerW = parseFloat(hvacPowerW);
  if (v2vPacketLossSlider !== undefined) vehicleState.v2vPacketLossSlider = parseFloat(v2vPacketLossSlider);
  if (byzantineAttackActive !== undefined) vehicleState.byzantineAttackActive = Boolean(byzantineAttackActive);

  if (triggerBrakeWave) {
    vehicleState.leadVehicleBraking = true;
    vehicleState.speedKmh = Math.max(30, vehicleState.speedKmh - 35);
    setTimeout(() => {
      vehicleState.leadVehicleBraking = false;
      vehicleState.speedKmh = 92.5;
    }, 4500);
  }

  res.json({ success: true, vehicleState });
});

// SSE Telemetry Stream
app.get('/api/telemetry', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  sseClients.add(res);

  req.on('close', () => {
    sseClients.delete(res);
  });
});

app.listen(PORT, () => {
  console.log('================================================================');
  console.log(`SMART-EV 2.0 Web Cockpit listening on http://localhost:${PORT}`);
  console.log(`Serving /public, /out, and /api endpoints`);
  console.log('================================================================');
});
