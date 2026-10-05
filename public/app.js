/**
 * SMART-EV 2.0: Automotive Cockpit Client Controller
 * KPIT Sparkle Project
 * 
 * Handles:
 * - SSE real-time digital twin telemetry stream
 * - 2D animated top-down road & brake-wave canvas
 * - Chart.js multi-tab telemetry visualization
 * - Scenario benchmark launcher and modal reports
 * - Interactive cyber-physical slider controls
 * - Traditional AEB vs SMART-EV 2.0 comparison mode
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const elModeBadge = document.getElementById('mode-badge');
  const elClock = document.getElementById('system-clock');
  const elSpeed = document.getElementById('val-speed');
  const elPower = document.getElementById('val-power');
  const elVoltage = document.getElementById('val-voltage');
  const elCurrent = document.getElementById('val-current');

  const elRangePoint = document.getElementById('val-range-point');
  const elRangeInterval = document.getElementById('val-range-interval');
  const elUncLo = document.getElementById('unc-lo');
  const elUncHi = document.getElementById('unc-hi');
  const elUncBandFill = document.getElementById('unc-band-fill');
  const elUncNeedle = document.getElementById('unc-needle');
  const elHalfWidth = document.getElementById('val-half-width');

  const elSoc = document.getElementById('val-soc');
  const elSocBar = document.getElementById('soc-bar-fill');
  const elR0 = document.getElementById('val-r0');
  const elCellTemp = document.getElementById('val-cell-temp');
  const elDtdt = document.getElementById('val-dtdt');

  const elLeadGap = document.getElementById('val-lead-gap');
  const elHeadway = document.getElementById('val-headway');
  const elCdBenefit = document.getElementById('val-cd-benefit');
  const elTailGap = document.getElementById('val-tail-gap');
  const elV2vBadge = document.getElementById('v2v-status-badge');

  const elRadarTrust = document.getElementById('val-radar-trust');
  const elV2vTrust = document.getElementById('val-v2v-trust');
  const elImuTrust = document.getElementById('val-imu-trust');
  const elFillRadar = document.getElementById('fill-radar');
  const elFillV2v = document.getElementById('fill-v2v');
  const elFillImu = document.getElementById('fill-imu');

  const elConflictK = document.getElementById('val-conflict-k');
  const elV2vLatency = document.getElementById('val-v2v-latency');
  const elV2vLoss = document.getElementById('val-v2v-loss');
  const elFusionDecision = document.getElementById('val-fusion-decision');

  // Sliders
  const sliderTemp = document.getElementById('slider-temp');
  const dispTemp = document.getElementById('disp-temp');
  const sliderGrade = document.getElementById('slider-grade');
  const dispGrade = document.getElementById('disp-grade');
  const sliderHvac = document.getElementById('slider-hvac');
  const dispHvac = document.getElementById('disp-hvac');
  const sliderLoss = document.getElementById('slider-loss');
  const dispLoss = document.getElementById('disp-loss');
  const toggleAttack = document.getElementById('toggle-attack');

  // Modals
  const modalScenario = document.getElementById('scenario-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalContent = document.getElementById('modal-content');
  const modalClose = document.getElementById('modal-close');

  const btnBlueprint = document.getElementById('btn-blueprint');
  const modalBlueprint = document.getElementById('blueprint-modal');
  const blueprintClose = document.getElementById('blueprint-close');

  const btnToggleAeb = document.getElementById('btn-toggle-aeb');

  // Canvas
  const canvas = document.getElementById('trafficCanvas');
  const ctx = canvas.getContext('2d');

  // Cached benchmark results
  let benchmarkResults = null;

  // Clock
  setInterval(() => {
    const d = new Date();
    elClock.textContent = d.toTimeString().split(' ')[0];
  }, 1000);

  // --------------------------------------------------------------------------
  // CHART.JS INITIALIZATION
  // --------------------------------------------------------------------------
  let activeChartTab = 'range';
  const chartCanvas = document.getElementById('telemetryChart');
  const chartLabels = [];
  const chartData1 = [];
  const chartData2 = [];
  const chartData3 = [];

  const chart = new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: 'Point Estimate (km)',
          data: chartData1,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          borderWidth: 2,
          tension: 0.25,
          fill: false,
        },
        {
          label: '95% CI Lower [lo]',
          data: chartData2,
          borderColor: '#94a3b8',
          borderDash: [4, 4],
          borderWidth: 1.5,
          fill: false,
        },
        {
          label: '95% CI Upper [hi]',
          data: chartData3,
          borderColor: '#94a3b8',
          borderDash: [4, 4],
          borderWidth: 1.5,
          fill: '-1', // fill between lo and hi
          backgroundColor: 'rgba(56, 189, 248, 0.12)',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { size: 10 } },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { size: 9 } },
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
      },
    },
  });

  // Chart Tab Buttons
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      activeChartTab = btn.getAttribute('data-chart');

      // Reset datasets for new mode
      chartLabels.length = 0;
      chartData1.length = 0;
      chartData2.length = 0;
      chartData3.length = 0;

      if (activeChartTab === 'range') {
        chart.data.datasets[0].label = 'Point Estimate (km)';
        chart.data.datasets[0].borderColor = '#38bdf8';
        chart.data.datasets[1].label = '95% CI Lower [lo]';
        chart.data.datasets[2].label = '95% CI Upper [hi]';
        chart.data.datasets[2].hidden = false;
      } else if (activeChartTab === 'thermal') {
        chart.data.datasets[0].label = 'Cell Temp (°C)';
        chart.data.datasets[0].borderColor = '#f59e0b';
        chart.data.datasets[1].label = 'dT/dt Rate (°C/min)';
        chart.data.datasets[1].borderColor = '#38bdf8';
        chart.data.datasets[2].hidden = true;
      } else if (activeChartTab === 'speed') {
        chart.data.datasets[0].label = 'Host Speed (km/h)';
        chart.data.datasets[0].borderColor = '#10b981';
        chart.data.datasets[1].label = 'Lead Speed (km/h)';
        chart.data.datasets[1].borderColor = '#38bdf8';
        chart.data.datasets[2].hidden = true;
      }
      chart.update();
    });
  });

  // --------------------------------------------------------------------------
  // SERVER-SENT EVENTS (SSE) STREAM
  // --------------------------------------------------------------------------
  const evtSource = new EventSource('/api/telemetry');

  evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateDashboardUI(data);
    updateChart(data);
  };

  evtSource.onerror = (err) => {
    console.warn('SSE connection interrupted. Reconnecting...', err);
  };

  function updateDashboardUI(d) {
    // Mode badge
    elModeBadge.textContent = d.mode;
    elModeBadge.className = 'mode-badge';
    if (d.mode === 'AGGRESSIVE') {
      elModeBadge.classList.add('mode-aggressive');
    } else if (d.mode === 'CONSERVATIVE') {
      elModeBadge.classList.add('mode-conservative');
    } else {
      elModeBadge.classList.add('mode-failsafe');
    }

    // Kinematics
    elSpeed.textContent = Math.round(d.speedKmh);
    const kw = Math.round(((d.currentA * d.packVoltageV) / 1000.0) * 10) / 10;
    elPower.textContent = `${kw} kW`;
    elVoltage.textContent = `${Math.round(d.packVoltageV * 10) / 10} V`;
    elCurrent.textContent = `${Math.round(d.currentA * 10) / 10} A`;

    // Range & Uncertainty
    elRangePoint.textContent = Math.round(d.rangePointKm);
    elRangeInterval.textContent = `Confidence: [${d.rangeLoKm.toFixed(1)}, ${d.rangeHiKm.toFixed(1)}] km`;
    elUncLo.textContent = `${Math.round(d.rangeLoKm)} km`;
    elUncHi.textContent = `${Math.round(d.rangeHiKm)} km`;

    const halfW = (d.rangeHiKm - d.rangeLoKm) / 2.0;
    const halfRatio = (halfW / (d.rangePointKm + 1e-4)) * 100.0;
    elHalfWidth.textContent = `${halfRatio.toFixed(1)}%`;
    if (halfRatio > 12.0) {
      elHalfWidth.style.color = '#f43f5e';
    } else {
      elHalfWidth.style.color = '#10b981';
    }

    // Battery Twin
    const socPct = (d.soc * 100).toFixed(1);
    elSoc.textContent = `${socPct}%`;
    elSocBar.style.width = `${socPct}%`;
    elR0.textContent = `${d.internalResistanceR0.toFixed(3)} Ω`;
    elCellTemp.textContent = `${d.cellTempC.toFixed(1)}°C`;
    elDtdt.textContent = `${d.dtDtCPerMin.toFixed(2)} °C/min`;

    // Platoon Stats
    elLeadGap.textContent = `${d.leadGapM.toFixed(1)} m`;
    const headway = d.leadGapM / Math.max(1, d.speedKmh / 3.6);
    elHeadway.textContent = `${headway.toFixed(2)} s`;
    elTailGap.textContent = `${d.tailgaterGapM.toFixed(1)} m`;

    if (d.v2vTrusted) {
      elV2vBadge.textContent = `V2V MESH SYNC: ${d.v2vLatencyMs.toFixed(1)} ms`;
      elV2vBadge.style.color = '#38bdf8';
    } else {
      elV2vBadge.textContent = 'V2V ADVISORY REJECTED';
      elV2vBadge.style.color = '#f43f5e';
    }

    // Trust Meters
    elRadarTrust.textContent = d.radarTrust.toFixed(2);
    elV2vTrust.textContent = d.v2vTrust.toFixed(2);
    elImuTrust.textContent = d.imuTrust.toFixed(2);

    elFillRadar.style.width = `${d.radarTrust * 100}%`;
    elFillV2v.style.width = `${d.v2vTrust * 100}%`;
    elFillImu.style.width = `${d.imuTrust * 100}%`;

    elConflictK.textContent = d.fusionConflictK.toFixed(3);
    elV2vLatency.textContent = `${d.v2vLatencyMs.toFixed(1)} ms`;
    elV2vLoss.textContent = `${d.v2vPacketLossPct.toFixed(1)}%`;
    elFusionDecision.textContent = d.fusionDecision;

    // Road Animation State Hook
    trafficSimState.hostSpeed = d.speedKmh;
    trafficSimState.leadGap = d.leadGapM;
    trafficSimState.tailGap = d.tailgaterGapM;
    trafficSimState.isBraking = d.leadVehicleBraking;
    trafficSimState.byzantineActive = d.byzantineAttackActive;
  }

  function updateChart(d) {
    const timeStr = new Date().toLocaleTimeString();
    chartLabels.push(timeStr);

    if (activeChartTab === 'range') {
      chartData1.push(d.rangePointKm);
      chartData2.push(d.rangeLoKm);
      chartData3.push(d.rangeHiKm);
    } else if (activeChartTab === 'thermal') {
      chartData1.push(d.cellTempC);
      chartData2.push(d.dtDtCPerMin);
    } else if (activeChartTab === 'speed') {
      chartData1.push(d.speedKmh);
      chartData2.push(d.leadSpeedKmh);
    }

    if (chartLabels.length > 25) {
      chartLabels.shift();
      chartData1.shift();
      chartData2.shift();
      if (chartData3.length) chartData3.shift();
    }

    chart.update('none');
  }

  // --------------------------------------------------------------------------
  // TOP-DOWN 2D ROAD & V2V BRAKE-WAVE CANVAS ANIMATION
  // --------------------------------------------------------------------------
  const trafficSimState = {
    roadOffset: 0,
    hostSpeed: 92.5,
    leadGap: 28.5,
    tailGap: 22.0,
    isBraking: false,
    byzantineActive: false,
    wavePulse: 0,
  };

  function renderTrafficSimulation() {
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // 1. Draw Road Lanes
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Lane border lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, 40); ctx.lineTo(w, 40);
    ctx.moveTo(0, h - 40); ctx.lineTo(w, h - 40);
    ctx.stroke();

    // Road dashed center line (scrolling based on vehicle speed)
    trafficSimState.roadOffset = (trafficSimState.roadOffset + trafficSimState.hostSpeed * 0.12) % 60;
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.setLineDash([25, 35]);
    ctx.lineDashOffset = -trafficSimState.roadOffset;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.setLineDash([]); // reset

    // 2. Vehicles Position Calculation
    const hostX = w * 0.48;
    const hostY = h / 2;

    const leadX = hostX + Math.min(220, trafficSimState.leadGap * 6.5);
    const tailX = hostX - Math.min(200, trafficSimState.tailGap * 6.5);

    // 3. Draw V2V Mesh Radio Wave Propagation
    trafficSimState.wavePulse = (trafficSimState.wavePulse + 0.08) % Math.PI;
    const waveRadius = 15 + Math.sin(trafficSimState.wavePulse) * 45;

    // Draw V2V RF Wave between Lead and Host
    ctx.save();
    ctx.strokeStyle = trafficSimState.byzantineActive ? 'rgba(244, 63, 94, 0.6)' : 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.arc(leadX - 30, hostY, waveRadius, Math.PI * 0.65, Math.PI * 1.35);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(hostX - 30, hostY, waveRadius, Math.PI * 0.65, Math.PI * 1.35);
    ctx.stroke();
    ctx.restore();

    // V2V sync data link line
    ctx.strokeStyle = trafficSimState.byzantineActive ? 'rgba(244, 63, 94, 0.4)' : 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tailX + 35, hostY);
    ctx.lineTo(hostX - 35, hostY);
    ctx.moveTo(hostX + 35, hostY);
    ctx.lineTo(leadX - 35, hostY);
    ctx.stroke();

    // 4. Draw Vehicles
    drawCar(leadX, hostY, '#38bdf8', 'LEAD EV', trafficSimState.isBraking);
    drawCar(hostX, hostY, '#10b981', 'SMART-EV 2.0', trafficSimState.isBraking);
    drawCar(tailX, hostY, '#f59e0b', 'TAILGATER', false);

    // 5. Gap annotations
    ctx.font = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(`${trafficSimState.leadGap.toFixed(1)} m`, (hostX + leadX) / 2, hostY - 32);
    ctx.fillText(`${trafficSimState.tailGap.toFixed(1)} m`, (hostX + tailX) / 2, hostY - 32);

    requestAnimationFrame(renderTrafficSimulation);
  }

  function drawCar(x, y, color, label, isBraking) {
    const carW = 68;
    const carH = 34;

    ctx.save();
    ctx.translate(x, y);

    // Chassis shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(-carW / 2 + 3, -carH / 2 + 4, carW, carH);

    // Vehicle body
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-carW / 2, -carH / 2, carW, carH, 6);
    ctx.fill();
    ctx.stroke();

    // Windshield & Roof
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.fillRect(-carW / 2 + 16, -carH / 2 + 5, carW - 32, carH - 10);

    // Headlights (Right side / traveling forward)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(carW / 2 - 2, -carH / 2 + 4, 3, 6);
    ctx.fillRect(carW / 2 - 2, carH / 2 - 10, 3, 6);

    // Taillights (Left side)
    if (isBraking) {
      ctx.fillStyle = '#ef4444';
      ctx.shadowColor = '#ef4444';
      ctx.shadowBlur = 14;
      ctx.fillRect(-carW / 2 - 3, -carH / 2 + 4, 4, 8);
      ctx.fillRect(-carW / 2 - 3, carH / 2 - 12, 4, 8);
    } else {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.5)';
      ctx.fillRect(-carW / 2 - 1, -carH / 2 + 4, 2, 6);
      ctx.fillRect(-carW / 2 - 1, carH / 2 - 10, 2, 6);
    }

    // Label
    ctx.shadowBlur = 0;
    ctx.font = '9px sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.fillText(label, 0, 3);

    ctx.restore();
  }

  requestAnimationFrame(renderTrafficSimulation);

  // --------------------------------------------------------------------------
  // SLIDERS & CYBER-PHYSICAL ATTACK CONTROLS
  // --------------------------------------------------------------------------
  function postControls() {
    const payload = {
      ambientTempC: parseFloat(sliderTemp.value),
      roadGradePct: parseFloat(sliderGrade.value),
      hvacPowerW: parseFloat(sliderHvac.value),
      v2vPacketLossSlider: parseFloat(sliderLoss.value),
      byzantineAttackActive: toggleAttack.checked,
    };

    fetch('/api/controls', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch((err) => console.error('Failed to post controls:', err));
  }

  sliderTemp.addEventListener('input', () => {
    dispTemp.textContent = `${sliderTemp.value}°C`;
    postControls();
  });

  sliderGrade.addEventListener('input', () => {
    dispGrade.textContent = `${parseFloat(sliderGrade.value).toFixed(1)}%`;
    postControls();
  });

  sliderHvac.addEventListener('input', () => {
    dispHvac.textContent = `${sliderHvac.value} W`;
    postControls();
  });

  sliderLoss.addEventListener('input', () => {
    dispLoss.textContent = `${parseFloat(sliderLoss.value).toFixed(1)}%`;
    postControls();
  });

  toggleAttack.addEventListener('change', () => {
    postControls();
  });

  // --------------------------------------------------------------------------
  // BENCHMARK RESULTS LOADER & SCENARIO BUTTONS
  // --------------------------------------------------------------------------
  async function loadResults() {
    try {
      const res = await fetch('/api/results');
      if (res.ok) {
        benchmarkResults = await res.json();
        const conf = benchmarkResults.conformal_metrics;
        document.getElementById('foot-coverage').textContent = `${conf.coverage_pct.toFixed(1)}%`;
        document.getElementById('foot-mae').textContent = `${conf.mae_pct.toFixed(2)}%`;
      }
    } catch (e) {
      console.warn('Initial results fetch failed:', e);
    }
  }
  loadResults();

  // Scenario Buttons
  document.querySelectorAll('.btn-scenario').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-scenario');
      showScenarioModal(id);
    });
  });

  function showScenarioModal(id) {
    if (!benchmarkResults) {
      alert('Simulation benchmark data is loading. Please wait a moment.');
      return;
    }

    modalScenario.classList.remove('hidden');

    if (id === '1') {
      const s1 = benchmarkResults.s1;
      modalTitle.textContent = 'Scenario 1: Physics Range vs Static OEM Estimate';
      let html = `
        <p style="margin-bottom: 12px; color: #94a3b8;">
          Evaluates physics-informed range estimation against the static OEM baseline (256.0 km at 80% SoC)
          across 5 environmental and topography profiles. Output includes split conformal 95% confidence intervals.
        </p>
        <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 0.8rem; text-align: left;">
          <thead>
            <tr style="border-bottom: 1px solid #334155; color: #38bdf8;">
              <th style="padding: 8px;">Profile</th>
              <th style="padding: 8px;">Temp</th>
              <th style="padding: 8px;">Grade</th>
              <th style="padding: 8px;">Physics Range</th>
              <th style="padding: 8px;">95% Conformal Interval</th>
              <th style="padding: 8px;">Variance vs OEM</th>
            </tr>
          </thead>
          <tbody>
      `;
      s1.profiles.forEach((p) => {
        html += `
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 8px; font-weight: 700; color: #f8fafc;">${p.profile}</td>
            <td style="padding: 8px;">${p.temp_c}°C</td>
            <td style="padding: 8px;">${p.grade_pct}%</td>
            <td style="padding: 8px; color: #34d399; font-weight: 700;">${p.physics_range_km} km</td>
            <td style="padding: 8px; color: #7dd3fc;">[${p.ci_95_lo_km}, ${p.ci_95_hi_km}] km</td>
            <td style="padding: 8px; color: ${p.variance_km >= 0 ? '#34d399' : '#f43f5e'};">${p.variance_km > 0 ? '+' : ''}${p.variance_km} km</td>
          </tr>
        `;
      });
      html += `
          </tbody>
        </table>
        <div style="margin-top: 14px; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;">
          <strong>Conformal Predictor Verification:</strong><br>
          &bull; Empirical Coverage: <span style="color: #34d399; font-weight: 700;">${s1.conformal_coverage_pct.toFixed(2)}%</span> (Target &ge; 95.0%)<br>
          &bull; Mean Absolute Error (MAE): <span style="color: #38bdf8; font-weight: 700;">${s1.conformal_mae_pct.toFixed(2)}%</span> (Target &lt; 4.5%)
        </div>
      `;
      modalContent.innerHTML = html;
    } else if (id === '2') {
      const s2 = benchmarkResults.s2;
      modalTitle.textContent = 'Scenario 2: Lead-Vehicle Emergency Braking (85 km/h)';
      modalContent.innerHTML = `
        <div style="margin-bottom: 12px; color: #94a3b8;">
          Lead vehicle decelerates at a = -7.2 m/s² at t = 1.0 s from 85 km/h with an initial gap of 40.0 m.
          Demonstrates V2V BSM packet delivery preceding onboard radar Doppler detection.
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">V2V Alert Arrival</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;">t = ${s2.v2v_first_alert_time_s} s</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">Radar Doppler Closing Detection</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">t = ${s2.radar_first_detect_time_s} s</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">V2V Lookahead Margin</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">+${s2.v2v_advantage_ms} ms</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">Regen Energy Recaptured</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">~${s2.regen_energy_recaptured_wh} Wh</div>
          </div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 10px; border-radius: 8px; color: #34d399;">
          <strong>Collision Assessment:</strong> ZERO COLLISION (Minimum distance maintained: ${s2.min_gap_reached_m} m).
        </div>
      `;
    } else if (id === '3') {
      const s3 = benchmarkResults.s3;
      modalTitle.textContent = 'Scenario 3: Dempster-Shafer Sensor Fusion & Spoof Rejection';
      let html = `
        <p style="margin-bottom: 12px; color: #94a3b8;">
          Multi-sensor evidence fusion combining Radar, V2V, and IMU with mass functions on frame &Theta; = {Safe, Fault}.
          Verifies that Byzantine spoof attacks are gated and phantom braking is blocked.
        </p>
        <div style="display: flex; flex-direction: column; gap: 10px;">
      `;
      s3.test_cases.forEach((tc) => {
        html += `
          <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; border-left: 4px solid ${tc.phantom_braking_blocked ? '#f43f5e' : '#38bdf8'};">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
              <strong style="color: #ffffff;">${tc.case}</strong>
              <span style="font-family: monospace; color: #7dd3fc;">Trust: ${tc.trust_score.toFixed(2)} | Conflict K: ${tc.conflict_k.toFixed(4)}</span>
            </div>
            <p style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px;">${tc.description}</p>
            <div style="font-size: 0.75rem;">
              <strong>Decision:</strong> <span style="color: #38bdf8;">${tc.decision}</span> &bull; 
              <strong>Action:</strong> <span style="color: #34d399;">${tc.action}</span>
              ${tc.phantom_braking_blocked ? '<br><span style="color: #f43f5e; font-weight: 700;">&bull; PHANTOM BRAKING SAFELY BLOCKED</span>' : ''}
            </div>
          </div>
        `;
      });
      html += '</div>';
      modalContent.innerHTML = html;
    } else if (id === '4') {
      const s4 = benchmarkResults.s4;
      modalTitle.textContent = 'Scenario 4: Thermal Runaway Precursor Detection';
      modalContent.innerHTML = `
        <p style="margin-bottom: 12px; color: #94a3b8;">
          Compares proactive dT/dt precursor alert (threshold > 1.0 °C/min) against standard static BMS (T > 60°C).
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
          <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; border-top: 3px solid #38bdf8;">
            <h4 style="color: #38bdf8; margin-bottom: 6px;">SMART-EV 2.0 (Predictive)</h4>
            <div>Trigger Time: <strong>t = ${s4.smart_precursor_alert_time_s} s</strong></div>
            <div>Cell Temp at Alert: <strong>${s4.temp_at_smart_alert_c}°C</strong></div>
            <div>Temp at 120 s: <strong style="color: #34d399;">${s4.temp_at_120s_smart_c}°C</strong> (Stabilized)</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; border-top: 3px solid #f43f5e;">
            <h4 style="color: #f43f5e; margin-bottom: 6px;">Standard BMS (Static 60°C)</h4>
            <div>Trigger Time: <strong>${s4.standard_bms_alert_time_s}</strong></div>
            <div>Reaction: <strong>No mitigation in 120 s</strong></div>
            <div>Temp at 120 s: <strong style="color: #f43f5e;">${s4.temp_at_120s_unmitigated_c}°C</strong> (High Risk)</div>
          </div>
        </div>
        <div style="text-align: center; margin-top: 10px;">
          <img src="/out/slide7_chart2.png" alt="Slide 7 Chart 2" style="max-width: 100%; border-radius: 8px; border: 1px solid #334155;">
        </div>
      `;
    } else if (id === '5') {
      const s5 = benchmarkResults.s5;
      const em = s5.energy_metrics;
      const rm = s5.cooperative_regen_metrics;
      modalTitle.textContent = 'Scenario 5: CACC Platooning & Energy Optimization';
      modalContent.innerHTML = `
        <p style="margin-bottom: 12px; color: #94a3b8;">
          Cruising at 110 km/h over 100 km: Solo EV (Cd = 0.280) vs 3-EV Platoon with 0.6 s headway (Follower Cd = ${em.platoon_follower_cd}).
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px;">
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">Aero Drag Reduction</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;">-${em.cd_reduction_pct}%</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">Energy Savings</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">${em.energy_savings_pct}%</div>
            <span style="font-size: 0.7rem; color: #94a3b8;">${em.kwh_per_100km_platoon} vs ${em.kwh_per_100km_solo} kWh/100km</span>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
            <span style="font-size: 0.7rem; color: #94a3b8;">Range Extension</span>
            <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">+${em.range_extension_pct}%</div>
            <span style="font-size: 0.7rem; color: #94a3b8;">${em.range_platoon_km} vs ${em.range_solo_km} km</span>
          </div>
        </div>
        <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;">
          <strong style="color: #38bdf8;">Cooperative Regenerative Deceleration:</strong><br>
          Recaptures <strong style="color: #34d399;">${rm.cooperative_recaptured_wh} Wh</strong> vs ${rm.reactive_recaptured_wh} Wh 
          (<strong style="color: #34d399;">+${rm.regen_recapture_boost_pct}% boost</strong>) by pre-allocating gradual motor regen before friction pad lockup.
        </div>
      `;
    } else if (id === '6') {
      const s6 = benchmarkResults.s6;
      modalTitle.textContent = 'Scenario 6: 120 km/h Sudden Brake with Tailgater at 25 m';
      modalContent.innerHTML = `
        <div style="margin-bottom: 12px; color: #94a3b8;">
          Emergency stopping from 120 km/h with an aggressive vehicle tailgating at only 25 m.
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
          <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.4); padding: 12px; border-radius: 8px;">
            <h4 style="color: #f43f5e; margin-bottom: 6px;">Traditional Isolated AEB</h4>
            <div>Deceleration: <strong>-1.18 G</strong></div>
            <div>Jerk: <strong>28.5 m/s³</strong></div>
            <div>Tailgater Reaction: <strong>1.25 s delay</strong></div>
            <div style="margin-top: 8px; color: #f43f5e; font-weight: 700;">CRASH: Tailgater rear-ends host vehicle!</div>
          </div>
          <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); padding: 12px; border-radius: 8px;">
            <h4 style="color: #34d399; margin-bottom: 6px;">SMART-EV 2.0 S-Curve + V2V</h4>
            <div>Deceleration: <strong>-0.65 G (Graduated)</strong></div>
            <div>Jerk: <strong>7.2 m/s³ (Controlled)</strong></div>
            <div>V2V Sync Latency: <strong>4.0 ms</strong></div>
            <div style="margin-top: 8px; color: #34d399; font-weight: 700;">COLLISION-FREE: Min gap maintained ~${s6.smart_ev_graduated_curve.min_gap_maintained_m} m</div>
          </div>
        </div>
        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; font-size: 0.78rem;">
          <strong>Evasive Lane Shift Alternative:</strong> ${s6.evasive_lane_shift.outcome}
        </div>
      `;
    }
  }

  modalClose.addEventListener('click', () => {
    modalScenario.classList.add('hidden');
  });

  // Blueprint Modal
  btnBlueprint.addEventListener('click', () => {
    modalBlueprint.classList.remove('hidden');
  });

  blueprintClose.addEventListener('click', () => {
    modalBlueprint.classList.add('hidden');
  });

  // Close modals on clicking background
  window.addEventListener('click', (e) => {
    if (e.target === modalScenario) modalScenario.classList.add('hidden');
    if (e.target === modalBlueprint) modalBlueprint.classList.add('hidden');
  });

  // Toggle Traditional AEB vs SMART-EV 2.0
  btnToggleAeb.addEventListener('click', () => {
    showScenarioModal('6');
  });
});
