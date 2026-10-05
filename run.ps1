# SMART-EV 2.0: Installation and Runner Script
# KPIT Sparkle Project
# Windows 11 PowerShell Environment

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " SMART-EV 2.0: FULL-SYSTEM LAUNCHER" -ForegroundColor Cyan
Write-Host " Confidence-Calibrated Dual-Loop Energy & Safety Architecture" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Resolve Python Binary
$PYTHON_BIN = ""
if (Test-Path "C:\Program Files\KiCad\10.0\bin\python.exe") {
    $PYTHON_BIN = "C:\Program Files\KiCad\10.0\bin\python.exe"
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PYTHON_BIN = "python"
} elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $PYTHON_BIN = "python3"
} else {
    Write-Error "Python 3.10+ could not be located. Please install Python or ensure it is in PATH."
    exit 1
}

Write-Host "`n[1/4] Using Python executable: $PYTHON_BIN" -ForegroundColor Green
& $PYTHON_BIN --version

# 2. Check and Install Node Dependencies
Write-Host "`n[2/4] Verifying Node.js and Express server dependencies..." -ForegroundColor Green
if (-not (Test-Path "node_modules\express")) {
    Write-Host "Installing npm packages..." -ForegroundColor Yellow
    npm.cmd install
} else {
    Write-Host "Node modules already present." -ForegroundColor DarkGray
}

# 3. Execute Unit Tests
Write-Host "`n[3/4] Running Pytest Verification Suite..." -ForegroundColor Green
& $PYTHON_BIN -m pytest -v tests/test_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some unit tests failed, continuing to benchmark..." -ForegroundColor Yellow
} else {
    Write-Host "All unit tests passed successfully." -ForegroundColor Green
}

# 4. Execute Full Vehicle Simulation & Slide-7 Exporter
Write-Host "`n[4/4] Executing Simulation Engine and Generating Slide-7 Artifacts..." -ForegroundColor Green
& $PYTHON_BIN sim/run_all.py

if (Test-Path "out\results.json") {
    Write-Host "`nArtifacts verified:" -ForegroundColor Cyan
    Get-ChildItem "out" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
}

# 5. Start Express Web Cockpit
Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Launching SMART-EV 2.0 Web Cockpit on http://localhost:3000" -ForegroundColor Green
Write-Host " Press Ctrl+C to terminate server" -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor Cyan

$portInUse = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "Web Cockpit is already running on http://localhost:3000 (Process ID: $($portInUse[0].OwningProcess))." -ForegroundColor Green
    Write-Host "Open http://localhost:3000 in your browser to view the cockpit." -ForegroundColor Cyan
} else {
    node server/server.js
}

