# LocalVid Studio - Windows setup. Run through setup.bat (double-click), or:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Say($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. Python
Say "1/5  Looking for Python 3.12 / 3.11 / 3.10"
$pyVersion = $null
foreach ($v in @("3.12", "3.11", "3.10")) {
    try {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $pyVersion = $v; break }
    } catch { }
}
if (-not $pyVersion) {
    Fail "No Python 3.10-3.12 found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH' and keep the 'py launcher' option), then run setup.bat again."
}
Write-Host "Using Python $pyVersion"

# ---------------------------------------------------------------- 2. venv
Say "2/5  Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py "-$pyVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}
$py = (Resolve-Path ".venv\Scripts\python.exe").Path
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed (check your internet connection)." }

# ---------------------------------------------------------------- 3. PyTorch (CUDA 12.8 build for RTX 50-series)
Say "3/5  Installing PyTorch with CUDA 12.8 (needed for RTX 50-series GPUs)"
& $py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Fail "PyTorch install failed. Open https://pytorch.org/get-started/locally/ , pick Windows / Pip / CUDA, and run the command it shows using $py -m pip ..."
}

# ---------------------------------------------------------------- 4. the rest
Say "4/5  Installing the other packages (this takes a few minutes)"
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installing requirements.txt failed. Scroll up for the first error and see docs/SETUP.md (Troubleshooting)." }

# ---------------------------------------------------------------- 5. checks
Say "5/5  Checking ffmpeg and your GPU"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg is not on PATH. The app can use a bundled copy, but a full install is better:" -ForegroundColor Yellow
    Write-Host "    winget install Gyan.FFmpeg      (then open a new terminal)" -ForegroundColor Yellow
}
& $py scripts\check_gpu.py
if ($LASTEXITCODE -ne 0) { Write-Host "`nGPU check reported a problem - fix it before generating AI clips (see message above)." -ForegroundColor Yellow }

Write-Host "`nSetup finished. Start the app with run.bat" -ForegroundColor Green
