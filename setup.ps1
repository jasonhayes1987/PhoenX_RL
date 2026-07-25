# setup.ps1 - Automated setup script for PhoenX RL environment
# Run this in the project root after cloning the repo.

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Immunize the whole run from per-user-site pollution. This is what was
# producing the recurring "cannot import name 'Sentinel' from
# 'typing_extensions'" errors and breaking conda's entry-point plugins.
$env:PYTHONNOUSERSITE         = "1"
$env:PYTHONDONTWRITEBYTECODE  = "1"

# Enable Windows long-path support. Required because the isaacsim extras
# ship wheels with file paths > 260 chars (e.g. isaacsim.replicator.caption.core
# test_data trees), which pip otherwise fails on with "[Errno 2] No such file
# or directory" + a hint pointing here. One-time HKLM registry write; needs
# Administrator. Takes effect immediately for child processes, no reboot.
$lpKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
if ((Get-ItemProperty -Path $lpKey -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled -ne 1) {
    Write-Host "Enabling Windows long-path support (requires Administrator)..." -ForegroundColor Yellow
    try {
        Set-ItemProperty -Path $lpKey -Name LongPathsEnabled -Value 1 -Type DWord
    } catch {
        throw "Could not set LongPathsEnabled in HKLM. Re-run setup.ps1 in an elevated PowerShell (Run as Administrator) one time; subsequent runs do not need elevation."
    }
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
$projectRoot    = $PSScriptRoot
# Pin CWD to the repo. Poetry resolves pyproject.toml from the current working
# directory, and when this script is launched elevated (Start-Process -Verb
# RunAs) the CWD defaults to C:\Windows\System32, not the repo. This makes every
# CWD-relative step correct no matter how the script is invoked.
Set-Location $projectRoot
$envName        = "rl_env"
$minicondaPath  = Join-Path $env:USERPROFILE "miniforge3"
$envPrefix      = Join-Path $minicondaPath   "envs\$envName"
$condaExe       = Join-Path $minicondaPath   "Scripts\conda.exe"
$envYml         = Join-Path $projectRoot     "environment.yml"

# Pinned Miniforge3 (conda-forge-only distribution). Avoids the
# repo.anaconda.com Terms-of-Service requirement entirely.
$minicondaInstaller = "Miniforge3-26.3.2-3-Windows-x86_64.exe"
$minicondaUrl       = "https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/$minicondaInstaller"

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

Write-Host "Setting up PhoenX RL environment..." -ForegroundColor Green

# -----------------------------------------------------------------------------
# Step 0: Install Miniforge (idempotent, non-destructive)
# -----------------------------------------------------------------------------
if (Test-Path $condaExe) {
    Write-Host "Found existing Miniforge at $minicondaPath, skipping install." -ForegroundColor DarkGray
} else {
    Write-Host "Installing Miniforge to $minicondaPath ..." -ForegroundColor Yellow
    $installerPath = Join-Path $env:TEMP $minicondaInstaller
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath `
        -ArgumentList @("/S", "/D=$minicondaPath") -Wait
    if (-not (Test-Path $condaExe)) {
        throw "Miniforge install did not produce $condaExe"
    }
}

# Put this conda first on PATH for the rest of the script.
$env:PATH = "$minicondaPath;$minicondaPath\Scripts;$minicondaPath\Library\bin;$env:PATH"

& $condaExe --version
Assert-Success "conda --version"

# -----------------------------------------------------------------------------
# Step 1: Create or update the conda env (idempotent, addressed by prefix)
# Using -p (not -n) so we never collide with another rl_env that might exist
# under a different miniconda/miniforge installation on this machine.
# -----------------------------------------------------------------------------
if (-not (Test-Path $envYml)) { throw "environment.yml not found at $envYml" }

if (Test-Path $envPrefix) {
    Write-Host "Updating existing conda env at $envPrefix ..." -ForegroundColor Yellow
    & $condaExe env update -p $envPrefix -f $envYml --prune
    Assert-Success "conda env update"
} else {
    Write-Host "Creating conda env at $envPrefix ..." -ForegroundColor Yellow
    & $condaExe env create  -p $envPrefix -f $envYml
    Assert-Success "conda env create"
}

# -----------------------------------------------------------------------------
# Step 2: Verify the env
# -----------------------------------------------------------------------------
$envPython = Join-Path $envPrefix "python.exe"
if (-not (Test-Path $envPython)) {
    throw "Expected $envPython after env creation, but it does not exist."
}
& $condaExe run --no-capture-output -p $envPrefix python --version
Assert-Success "python --version"

# -----------------------------------------------------------------------------
# Step 3: Install Poetry + project dependencies
#   - Do NOT call `poetry lock` unconditionally. The committed poetry.lock is
#     authoritative; we only refresh it (in place, no version bumps) if it is
#     out of date with pyproject.toml.
#   - Use a SHORT, AppData-free Poetry cache. `%USERPROFILE%\.poetry-cache`
#     avoids both the original Permission-Denied on %LOCALAPPDATA%\pypoetry and
#     the MAX_PATH blowups from deep repo paths + long wheel filenames (glfw).
#   - Invoke Poetry via `python -m poetry` from THIS env, never a stray
#     poetry.exe on PATH.
#   - Activate via `conda run` so cxx-compiler / vcvars are configured for
#     any source builds Poetry triggers (e.g. envpool).
# -----------------------------------------------------------------------------
Write-Host "Installing Poetry and project dependencies..." -ForegroundColor Yellow

$env:POETRY_CACHE_DIR          = Join-Path $env:USERPROFILE ".poetry-cache"
$env:POETRY_VIRTUALENVS_CREATE = "false"
$env:POETRY_NO_INTERACTION     = "1"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install --upgrade pip
Assert-Success "pip upgrade"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install "poetry==1.8.3"
Assert-Success "pip install poetry"

# Poetry refuses to install if poetry.lock is out of date with respect to
# pyproject.toml ("pyproject.toml changed significantly since poetry.lock was
# last generated"). Detect that explicitly and refresh the lock in place
# WITHOUT bumping any pinned versions, then install. We do this via
# `conda run -p $envPrefix` so Poetry uses rl_env's Python 3.11 (the only
# interpreter on this machine that satisfies pyproject.toml's
# `python = ">=3.11,<3.12"` constraint).
& $condaExe run --no-capture-output -p $envPrefix `
    python -m poetry check --lock
if ($LASTEXITCODE -ne 0) {
    Write-Host "poetry.lock is out of date with pyproject.toml; refreshing via 'poetry lock --no-update'..." -ForegroundColor Yellow
    & $condaExe run --no-capture-output -p $envPrefix `
        python -m poetry lock --no-update
    Assert-Success "poetry lock --no-update"
    Write-Host "Note: poetry.lock was regenerated. Consider committing it." -ForegroundColor DarkYellow
}

& $condaExe run --no-capture-output -p $envPrefix `
    python -m poetry install --with dev
Assert-Success "poetry install"

# -----------------------------------------------------------------------------
# Step 4: Special packages (PyTorch CUDA, IsaacLab, EnvPool, Gymnasium-Robotics)
# -----------------------------------------------------------------------------
Write-Host "Installing special packages (torch, isaaclab, envpool, gymnasium-robotics)..." -ForegroundColor Yellow

$env:OMNI_KIT_ACCEPT_EULA = "yes"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install -U `
        "torch==2.7.0" "torchvision==0.22.0" `
        --index-url "https://download.pytorch.org/whl/cu128"
Assert-Success "pip install torch"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install `
        "isaaclab[isaacsim]==2.3.0" `
        --extra-index-url "https://pypi.nvidia.com"
Assert-Success "pip install isaaclab"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install "envpool>=1.0.1"
Assert-Success "pip install envpool"

& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install `
        "git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0"
Assert-Success "pip install gymnasium-robotics"

# Some of the above can downgrade filelock; bring it back up.
& $condaExe run --no-capture-output -p $envPrefix `
    python -m pip install -U "filelock>=3.20.1"
Assert-Success "pip install filelock"

# -----------------------------------------------------------------------------
# Step 5: Verify
# -----------------------------------------------------------------------------
Write-Host "Verifying setup..." -ForegroundColor Yellow
& $condaExe run --no-capture-output -p $envPrefix `
    python -c "import os; os.environ['OMNI_KIT_ACCEPT_EULA']='yes'; import torch, isaaclab, gymnasium_robotics; print('Imports OK'); print('CUDA available:', torch.cuda.is_available())"
Assert-Success "import verification"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "    conda activate `"$envPrefix`"" -ForegroundColor Cyan
