<#
.SYNOPSIS
    Sets up a PhoenX RL environment on Windows.

.DESCRIPTION
    Creates a conda environment, installs PhoenX and its dependencies, and
    optionally installs Isaac Sim / Isaac Lab. All package installation goes
    through pip against the standard pyproject.toml — there is no Poetry and no
    environment.yml. Safe to re-run; every step is idempotent.

.EXAMPLE
    .\setup.ps1
    Gymnasium mode with dev tooling. No Isaac. Fast (~5 min).

.EXAMPLE
    .\setup.ps1 -Isaac -Docs
    Everything: Isaac Sim/Lab, dev tooling, docs tooling. Slow (~30-60 min).

.EXAMPLE
    .\setup.ps1 -EnvName phoenx-test -NonEditable
    Throwaway env, non-editable install — reproduces exactly what an end user
    gets from `pip install`. Use this to test a release before pushing.
#>

param(
    # Install Isaac Sim + Isaac Lab (large download, requires NVIDIA GPU).
    [switch]$Isaac,

    # Install docs tooling (mkdocs-material, mkdocstrings).
    [switch]$Docs,

    # Skip dev tooling (pytest, black, isort, pylint). Dev is ON by default.
    [switch]$NoDev,

    # Install non-editable, i.e. copy into site-packages instead of linking to
    # the source tree. Use to verify the real end-user install path.
    [switch]$NonEditable,

    # Conda environment name. Override to build a throwaway test env.
    [string]$EnvName = "phoenx",

    # Python version. Must match your Isaac Sim release:
    #   Isaac Sim 4.5 -> 3.10   |   5.X -> 3.11   |   6.X -> 3.12
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# -----------------------------------------------------------------------------
# Environment hygiene
# -----------------------------------------------------------------------------

# Ignore per-user site-packages. Prevents a stale user-level typing_extensions
# from shadowing the env's copy ("cannot import name 'Sentinel'").
$env:PYTHONNOUSERSITE        = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

# Short pip cache path. Deep repo paths + long wheel filenames (glfw, isaacsim)
# blow past MAX_PATH inside %LOCALAPPDATA%\pip.
$env:PIP_CACHE_DIR = Join-Path $env:USERPROFILE ".pip-cache"

# Isaac Sim refuses to import without this.
$env:OMNI_KIT_ACCEPT_EULA = "yes"

# Windows long-path support. isaacsim extras ship files > 260 chars, which pip
# fails on with a misleading "[Errno 2] No such file or directory".
# One-time HKLM write; needs Administrator once, then never again.
$lpKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
$lpVal = (Get-ItemProperty -Path $lpKey -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($lpVal -ne 1) {
    if ($Isaac) {
        Write-Host "Enabling Windows long-path support (requires Administrator)..." -ForegroundColor Yellow
        try {
            Set-ItemProperty -Path $lpKey -Name LongPathsEnabled -Value 1 -Type DWord
        } catch {
            throw "Could not set LongPathsEnabled. Re-run setup.ps1 once in an elevated PowerShell (Run as Administrator); later runs do not need elevation."
        }
    } else {
        Write-Host "Note: long paths are disabled. Not needed for this install, but -Isaac will require it." -ForegroundColor DarkGray
    }
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
$projectRoot = $PSScriptRoot
Set-Location $projectRoot   # so relative paths work even when launched elevated

$minicondaPath = Join-Path $env:USERPROFILE "miniforge3"
$envPrefix     = Join-Path $minicondaPath   "envs\$EnvName"
$condaExe      = Join-Path $minicondaPath   "Scripts\conda.exe"

# Miniforge = conda-forge-only distribution; avoids the repo.anaconda.com ToS.
$minicondaInstaller = "Miniforge3-26.3.2-3-Windows-x86_64.exe"
$minicondaUrl       = "https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/$minicondaInstaller"

# Pinned versions — these are known-good on this machine. Bump deliberately.
$torchSpec       = @("torch==2.7.0", "torchvision==0.22.0")
$torchIndex      = "https://download.pytorch.org/whl/cu128"
$isaaclabSpec    = "isaaclab[isaacsim]==2.3.0"
$nvidiaIndex     = "https://pypi.nvidia.com"
$gymRoboticsSpec = "git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0"

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Invoke-EnvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $condaExe run --no-capture-output -p $envPrefix python @Args
}

Write-Host "Setting up PhoenX RL ($EnvName, Python $PythonVersion)..." -ForegroundColor Green

# -----------------------------------------------------------------------------
# Step 0: Miniforge
# -----------------------------------------------------------------------------
if (Test-Path $condaExe) {
    Write-Host "Found Miniforge at $minicondaPath." -ForegroundColor DarkGray
} else {
    Write-Host "Installing Miniforge to $minicondaPath ..." -ForegroundColor Yellow
    $installerPath = Join-Path $env:TEMP $minicondaInstaller
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -ArgumentList @("/S", "/D=$minicondaPath") -Wait
    if (-not (Test-Path $condaExe)) { throw "Miniforge install did not produce $condaExe" }
}

$env:PATH = "$minicondaPath;$minicondaPath\Scripts;$minicondaPath\Library\bin;$env:PATH"

# -----------------------------------------------------------------------------
# Step 1: Conda env — Python version + isolation only. No packages.
# Addressed by prefix (-p) so it can't collide with an rl_env from another
# conda installation on the machine.
# -----------------------------------------------------------------------------
if (Test-Path $envPrefix) {
    Write-Host "Using existing env at $envPrefix." -ForegroundColor DarkGray
} else {
    Write-Host "Creating conda env at $envPrefix ..." -ForegroundColor Yellow
    & $condaExe create -y -p $envPrefix "python=$PythonVersion"
    Assert-Success "conda create"
}

Invoke-EnvPython --version
Assert-Success "python --version"

Invoke-EnvPython -m pip install --upgrade pip
Assert-Success "pip upgrade"

# -----------------------------------------------------------------------------
# Step 2: PyTorch with CUDA — BEFORE the project install.
# pyproject declares `torch>=2.5`. Installing the CUDA build first means the
# project install sees that constraint already satisfied and leaves it alone.
# Reversing this order downloads a second, CPU-only torch (~2.5 GB wasted) and
# then overwrites it.
# -----------------------------------------------------------------------------
Write-Host "Installing PyTorch (CUDA) ..." -ForegroundColor Yellow
Invoke-EnvPython -m pip install -U @torchSpec --index-url $torchIndex
Assert-Success "pip install torch"

# -----------------------------------------------------------------------------
# Step 3: PhoenX itself, from pyproject.toml
# -----------------------------------------------------------------------------
$extras = @()
if (-not $NoDev) { $extras += "dev" }
if ($Docs)       { $extras += "docs" }
$extraSpec = if ($extras.Count -gt 0) { "[" + ($extras -join ",") + "]" } else { "" }

if ($NonEditable) {
    Write-Host "Installing PhoenX (non-editable) with extras: $($extras -join ', ') ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install ".$extraSpec"
} else {
    Write-Host "Installing PhoenX (editable) with extras: $($extras -join ', ') ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install -e ".$extraSpec"
}
Assert-Success "pip install PhoenX"

# -----------------------------------------------------------------------------
# Step 4: Optional Isaac Sim / Isaac Lab
# Cannot live in pyproject dependencies: it comes from NVIDIA's package index,
# which PEP 508 metadata has no way to express portably.
# -----------------------------------------------------------------------------
if ($Isaac) {
    Write-Host "Installing Isaac Lab + Isaac Sim (large, be patient) ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install $isaaclabSpec --extra-index-url $nvidiaIndex
    Assert-Success "pip install isaaclab"

    # Isaac's resolver commonly downgrades filelock. Restore it afterwards.
    Invoke-EnvPython -m pip install -U "filelock>=3.20.1"
    Assert-Success "pip install filelock"
}

# -----------------------------------------------------------------------------
# Step 5: Packages that need a non-PyPI source or are platform-fragile
# -----------------------------------------------------------------------------
Write-Host "Installing Gymnasium-Robotics ..." -ForegroundColor Yellow
Invoke-EnvPython -m pip install $gymRoboticsSpec
Assert-Success "pip install gymnasium-robotics"

# envpool publishes Linux wheels only. Attempt it, but never fail the setup:
# on Windows the correct outcome is "not installed" plus a clear warning.
Write-Host "Attempting envpool (Linux-only; expected to skip on Windows) ..." -ForegroundColor Yellow
Invoke-EnvPython -m pip install "envpool>=1.0.1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "envpool unavailable on this platform — continuing. Vectorized envs will fall back to Gymnasium's own vector API." -ForegroundColor DarkYellow
    $global:LASTEXITCODE = 0
}

# -----------------------------------------------------------------------------
# Step 6: Verify
# -----------------------------------------------------------------------------
Write-Host "Verifying ..." -ForegroundColor Yellow

# Import from a directory OUTSIDE the repo. This is the real proof that the
# package is installed rather than merely present on disk — it would have
# failed under the old sys.path.insert approach.
Push-Location $env:TEMP
try {
    Invoke-EnvPython -c "import phoenx; print('phoenx', phoenx.__version__, 'imports cleanly from', __import__('os').getcwd())"
    Assert-Success "import phoenx"
} finally {
    Pop-Location
}

Invoke-EnvPython -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Assert-Success "torch check"

if ($Isaac) {
    Invoke-EnvPython -c "import isaaclab; print('isaaclab imports OK')"
    Assert-Success "isaaclab check"
}

# Console entry points exist and are on PATH inside the env.
& $condaExe run --no-capture-output -p $envPrefix phoenx-train --help | Out-Null
Assert-Success "phoenx-train --help"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "    conda activate `"$envPrefix`"" -ForegroundColor Cyan
Write-Host "Then try:" -ForegroundColor Green
Write-Host "    phoenx-train --config configs/LunarLander-v3/sac.yml" -ForegroundColor Cyan