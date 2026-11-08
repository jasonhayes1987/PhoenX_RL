<#
.SYNOPSIS
    Sets up a PhoenX RL environment on Windows.

.DESCRIPTION
    Creates a conda environment, then installs packages in this order: optional
    Isaac Sim / Isaac Lab, CUDA PyTorch (Gymnasium mode only), then PhoenX
    from pyproject.toml. Isaac mode installs through uv with Isaac Lab 3.0's
    overrides, which already pin torch 2.11.0 and the cu130 wheels. PhoenX
    installs last so its floor constraints see already-satisfied versions.

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

.EXAMPLE
    .\setup.ps1 -CondaRoot E:\Miniconda3 -EnvName rl_env
    Build the env under an existing conda installation instead of the default
    Miniforge location.
#>

param(
    # Install Isaac Sim + Isaac Lab (large download, requires NVIDIA GPU).
    [switch]$Isaac,

    # Install docs tooling (mkdocs-material, mkdocstrings).
    [switch]$Docs,

    # Skip dev tooling (pytest, black, isort, pylint, ruff). Dev is ON by default.
    [switch]$NoDev,

    # Install non-editable, i.e. copy into site-packages instead of linking to
    # the source tree. Use to verify the real end-user install path.
    [switch]$NonEditable,

    # Conda environment name. Override to build a throwaway test env.
    [string]$EnvName = "phoenx",

    # Python version. Isaac Lab 3.0 / Isaac Sim 6.X require 3.12.
    [string]$PythonVersion = "3.12",

    # Root of the conda installation to use. Defaults to Miniforge in the user profile;
    # point it at an existing installation to build the env there instead.
    [string]$CondaRoot = (Join-Path $env:USERPROFILE "miniforge3")
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

$envPrefix = Join-Path $CondaRoot "envs\$EnvName"
$condaExe  = Join-Path $CondaRoot "Scripts\conda.exe"

# Miniforge = conda-forge-only distribution; avoids the repo.anaconda.com ToS.
$minicondaInstaller = "Miniforge3-26.3.2-3-Windows-x86_64.exe"
$minicondaUrl       = "https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/$minicondaInstaller"

# Pinned versions — these are known-good on this machine. Bump deliberately.
$torchSpec    = @("torch==2.11.0", "torchvision==0.26.0")
$torchIndex   = "https://download.pytorch.org/whl/cu130"
$isaaclabSpec = "isaaclab[isaacsim,all]==3.0.0"
$nvidiaIndex  = "https://pypi.nvidia.com"
# pip has no equivalent of uv's --overrides. This file pins torch 2.11.0,
# torchvision 0.26.0, and numpy>=2 for the Isaac Lab 3.0 stack.
$isaacOverrides = "https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Invoke-EnvPython {
    & $condaExe run --no-capture-output -p $envPrefix python @args
}

Write-Host "Setting up PhoenX RL ($EnvName, Python $PythonVersion)..." -ForegroundColor Green

# -----------------------------------------------------------------------------
# Step 0: Miniforge
# -----------------------------------------------------------------------------
if (Test-Path $condaExe) {
    Write-Host "Found conda at $CondaRoot." -ForegroundColor DarkGray
} else {
    Write-Host "Installing Miniforge to $CondaRoot ..." -ForegroundColor Yellow
    $installerPath = Join-Path $env:TEMP $minicondaInstaller
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -ArgumentList @("/S", "/D=$CondaRoot") -Wait
    if (-not (Test-Path $condaExe)) { throw "Miniforge install did not produce $condaExe" }
}


# -----------------------------------------------------------------------------
# Step 1: Conda env — Python version + isolation only. No packages.
# Addressed by prefix (-p) so it can't collide with a same-named env from
# another conda installation on the machine.
# -----------------------------------------------------------------------------
if (Test-Path $envPrefix) {
    Write-Host "Using existing env at $envPrefix." -ForegroundColor DarkGray
} else {
    Write-Host "Creating conda env at $envPrefix ..." -ForegroundColor Yellow
    & $condaExe create -y -p $envPrefix "python=$PythonVersion"
    Assert-Success "conda create"
}

# Assert conda run really drives the ENV's interpreter.
$probe = & $condaExe run -p $envPrefix python -c "import sys; print(sys.prefix)" 2>&1 | Out-String
Assert-Success "python interpreter probe"
# Last non-empty line, so a stray conda warning on stderr can't fail the compare.
$resolvedPrefix = ($probe -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
if ($resolvedPrefix -ne $envPrefix) {
    throw "Environment mismatch: expected the interpreter at '$envPrefix' but conda run resolved sys.prefix='$resolvedPrefix'. Refusing to install into the wrong environment."
}
Write-Host "Interpreter: $resolvedPrefix" -ForegroundColor DarkGray

Invoke-EnvPython -m pip install --upgrade pip
Assert-Success "pip upgrade"

# -----------------------------------------------------------------------------
# Step 2: Optional Isaac Sim / Isaac Lab
# -----------------------------------------------------------------------------
if ($Isaac) {
    Write-Host "Installing uv ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install uv
    Assert-Success "pip install uv"

    Write-Host "Installing Isaac Lab + Isaac Sim (large, be patient) ..." -ForegroundColor Yellow
    $envPython = Join-Path $envPrefix "python.exe"
    & $condaExe run --no-capture-output -p $envPrefix uv pip install --python $envPython $isaaclabSpec --overrides $isaacOverrides --extra-index-url $nvidiaIndex --torch-backend cu130 --index-strategy unsafe-best-match
    Assert-Success "uv pip install isaaclab"

    # Isaac Sim 5.1's pip resolver always downgraded filelock. The 3.0 uv
    # overrides do not pin it, so restore the floor only when the install
    # left it missing or below 3.20.1.
    # One line: conda run rejects python -c arguments that contain newlines.
    $filelockCode = "import importlib.metadata as m; d={x.metadata['Name'].lower(): x.version for x in m.distributions()}; v=d.get('filelock',''); print('missing' if not v else (('ok' if tuple(int(p) for p in v.split('.')[:3])>=(3,20,1) else 'low')+' '+v))"
    $filelockReport = & $condaExe run -p $envPrefix python -c $filelockCode
    Assert-Success "filelock version check"
    $filelockLine = ($filelockReport | Out-String).Trim().Split("`n")[-1].Trim()
    Write-Host "filelock after Isaac install: $filelockLine" -ForegroundColor DarkGray
    if ($filelockLine -notlike "ok *") {
        Write-Host "Restoring filelock>=3.20.1 ..." -ForegroundColor Yellow
        Invoke-EnvPython -m pip install -U "filelock>=3.20.1"
        Assert-Success "pip install filelock"
    }
}

# -----------------------------------------------------------------------------
# Step 3: PyTorch with CUDA — Gymnasium mode only.
# Isaac mode already received torch 2.11.0 cu130 from the uv overrides.
# PyPI torch on Windows is CPU-only; the cu130 index is what makes CUDA work.
# Installed before PhoenX so pyproject's torch>=2.5 floor is already met.
# -----------------------------------------------------------------------------
if (-not $Isaac) {
    Write-Host "Installing PyTorch (CUDA) ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install -U @torchSpec --index-url $torchIndex
    Assert-Success "pip install torch"
}

# -----------------------------------------------------------------------------
# Step 4: Build prerequisites
# gymnasium[box2d] pins box2d-py==2.3.5, which ships no cp312 Windows wheel and
# so builds from source — and that build shells out to SWIG. Installed from
# conda-forge rather than PyPI on purpose: the pip `swig` package is a Python
# console-script shim that dies with "No module named 'swig'" inside pip's build
# isolation, while conda-forge ships a native swig.exe that isolation can't break.
# -----------------------------------------------------------------------------
Write-Host "Installing build prerequisites (swig, needed to build box2d-py) ..." -ForegroundColor Yellow
& $condaExe install -y -p $envPrefix -c conda-forge swig
Assert-Success "conda install swig"

# -----------------------------------------------------------------------------
# Step 5: PhoenX itself, from pyproject.toml
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

# Record which env was built so activation scripts can find it. Missing recorder
# is non-fatal — a successful install that merely lacks a convenience record
# should not be reported as a failed install.
$useEnvScript = Join-Path $PSScriptRoot "scripts\use-env.ps1"
$recorded = $false
if (Test-Path $useEnvScript) {
    # try/catch, not Assert-Success: PowerShell script invocations do not set
    # $LASTEXITCODE the way native executables do; use-env.ps1 throws on failure.
    try {
        & $useEnvScript -CondaRoot $CondaRoot -EnvName $EnvName -Quiet
        $recorded = $true
    } catch {
        Write-Host "Warning: could not record environment ($_). Activation scripts may not pick it up automatically." -ForegroundColor Yellow
    }
} else {
    Write-Host "Warning: scripts\use-env.ps1 not found; skipping environment record. Activation scripts may not pick it up automatically." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
if ($recorded) {
    Write-Host "The environment was recorded in .phoenx-env so newly opened terminals pick it up." -ForegroundColor Green
}
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "    PowerShell:  .\scripts\activate.ps1" -ForegroundColor Cyan
Write-Host "    Git Bash:    source scripts/activate.sh" -ForegroundColor Cyan
Write-Host "Or fall back to:" -ForegroundColor Green
Write-Host "    conda activate `"$envPrefix`"" -ForegroundColor Cyan
Write-Host "Then try:" -ForegroundColor Green
Write-Host "    phoenx-train --config LunarLanderContinuous-v3/sac.yml" -ForegroundColor Cyan
