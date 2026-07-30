<#
.SYNOPSIS
    Sets up a PhoenX RL environment on Windows.

.DESCRIPTION
    Creates a conda environment, then installs packages in this order: optional
    Isaac Sim / Isaac Lab, CUDA PyTorch, then PhoenX from pyproject.toml.
    Isaac first + torch -U last among those two keeps the cu128 build; PhoenX
    last so its floor constraints see already-satisfied versions.

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
$torchSpec    = @("torch==2.7.0", "torchvision==0.22.0")
$torchIndex   = "https://download.pytorch.org/whl/cu128"
$isaaclabSpec = "isaaclab[isaacsim,all]==2.3.2.post1"
$nvidiaIndex  = "https://pypi.nvidia.com"

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

# Deliberately a simple function, not an advanced one. Declaring a [Parameter()]
# attribute pulls in PowerShell's common parameters, and `pip install -e .` then
# dies with "the parameter name 'e' is ambiguous" (-ErrorAction/-ErrorVariable).
# A plain function passes everything through the automatic $args untouched.
function Invoke-EnvPython {
    & $condaExe run --no-capture-output -p $envPrefix python @args
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

# Deliberately NOT prepending Miniforge to PATH. Every conda call below uses the
# absolute $condaExe, while `conda run -p <prefix> python` resolves `python` from
# PATH — so putting base Miniforge first makes it run the BASE interpreter and
# install every package into the wrong environment.

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

# Assert conda run really drives the ENV's interpreter. If PATH ordering makes it
# resolve the base interpreter instead, every package below installs into the
# wrong environment and the only symptom is a confusing "no matching
# distribution" error much later. Fail loudly and immediately instead.
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
    Write-Host "Installing Isaac Lab + Isaac Sim (large, be patient) ..." -ForegroundColor Yellow
    Invoke-EnvPython -m pip install $isaaclabSpec --extra-index-url $nvidiaIndex
    Assert-Success "pip install isaaclab"

    # Isaac's resolver commonly downgrades filelock. Restore it afterwards.
    Invoke-EnvPython -m pip install -U "filelock>=3.20.1"
    Assert-Success "pip install filelock"
}

# -----------------------------------------------------------------------------
# Step 3: PyTorch with CUDA — AFTER Isaac (so this cu128 build survives
# isaacsim's own torch pin) and BEFORE PhoenX (so pyproject's torch>=2.5
# floor is already satisfied and the project install leaves it alone).
# Unconditional: PyPI torch on Windows is CPU-only; the cu128 index is what
# makes CUDA work in both Gymnasium and Isaac modes.
# -----------------------------------------------------------------------------
Write-Host "Installing PyTorch (CUDA) ..." -ForegroundColor Yellow
Invoke-EnvPython -m pip install -U @torchSpec --index-url $torchIndex
Assert-Success "pip install torch"

# -----------------------------------------------------------------------------
# Step 4: Build prerequisites
# gymnasium[box2d] pins box2d-py==2.3.5, which ships no cp311 Windows wheel and
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

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "    conda activate `"$envPrefix`"" -ForegroundColor Cyan
Write-Host "Then try:" -ForegroundColor Green
# Bundled example name, not a repo path: the top-level configs/ tree is personal
# and untracked, so a fresh clone has no such directory.
Write-Host "    phoenx-train --config LunarLanderContinuous-v3/sac.yml" -ForegroundColor Cyan
