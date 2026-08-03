<#
.SYNOPSIS
    Activates the PhoenX conda environment recorded in .phoenx-env.

.DESCRIPTION
    Intended as a VS Code / Cursor integrated-terminal startup command.
    Reads the machine-local record written by use-env.ps1 (or setup.ps1),
    initializes conda from the RECORDED root (PATH may point at a different
    conda install), and activates by prefix.

.NOTES
    Must not throw on missing record: this runs at terminal open. A red
    exception wall would make every new terminal look broken.
#>

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Locate / parse .phoenx-env
# -----------------------------------------------------------------------------

function Get-PhoenXEnvRecord {
    <#
    .SYNOPSIS
        Parse <repoRoot>\.phoenx-env into a hashtable of KEY -> value.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $record = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.StartsWith("#")) { continue }
        if ($line -match "^(\w+)='(.*)'$") {
            $record[$Matches[1]] = $Matches[2]
        }
    }
    return $record
}

function Test-PhoenXPrefixMatch {
    <#
    .SYNOPSIS
        Compare two env prefixes case-insensitively, ignoring a trailing separator.
    #>
    param(
        [string]$Expected,
        [string]$Actual
    )

    if (-not $Expected -or -not $Actual) { return $false }
    $a = $Expected.TrimEnd('\', '/')
    $b = $Actual.TrimEnd('\', '/')
    return ($a -ieq $b)
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$recordPath = Join-Path $repoRoot ".phoenx-env"

$EnvPrefix = $null
$CondaRoot = $null

if (Test-Path -LiteralPath $recordPath) {
    $record = Get-PhoenXEnvRecord -Path $recordPath
    $EnvPrefix = $record["PHOENX_ENV_PREFIX"]
    $CondaRoot = $record["PHOENX_CONDA_ROOT"]
}
elseif ($env:PHOENX_ENV_PREFIX) {
    # Fallback: process already has the vars (e.g. parent shell exported them).
    $EnvPrefix = $env:PHOENX_ENV_PREFIX
    $CondaRoot = $env:PHOENX_CONDA_ROOT
}
else {
    Write-Host "No .phoenx-env record found. Run scripts\use-env.ps1 -Prefix <path> or setup.ps1 first." -ForegroundColor Yellow
    return
}

if (-not $EnvPrefix) {
    # Distinct from the missing-file case above: the record (or the inherited
    # variables) exist but carry no prefix, so point at the bad content rather
    # than sending the user looking for a file that is already there.
    Write-Host "PHOENX_ENV_PREFIX is missing from the .phoenx-env record. Re-run scripts\use-env.ps1 -Prefix <path>." -ForegroundColor Yellow
    return
}

# Stale per-user site-packages (typing_extensions) can shadow conda's copy and
# break conda plugins with "cannot import name 'Sentinel'".
$env:PYTHONNOUSERSITE = "1"

if (-not $CondaRoot) {
    Write-Host "PHOENX_CONDA_ROOT is unset; cannot locate conda.exe. Re-run use-env.ps1." -ForegroundColor Yellow
    return
}

# Always load the PowerShell hook from the RECORDED root. PATH may expose
# conda.bat (Application) which cannot mutate this session — Get-Command is
# not a reliable signal that conda shell functions are already installed.
try {
    (& "$CondaRoot\Scripts\conda.exe" shell.powershell hook) | Out-String | Invoke-Expression
    conda activate $EnvPrefix
}
catch {
    Write-Host ("Failed to activate PhoenX env '{0}': {1}" -f $EnvPrefix, $_.Exception.Message) -ForegroundColor Yellow
    return
}

# conda.bat / a broken hook can leave CONDA_PREFIX at base while activate
# appears to succeed — only claim success when the prefix actually matches.
if (Test-PhoenXPrefixMatch -Expected $EnvPrefix -Actual $env:CONDA_PREFIX) {
    Write-Host "Activated PhoenX env: $EnvPrefix" -ForegroundColor Green
}
else {
    Write-Host ("Activation mismatch: requested '{0}' but CONDA_PREFIX is '{1}'." -f $EnvPrefix, $env:CONDA_PREFIX) -ForegroundColor Yellow
}
