# setup.ps1 - Automated setup script for PhoenX RL environment
# Run this in the project root after cloning the repo

Write-Host "Setting up PhoenX RL environment..." -ForegroundColor Green

# Step 0: Install Miniconda
Write-Host "Installing Miniconda..." -ForegroundColor Yellow
$minicondaPath = "$env:USERPROFILE\miniconda3"

# Remove existing installation if it exists
if (Test-Path $minicondaPath) {
    Write-Host "Removing existing Miniconda installation..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $minicondaPath
}

$minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$installerPath = "$env:TEMP\miniconda_installer.exe"
Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
Start-Process -FilePath $installerPath -ArgumentList "/S /D=$minicondaPath" -Wait

# Update PATH for the rest of the script
$env:PATH = "$minicondaPath;$minicondaPath\Scripts;$env:PATH"

# Verify conda is available
& "$minicondaPath\Scripts\conda.exe" --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Conda installation failed or PATH not updated correctly" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 1: Accept conda Terms of Service
Write-Host "Accepting conda Terms of Service..." -ForegroundColor Yellow
& "$minicondaPath\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
& "$minicondaPath\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
& "$minicondaPath\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# Step 2: Create conda environment
Write-Host "Creating conda environment from environment.yml..." -ForegroundColor Yellow
& "$minicondaPath\Scripts\conda.exe" env create -f environment.yml --yes
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 3: Verify environment creation
Write-Host "Verifying conda environment..." -ForegroundColor Yellow
& "$minicondaPath\Scripts\conda.exe" env list
& "$minicondaPath\Scripts\conda.exe" run -n rl_env python --version

# Step 4: Install Poetry and dependencies using conda environment
Write-Host "Installing Poetry and dependencies using conda environment..." -ForegroundColor Yellow

# Create temporary script file
$tempScript = "$env:TEMP\poetry_setup.py"
@"
import subprocess
import sys
import os

# Install Poetry
print("Installing Poetry...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'poetry==1.8.3'], check=True)

# Upgrade pip
print("Upgrading pip...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)

# Configure Poetry
print("Configuring Poetry...")
os.environ['PATH'] += os.pathsep + os.path.expanduser('~/AppData/Roaming/Python/Scripts')
subprocess.run(['poetry', 'config', 'virtualenvs.create', 'false'], check=True)

# Lock and install Poetry dependencies
print("Locking Poetry dependencies...")
subprocess.run(['poetry', 'lock'], check=True)

print("Installing Poetry dependencies...")
subprocess.run(['poetry', 'install', '--with', 'dev'], check=True)

print("Poetry setup completed successfully!")
"@ | Out-File -FilePath $tempScript -Encoding UTF8

# Run the script using conda
& "$minicondaPath\Scripts\conda.exe" run -n rl_env python $tempScript
if ($LASTEXITCODE -ne 0) {
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    exit $LASTEXITCODE
}

# Clean up temporary script
Remove-Item $tempScript -ErrorAction SilentlyContinue

# Step 5: Install special packages manually
Write-Host "Installing special packages (torch, torchvision, isaaclab, gymnasium-robotics)..." -ForegroundColor Yellow

# Create temporary script file for special packages
$tempPackagesScript = "$env:TEMP\special_packages.py"
@"
import subprocess
import sys
import os

print("Installing PyTorch CUDA...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'torch==2.7.0', 'torchvision==0.22.0', '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)

print("Installing IsaacLab...")
# Set environment variable to accept EULA automatically
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'yes'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'isaaclab[isaacsim]==2.3.0', '--extra-index-url', 'https://pypi.nvidia.com'], check=True)

print("Installing EnvPool...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'envpool>=1.0.1'], check=True)

print("Installing Gymnasium Robotics...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0'], check=True)

print("Updating filelock to resolve version conflicts...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'filelock>=3.20.1'], check=True)

print("Special packages installation completed!")
"@ | Out-File -FilePath $tempPackagesScript -Encoding UTF8

# Run the script using conda
& "$minicondaPath\Scripts\conda.exe" run -n rl_env python $tempPackagesScript
if ($LASTEXITCODE -ne 0) {
    Remove-Item $tempPackagesScript -ErrorAction SilentlyContinue
    exit $LASTEXITCODE
}

# Clean up temporary script
Remove-Item $tempPackagesScript -ErrorAction SilentlyContinue

# Step 6: Verify setup
Write-Host "Verifying setup..." -ForegroundColor Yellow
& "$minicondaPath\Scripts\conda.exe" run -n rl_env python -c "import os; os.environ['OMNI_KIT_ACCEPT_EULA'] = 'yes'; import torch, isaaclab, gymnasium_robotics; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete! Activate rl_env and start developing." -ForegroundColor Green
