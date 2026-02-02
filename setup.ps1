# setup.ps1 - Automated setup script for PhoenX RL environment
# Run this in the project root after cloning the repo

Write-Host "Setting up PhoenX RL environment..." -ForegroundColor Green

# Step 1: Create conda environment
Write-Host "Creating conda environment from environment.yml..." -ForegroundColor Yellow
conda env create -f environment.yml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: Activate environment
Write-Host "Activating rl_env..." -ForegroundColor Yellow
conda activate rl_env

# Step 3: Ensure latest version of pip
Write-Host "Ensuring latest version of pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 4: Install Poetry
Write-Host "Installing Poetry..." -ForegroundColor Yellow
curl.exe -sSL https://install.python-poetry.org | python - --version 1.8.3
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 5: Configure Poetry
Write-Host "Configuring Poetry..." -ForegroundColor Yellow
$env:PATH = "$env:PATH;$env:APPDATA\Python\Scripts"
poetry config virtualenvs.create false

# Step 6: Lock Poetry
poetry lock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 7: Install Poetry dependencies
Write-Host "Installing Poetry dependencies..." -ForegroundColor Yellow
poetry install --with dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 8: Install special packages manually
Write-Host "Installing special packages (torch, torchvision, isaaclab, gymnasium-robotics)..." -ForegroundColor Yellow
# PyTorch CUDA
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# IsaacLab
pip install isaaclab[isaacsim]==2.3.0 --extra-index-url https://pypi.nvidia.com
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Gymnasium Robotics
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 6: Verify setup
Write-Host "Verifying setup..." -ForegroundColor Yellow
python -c "import torch, isaaclab, gymnasium_robotics; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete! Activate rl_env and start developing." -ForegroundColor Green
