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

# Step 3: Configure Poetry
Write-Host "Configuring Poetry..." -ForegroundColor Yellow
poetry config virtualenvs.create false

# Step 4: Install Poetry dependencies
Write-Host "Installing Poetry dependencies..." -ForegroundColor Yellow
poetry install --with dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 5: Install special packages manually
Write-Host "Installing special packages (torch, torchvision, isaaclab, gymnasium-robotics)..." -ForegroundColor Yellow

# PyTorch CUDA
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# IsaacLab
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Gymnasium Robotics
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 6: Verify setup
Write-Host "Verifying setup..." -ForegroundColor Yellow
poetry run python -c "import torch, isaaclab, gymnasium_robotics; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete! Activate rl_env and start developing." -ForegroundColor Green
