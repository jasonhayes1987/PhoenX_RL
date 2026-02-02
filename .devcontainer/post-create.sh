#!/bin/bash
set -e  # Exit on error

echo "Setting up PhoenX RL environment..."

# Step 1: Create Conda environment from environment.yml
echo "Creating Conda environment from environment.yml..."
conda env create -f environment.yml

# Step 2: Activate environment
echo "Activating rl_env..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl_env

# Step 3: Configure Poetry (disable virtualenv creation since we're in Conda)
echo "Configuring Poetry..."
poetry config virtualenvs.create false

# Step 4: Install Poetry dependencies
echo "Installing Poetry dependencies..."
poetry install --with dev

# Step 5: Install special packages manually
echo "Installing special packages (torch, torchvision, isaaclab, gymnasium-robotics)..."

# PyTorch: Use CPU version (no GPU in Codespaces; change to cu128 if on GPU cloud)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu

# IsaacLab: Install base (Isaac Sim integration may fail without GPU/Isaac Sim installed)
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
# Note: Skip [isaacsim] extra to avoid GPU-dependent errors; add it if needed

# Gymnasium Robotics
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0

# Step 6: Verify setup (CUDA will be False due to no GPU)
echo "Verifying setup..."
poetry run python -c "import torch, isaaclab, gymnasium_robotics; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"

echo "Setup complete! Environment is ready (activate with 'conda activate rl_env')."
