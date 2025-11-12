#!/bin/bash

# Ensure Conda is available
source /opt/conda/etc/profile.d/conda.sh

# Create and activate the Conda environment from environment.yml
conda env create -f environment.yml
conda activate rl_env

# Upgrade pip
python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch (matching your guide)
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Lab and Isaac Sim dependencies
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com

# Install Gymnasium Robotics from GitHub
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0

# Optional: Verify installations
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python -c "import isaaclab; print('Isaac Lab installed')"
python -c "import gymnasium_robotics; print('Gymnasium Robotics installed')"

echo "Setup complete! Environment 'rl_env' is ready."