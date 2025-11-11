# Installation Guide

This guide provides step-by-step instructions for setting up the PhoenX RL environment with all necessary dependencies.

## Prerequisites

- Miniconda or Anaconda installed on your system
- CUDA-compatible GPU (recommended for optimal performance)
- At least 16GB of RAM
- Sufficient disk space (~50GB free)

## Installation Steps

### 1. Create Conda Environment

Navigate to the project directory and create the conda environment:

```
cd E:\Documents\Programming\Projects\Reinforcement\PhoenX_RL\src
conda env create -f environment.yml
```

### 2. Activate Environment

Activate the newly created environment:

```
conda activate rl_env
```

### 3. Upgrade pip

Ensure pip is up to date:

```
python.exe -m pip install --upgrade pip
```

### 4. Install CUDA-Enabled PyTorch

Install PyTorch with CUDA 12.8 support:

```
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install Isaac Lab and Isaac Sim

Install Isaac Lab along with Isaac Sim dependencies:

```
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```

### 6. Install Gymnasium Robotics

'''
-m pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0
'''

## Verification

After installation, you can verify the setup by running:

```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- The installation may take 30-60 minutes depending on your internet connection and system performance
- Ensure you have accepted the Isaac Sim EULA before installation
- If you encounter permission issues, consider running the commands with administrator privileges