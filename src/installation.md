# Installation Guide

This guide provides step-by-step instructions for setting up the PhoenX RL environment with all necessary dependencies.

## Prerequisites

- Miniconda or Anaconda installed on your system
- CUDA-compatible GPU (recommended for optimal performance)
- At least 16GB of RAM
- Sufficient disk space (~50GB free)

## Installation Steps

### Automated Setup (Recommended)

For a quick setup, run the automated script in the project root:

```
./setup.ps1  # On Windows PowerShell (run as administrator if needed)
```

This script will:
- Create the conda environment from environment.yml
- Configure Poetry
- Install all dependencies (Poetry + manual pip for specials)
- Verify the setup

If the script fails, follow the manual steps below.

### Manual Setup Steps

#### 1. Create Conda Environment

Navigate to the project directory and create the conda environment:

```
cd {PATH TO REPO}\PhoenX_RL
conda env create -f environment.yml
```

#### 2. Activate Environment

Activate the newly created environment:

```
conda activate rl_env
```

#### 3. Configure Poetry

Set Poetry to use the conda environment:

```
poetry config virtualenvs.create false
poetry install --with dev
```

#### 4. Install Special Packages

Install PyTorch, IsaacLab, and Gymnasium Robotics:

```
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0
```

## Verification

After installation, you can verify the setup by running:

```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- The installation may take 30-60 minutes depending on your internet connection and system performance
- Ensure you have accepted the Isaac Sim EULA before installation
- If you encounter permission issues, consider running the commands with administrator privileges