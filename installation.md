# Installation Guide

This guide provides step-by-step instructions for setting up the PhoenX RL environment with all necessary dependencies.

## Prerequisites

- Windows 10/11
- Miniconda or Anaconda installed on your system (and available on your `PATH`)
- CUDA-compatible GPU + NVIDIA drivers (recommended for best performance)
- At least 16GB of RAM
- Sufficient disk space (GPU/Isaac-related installs can be large)

## Installation Steps

### 0. Clone the repository

```
git clone <your-repo-url>
cd PhoenX_RL
```

### Automated Setup (Recommended)

For a quick setup, run the automated script in the project root:

```
.\setup.ps1  # Windows PowerShell
```

This script will:
- Create the conda environment from environment.yml
- Configure Poetry
- Install all dependencies (Poetry + manual pip for specials)
- Verify the setup

Notes:
- If `conda activate rl_env` fails inside PowerShell, you may need to run `conda init powershell` and restart your shell, or run the script from an “Anaconda Prompt”.
- The script installs CUDA-enabled PyTorch wheels by default.

If the script fails, follow the manual steps below.

### Manual Setup Steps

#### 1. Create Conda Environment

Navigate to the project directory and create the conda environment:

```
cd path\to\PhoenX_RL
conda env create -f environment.yml
```

#### 2. Activate Environment

Activate the newly created environment:

```
conda activate rl_env
```

#### 3. Configure Poetry

If Poetry is not installed yet, install it first (the automated script uses this installer):

```
curl.exe -sSL https://install.python-poetry.org | python - --version 1.8.3
```

Then configure Poetry to use the active conda environment and install dependencies:

```
poetry config virtualenvs.create false
poetry install --with dev
```

#### 4. (Optional) Lock dependencies
If you want to regenerate `poetry.lock` from `pyproject.toml`:

```
poetry lock
```

#### 5. Install Special Packages

Install PyTorch, IsaacLab, and Gymnasium Robotics (these are installed explicitly in `setup.ps1`):

```
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim]==2.3.0 --extra-index-url https://pypi.nvidia.com
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0
```

## Verification

After installation, you can verify the setup by running:

```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- The installation may take 30-60 minutes depending on your internet connection and system performance.
- If you install `isaaclab[isaacsim]`, ensure you have accepted any required NVIDIA/Isaac Sim EULA(s) for your setup.
- If you encounter permission issues, consider running PowerShell as Administrator.