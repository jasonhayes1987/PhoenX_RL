# Use the CUDA 12.6.2 devel base image
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    build-essential \
    htop \
    git \
    swig \
    cmake \
    gcc \
    curl \
    nodejs \
    npm \
    libglfw3-dev \
    libglew-dev \
    libopengl0 \
    python3-opengl \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdb \
    mesa-utils \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    # nsight-systems-2024.5.1 \
    # nsight-compute-2024.3.2 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Nsight Systems and Compute via apt (compatible with CUDA 12.6 on Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    gpg --dearmor -o /usr/share/keyrings/nvidia-devtools-archive-keyring.gpg 7fa2af80.pub && \
    rm 7fa2af80.pub && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-archive-keyring.gpg] http://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && apt-get install -y nsight-systems-cli nsight-compute && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init

# Set PATH to include conda
ENV PATH=/opt/conda/bin:$PATH

# Copy over environment.yml for RL API and app
COPY src/environment.yml .

# Accept Anaconda Terms of Service for the channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment from environment.yml
RUN conda env create -f environment.yml

# Set the default environment (replace 'rl_env' with the name in your environment.yml)
ENV CONDA_DEFAULT_ENV=rl_env

# Use the conda environment in subsequent commands
SHELL ["conda", "run", "-n", "rl_env", "/bin/bash", "-c"]

# Install PyTorch with CUDA support (ensure it's CUDA 12.6 compatible; adjust if needed)
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Gymnasium Robotics directly from Repo for Mujoco v4 support
RUN pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0

# Install additional Python packages for profiling (e.g., TensorBoard for Torch Profiler visualization)
RUN pip install tensorboard torch-tb-profiler

# # Set environment variables for MuJoCo
ENV MUJOCO_GL="egl"
ENV PYOPENGL_PLATFORM="egl"
# ENV MUJOCO_GL="osmesa"
# ENV PYOPENGL_PLATFORM="osmesa"
# ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco-3.1.4/lib:${LD_LIBRARY_PATH}"
# ENV MUJOCO_PATH="/root/.mujoco/mujoco-3.1.4"

# Add app directory to python path
ENV PYTHONPATH="/workspaces/PhoenX_RL"

# Run accept license command for AutoROM do download all roms needed for Atari ale
RUN AutoROM --accept-license --install-dir /opt/conda/envs/rl_env/lib/python3.12/site-packages/ale_py/roms

# Set working directory
WORKDIR /app

# Copy your app from your host to the container
COPY src/ /app/

# Verify CUDA installation with nvcc
RUN nvcc --version

# Verify CUDA availability in PyTorch
# RUN python -c "import torch; print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of CUDA devices: {torch.cuda.device_count()}')"

# (Optional) Set the entry point to your application
# CMD ["conda", "run", "-n", "rl_env", "python", "your_script.py"]
