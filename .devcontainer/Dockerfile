# Use the CUDA 12.4.1 devel base image
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
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init

# Set PATH to include conda
ENV PATH=/opt/conda/bin:$PATH

# Copy over environment.yml for RL API and app
COPY src/environment.yml .

# Create the conda environment from environment.yml
RUN conda env create -f environment.yml

# Set the default environment (replace 'rl_env' with the name in your environment.yml)
ENV CONDA_DEFAULT_ENV=rl_env

# Use the conda environment in subsequent commands
SHELL ["conda", "run", "-n", "rl_env", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install Gymnasium Robotics directly from Repo for Mujoco v4 support
RUN pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git

# # Install MuJoCo
# RUN mkdir -p /root/.mujoco && \
#     wget https://github.com/google-deepmind/mujoco/releases/download/3.1.4/mujoco-3.1.4-linux-x86_64.tar.gz -O mujoco.tar.gz && \
#     tar -xf mujoco.tar.gz -C /root/.mujoco && \
#     rm mujoco.tar.gz && \
#     ln -sf /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1.4 /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1 && \
#     ln -sf /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1 /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so

# # Set environment variables for MuJoCo
# ENV MUJOCO_GL="egl"
# ENV PYOPENGL_PLATFORM="egl"
# ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco-3.1.4/lib:${LD_LIBRARY_PATH}"
# ENV MUJOCO_PATH="/root/.mujoco/mujoco-3.1.4"

# Add app directory to python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces/RL_Agents/src/app"

# Run accept license command for AutoROM do download all roms needed for Atari ale
RUN AutoROM --accept-license --install-dir /opt/conda/envs/rl_env/lib/python3.10/site-packages/ale_py/roms

# Set working directory
WORKDIR /app

# Copy your app from your host to the container
COPY src/ /app/

# Verify CUDA installation with nvcc
# RUN nvcc --version

# Verify CUDA availability in PyTorch
# RUN python -c "import torch; print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of CUDA devices: {torch.cuda.device_count()}')"

# (Optional) Set the entry point to your application
# CMD ["conda", "run", "-n", "rl_env", "python", "your_script.py"]
