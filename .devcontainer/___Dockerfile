# Use the CUDA 12.5 devel base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

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
    python3-opengl \
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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init

# Set PATH to include conda
ENV PATH /opt/conda/bin:$PATH

# Create a conda environment and install specific versions of PyTorch in a single RUN command
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n myenv python=3.10 -y && \
    conda activate myenv && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia && \
    conda install -y -c conda-forge libstdcxx-ng"

# Ignore uninstall of blinker
RUN pip install --ignore-installed blinker

# Get and install Mujoco
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/google-deepmind/mujoco/releases/download/3.1.4/mujoco-3.1.4-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xf mujoco.tar.gz -C /root/.mujoco && \
    rm mujoco.tar.gz && \
    echo "export LD_LIBRARY_PATH=/root/.mujoco/mujoco-3.1.4/lib:$LD_LIBRARY_PATH" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/root/.mujoco/mujoco-3.1.4/lib:$LD_LIBRARY_PATH" >> /root/.profile && \
    ln -sf /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1.4 /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1 && \
    ln -sf /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so.3.1 /root/.mujoco/mujoco-3.1.4/lib/libmujoco.so

# Set environment variables for Mujoco
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-3.1.4/lib:${LD_LIBRARY_PATH}
ENV MUJOCO_PATH=/root/.mujoco/mujoco-3.1.4
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=/usr/local/cuda/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy over requirements for RL API and app
COPY src/requirements.txt .

# Activate the conda environment and install packages in requirements
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate myenv && \
    pip install --no-cache-dir -r requirements.txt"

# Install mpi4py compatible with Python 3.10
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate myenv && \
    conda install -y mpi4py=3.1.4"

# Set working directory
WORKDIR /app

# Copy your app from your host to the container
COPY src/ /

# Verify CUDA installation with nvcc
RUN nvcc --version

# Verify CUDA availability in PyTorch
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && python -c \"import torch; print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of CUDA devices: {torch.cuda.device_count()}')\""

# Set the entry point to your application (modify as needed)
# CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && python your_script.py"]
