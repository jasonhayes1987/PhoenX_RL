FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && \
    apt-get install -y sudo

#  I am suprised this is needed
RUN conda init

# Install update and install git
# RUN apt-get update && apt-get install -y git

# Install system packages
RUN apt-get update && apt-get install -y \
    #build-essential \
    #ca-certificates \
    #gdb \
    #gfortran \
    htop \
    git \
    swig \
    wget \
    libglfw3 \
    libglew-dev \
    libgl1-mesa-glx \
    libosmesa6 \
    libopenmpi-dev \
    libglib2.0-0 \
    # libosmesa6-dev \
    # mesa-utils libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# install libstdcxx-ng from conda-forge
RUN conda install -y -c conda-forge libstdcxx-ng

# Update UCX
# RUN mkdir -p /tmp/UCX-build \
#     && cd /tmp/UCX-build \
#     && wget https://github.com/openucx/ucx/releases/download/v1.16.0/ucx-1.16.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 -O ucx.tar.bz2 \
#     && tar xjf ucx.tar.bz2 \
#     && dpkg -i ucx-xpmem-1.16.0.deb \
#     && rm -rf /tmp/UCX-build

# Build Open MPI with Cuda support
# ARG MPI_VERSION="5.0.3"
# ARG MPI_CONFIGURE_OPTIONS="--build=x86_64-pc-linux-gnu --enable-fast=all,O3 --prefix=/usr --with-cuda=/usr/local/cuda"
# ARG MPI_MAKE_OPTIONS="-j$(nproc)"

# RUN mkdir -p /tmp/openmpi-build \
#       && cd /tmp/openmpi-build \
#       && MPI_VER_MM="${MPI_VERSION%.*}" \
#       && wget http://www.openmpi.org/software/ompi/v${MPI_VER_MM}/downloads/openmpi-${MPI_VERSION}.tar.bz2 \
#       && tar xjf openmpi-${MPI_VERSION}.tar.bz2 \
#       && cd openmpi-${MPI_VERSION} \
#       && ./configure ${MPI_CONFIGURE_OPTIONS} \
#       && make ${MPI_MAKE_OPTIONS} \
#       && make install \
#       && ldconfig \
#       && cd / \
#       && rm -rf /tmp/openmpi-build

# Set environment variables
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Run the conda command to set the environment variables
RUN conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# RUN conda deactivate && conda activate

# Get and install Mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://github.com/google-deepmind/mujoco/releases/download/3.1.4/mujoco-3.1.4-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco314/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Ignore uninstall of blinker
RUN pip install --ignore-installed blinker

# Copy over requirements for RL API and app
COPY src/requirements.txt .

# Install packages in requirements
RUN pip install --no-cache-dir -r requirements.txt

RUN conda install -y mpi4py

# install cuda aware version of mpi
# RUN conda install -c conda-forge mpi4py openmpi

# upgrade mujoco to 3.1.4 python bindings
RUN pip uninstall mujoco-py -y
RUN pip install mujoco==3.1.4

# # Create a new user and add to sudo group
# RUN adduser --disabled-password --gecos '' vscode && \
#     adduser vscode sudo && \
#     echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set working directory
WORKDIR /app

# RUN chown -R vscode:vscode /app

# Copy your app from your host to the container
COPY src/ /
# COPY --chown=vscode:vscode src/ .

# USER vscode