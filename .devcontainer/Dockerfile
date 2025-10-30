# Base on official Isaac Lab image (headless Isaac Sim + Isaac Lab pre-installed)
FROM nvcr.io/nvidia/isaac-lab:2.2.0
# If 2.2.1 is available to you, prefer it:
# FROM nvcr.io/nvidia/isaac-lab:2.2.1

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y libosmesa6-dev && rm -rf /var/lib/apt/lists/*

# Install Vulkan runtime and tools
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

# Create NVIDIA Vulkan ICD JSON (points to libGLX_nvidia.so.0 after mounting/symlink)
# RUN mkdir -p /etc/vulkan/icd.d && \
#     printf '{ "file_format_version": "1.0.0", "ICD": { "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0", "api_version": "1.3.204" } }' > /etc/vulkan/icd.d/nvidia_icd.json

# Set env vars for Vulkan loader and Mesa (force NVIDIA adapter)
# ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# ENV MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# Optional: Suppress XDG/DISPLAY warnings (create runtime dir)
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR

WORKDIR /workspaces/PhoenX_RL

# Use Isaac Sim Python
RUN /isaac-sim/python.sh -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r requirements.txt

# Explicitly install AutoROM in Isaac Sim Python (ensures it's available)
# RUN /isaac-sim/python.sh -m pip install --no-cache-dir AutoROM

# RUN /isaac-sim/python.sh AutoROM --accept-license

# Tell ale-py where the ROMs are (root is the user in this image)
ENV ALE_PY_ROM_PATH=/root/.local/share/AutoROM/

# (Optional) Gymnasium Robotics
RUN /isaac-sim/python.sh -m pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.0



# Mujoco headless (if you use mujoco outside Isaac Sim)
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Keep your project on sys.path
ENV PYTHONPATH=/workspaces/PhoenX_RL