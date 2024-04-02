FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

#  I am suprised this is needed
RUN conda init

# Set the working directory to /app
WORKDIR /app

COPY src/requirements.txt .

# Install update and install git
# RUN apt-get update && apt-get install -y git

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    swig \
    && rm -rf /var/lib/apt/lists/*
    
# Install swig
# RUN pip install swig

# Ignore uninstall of blinker
RUN pip install --ignore-installed blinker

# Install any additional dependencies (if needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app from your host to the container
COPY src/ .