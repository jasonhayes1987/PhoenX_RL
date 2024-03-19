# Start from the TensorFlow image optimized for GPU
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
# COPY requirements.txt /app/

# Update pip
RUN pip install --upgrade pip

# Install swig
# RUN pip install swig

# Ignore blinker uninstall
# RUN pip install --ignore-installed blinker

# Install any dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . /app

# Command to run your application
# CMD ["python", "your_dash_app_script.py"]
