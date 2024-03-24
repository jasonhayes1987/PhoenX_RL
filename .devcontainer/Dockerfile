# Use the TensorFlow image as the base
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory to /app
WORKDIR /app

# Copy requiremtents.txt to the container
COPY requirements.txt /app/

# Install update and install git
RUN apt-get update && apt-get install -y git
    
# Install swig
RUN pip install swig

# Ignore uninstall of blinker
RUN pip install --ignore-installed blinker

# Install any additional dependencies (if needed)
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git

# Copy your app from your host to the container
COPY src/app /app



# Command to run your application (customize as needed)
# CMD ["python", "your_app.py"]
