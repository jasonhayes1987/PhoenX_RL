#!/bin/bash
# start_vnc.sh -- Start VNC server for Isaac Sim GUI access

# Set display :99 (port 5901), resolution, and allow non-localhost connections
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
sleep 2  # Wait for Xvfb to be ready

# Use x11vnc to SHARE the Xvfb display at :99 (instead of tightvncserver creating a new :1)
x11vnc -display :99 -geometry 1920x1080 -forever -shared -rfbport 5901 -passwd vncpass

echo "VNC server started on :99 (port 5901) – sharing Xvfb display"
# Keep the script running (background-friendly; post-create command will continue)
tail -f /dev/null