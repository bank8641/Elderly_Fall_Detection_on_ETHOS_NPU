#!/bin/bash
#
# Fall Detection GUI Launcher
# Runs on NXP i.MX93 with HDMI display via Weston
#

cd /home/root/elderly_fall_detection

echo "========================================"
echo "  Fall Detection System - GUI Mode"
echo "  NXP i.MX93 + Ethos-U65 NPU"
echo "========================================"

# Check if Weston is running
if ! pgrep -x weston > /dev/null; then
    echo "Starting Weston..."
    weston --tty=1 &
    sleep 3
fi

# Set display for Wayland
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0

# Run the OpenCV GUI application
echo "Starting Fall Detection..."
python3 app/main.py

echo "Exited."
