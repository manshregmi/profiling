#!/bin/bash
# setup.sh - One-click environment setup for YOLO v13 + Monsoon profiling
# Usage: chmod +x setup.sh && ./setup.sh

set -e  # Exit immediately if any command fails

echo "=============================================="
echo "  YOLO v13 + Monsoon Profiling Setup Script  "
echo "=============================================="

# --- 1. System-level dependencies (Linux only) ---
echo "[1/5] Installing system dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    udev

# --- 2. Upgrade pip ---
echo "[2/5] Upgrading pip..."
python3 -m pip install --upgrade pip

# --- 3. Install PyTorch (architecture-aware) ---
ARCH=$(uname -m)
echo "[3/5] Detected architecture: $ARCH"

if [[ "$ARCH" == "aarch64" ]]; then
    echo "    -> ARM64 (NVIDIA Jetson/Orin) detected."
    echo "    -> Installing PyTorch for Jetson (JetPack 5.x / 6.x)..."
    # NVIDIA's official Jetson PyTorch wheel index
    python3 -m pip install --index-url https://download.pytorch.org/whl/jetson torch torchvision
else
    echo "    -> x86_64 (standard PC) detected."
    echo "    -> Installing PyTorch from PyPI..."
    python3 -m pip install torch torchvision
fi

# --- 4. Install remaining Python dependencies from requirements.txt ---
echo "[4/5] Installing remaining Python packages..."
python3 -m pip install -r requirements.txt

# --- 5. Set up Monsoon USB permissions ---
echo "[5/5] Setting up Monsoon USB permissions (requires sudo)..."
# Monsoon Solutions Inc. USB Vendor ID is typically 0b1e
# This udev rule allows non-root users to access the device
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0b1e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-monsoon.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add current user to plugdev group (so you don't need sudo to run the script later)
sudo usermod -a -G plugdev $USER

echo "=============================================="
echo "  ✅ Setup completed successfully!"
echo "=============================================="
echo ""
echo "IMPORTANT: For USB permissions to take effect, either:"
echo "  1. Log out and log back in, OR"
echo "  2. Run: sudo reboot"
echo ""
echo "After reboot, you can run the profiler with:"
echo "  python3 profile_yolo_v13.py"
echo "=============================================="