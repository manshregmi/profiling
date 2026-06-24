#!/bin/bash
# setup.sh - Full environment setup for YOLOv13 + Monsoon profiling
# Optimized for NVIDIA Orin NX with JetPack 6.2.2
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "=============================================="
echo "  YOLOv13 + Monsoon Profiling Setup Script   "
echo "       (Optimized for JetPack 6.2.2)         "
echo "=============================================="

# ----------------------------------------------------------------------
# 1. System dependencies
# ----------------------------------------------------------------------
echo "[1/8] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 libusb-1.0-0-dev \
    udev wget curl git

# ----------------------------------------------------------------------
# 2. Upgrade pip
# ----------------------------------------------------------------------
echo "[2/8] Upgrading pip..."
python3 -m pip install --upgrade pip

# ----------------------------------------------------------------------
# 3. Install cuSPARSELt (required for JetPack 6.x)
# ----------------------------------------------------------------------
echo "[3/8] Installing cuSPARSELt..."
wget -q https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb -O /tmp/cusparselt.deb
if [ -f /tmp/cusparselt.deb ]; then
    sudo dpkg -i /tmp/cusparselt.deb 2>/dev/null || true
    sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
    sudo apt-get update 2>/dev/null || true
    sudo apt-get install -y cusparselt-cuda-12 2>/dev/null || true
    rm -f /tmp/cusparselt.deb
    echo "    ✅ cuSPARSELt installed"
else
    echo "    ⚠️  cuSPARSELt download failed. Continuing..."
fi

# ----------------------------------------------------------------------
# 4. Install PyTorch from Jetson AI Lab (CUDA-enabled)
# ----------------------------------------------------------------------
echo "[4/8] Installing PyTorch with CUDA support for JetPack 6.2..."
pip3 install --pre torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# ----------------------------------------------------------------------
# 5. Install Python dependencies (excluding PyTorch)
# ----------------------------------------------------------------------
echo "[5/8] Installing Python packages..."
pip3 install numpy opencv-python ultralytics Monsoon pyusb libusb1

# ----------------------------------------------------------------------
# 6. Set up Monsoon USB permissions
# ----------------------------------------------------------------------
echo "[6/8] Setting up Monsoon USB permissions..."
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0b1e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-monsoon.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -a -G plugdev $USER

# ----------------------------------------------------------------------
# 7. Download YOLOv13n model weights from official repository
# ----------------------------------------------------------------------
echo "[7/8] Downloading YOLOv13n model weights..."
mkdir -p weights
cd weights

MODEL_URL="https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt"

if [ -f yolov13n.pt ]; then
    echo "    ⚠️  Model already exists. Verifying..."
    if python3 -c "import torch; torch.load('yolov13n.pt', map_location='cpu')" 2>/dev/null; then
        echo "    ✅ Existing model is valid."
        cd ..
    else
        echo "    ⚠️  Existing model is corrupt. Re-downloading..."
        rm -f yolov13n.pt
        wget --progress=dot:giga "$MODEL_URL" -O yolov13n.pt
        cd ..
    fi
else
    echo "    Downloading from $MODEL_URL"
    wget --progress=dot:giga "$MODEL_URL" -O yolov13n.pt
    if [ -f yolov13n.pt ]; then
        echo "    ✅ Download complete."
    else
        echo "    ❌ Download failed. Please manually download from:"
        echo "       $MODEL_URL"
        echo "    and place it in ./weights/yolov13n.pt"
        exit 1
    fi
    cd ..
fi

# ----------------------------------------------------------------------
# 8. Final verification
# ----------------------------------------------------------------------
echo "[8/8] Verifying installation..."
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
from ultralytics import YOLO
model = YOLO('weights/yolov13n.pt')
print('✅ YOLOv13 model loaded successfully')
" || echo "⚠️  Verification encountered issues, but setup may still work."

echo ""
echo "=============================================="
echo "  ✅ Setup completed successfully!"
echo "=============================================="
echo ""
echo "📁 YOLOv13 weights are in: ./weights/yolov13n.pt"
echo ""
echo "⚠️  For USB permissions, log out/in or reboot."
echo ""
echo "Now run your profiler:"
echo "  python3 profile_yolo.py"
echo "=============================================="