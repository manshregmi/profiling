#!/bin/bash

set -e

echo "========================================"
echo "  FIXING JETSON YOLOv13 ENVIRONMENT"
echo "========================================"

# ============================================================
# 1. System tools (DNS + networking fix)
# ============================================================

echo "[1/8] Installing network utilities..."

sudo apt update

sudo apt install -y \
    dnsutils \
    iputils-ping \
    curl \
    wget \
    git \
    nano


# ============================================================
# 2. Fix DNS (Jetson common issue)
# ============================================================

echo "[2/8] Fixing DNS resolution..."

sudo bash -c 'cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 1.1.1.1
EOF'

echo "DNS updated"


# ============================================================
# 3. System dependencies
# ============================================================

echo "[3/8] Installing system libraries..."

sudo apt install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    udev


# ============================================================
# 4. Upgrade pip
# ============================================================

echo "[4/8] Upgrading pip..."

python3 -m pip install --upgrade pip


# ============================================================
# 5. Remove broken PyTorch (VERY IMPORTANT)
# ============================================================

echo "[5/8] Removing broken PyTorch builds..."

pip3 uninstall -y torch torchvision torchaudio || true

pip3 cache purge || true


# ============================================================
# 6. Install CORRECT Jetson PyTorch (CUDA 12 compatible)
# ============================================================

echo "[6/8] Installing Jetson-compatible PyTorch..."

pip3 install --no-cache-dir torch torchvision


# fallback if above fails
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Fallback PyTorch install..."
    pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision
fi


# ============================================================
# 7. Install YOLOv13 repo (NOT pip ultralytics)
# ============================================================

echo "[7/8] Installing YOLOv13..."

if [ ! -d "yolov13" ]; then
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

pip3 install -e .

cd ..


# ============================================================
# 8. Python dependencies for YOLO + profiling
# ============================================================

echo "[8/8] Installing Python dependencies..."

pip3 install \
    numpy \
    opencv-python \
    pyusb \
    libusb1 \
    Monsoon \
    huggingface_hub \
    tqdm \
    pyyaml \
    requests


# ============================================================
# 9. Verification
# ============================================================

echo "========================================"
echo " VERIFYING INSTALLATION"
echo "========================================"

python3 - <<EOF
import torch

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from ultralytics import YOLO

model = YOLO("yolov13/weights/yolov13n.pt")

print("YOLOv13 loaded successfully")
EOF


echo "========================================"
echo " SETUP COMPLETE"
echo "========================================"
echo ""
echo "If CUDA is still False:"
echo "  -> reboot Jetson"
echo "  -> re-run script"
echo ""
echo "Run your profiler:"
echo "  python3 profile_yolo.py"