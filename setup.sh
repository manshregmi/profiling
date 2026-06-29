#!/bin/bash
# setup.sh - Complete pipeline setup for Jetson Nano (Python 3.6, GPU)
# Works around the timm version pin by explicitly installing 0.6.12.

set -e

echo "==== Starting Jetson Nano pipeline setup (Python 3.6) ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# Clean root cache, set pip to external media
sudo apt autoremove -y
sudo apt clean
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Fix broken packages
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# Minimal system libraries
sudo apt install -y \
    build-essential git wget \
    libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip python3-venv

# Virtual environment
VENV_NAME="yolov13_env"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# PyTorch 1.9.0 (official for Python 3.6, CUDA 10.2)
pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# Core packages (all have aarch64 wheels for Python 3.6)
pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install --no-cache-dir opencv-python-headless==4.5.5.64
pip install --no-cache-dir Jetson.GPIO monsoon

# YOLOv13
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "Patching requirements.txt..."

# 1. Fix torch pin to allow any >=1.9.0
sed -i 's/torch==2.2.2/torch>=1.9.0/' requirements.txt

# 2. Remove flash_attn (ARM64 incompatible)
sed -i '/flash_attn/d' requirements.txt

# 3. Remove timm completely (we'll install a compatible version separately)
sed -i '/timm/d' requirements.txt

# Create patched requirements without torch
grep -v "^torch" requirements.txt > requirements_patched.txt

# Install all other dependencies (except torch and timm)
pip install --no-cache-dir -r requirements_patched.txt

# Install timm 0.6.12 explicitly (last version compatible with Python 3.6)
pip install --no-cache-dir timm==0.6.12

# Install YOLOv13 package (no torch reinstall)
pip install --no-cache-dir --no-deps -e .

# Download YOLOv13 Nano weights
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# Verify
echo "==== Verification ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"

echo ""
echo "==== Setup complete! ===="
echo "Activate with: source $VENV_NAME/bin/activate"