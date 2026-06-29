#!/bin/bash
# setup.sh - Complete environment on external drive, no root space used

set -e

BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Free root space and fix apt
# ------------------------------------------------------------------
sudo apt clean
sudo apt autoclean
sudo rm -rf /var/lib/apt/lists/*
sudo apt update --fix-missing

# Set apt cache to external drive (if possible)
# But easier: just free up enough space by cleaning and removing old kernels if needed
# We'll also set TMPDIR and pip cache to external
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
mkdir -p "$PIP_CACHE_DIR"
PIP_OPTS="--no-cache-dir --cache-dir $PIP_CACHE_DIR"

# ------------------------------------------------------------------
# 2. Install minimal system packages (no multimedia, no libomp)
# ------------------------------------------------------------------
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libopenblas-dev \
    libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev \
    python3-pip \
    python3-venv

# ------------------------------------------------------------------
# 3. Install gdown for Google Drive downloads
# ------------------------------------------------------------------
pip3 install --user gdown
export PATH="$HOME/.local/bin:$PATH"

# ------------------------------------------------------------------
# 4. Recreate virtual environment
# ------------------------------------------------------------------
rm -rf yolov13_env
python3 -m venv yolov13_env --clear
source yolov13_env/bin/activate

# ------------------------------------------------------------------
# 5. Download and install PyTorch 1.9.0 from Qengineering (Google Drive)
# ------------------------------------------------------------------
echo "Downloading PyTorch 1.9.0 from Qengineering (Google Drive)..."
gdown --id 1e9FDGt2zGS5C5Pms7wzHYRb0HuupngK1 -O torch.whl
pip install $PIP_OPTS torch.whl
rm torch.whl

# ------------------------------------------------------------------
# 6. Download and install torchvision 0.10.0
# ------------------------------------------------------------------
echo "Downloading torchvision 0.10.0 from Qengineering (Google Drive)..."
gdown --id 19UbYsKHhKnyeJ12VPUwcSvoxJaX7jQZ2 -O torchvision.whl
pip install $PIP_OPTS torchvision.whl
rm torchvision.whl

# ------------------------------------------------------------------
# 7. Install other core packages (aarch64 wheels)
# ------------------------------------------------------------------
pip install $PIP_OPTS numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install $PIP_OPTS opencv-python-headless==4.5.5.64
pip install $PIP_OPTS Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 8. YOLOv13
# ------------------------------------------------------------------
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi
cd yolov13
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt
pip install $PIP_OPTS -r requirements.txt
pip install $PIP_OPTS timm==0.6.12
pip install $PIP_OPTS --no-deps -e .
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi
cd ..

# ------------------------------------------------------------------
# 9. Verify
# ------------------------------------------------------------------
python -c "import torch, cv2, ultralytics; print('All good')"
echo "Setup complete. Activate with: source yolov13_env/bin/activate"