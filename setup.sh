#!/bin/bash
# setup.sh - Install everything on external media with space management

set -e

echo "==== Starting Jetson Nano environment setup (Python 3.6) ===="
echo "Current directory: $(pwd)"

# ------------------------------------------------------------------
# 1. Free some space on root
# ------------------------------------------------------------------
sudo apt autoremove -y
sudo apt clean
rm -rf ~/.cache/pip

# ------------------------------------------------------------------
# 2. Set pip to use the external media for cache and temporary files
# ------------------------------------------------------------------
export PIP_CACHE_DIR="$(pwd)/pip_cache"
export TMPDIR="$(pwd)/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "Using PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "Using TMPDIR=$TMPDIR"

# ------------------------------------------------------------------
# 3. Fix broken packages (no new deps)
# ------------------------------------------------------------------
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# ------------------------------------------------------------------
# 4. Install minimal system libraries
# ------------------------------------------------------------------
sudo apt install -y \
    build-essential \
    git \
    wget \
    libopenblas-dev \
    libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev \
    python3-pip \
    python3-venv

# ------------------------------------------------------------------
# 5. Create virtual environment in the current (external) directory
# ------------------------------------------------------------------
VENV_NAME="yolov13_env"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 6. Install Python packages (compatible with 3.6, using external cache)
# ------------------------------------------------------------------
pip install --upgrade pip --no-cache-dir

# PyTorch 1.9.0 for JetPack 4.6 (CUDA 10.2) – has aarch64 wheel
pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# Core packages – use known good versions with aarch64 wheels
pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm

# OpenCV – use headless + known version that has aarch64 wheel
pip install --no-cache-dir opencv-python-headless==4.5.5.64

# GPIO and Monsoon
pip install --no-cache-dir Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 7. YOLOv13 – use a version of Ultralytics that supports Python 3.6
# ------------------------------------------------------------------
if [ -d "yolov13" ]; then
    echo "yolov13 already exists, pulling latest..."
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

# Patch requirements: remove flash-attn, pin ultralytics to 8.0.20
cat requirements.txt | grep -v flash_attn > requirements_patched.txt
echo "ultralytics==8.0.20" >> requirements_patched.txt
pip install --no-cache-dir -r requirements_patched.txt

# Install YOLOv13 package
pip install --no-cache-dir -e .

# Download weights if not present
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# ------------------------------------------------------------------
# 8. Verify installation
# ------------------------------------------------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"
python -c "import Jetson.GPIO, monsoon; print('All modules OK')"

echo ""
echo "==== Setup complete! ===="
echo "Activate the environment with: source $VENV_NAME/bin/activate"
echo "All files are stored in: $(pwd)"