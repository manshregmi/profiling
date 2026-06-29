#!/bin/bash
# setup_final.sh - Complete environment for YOLOv13 pipeline (Python 3.6, GPU)

set -e

echo "==== Starting Jetson Nano pipeline setup (Python 3.6) ===="
BASE_DIR="$(pwd)"

# Clean up root cache and use external media for pip caches
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Free space
sudo apt autoremove -y
sudo apt clean

# Fix broken packages
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# Minimal system libraries (no problematic multimedia packages)
sudo apt install -y \
    build-essential git wget \
    libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip python3-venv

# Create virtual environment
VENV_NAME="yolov13_env"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Install PyTorch 1.9.0 (official Jetson wheel, CUDA 10.2)
pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# Other core packages (all have aarch64 wheels for Python 3.6)
pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install --no-cache-dir opencv-python-headless==4.5.5.64
pip install --no-cache-dir Jetson.GPIO monsoon

# Clone YOLOv13
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

# Patch requirements: allow any torch >=1.9.0, remove flash_attn
sed -i 's/torch==2.2.2/torch>=1.9.0/' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Create a requirements file without torch (we already have it)
grep -v "^torch" requirements.txt > requirements_patched.txt

# Install other dependencies
pip install --no-cache-dir -r requirements_patched.txt

# Install YOLOv13 itself (no torch reinstall)
pip install --no-cache-dir --no-deps -e .

# Download Nano weights
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# Verify
echo "==== Verification ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 OK')"

echo ""
echo "==== Setup complete! ===="
echo "Activate environment with: source $VENV_NAME/bin/activate"
echo "All files are in: $BASE_DIR"