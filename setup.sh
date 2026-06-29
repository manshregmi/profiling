#!/bin/bash
# setup.sh - Complete environment for YOLOv13 pipeline (Python 3.6, GPU)
# Fixed: removes timm version pin entirely.

set -e

echo "==== Starting Jetson Nano pipeline setup (Python 3.6) ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

sudo apt autoremove -y
sudo apt clean
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "Using PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "Using TMPDIR=$TMPDIR"

sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

sudo apt install -y \
    build-essential git wget \
    libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip python3-venv

VENV_NAME="yolov13_env"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install --no-cache-dir opencv-python-headless==4.5.5.64
pip install --no-cache-dir Jetson.GPIO monsoon

if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "Patching requirements.txt..."

# Remove strict torch pin
sed -i 's/torch==2.2.2/torch>=1.9.0/' requirements.txt

# Remove flash_attn
sed -i '/flash_attn/d' requirements.txt

# Remove timm entirely (we'll install it separately if needed)
sed -i '/timm/d' requirements.txt

# Create patched requirements without torch
grep -v "^torch" requirements.txt > requirements_patched.txt

# Install dependencies (timm will be picked up later if needed)
pip install --no-cache-dir -r requirements_patched.txt

# Install YOLOv13 (no torch)
pip install --no-cache-dir --no-deps -e .

# Download weights
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

echo "==== Verification ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 OK')"

echo ""
echo "Setup complete! Activate with: source $VENV_NAME/bin/activate"