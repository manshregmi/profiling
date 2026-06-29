#!/bin/bash
# setup.sh - Uses Qengineering Google Drive wheels

set -e
BASE_DIR="$(pwd)"
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
PIP_OPTS="--no-cache-dir --cache-dir $PIP_CACHE_DIR"

sudo apt update
sudo apt install -y build-essential git wget libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev python3-dev python3-pip python3-venv \
    libopenblas-base libopenmpi-dev libomp-dev

rm -rf yolov13_env
python3 -m venv yolov13_env --clear
source yolov13_env/bin/activate

# Install PyTorch 1.9.0 from Qengineering Google Drive
echo "Downloading PyTorch 1.9.0 from Qengineering..."
wget -O torch.whl "https://drive.google.com/uc?export=download&id=1e9FDGt2zGS5C5Pms7wzHYRb0HuupngK1"
pip install $PIP_OPTS torch.whl
rm torch.whl

# Install torchvision 0.10.0 from Qengineering Google Drive
echo "Downloading torchvision 0.10.0 from Qengineering..."
wget -O torchvision.whl "https://drive.google.com/uc?export=download&id=19UbYsKHhKnyeJ12VPUwcSvoxJaX7jQZ2"
pip install $PIP_OPTS torchvision.whl
rm torchvision.whl

# Install other packages
pip install $PIP_OPTS numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install $PIP_OPTS opencv-python-headless==4.5.5.64
pip install $PIP_OPTS Jetson.GPIO monsoon

# YOLOv13
if [ ! -d "yolov13" ]; then
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

# Verify
python -c "import torch, cv2, ultralytics; print('All good')"
echo "Setup complete. Activate with: source yolov13_env/bin/activate"