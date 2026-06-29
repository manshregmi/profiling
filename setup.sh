#!/bin/bash
# setup.sh - Robust installation with fallback

set -e
BASE_DIR="$(pwd)"
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
PIP_OPTS="--no-cache-dir --cache-dir $PIP_CACHE_DIR"

sudo apt update
sudo apt install -y build-essential git wget libopenblas-dev libatlas-base-dev libusb-1.0-0-dev python3-dev python3-pip python3-venv

rm -rf yolov13_env
python3 -m venv yolov13_env --clear
source yolov13_env/bin/activate

# ------------------------------------------------------------------
# PyTorch 1.9.0 – use wget, fallback to pip -f
# ------------------------------------------------------------------
echo "Installing PyTorch 1.9.0..."
if wget -q --show-progress -O torch.whl https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl; then
    pip install $PIP_OPTS torch.whl
    rm torch.whl
else
    echo "wget failed, trying pip with -f..."
    pip install $PIP_OPTS torch==1.9.0 -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl
fi

# ------------------------------------------------------------------
# torchvision 0.10.0 – use wget, fallback to pip -f
# ------------------------------------------------------------------
echo "Installing torchvision 0.10.0..."
if wget -q --show-progress -O torchvision.whl https://nvidia.box.com/shared/static/4h2ltvj6s8h9o9h8w7z5e7e8e8e8e8e8.whl; then
    pip install $PIP_OPTS torchvision.whl
    rm torchvision.whl
else
    echo "wget failed, trying pip with -f..."
    pip install $PIP_OPTS torchvision==0.10.0 -f https://nvidia.box.com/shared/static/4h2ltvj6s8h9o9h8w7z5e7e8e8e8e8e8.whl
fi

# ------------------------------------------------------------------
# Other packages (all aarch64 wheels)
# ------------------------------------------------------------------
pip install $PIP_OPTS numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install $PIP_OPTS opencv-python-headless==4.5.5.64
pip install $PIP_OPTS Jetson.GPIO monsoon

# ------------------------------------------------------------------
# YOLOv13
# ------------------------------------------------------------------
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

# Verification
python -c "import torch, cv2, ultralytics; print('All good')"
echo "Setup complete. Activate with: source yolov13_env/bin/activate"