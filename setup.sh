#!/bin/bash
# setup.sh - Install PyTorch and all dependencies with direct URLs

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
# PyTorch 1.9.0 – try direct pip install from URL, then wget
# ------------------------------------------------------------------
TORCH_URL="https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl"
echo "Installing PyTorch 1.9.0 from $TORCH_URL..."
if pip install $PIP_OPTS "$TORCH_URL"; then
    echo "PyTorch installed via pip."
else
    echo "pip install failed, trying wget..."
    if wget --no-check-certificate -q --show-progress -O torch.whl "$TORCH_URL"; then
        pip install $PIP_OPTS torch.whl
        rm torch.whl
    else
        echo "ERROR: Could not download torch wheel."
        echo "Please manually download $TORCH_URL and place it in this folder, then run:"
        echo "  pip install torch.whl"
        exit 1
    fi
fi

# ------------------------------------------------------------------
# torchvision 0.10.0
# ------------------------------------------------------------------
TV_URL="https://nvidia.box.com/shared/static/4h2ltvj6s8h9o9h8w7z5e7e8e8e8e8e8.whl"
echo "Installing torchvision 0.10.0 from $TV_URL..."
if pip install $PIP_OPTS "$TV_URL"; then
    echo "torchvision installed via pip."
else
    echo "pip install failed, trying wget..."
    if wget --no-check-certificate -q --show-progress -O torchvision.whl "$TV_URL"; then
        pip install $PIP_OPTS torchvision.whl
        rm torchvision.whl
    else
        echo "ERROR: Could not download torchvision wheel."
        echo "Please manually download $TV_URL and place it in this folder, then run:"
        echo "  pip install torchvision.whl"
        exit 1
    fi
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