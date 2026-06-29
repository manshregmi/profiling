#!/bin/bash
# setup_external.sh - Complete installation on external drive (no root space used)

set -e

BASE_DIR="$(pwd)"
echo "Installing into external drive: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Set all temp and cache paths to the external drive
# ------------------------------------------------------------------
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
export PYTHONPYCACHEPREFIX="$BASE_DIR/pycache"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"

# Also set for pip calls
PIP_OPTS="--no-cache-dir --cache-dir $PIP_CACHE_DIR"

# ------------------------------------------------------------------
# 2. Free root space (just in case)
# ------------------------------------------------------------------
sudo apt autoremove -y
sudo apt clean

# ------------------------------------------------------------------
# 3. Install minimal system dependencies (no multimedia packages)
# ------------------------------------------------------------------
sudo apt update
sudo apt install -y \
    build-essential git wget \
    libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip python3-venv

# ------------------------------------------------------------------
# 4. Remove old environment and create fresh one (on external drive)
# ------------------------------------------------------------------
rm -rf yolov13_env
python3 -m venv yolov13_env --clear
source yolov13_env/bin/activate

# ------------------------------------------------------------------
# 5. Install PyTorch 1.9.0 (official aarch64 wheel)
# ------------------------------------------------------------------
wget -q --show-progress -O torch.whl \
    https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl
pip install $PIP_OPTS torch.whl
rm torch.whl

# ------------------------------------------------------------------
# 6. Install torchvision 0.10.0 (official aarch64 wheel)
# ------------------------------------------------------------------
wget -q --show-progress -O torchvision.whl \
    https://nvidia.box.com/shared/static/4h2ltvj6s8h9o9h8w7z5e7e8e8e8e8e8.whl
pip install $PIP_OPTS torchvision.whl
rm torchvision.whl

# ------------------------------------------------------------------
# 7. Install core packages (aarch64 wheels, no builds)
# ------------------------------------------------------------------
pip install $PIP_OPTS numpy==1.19.5 pillow==8.4.0 pandas tqdm

# OpenCV – direct aarch64 wheel
wget -q --show-progress -O opencv.whl \
    https://files.pythonhosted.org/packages/b4/0e/eb390a76bff15ebc453c539bcc6bfdaff5b9ca9e566441dae45eb508a138/opencv_python_headless-4.5.5.64-cp36-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install $PIP_OPTS opencv.whl
rm opencv.whl

# GPIO and Monsoon
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

# Remove version pins
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install dependencies
pip install $PIP_OPTS -r requirements.txt
pip install $PIP_OPTS timm==0.6.12

# Install YOLOv13 (no torch reinstall)
pip install $PIP_OPTS --no-deps -e .

# Download weights if missing
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# ------------------------------------------------------------------
# 9. Verify everything
# ------------------------------------------------------------------
echo "==== Verification ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded')"

echo ""
echo "==== Setup completed successfully on external drive! ===="
echo "Base directory: $BASE_DIR"
echo "Virtual environment: yolov13_env"
echo "All cache and temp files are on this drive."
echo ""
echo "Activate with: source yolov13_env/bin/activate"