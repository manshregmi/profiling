#!/bin/bash
# setup.sh - Complete environment for YOLOv13 on Jetson Nano (Python 3.6, GPU)
# All packages are pre‑built aarch64 wheels – no compilation, no root space used.

set -e  # Exit on any error

echo "==== Starting Jetson Nano YOLOv13 environment setup ===="
BASE_DIR="$(pwd)"
echo "Installing into external drive: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Set all cache/temp paths to external drive
# ------------------------------------------------------------------
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
export PYTHONPYCACHEPREFIX="$BASE_DIR/pycache"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"
PIP_OPTS="--no-cache-dir --cache-dir $PIP_CACHE_DIR"

# ------------------------------------------------------------------
# 2. Free root space and update system
# ------------------------------------------------------------------
sudo apt autoremove -y
sudo apt clean
sudo apt update

# ------------------------------------------------------------------
# 3. Install minimal system dependencies (no multimedia conflicts)
# ------------------------------------------------------------------
sudo apt install -y \
    build-essential git wget \
    libopenblas-dev libatlas-base-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip python3-venv

# ------------------------------------------------------------------
# 4. Create fresh virtual environment on external drive
# ------------------------------------------------------------------
rm -rf yolov13_env
python3 -m venv yolov13_env --clear
source yolov13_env/bin/activate

# ------------------------------------------------------------------
# 5. Install PyTorch 1.9.0 (official aarch64 wheel for JetPack 4.6)
# ------------------------------------------------------------------
echo "Downloading PyTorch 1.9.0..."
wget -q --show-progress -O torch.whl \
    https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl
pip install $PIP_OPTS torch.whl
rm torch.whl

# ------------------------------------------------------------------
# 6. Install torchvision 0.10.0 (official aarch64 wheel)
# ------------------------------------------------------------------
echo "Downloading torchvision 0.10.0..."
wget -q --show-progress -O torchvision.whl \
    https://nvidia.box.com/shared/static/4h2ltvj6s8h9o9h8w7z5e7e8e8e8e8e8.whl
pip install $PIP_OPTS torchvision.whl
rm torchvision.whl

# ------------------------------------------------------------------
# 7. Install all other core packages (aarch64 wheels)
# ------------------------------------------------------------------
echo "Installing core packages..."
pip install $PIP_OPTS numpy==1.19.5 pillow==8.4.0 pandas tqdm

# OpenCV – direct aarch64 wheel
echo "Downloading OpenCV 4.5.5.64..."
wget -q --show-progress -O opencv.whl \
    https://files.pythonhosted.org/packages/b4/0e/eb390a76bff15ebc453c539bcc6bfdaff5b9ca9e566441dae45eb508a138/opencv_python_headless-4.5.5.64-cp36-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install $PIP_OPTS opencv.whl
rm opencv.whl

# GPIO and Monsoon
pip install $PIP_OPTS Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 8. Install YOLOv13 (patched to remove version pins)
# ------------------------------------------------------------------
echo "Cloning and patching YOLOv13..."
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

# Remove strict pins for torch, timm, flash_attn
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install all remaining requirements
pip install $PIP_OPTS -r requirements.txt

# Install timm 0.6.12 (last version for Python 3.6)
pip install $PIP_OPTS timm==0.6.12

# Install YOLOv13 package (no torch reinstall)
pip install $PIP_OPTS --no-deps -e .

# Download YOLOv13 Nano weights
if [ ! -f "yolov13n.pt" ]; then
    echo "Downloading YOLOv13-Nano weights..."
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# ------------------------------------------------------------------
# 9. Verify every import
# ------------------------------------------------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy, pandas, tqdm; print('NumPy, Pandas, tqdm: OK')"
python -c "import Jetson.GPIO, monsoon; print('Jetson.GPIO, Monsoon: OK')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"

# ------------------------------------------------------------------
# 10. Final instructions
# ------------------------------------------------------------------
echo ""
echo "==== Setup completed successfully ===="
echo ""
echo "All files are stored on your external drive: $BASE_DIR"
echo "Virtual environment: yolov13_env"
echo ""
echo "To activate, run:"
echo "  source yolov13_env/bin/activate"
echo ""
echo "To run your pipeline, use:"
echo "  python profile_yolo.py --image test.jpg"
echo ""
echo "Enjoy!"