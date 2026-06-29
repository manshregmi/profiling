#!/bin/bash
# setup_complete.sh - Complete environment setup for YOLOv13 on Jetson Nano
# Python 3.6, GPU acceleration, aarch64 wheels only.

set -e  # Exit on any error

echo "==== Starting Jetson Nano YOLOv13 environment setup (Python 3.6, GPU) ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Clean up and set pip cache to external media
# ------------------------------------------------------------------
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Free root space
sudo apt autoremove -y
sudo apt clean

# Fix broken packages
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# ------------------------------------------------------------------
# 2. Install minimal system dependencies (no multimedia conflicts)
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
# 3. Remove any existing broken environment and create a fresh one
# ------------------------------------------------------------------
if [ -d "yolov13_env" ]; then
    echo "Removing old environment..."
    rm -rf yolov13_env
fi

python3 -m venv yolov13_env
source yolov13_env/bin/activate

# ------------------------------------------------------------------
# 4. Install PyTorch 1.9.0 (official NVIDIA aarch64 wheel for JetPack 4.6)
# ------------------------------------------------------------------
pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# ------------------------------------------------------------------
# 5. Install other core packages (all have aarch64 wheels for Python 3.6)
# ------------------------------------------------------------------
pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install --no-cache-dir opencv-python-headless==4.5.5.64
pip install --no-cache-dir Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 6. Patch and install YOLOv13
# ------------------------------------------------------------------
if [ -d "yolov13" ]; then
    echo "yolov13 directory exists. Pulling latest..."
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "Patching requirements.txt..."

# Remove strict version pins for torch, timm, flash_attn
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install all remaining dependencies (no torch, no timm pin)
pip install --no-cache-dir -r requirements.txt

# Install timm 0.6.12 (the last version compatible with Python 3.6)
pip install --no-cache-dir timm==0.6.12

# Install YOLOv13 package itself (do not reinstall torch)
pip install --no-cache-dir --no-deps -e .

# Download YOLOv13 Nano weights if missing
if [ ! -f "yolov13n.pt" ]; then
    echo "Downloading YOLOv13-Nano weights..."
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
else
    echo "yolov13n.pt already exists."
fi

cd ..

# ------------------------------------------------------------------
# 7. Verify the installation
# ------------------------------------------------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import Jetson.GPIO, monsoon; print('Jetson.GPIO and Monsoon: OK')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"

# ------------------------------------------------------------------
# 8. Final instructions
# ------------------------------------------------------------------
echo ""
echo "==== Setup completed successfully! ===="
echo ""
echo "All files are stored in: $BASE_DIR"
echo "Virtual environment: yolov13_env"
echo ""
echo "To activate the environment, run:"
echo "  source yolov13_env/bin/activate"
echo ""
echo "To test YOLOv13 detection, run:"
echo "  cd yolov13"
echo "  python -c \"from ultralytics import YOLO; model = YOLO('yolov13n.pt'); results = model('path/to/your/image.jpg'); print('Detection done.')\""
echo ""
echo "To run your full pipeline with latency profiling, use:"
echo "  python profile_yolo.py --image test.jpg"
echo ""
echo "Enjoy!"