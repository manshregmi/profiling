#!/bin/bash
# setup.sh - One‑shot environment setup for Jetson Nano (Ubuntu 18.04)
# This script will:
#   - Fix broken packages
#   - Install Python 3.9 and minimal build tools
#   - Create a virtual environment (Python 3.9)
#   - Install all Python packages (PyTorch, OpenCV, YOLOv13, etc.)
#   - Download YOLOv13 model weights

set -e  # Stop on error

echo "==== Starting Jetson Nano environment setup ===="

# ------------------------------------------------------------------
# 1. Fix broken packages and remove problematic system packages
# ------------------------------------------------------------------
echo "==== Fixing system packages ===="
sudo apt --fix-broken install -y
sudo dpkg --configure -a
sudo apt install -f -y

# Remove curl, libjpeg-dev, libtiff-dev – they cause conflicts and are not needed
echo "Removing non‑essential packages that may cause conflicts..."
sudo apt remove --purge -y curl libjpeg-dev libtiff-dev || true

# ------------------------------------------------------------------
# 2. Install minimal system dependencies
# ------------------------------------------------------------------
echo "==== Installing essential system packages ===="
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-dev \
    python3-pip \
    python3-venv \
    software-properties-common \
    libopenblas-dev \
    libatlas-base-dev \
    libusb-1.0-0-dev

# ------------------------------------------------------------------
# 3. Install Python 3.9 (if not already present)
# ------------------------------------------------------------------
echo "==== Installing Python 3.9 ===="
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 not found. Adding deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev
else
    echo "Python 3.9 already installed."
fi

# ------------------------------------------------------------------
# 4. Create virtual environment
# ------------------------------------------------------------------
VENV_NAME="yolov13_env"
echo "==== Creating virtual environment: $VENV_NAME (Python 3.9) ===="
python3.9 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 5. Install Python packages inside the virtual environment
# ------------------------------------------------------------------
echo "==== Installing Python packages ===="
pip install --upgrade pip

# Detect L4T version to choose correct PyTorch wheel
L4T_VERSION=$(head -n1 /etc/nv_tegra_release | grep -o "REVISION: [0-9.]*" | cut -d' ' -f2)
echo "Detected L4T version: $L4T_VERSION"

if [[ "$L4T_VERSION" == "32."* ]]; then
    # JetPack 4.x (CUDA 10.2)
    echo "Installing PyTorch for JetPack 4.x (CUDA 10.2)"
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
elif [[ "$L4T_VERSION" == "35."* ]]; then
    # JetPack 5.0+ (CUDA 11.4)
    echo "Installing PyTorch for JetPack 5.x (CUDA 11.4)"
    pip install torch==1.12.0 torchvision==0.13.0 -f https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
else
    echo "Unknown L4T version. Defaulting to JetPack 4.x (CUDA 10.2) PyTorch."
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
fi

# Other core packages
pip install opencv-python pillow numpy pandas tqdm

# GPIO and Monsoon
pip install Jetson.GPIO
pip install monsoon

# ------------------------------------------------------------------
# 6. Clone and install YOLOv13
# ------------------------------------------------------------------
echo "==== Setting up YOLOv13 ===="
if [ -d "yolov13" ]; then
    echo "yolov13 directory already exists. Pulling latest..."
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

# Remove flash-attention from requirements (ARM64 incompatible)
if grep -q "flash_attn" requirements.txt; then
    echo "Removing flash-attention from requirements.txt for ARM64 compatibility."
    grep -v "flash_attn" requirements.txt > requirements_arm64.txt
    pip install -r requirements_arm64.txt
else
    pip install -r requirements.txt
fi

# Install the YOLOv13 package itself
pip install -e .

# Download YOLOv13 Nano weights (skip if already there)
echo "==== Downloading YOLOv13-Nano weights ===="
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
else
    echo "yolov13n.pt already exists."
fi

cd ..

# ------------------------------------------------------------------
# 7. Verify installation
# ------------------------------------------------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import Jetson.GPIO as GPIO; print('Jetson.GPIO: OK')"
python -c "import monsoon; print('Monsoon: OK')"
python -c "from ultralytics import YOLO; print('Ultralytics/YOLOv13: OK')"

# ------------------------------------------------------------------
# 8. Final instructions
# ------------------------------------------------------------------
echo ""
echo "==== Setup completed successfully! ===="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To test YOLOv13 detection, run:"
echo "  cd yolov13"
echo "  python -c \"from ultralytics import YOLO; model = YOLO('yolov13n.pt'); results = model('path/to/your/image.jpg'); print('Detection done.')\""
echo ""
echo "Enjoy!"