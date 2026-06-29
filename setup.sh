#!/bin/bash
# setup_environment.sh - Full environment setup for YOLOv13 pipeline on Jetson Nano
# This script installs Python 3.9, creates a virtual environment, and installs all dependencies.

set -e  # Exit on error

# ----------------------------
# 1. System packages
# ----------------------------
echo "==== Updating system packages ===="
sudo apt update
sudo apt upgrade -y

echo "==== Installing essential build tools and libraries ===="
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    python3-dev \
    python3-pip \
    python3-venv

# ----------------------------
# 2. Install Python 3.9 (if not already available)
# ----------------------------
echo "==== Checking Python 3.9 ===="
if command -v python3.9 &> /dev/null; then
    echo "Python 3.9 is already installed."
else
    echo "Python 3.9 not found. Installing from deadsnakes PPA..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev
fi

# ----------------------------
# 3. Create virtual environment
# ----------------------------
VENV_NAME="yolov13_env"
echo "==== Creating virtual environment: $VENV_NAME (Python 3.9) ===="
python3.9 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# ----------------------------
# 4. Upgrade pip and install core packages
# ----------------------------
echo "==== Installing Python packages inside virtual environment ===="
pip install --upgrade pip

# Install PyTorch for Jetson (ARM64, CUDA 10.2 or 11.4 depending on JetPack)
# For JetPack 4.6 (CUDA 10.2) use the official NVIDIA wheel.
# For JetPack 5.0+ (CUDA 11.4) adjust accordingly.
# We'll detect the L4T version to choose the correct wheel.

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
    echo "Unknown L4T version, defaulting to JetPack 4.x PyTorch 1.10.0"
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
fi

# Install other common packages
pip install opencv-python pillow numpy pandas tqdm

# ----------------------------
# 5. Clone and install YOLOv13
# ----------------------------
echo "==== Cloning YOLOv13 repository ===="
if [ -d "yolov13" ]; then
    echo "yolov13 directory already exists, pulling latest..."
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "==== Installing YOLOv13 dependencies ===="
# Remove flash-attention from requirements if on ARM64 (Jetson)
if grep -q "flash_attn" requirements.txt; then
    echo "Detected flash-attention in requirements. Removing it for ARM64 compatibility."
    grep -v "flash_attn" requirements.txt > requirements_arm64.txt
    pip install -r requirements_arm64.txt
else
    pip install -r requirements.txt
fi

# Install the YOLOv13 package itself
pip install -e .

# ----------------------------
# 6. Download model weights (Nano variant)
# ----------------------------
echo "==== Downloading YOLOv13-Nano weights ===="
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
else
    echo "yolov13n.pt already exists."
fi

# Optionally download other variants (S, L, X) if needed
# wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt
# wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt
# wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt

cd ..

# ----------------------------
# 7. Final instructions
# ----------------------------
echo "==== Setup complete! ===="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To test YOLOv13, run:"
echo "  cd yolov13"
echo "  python -c \"from ultralytics import YOLO; model = YOLO('yolov13n.pt'); results = model('path/to/your/image.jpg'); results[0].show()\""
echo ""
echo "Enjoy!"