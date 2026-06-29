#!/bin/bash
# setup.sh - Minimal environment setup for YOLOv13 pipeline on Jetson Nano

set -e

# 0. Fix GPG key for deadsnakes
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 || true

# 1. Minimal system packages (no multimedia dependencies)
echo "==== Installing essential system packages ===="
sudo apt update
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
    python3-dev \
    python3-pip \
    python3-venv \
    software-properties-common \
    libusb-1.0-0-dev

# 2. Install Python 3.9 (if missing)
if ! command -v python3.9 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev
fi

# 3. Create virtual environment
VENV_NAME="yolov13_env"
python3.9 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 4. Install Python packages
pip install --upgrade pip

# PyTorch for Jetson (detect L4T version)
L4T_VERSION=$(head -n1 /etc/nv_tegra_release | grep -o "REVISION: [0-9.]*" | cut -d' ' -f2)
if [[ "$L4T_VERSION" == "32."* ]]; then
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
elif [[ "$L4T_VERSION" == "35."* ]]; then
    pip install torch==1.12.0 torchvision==0.13.0 -f https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
else
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
fi

pip install opencv-python pillow numpy pandas tqdm Jetson.GPIO monsoon

# 5. YOLOv13
if [ ! -d "yolov13" ]; then
    git clone https://github.com/iMoonLab/yolov13.git
fi
cd yolov13
# Remove flash-attn from requirements
grep -v "flash_attn" requirements.txt > requirements_arm64.txt || true
pip install -r requirements_arm64.txt
pip install -e .
# Download weights
wget -nc https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
cd ..

# 6. Verify
python -c "import torch, cv2, monsoon, Jetson.GPIO; from ultralytics import YOLO; print('All OK')"

echo "Done! Activate with: source $VENV_NAME/bin/activate"