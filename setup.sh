#!/bin/bash
# setup_environment.sh - Complete environment setup for YOLOv13 pipeline on Jetson Nano
# Includes all dependencies for the pipeline, power monitoring, and CSV logging.

set -e  # Exit on error

# ----------------------------
# 1. System packages
# ----------------------------
echo "==== Updating system packages ===="
sudo apt update
sudo apt upgrade -y

echo "==== Installing essential build tools, libraries, and USB support ===="
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
    libusb-1.0-0-dev \
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

# Install core data science and vision packages
echo "==== Installing vision, data, and GPIO packages ===="
pip install opencv-python pillow numpy pandas tqdm

# Install Jetson GPIO (for hardware triggering)
pip install Jetson.GPIO

# Install Monsoon power monitor Python library (requires libusb)
echo "==== Installing Monsoon power monitor library ===="
pip install monsoon

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
# 7. Verify installation
# ----------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import Jetson.GPIO as GPIO; print('Jetson.GPIO: OK')"
python -c "import monsoon; print('Monsoon: OK')"
python -c "from ultralytics import YOLO; print('Ultralytics/YOLOv13: OK')"

# ----------------------------
# 8. Final instructions
# ----------------------------
echo ""
echo "==== Setup complete! ===="
echo ""
echo "✅ All dependencies installed:"
echo "   - PyTorch + torchvision (JetPack-specific)"
echo "   - OpenCV, Pillow, NumPy, Pandas"
echo "   - Jetson.GPIO (hardware triggers)"
echo "   - Monsoon (power monitor API)"
echo "   - YOLOv13 (with Nano weights downloaded)"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To profile your pipeline and generate CSV logs, run:"
echo "  python pipeline_profiler.py --image test.jpg"
echo ""
echo "The script will create:"
echo "  - latency_log.csv  (timestamps and durations for each stage)"
echo "  - monsoon_data.csv (power samples, if Monsoon is connected)"
echo ""
echo "Enjoy!"