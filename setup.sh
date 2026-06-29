#!/bin/bash
# setup.sh - Minimal environment setup for Jetson Nano (ARM64)

set -e

echo "==== Starting Jetson Nano environment setup ===="

# 1. Fix system packages and resolve conflicts
echo "==== Resolving package conflicts ===="
sudo apt update
sudo apt dist-upgrade -y
sudo apt install -f -y
sudo dpkg --configure -a

# 2. Install minimal essential build dependencies (no problematic -dev packages)
echo "==== Installing essential build dependencies ===="
sudo apt install -y \
    build-essential \
    wget \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libopenblas-dev \
    libatlas-base-dev \
    libusb-1.0-0-dev

# 3. Build Python 3.9 from source (without optimizations for speed)
PYTHON_VERSION="3.9.18"
PYTHON_SRC="/tmp/Python-${PYTHON_VERSION}"

echo "==== Building Python ${PYTHON_VERSION} from source ===="
if command -v python3.9 &> /dev/null; then
    echo "Python 3.9 already installed, skipping build."
else
    cd /tmp
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    tar -xzf "Python-${PYTHON_VERSION}.tgz"
    cd "Python-${PYTHON_VERSION}"
    ./configure --enable-shared --prefix=/usr/local
    make -j$(nproc)
    sudo make altinstall
    # Clean up
    cd /tmp
    rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"
    sudo ldconfig
    echo "Python 3.9 installed successfully."
fi

python3.9 --version

# 4. Create virtual environment
VENV_NAME="yolov13_env"
echo "==== Creating virtual environment: $VENV_NAME ===="
python3.9 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 5. Install Python packages
echo "==== Installing Python packages ===="
pip install --upgrade pip

# PyTorch for Jetson (detect L4T version)
L4T_VERSION=$(head -n1 /etc/nv_tegra_release | grep -o "REVISION: [0-9.]*" | cut -d' ' -f2)
echo "Detected L4T version: $L4T_VERSION"

if [[ "$L4T_VERSION" == "32."* ]]; then
    echo "Installing PyTorch for JetPack 4.x (CUDA 10.2)"
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
elif [[ "$L4T_VERSION" == "35."* ]]; then
    echo "Installing PyTorch for JetPack 5.x (CUDA 11.4)"
    pip install torch==1.12.0 torchvision==0.13.0 -f https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
else
    echo "Unknown L4T version. Defaulting to JetPack 4.x PyTorch."
    pip install torch==1.10.0 torchvision==0.11.1 -f https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
fi

pip install opencv-python pillow numpy pandas tqdm
pip install Jetson.GPIO
pip install monsoon

# 6. YOLOv13
echo "==== Setting up YOLOv13 ===="
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13
# Remove flash-attention
if grep -q "flash_attn" requirements.txt; then
    grep -v "flash_attn" requirements.txt > requirements_arm64.txt
    pip install -r requirements_arm64.txt
else
    pip install -r requirements.txt
fi
pip install -e .

if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi
cd ..

# 7. Verify
echo "==== Verifying installation ===="
python -c "import torch, cv2, numpy, pandas, tqdm, Jetson.GPIO, monsoon; from ultralytics import YOLO; print('All OK')"

echo "==== Setup complete! ===="
echo "Activate with: source $VENV_NAME/bin/activate"