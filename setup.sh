#!/bin/bash
# setup.sh - Clean environment using Python 3.6 (system default)

set -e

echo "==== Starting Jetson Nano environment setup (Python 3.6) ===="

# 1. Fix broken packages (no new deps)
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# 2. Install only absolutely essential system libraries
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

# 3. Create virtual environment with Python 3.6
VENV_NAME="yolov13_env"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 4. Install Python packages (compatible with 3.6)
pip install --upgrade pip

# PyTorch 1.9.0 for JetPack 4.6 (CUDA 10.2) – works with Python 3.6
pip install torch==1.9.0 torchvision==0.10.0 -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# Core packages
pip install opencv-python pillow numpy pandas tqdm

# GPIO and Monsoon
pip install Jetson.GPIO
pip install monsoon

# 5. YOLOv13 – use a version of Ultralytics that supports Python 3.6
echo "==== Cloning and installing YOLOv13 ===="
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

# Patch requirements: remove flash-attn, and pin ultralytics to 8.0.20 (last 3.6-compatible)
cat requirements.txt | grep -v flash_attn > requirements_patched.txt
echo "ultralytics==8.0.20" >> requirements_patched.txt
pip install -r requirements_patched.txt

# Install YOLOv13 package
pip install -e .

# Download weights
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# 6. Verify
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"
python -c "import Jetson.GPIO, monsoon; print('All modules OK')"

echo ""
echo "==== Setup complete! ===="
echo "Activate the environment with: source $VENV_NAME/bin/activate"
echo "Then run your pipeline as usual."