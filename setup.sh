#!/bin/bash
# setup.sh - Complete environment for YOLOv13 pipeline (Python 3.6, GPU)
# Works on Jetson Nano, JetPack 4.6, Ubuntu 18.04.
# All files are stored in the current working directory.

set -e

echo "==== Starting Jetson Nano pipeline setup (Python 3.6) ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Clean up root filesystem and set pip cache to external media
# ------------------------------------------------------------------
sudo apt autoremove -y
sudo apt clean

export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "Using PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "Using TMPDIR=$TMPDIR"

# ------------------------------------------------------------------
# 2. Fix broken packages (if any)
# ------------------------------------------------------------------
sudo apt update
sudo apt --fix-broken install -y
sudo apt install -f -y

# ------------------------------------------------------------------
# 3. Install minimal system libraries (no multimedia conflicts)
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
# 4. Create virtual environment with Python 3.6
# ------------------------------------------------------------------
VENV_NAME="yolov13_env"
echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 5. Install PyTorch 1.9.0 (official Jetson wheel, CUDA 10.2)
# ------------------------------------------------------------------
pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0 \
    -f https://nvidia.box.com/shared/static/ssad6uxn4kcxv80e1xj99nrw3qkw5ss7.whl

# ------------------------------------------------------------------
# 6. Install core Python packages (all have aarch64 wheels for Python 3.6)
# ------------------------------------------------------------------
pip install --no-cache-dir numpy==1.19.5 pillow==8.4.0 pandas tqdm
pip install --no-cache-dir opencv-python-headless==4.5.5.64
pip install --no-cache-dir Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 7. Clone and patch YOLOv13
# ------------------------------------------------------------------
if [ -d "yolov13" ]; then
    echo "yolov13 directory already exists. Pulling latest..."
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "Patching YOLOv13 requirements..."

# Remove strict version pin for torch → allow any >=1.9.0
sed -i 's/torch==2.2.2/torch>=1.9.0/' requirements.txt

# Remove flash-attention (ARM64 incompatible)
sed -i '/flash_attn/d' requirements.txt

# Downgrade timm to a Python 3.6‑compatible version (0.8.0)
sed -i 's/timm==1.0.14/timm==0.8.0/' requirements.txt

# Create a requirements file without the torch line (we already have it)
grep -v "^torch" requirements.txt > requirements_patched.txt

# Install all other dependencies
pip install --no-cache-dir -r requirements_patched.txt

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
# 8. Verify everything works
# ------------------------------------------------------------------
echo "==== Verifying installation ===="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"
python -c "import Jetson.GPIO, monsoon; print('Jetson.GPIO and Monsoon: OK')"

# ------------------------------------------------------------------
# 9. Final instructions
# ------------------------------------------------------------------
echo ""
echo "==== Setup completed successfully! ===="
echo ""
echo "All files are stored in: $BASE_DIR"
echo "Virtual environment: $VENV_NAME"
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To test YOLOv13 detection, run:"
echo "  cd yolov13"
echo "  python -c \"from ultralytics import YOLO; model = YOLO('yolov13n.pt'); results = model('path/to/your/image.jpg'); print('Detection done.')\""
echo ""
echo "To run your full pipeline with latency profiling, use:"
echo "  python pipeline_profiler.py --image test.jpg"
echo ""
echo "Enjoy!"