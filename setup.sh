#!/bin/bash
# setup_python310.sh - Install Python 3.10 into media folder,
# create a virtual environment, install CPU-only PyTorch + YOLOv13.

set -e

echo "==== Setting up Python 3.10 environment in media folder ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Install system build dependencies (if not already)
# ------------------------------------------------------------------
sudo apt update
sudo apt install -y \
    build-essential wget git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev libncurses5-dev \
    libgdbm-dev libnss3-dev uuid-dev tk-dev \
    libopenblas-dev libatlas-base-dev libusb-1.0-0-dev

# ------------------------------------------------------------------
# 2. Build Python 3.10 from source into the media folder
# ------------------------------------------------------------------
PYTHON_VERSION="3.10.14"
INSTALL_PREFIX="$BASE_DIR/python310"

if [ -d "$INSTALL_PREFIX" ]; then
    echo "Python 3.10 already installed at $INSTALL_PREFIX. Skipping build."
else
    echo "Building Python $PYTHON_VERSION from source..."
    cd /tmp
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    tar -xzf "Python-${PYTHON_VERSION}.tgz"
    cd "Python-${PYTHON_VERSION}"
    ./configure --prefix="$INSTALL_PREFIX" --enable-shared --with-ensurepip=install
    make -j$(nproc)
    make install
    cd /tmp
    rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"
    echo "Python 3.10 installed at $INSTALL_PREFIX"
fi

# ------------------------------------------------------------------
# 3. Create a virtual environment using this Python
# ------------------------------------------------------------------
export PATH="$INSTALL_PREFIX/bin:$PATH"
VENV_NAME="py310_env"
"$INSTALL_PREFIX/bin/python3.10" -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 4. Install Python packages (CPU-only PyTorch)
# ------------------------------------------------------------------
pip install --upgrade pip

# Install CPU-only PyTorch (latest stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other common packages
pip install numpy pandas opencv-python pillow tqdm matplotlib seaborn scikit-learn

# Optional: install Jetson.GPIO and monsoon (they work with Python 3.10 too)
pip install Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 5. Install YOLOv13 (patched to remove strict version pins)
# ------------------------------------------------------------------
if [ -d "yolov13" ]; then
    cd yolov13 && git pull && cd ..
else
    git clone https://github.com/iMoonLab/yolov13.git
fi

cd yolov13

echo "Patching YOLOv13 requirements for Python 3.10..."

# Remove version pins for torch, timm, and flash_attn
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install everything else
pip install -r requirements.txt

# Install YOLOv13 itself
pip install -e .

# Download weights
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# ------------------------------------------------------------------
# 6. Verify
# ------------------------------------------------------------------
echo "==== Verification ===="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 OK')"

echo ""
echo "==== Setup complete! ===="
echo "Python 3.10 is installed in: $INSTALL_PREFIX"
echo "Virtual environment is in: $VENV_NAME"
echo "Activate with: source $VENV_NAME/bin/activate"
echo ""
echo "IMPORTANT: This uses CPU‑only PyTorch – YOLOv13 will run slowly on Jetson Nano."
echo "If you need GPU acceleration, please use the Python 3.6 environment (yolov13_env)."