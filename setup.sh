#!/bin/bash
# setup_py310.sh - Install Python 3.10 via pyenv into media folder,
# create virtual environment, install CPU-only PyTorch and YOLOv13.

set -e

echo "==== Installing Python 3.10 environment in media folder ===="
BASE_DIR="$(pwd)"
echo "Installing into: $BASE_DIR"

# ------------------------------------------------------------------
# 1. Install minimal system build dependencies (skip errors)
# ------------------------------------------------------------------
sudo apt update
sudo apt install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
    libopenblas-dev libatlas-base-dev || true

# ------------------------------------------------------------------
# 2. Install pyenv into the current directory
# ------------------------------------------------------------------
if [ ! -d ".pyenv" ]; then
    echo "Cloning pyenv..."
    git clone https://github.com/pyenv/pyenv.git .pyenv
else
    echo "pyenv already exists, updating..."
    cd .pyenv && git pull && cd ..
fi

export PYENV_ROOT="$BASE_DIR/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ------------------------------------------------------------------
# 3. Install Python 3.10.14 (if not already installed)
# ------------------------------------------------------------------
if [ ! -d ".pyenv/versions/3.10.14" ]; then
    echo "Building Python 3.10.14 (this takes 20-30 minutes)..."
    pyenv install 3.10.14
else
    echo "Python 3.10.14 already installed."
fi

# ------------------------------------------------------------------
# 4. Create virtual environment using this Python
# ------------------------------------------------------------------
pyenv shell 3.10.14
VENV_NAME="py310_env"
if [ ! -d "$VENV_NAME" ]; then
    python -m venv "$VENV_NAME"
fi
source "$VENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 5. Install Python packages (CPU-only PyTorch)
# ------------------------------------------------------------------
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas opencv-python pillow tqdm matplotlib scikit-learn
pip install timm Jetson.GPIO monsoon

# ------------------------------------------------------------------
# 6. Install YOLOv13 (patched)
# ------------------------------------------------------------------
if [ ! -d "yolov13" ]; then
    git clone https://github.com/iMoonLab/yolov13.git
else
    cd yolov13 && git pull && cd ..
fi

cd yolov13

# Remove strict version pins for torch, timm, flash_attn
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install dependencies
pip install -r requirements.txt

# Install YOLOv13 itself
pip install -e .

# Download weights if missing
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# ------------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------------
echo "==== Verification ===="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('YOLOv13 loaded successfully')"

echo ""
echo "==== Setup complete! ===="
echo "Python 3.10 is installed in: $BASE_DIR/.pyenv/versions/3.10.14"
echo "Virtual environment is in: $BASE_DIR/$VENV_NAME"
echo "Activate with: source $VENV_NAME/bin/activate"
echo ""
echo "IMPORTANT: This uses CPU‑only PyTorch – YOLOv13 will run on CPU (slow on Nano)."
echo "If you need GPU acceleration, please use the Python 3.6 environment (yolov13_env)."