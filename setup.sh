#!/bin/bash
# setup_py310_final.sh - Complete Python 3.10 setup with pyenv

set -e
BASE_DIR="$(pwd)"

# Fix broken packages
sudo apt update
sudo apt --fix-broken install -y || true
sudo apt install -f -y || true

# Install minimal build deps (skip errors)
sudo apt install -y \
    build-essential make wget git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libffi-dev liblzma-dev libopenblas-dev libatlas-base-dev \
    libxml2-dev libxmlsec1-dev tk-dev || true

# pyenv
if [ ! -d ".pyenv" ]; then
    git clone https://github.com/pyenv/pyenv.git .pyenv
fi
export PYENV_ROOT="$BASE_DIR/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Build Python 3.10 only if missing
if [ ! -d ".pyenv/versions/3.10.14" ]; then
    echo "Building Python 3.10.14 (takes 20-30 min)..."
    pyenv install 3.10.14
fi

# Create venv
pyenv shell 3.10.14
python -m venv py310_env --clear
source py310_env/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas opencv-python pillow tqdm matplotlib scikit-learn
pip install timm Jetson.GPIO monsoon

# YOLOv13
if [ ! -d "yolov13" ]; then
    git clone https://github.com/iMoonLab/yolov13.git
fi
cd yolov13
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt
pip install -r requirements.txt
pip install -e .
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi
cd ..

# Verify
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('✅ YOLOv13 works')"

echo "==== Done ===="
echo "Activate with: source py310_env/bin/activate"