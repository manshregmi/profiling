#!/bin/bash
# setup.sh - One-click environment setup for YOLO v13 + Monsoon profiling
# Specifically optimized for NVIDIA Orin NX with JetPack 6.2.2
# Usage: chmod +x setup.sh && ./setup.sh

set -e  # Exit immediately if any command fails

echo "=============================================="
echo "  YOLO v13 + Monsoon Profiling Setup Script  "
echo "       (Optimized for JetPack 6.2.2)         "
echo "=============================================="

# ----------------------------------------------------------------------
# 1. System-level dependencies
# ----------------------------------------------------------------------
echo "[1/6] Installing system dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    udev \
    wget \
    curl

# ----------------------------------------------------------------------
# 2. Upgrade pip
# ----------------------------------------------------------------------
echo "[2/6] Upgrading pip..."
python3 -m pip install --upgrade pip

# ----------------------------------------------------------------------
# 3. Install cuSPARSELt (required for JetPack 6.x)
# ----------------------------------------------------------------------
echo "[3/6] Installing cuSPARSELt for JetPack 6.2..."
# cuSPARSELt 0.8.1 is the compatible version for JetPack 6.2.2[reference:3]
wget -q https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb -O /tmp/cusparselt.deb
if [ -f /tmp/cusparselt.deb ]; then
    sudo dpkg -i /tmp/cusparselt.deb 2>/dev/null || true
    sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
    sudo apt-get update 2>/dev/null || true
    sudo apt-get install -y cusparselt-cuda-12 2>/dev/null || true
    rm -f /tmp/cusparselt.deb
    echo "    ✅ cuSPARSELt installed"
else
    echo "    ⚠️  Could not download cuSPARSELt. Continuing anyway..."
fi

# ----------------------------------------------------------------------
# 4. Install PyTorch for JetPack 6.2.2
# ----------------------------------------------------------------------
echo "[4/6] Installing PyTorch for JetPack 6.2.2..."

# Option A: Use the community build from YeQiao (PyTorch 2.3.0, cuDNN 9.3 compatible)[reference:4]
# This is the most reliable option for JetPack 6.2
echo "    -> Downloading community PyTorch build (cuDNN 9.3 compatible)..."
wget -q https://github.com/YeQiao/pytorch-jetson-orin-nano/releases/download/v2.3.0-jetson/pytorch-2.3.0-jetson-orin-nano.tar.gz -O /tmp/pytorch-jetson.tar.gz

if [ -f /tmp/pytorch-jetson.tar.gz ]; then
    tar -xzf /tmp/pytorch-jetson.tar.gz -C /tmp/
    cd /tmp/pytorch-jetson-dist
    ./install_pytorch.sh
    cd -
    rm -f /tmp/pytorch-jetson.tar.gz
    rm -rf /tmp/pytorch-jetson-dist
    echo "    ✅ PyTorch installed from community build"
else
    echo "    ⚠️  Community build download failed. Trying alternative method..."
    
    # Option B: Use the davidl-nv wheel (PyTorch 2.8.0)[reference:5]
    wget -q https://github.com/davidl-nv/torch/raw/refs/heads/main/torch-2.8/torch-2.8.0-cp310-cp310-linux_aarch64.whl -O /tmp/torch.whl
    wget -q https://github.com/davidl-nv/torch/raw/refs/heads/main/torch-2.8/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl -O /tmp/torchvision.whl
    
    if [ -f /tmp/torch.whl ]; then
        python3 -m pip install /tmp/torch.whl
        if [ -f /tmp/torchvision.whl ]; then
            python3 -m pip install /tmp/torchvision.whl
        fi
        rm -f /tmp/torch.whl /tmp/torchvision.whl
        echo "    ✅ PyTorch installed from davidl-nv wheel"
    else
        echo "    ❌ All PyTorch installation methods failed."
        echo "    Please install manually:"
        echo "    - Option 1: https://github.com/YeQiao/pytorch-jetson-orin-nano"
        echo "    - Option 2: https://github.com/davidl-nv/torch"
        exit 1
    fi
fi

# ----------------------------------------------------------------------
# 5. Install remaining Python dependencies
# ----------------------------------------------------------------------
echo "[5/6] Installing remaining Python packages..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
else
    python3 -m pip install numpy opencv-python ultralytics Monsoon pyusb libusb1
fi

# ----------------------------------------------------------------------
# 6. Set up Monsoon USB permissions
# ----------------------------------------------------------------------
echo "[6/6] Setting up Monsoon USB permissions (requires sudo)..."
# Monsoon Solutions Inc. USB Vendor ID is 0b1e
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0b1e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-monsoon.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add current user to plugdev group
sudo usermod -a -G plugdev $USER

# ----------------------------------------------------------------------
# 7. Verify installation
# ----------------------------------------------------------------------
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
" || echo "⚠️  PyTorch verification failed. Please check manually."

echo ""
echo "=============================================="
echo "  ✅ Setup completed successfully!"
echo "=============================================="
echo ""
echo "⚠️  IMPORTANT: For USB permissions to take effect:"
echo "  1. Log out and log back in, OR"
echo "  2. Run: sudo reboot"
echo ""
echo "After reboot, you can run the profiler with:"
echo "  python3 profile_yolo_v13.py"
echo "=============================================="