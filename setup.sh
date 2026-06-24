#!/bin/bash
# setup.sh - One-click environment setup for YOLO v13 + Monsoon profiling on Orin NX
# Usage: chmod +x setup.sh && ./setup.sh

set -e  # Exit immediately if any command fails

echo "=============================================="
echo "  YOLO v13 + Monsoon Profiling Setup Script  "
echo "        (Optimized for NVIDIA Orin NX)       "
echo "=============================================="

# --- 1. System-level dependencies (Linux only) ---
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
    wget

# --- 2. Upgrade pip ---
echo "[2/6] Upgrading pip..."
python3 -m pip install --upgrade pip

# --- 3. Install PyTorch for Orin NX (ARM64) ---
ARCH=$(uname -m)
echo "[3/6] Detected architecture: $ARCH"

if [[ "$ARCH" == "aarch64" ]]; then
    echo "    -> ARM64 (NVIDIA Orin NX) detected."
    echo "    -> Checking JetPack version..."
    
    # Read JetPack version from /etc/nv_tegra_release
    JETPACK_VER=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' | head -1)
    echo "    -> JetPack version: $JETPACK_VER"
    
    # Determine Python version
    PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "    -> Python version: $PYTHON_VER"
    
    # Set the correct PyTorch wheel URL based on JetPack version
    # You may need to adjust these URLs based on your exact JetPack release.
    # Check https://developer.download.nvidia.com/compute/redist/jp/ for the latest URLs.
    if [[ "$JETPACK_VER" == "R35" ]]; then
        # JetPack 5.x (CUDA 11.4)
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-${PYTHON_VER}-linux_aarch64.whl"
    elif [[ "$JETPACK_VER" == "R36" ]]; then
        # JetPack 6.x (CUDA 12.2+) - Example URL, verify for your specific version
        # For JetPack 6.2, use the community build (see alternative below)
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+42c1c3c.nv24.03-${PYTHON_VER}-linux_aarch64.whl"
    else
        echo "    -> Unknown JetPack version. Falling back to generic URL."
        echo "    -> Please manually set TORCH_URL in this script for your JetPack version."
        echo "    -> Visit: https://developer.download.nvidia.com/compute/redist/jp/"
        TORCH_URL=""
    fi
    
    if [[ -n "$TORCH_URL" ]]; then
        echo "    -> Installing PyTorch from: $TORCH_URL"
        python3 -m pip install --no-cache-dir $TORCH_URL
    else
        echo "    -> ERROR: Could not determine Torch URL. Please install manually."
        echo "    -> For JetPack 6.2, try the community build:"
        echo "       wget https://github.com/YeQiao/pytorch-jetson-orin-nano/releases/download/v2.3.0-jetson/pytorch-2.3.0-jetson-orin-nano.tar.gz"
        echo "       tar -xzf pytorch-2.3.0-jetson-orin-nano.tar.gz && cd pytorch-jetson-dist && ./install_pytorch.sh"
        exit 1
    fi
    
    # Install torchvision (optional but recommended)
    echo "    -> Installing torchvision..."
    # For JetPack 5.x:
    if [[ "$JETPACK_VER" == "R35" ]]; then
        TORCHVISION_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.0a0+5c8bc3a.nv23.06-${PYTHON_VER}-linux_aarch64.whl"
        python3 -m pip install --no-cache-dir $TORCHVISION_URL || echo "    -> torchvision install skipped (optional)"
    fi

else
    echo "    -> x86_64 (standard PC) detected."
    echo "    -> Installing PyTorch from PyPI..."
    python3 -m pip install torch torchvision
fi

# --- 4. Install remaining Python dependencies from requirements.txt ---
echo "[4/6] Installing remaining Python packages (requirements.txt)..."
python3 -m pip install -r requirements.txt

# --- 5. Set up Monsoon USB permissions ---
echo "[5/6] Setting up Monsoon USB permissions (requires sudo)..."
# Monsoon Solutions Inc. USB Vendor ID is typically 0b1e
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0b1e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-monsoon.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add current user to plugdev group (so you don't need sudo to run the script later)
sudo usermod -a -G plugdev $USER

# --- 6. Verify installation ---
echo "[6/6] Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}')" || echo "⚠️ PyTorch verification failed."

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