#!/bin/bash

# ============================================================
# YOLOv13 + Monsoon Setup for NVIDIA Jetson Orin NX
# JetPack 6.2.2
# ============================================================

set -e


echo "=============================================="
echo " YOLOv13 + Monsoon Setup"
echo " Jetson Orin NX / JetPack 6.2.2"
echo "=============================================="


# ------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------

echo "[1/8] Installing system dependencies..."

sudo apt-get update

sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    udev



# ------------------------------------------------------------
# 2. Upgrade pip
# ------------------------------------------------------------

echo "[2/8] Updating pip..."

python3 -m pip install --upgrade pip



# ------------------------------------------------------------
# 3. cuSPARSELt
# ------------------------------------------------------------

echo "[3/8] Installing cuSPARSELt..."

wget -q \
https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb \
-O /tmp/cusparselt.deb


if [ -f /tmp/cusparselt.deb ]; then

    sudo dpkg -i /tmp/cusparselt.deb || true

    sudo cp \
    /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg \
    /usr/share/keyrings/ || true


    sudo apt-get update

    sudo apt-get install -y \
        cusparselt-cuda-12 || true


    rm /tmp/cusparselt.deb

    echo "cuSPARSELt installed"

else

    echo "cuSPARSELt download failed"

fi




# ------------------------------------------------------------
# 4. PyTorch CUDA
# ------------------------------------------------------------

echo "[4/8] Installing Jetson PyTorch..."

pip3 install \
--pre torch torchvision \
--index-url https://pypi.jetson-ai-lab.dev/jp6/cu126



# ------------------------------------------------------------
# 5. Clone YOLOv13 repo
# ------------------------------------------------------------


echo "[5/8] Installing YOLOv13..."

if [ ! -d "yolov13" ]; then

    git clone https://github.com/iMoonLab/yolov13.git

fi


cd yolov13



# IMPORTANT:
# install repo version, NOT pip ultralytics

pip3 install \
numpy \
opencv-python \
pyusb \
libusb1 \
Monsoon


pip3 install -e .



cd ..



# ------------------------------------------------------------
# 6. Monsoon USB permissions
# ------------------------------------------------------------


echo "[6/8] Setting Monsoon permissions..."


sudo bash -c \
'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0b1e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-monsoon.rules'


sudo udevadm control --reload-rules

sudo udevadm trigger


sudo usermod -a -G plugdev $USER




# ------------------------------------------------------------
# 7. Download YOLOv13 weights
# ------------------------------------------------------------


echo "[7/8] Downloading YOLOv13n weights..."


mkdir -p weights


cd weights


if [ ! -f yolov13n.pt ]; then


wget \
https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt \
-O yolov13n.pt


else

echo "weights already exist"

fi


cd ..



# ------------------------------------------------------------
# 8. Verify
# ------------------------------------------------------------


echo "[8/8] Testing installation..."

python3 <<EOF

import torch

print("--------------------------------")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


from ultralytics import YOLO


model = YOLO(
    "weights/yolov13n.pt"
)


print("YOLOv13 loaded successfully")

EOF



echo ""
echo "=============================================="
echo " SETUP COMPLETE"
echo "=============================================="

echo ""
echo "Run:"
echo ""
echo "python3 profile_yolo.py"
echo ""

echo "If USB permissions fail, reboot:"
echo ""
echo "sudo reboot"