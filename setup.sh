# Activate the existing environment
source yolov13_env/bin/activate

# Go into yolov13
cd yolov13

# Remove strict version pins
sed -i '/torch==/d' requirements.txt
sed -i '/timm==/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt

# Install dependencies (no torch, no timm pin)
pip install --no-cache-dir -r requirements.txt

# Install timm 0.6.12 (compatible with Python 3.6)
pip install --no-cache-dir timm==0.6.12

# Install YOLOv13 (without reinstalling torch)
pip install --no-cache-dir --no-deps -e .

# Download weights if missing
if [ ! -f "yolov13n.pt" ]; then
    wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
fi

cd ..

# Verify
python -c "from ultralytics import YOLO; model = YOLO('yolov13/yolov13n.pt'); print('✅ YOLOv13 loaded successfully')"