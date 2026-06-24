import torch
import numpy as np
import time
import cv2
import csv
import os
import sys
from datetime import datetime
from ultralytics import YOLO

# --- Monsoon Power Monitor imports ---
import Monsoon
from Monsoon import sampleEngine

# ============================================================
# 0. Helper: check model exists
# ============================================================
def get_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'weights', 'yolov13n.pt')
    if not os.path.isfile(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run setup.sh first to download the model.")
        sys.exit(1)
    # Verify it loads
    try:
        torch.load(model_path, map_location='cpu')
        print(f"✅ Model found and valid: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Model file is corrupt: {e}")
        print("Please delete it and re-run setup.sh")
        sys.exit(1)

# ============================================================
# 1. Load image
# ============================================================
image_path = os.path.join(os.path.dirname(__file__), 'test.jpg')
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# ============================================================
# 2. Initialize YOLO model
# ============================================================
model_path = get_model_path()
model = YOLO(model_path)
model.to('cuda:0')
input_size = 640

# ============================================================
# 3. Initialize Monsoon
# ============================================================
HVPMSerialNo = 12345          # <-- REPLACE
HVMON = Monsoon.Monsoon()
HVMON.setup_usb(HVPMSerialNo, Monsoon.USB_protocol())
HVMON.fillStatusPacket()
HVMON.setVout(12.0)

HVengine = sampleEngine.SampleEngine(HVMON)
HVengine.startSampling()

# ============================================================
# 4. Profiling parameters
# ============================================================
WARMUP_ITERATIONS = 1000
TOTAL_ITERATIONS = 10000
FINAL_AVG_ITERATIONS = 9000

preprocess_latencies = []
yolo_latencies = []
preprocess_power = []
yolo_power = []

# ============================================================
# 5. Warm-up
# ============================================================
print(f"Warm-up ({WARMUP_ITERATIONS} iterations)...")
for _ in range(WARMUP_ITERATIONS):
    img = img_bgr.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    resized = cv2.resize(gray_float, (input_size, input_size))
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    tensor_3ch = tensor.repeat(1, 3, 1, 1)
    _ = model.predict(tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)

# ============================================================
# 6. Main loop
# ============================================================
print(f"Profiling ({TOTAL_ITERATIONS} iterations)...")
for i in range(TOTAL_ITERATIONS):
    # Preprocessing
    p_start = time.perf_counter()
    img = img_bgr.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    resized = cv2.resize(gray_float, (input_size, input_size))
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    tensor_3ch = tensor.repeat(1, 3, 1, 1)
    p_end = time.perf_counter()
    p_lat = (p_end - p_start) * 1000.0
    p_sample = HVengine.getLastSample()
    p_power = p_sample.power

    # YOLO inference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    _ = model.predict(tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)
    end_event.record()
    torch.cuda.synchronize()
    y_lat = start_event.elapsed_time(end_event)
    y_sample = HVengine.getLastSample()
    y_power = y_sample.power

    if i >= (TOTAL_ITERATIONS - FINAL_AVG_ITERATIONS):
        preprocess_latencies.append(p_lat)
        yolo_latencies.append(y_lat)
        preprocess_power.append(p_power)
        yolo_power.append(y_power)

    if (i+1) % 1000 == 0:
        print(f"  Completed {i+1}/{TOTAL_ITERATIONS}")

# ============================================================
# 7. Cleanup & stats
# ============================================================
HVengine.stopSampling()
HVMON.setVout(0)

avg_pp = np.mean(preprocess_latencies)
avg_yolo = np.mean(yolo_latencies)
avg_pp_pwr = np.mean(preprocess_power)
avg_yolo_pwr = np.mean(yolo_power)
std_pp = np.std(preprocess_latencies)
std_yolo = np.std(yolo_latencies)

# CSV export (same as before)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_csv = f"profile_raw_{ts}.csv"
with open(raw_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["iteration","preprocess_latency_ms","yolo_latency_ms",
                "preprocess_power_w","yolo_power_w"])
    for idx, (pl, yl, pp, yp) in enumerate(zip(preprocess_latencies,
                                               yolo_latencies,
                                               preprocess_power,
                                               yolo_power)):
        w.writerow([idx+1, pl, yl, pp, yp])
print(f"✅ Raw data saved: {raw_csv}")

sum_csv = f"profile_summary_{ts}.csv"
with open(sum_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["metric","value"])
    w.writerow(["timestamp", ts])
    w.writerow(["model","yolov13n.pt"])
    w.writerow(["total_iterations", TOTAL_ITERATIONS])
    w.writerow(["recorded_iterations", FINAL_AVG_ITERATIONS])
    w.writerow(["preprocess_avg_latency_ms", avg_pp])
    w.writerow(["preprocess_std_latency_ms", std_pp])
    w.writerow(["preprocess_avg_power_w", avg_pp_pwr])
    w.writerow(["preprocess_energy_joules", avg_pp_pwr * (avg_pp/1000)])
    w.writerow(["yolo_avg_latency_ms", avg_yolo])
    w.writerow(["yolo_std_latency_ms", std_yolo])
    w.writerow(["yolo_avg_power_w", avg_yolo_pwr])
    w.writerow(["yolo_energy_joules", avg_yolo_pwr * (avg_yolo/1000)])
    w.writerow(["total_avg_latency_ms", avg_pp + avg_yolo])
print(f"✅ Summary saved: {sum_csv}")

print("\n" + "="*60)
print(f"RESULTS (based on last {FINAL_AVG_ITERATIONS} iterations)")
print("="*60)
print(f"\n[Preprocessing]  Latency: {avg_pp:.3f} ms, Power: {avg_pp_pwr:.3f} W")
print(f"[YOLO Inference] Latency: {avg_yolo:.3f} ms, Power: {avg_yolo_pwr:.3f} W")
print(f"[Total]          {avg_pp + avg_yolo:.3f} ms")
print("="*60)