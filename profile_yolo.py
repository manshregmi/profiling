import torch
import numpy as np
import time
import cv2
import csv
import os
import sys
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# 0. MONSOON – correct imports
# ============================================================
import Monsoon
from Monsoon import sampleEngine

# ============================================================
# 1. Model path helper
# ============================================================
def get_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "weights", "yolov13n.pt")
    if not os.path.isfile(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run setup.sh to download it.")
        sys.exit(1)
    return model_path

# ============================================================
# 2. Load image
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# ============================================================
# 3. YOLOv13 load
# ============================================================
model_path = get_model_path()
device = 0 if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
model = YOLO(model_path)
input_size = 640

# ============================================================
# 4. MONSOON INIT (corrected)
# ============================================================
HVPMSerialNo = 33521  

HVMON = Monsoon.Monsoon()
HVMON.setup_usb(HVPMSerialNo, Monsoon.USB_protocol())
HVMON.fillStatusPacket()
HVMON.setVout(12.0)

HVengine = sampleEngine.SampleEngine(HVMON)
HVengine.startSampling()

# ============================================================
# 5. Parameters
# ============================================================
WARMUP_ITERATIONS = 200
TOTAL_ITERATIONS = 10000
FINAL_AVG_ITERATIONS = 9000

preprocess_latencies = []
yolo_latencies = []
preprocess_power = []
yolo_power = []

# ============================================================
# 6. Warmup
# ============================================================
print(f"Warmup ({WARMUP_ITERATIONS})...")
for _ in range(WARMUP_ITERATIONS):
    _ = model.predict(img_bgr, imgsz=input_size, device=device, verbose=False)

if torch.cuda.is_available():
    torch.cuda.synchronize()

# ============================================================
# 7. MAIN PROFILING LOOP
# ============================================================
print(f"Profiling ({TOTAL_ITERATIONS})...")

for i in range(TOTAL_ITERATIONS):
    # ---------- Preprocessing ----------
    p_start = time.perf_counter()
    resized = cv2.resize(img_bgr, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    p_end = time.perf_counter()
    p_latency = (p_end - p_start) * 1000.0
    p_power = HVengine.getLastSample().power

    # ---------- YOLO inference ----------
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    _ = model.predict(rgb, imgsz=input_size, device=device, verbose=False)

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        y_latency = start_event.elapsed_time(end_event)
    else:
        y_latency = 0

    y_power = HVengine.getLastSample().power

    # ---------- Store only last 9000 ----------
    if i >= TOTAL_ITERATIONS - FINAL_AVG_ITERATIONS:
        preprocess_latencies.append(p_latency)
        yolo_latencies.append(y_latency)
        preprocess_power.append(p_power)
        yolo_power.append(y_power)

    if (i + 1) % 1000 == 0:
        print(f"Completed {i+1}/{TOTAL_ITERATIONS}")

# ============================================================
# 8. Cleanup
# ============================================================
HVengine.stopSampling()
HVMON.setVout(0)

# ============================================================
# 9. Stats
# ============================================================
avg_pp = np.mean(preprocess_latencies)
avg_yolo = np.mean(yolo_latencies)
avg_pp_power = np.mean(preprocess_power)
avg_yolo_power = np.mean(yolo_power)
std_pp = np.std(preprocess_latencies)
std_yolo = np.std(yolo_latencies)

# ============================================================
# 10. CSV OUTPUT
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

raw_csv = f"profile_raw_{ts}.csv"
with open(raw_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "preprocess_latency_ms", "yolo_latency_ms",
                     "preprocess_power_w", "yolo_power_w"])
    for i, row in enumerate(zip(preprocess_latencies, yolo_latencies,
                                preprocess_power, yolo_power)):
        writer.writerow([i + 1, *row])
print(f"Saved: {raw_csv}")

summary_csv = f"profile_summary_{ts}.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["preprocess_avg_ms", avg_pp])
    writer.writerow(["preprocess_std_ms", std_pp])
    writer.writerow(["yolo_avg_ms", avg_yolo])
    writer.writerow(["yolo_std_ms", std_yolo])
    writer.writerow(["preprocess_power_w", avg_pp_power])
    writer.writerow(["yolo_power_w", avg_yolo_power])
    writer.writerow(["total_latency_ms", avg_pp + avg_yolo])
print(f"Saved: {summary_csv}")

# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n==============================")
print("RESULTS")
print("==============================")
print(f"Preprocess: {avg_pp:.3f} ms | {avg_pp_power:.3f} W")
print(f"YOLO      : {avg_yolo:.3f} ms | {avg_yolo_power:.3f} W")
print(f"TOTAL     : {avg_pp + avg_yolo:.3f} ms")
print("==============================")