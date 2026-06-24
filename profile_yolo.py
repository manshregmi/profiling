import torch
import numpy as np
import time
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

# --- Monsoon Power Monitor imports ---
import monsoon
from monsoon import sampling

# ============================================================
# 1. Load the REAL image ONCE (disk I/O excluded from timing)
# ============================================================
image_path = "/path/to/your/test_image.jpg"  # <-- CHANGE THIS
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# ============================================================
# 2. Initialize YOLO model on GPU
# ============================================================
model = YOLO('yolov13n.pt')
model.to('cuda:0')
input_size = 640

# ============================================================
# 3. Initialize Monsoon High Voltage Power Monitor
# ============================================================
HVPMSerialNo = 12345  # Replace with your Monsoon's serial number

HVMON = monsoon.Monsoon()
HVMON.setup_usb(HVPMSerialNo, monsoon.USB_protocol())
HVMON.fillStatusPacket()

# Set voltage to match your Orin's supply (e.g., 12V for barrel jack, 5V for USB-C)
HVMON.setVout(12.0)

# Create a sample engine to collect readings at 5000 Hz
HVengine = sampling.SampleEngine(HVMON)

# Optional: Save ALL raw Monsoon samples to a separate CSV (5000 Hz data)
# HVengine.enableCSVOutput("monsoon_full_profile.csv")

# Start collecting samples
HVengine.startSampling()

# ============================================================
# 4. Warm-up and iteration parameters
# ============================================================
WARMUP_ITERATIONS = 1000
TOTAL_ITERATIONS = 10000
FINAL_AVG_ITERATIONS = 9000

# ============================================================
# 5. Storage for results (renamed for clarity)
# ============================================================
preprocess_latencies = []      # Preprocessing latency (ms)
yolo_latencies = []            # YOLO inference latency (ms)
preprocess_power_readings = [] # Power during preprocessing phase (W)
yolo_power_readings = []       # Power during YOLO inference phase (W)

# ============================================================
# 6. Warm-up phase
# ============================================================
print(f"Starting warm-up ({WARMUP_ITERATIONS} iterations)...")
for _ in range(WARMUP_ITERATIONS):
    img_copy = img_bgr.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    resized = cv2.resize(gray_float, (input_size, input_size))
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    input_tensor_3ch = input_tensor.repeat(1, 3, 1, 1)

    _ = model.predict(input_tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)

# ============================================================
# 7. Main testing loop
# ============================================================
print(f"Starting official test ({TOTAL_ITERATIONS} iterations)...")

for i in range(TOTAL_ITERATIONS):
    # ---------- PHASE 1: PREPROCESSING (CPU) ----------
    preprocess_start = time.perf_counter()

    img_copy = img_bgr.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    resized = cv2.resize(gray_float, (input_size, input_size))
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    input_tensor_3ch = input_tensor.repeat(1, 3, 1, 1)

    preprocess_end = time.perf_counter()
    preprocess_latency_ms = (preprocess_end - preprocess_start) * 1000.0

    preprocess_sample = HVengine.getLastSample()
    preprocess_power_w = preprocess_sample.power

    # ---------- PHASE 2: YOLO INFERENCE (GPU) ----------
    yolo_start_event = torch.cuda.Event(enable_timing=True)
    yolo_end_event = torch.cuda.Event(enable_timing=True)
    yolo_start_event.record()

    results = model.predict(input_tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)

    yolo_end_event.record()
    torch.cuda.synchronize()
    yolo_latency_ms = yolo_start_event.elapsed_time(yolo_end_event)

    yolo_sample = HVengine.getLastSample()
    yolo_power_w = yolo_sample.power

    # ---------- Only record data for the last 9000 iterations ----------
    if i >= (TOTAL_ITERATIONS - FINAL_AVG_ITERATIONS):
        preprocess_latencies.append(preprocess_latency_ms)
        yolo_latencies.append(yolo_latency_ms)
        preprocess_power_readings.append(preprocess_power_w)
        yolo_power_readings.append(yolo_power_w)

    if (i + 1) % 1000 == 0:
        print(f"  Completed {i+1}/{TOTAL_ITERATIONS} iterations")

# ============================================================
# 8. Stop sampling and clean up
# ============================================================
HVengine.stopSampling()
HVMON.setVout(0)

# ============================================================
# 9. Calculate statistics (renamed variables)
# ============================================================
avg_preprocess_lat = np.mean(preprocess_latencies)
avg_yolo_lat = np.mean(yolo_latencies)
avg_preprocess_pwr = np.mean(preprocess_power_readings)
avg_yolo_pwr = np.mean(yolo_power_readings)

std_preprocess_lat = np.std(preprocess_latencies)
std_yolo_lat = np.std(yolo_latencies)
std_preprocess_pwr = np.std(preprocess_power_readings)
std_yolo_pwr = np.std(yolo_power_readings)

total_avg_lat = avg_preprocess_lat + avg_yolo_lat

# ============================================================
# 10. Save results to CSV (updated headers)
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 10a. Save per-iteration raw data ---
raw_filename = f"profile_raw_{timestamp}.csv"
with open(raw_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "iteration",
        "preprocess_latency_ms",
        "yolo_latency_ms",
        "preprocess_power_w",
        "yolo_power_w"
    ])
    for idx, (p_lat, y_lat, p_pwr, y_pwr) in enumerate(
        zip(preprocess_latencies, yolo_latencies, preprocess_power_readings, yolo_power_readings)
    ):
        writer.writerow([idx + 1, p_lat, y_lat, p_pwr, y_pwr])

print(f"✅ Raw iteration data saved to: {raw_filename}")

# --- 10b. Save summary statistics (updated metric names) ---
summary_filename = f"profile_summary_{timestamp}.csv"
with open(summary_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["timestamp", timestamp])
    writer.writerow(["image_path", image_path])
    writer.writerow(["model", "yolov13n.pt"])
    writer.writerow(["total_iterations", TOTAL_ITERATIONS])
    writer.writerow(["recorded_iterations", FINAL_AVG_ITERATIONS])
    writer.writerow(["warmup_iterations", WARMUP_ITERATIONS])
    writer.writerow(["input_size", input_size])
    writer.writerow([])  # blank row
    # Preprocessing metrics
    writer.writerow(["preprocess_avg_latency_ms", avg_preprocess_lat])
    writer.writerow(["preprocess_std_latency_ms", std_preprocess_lat])
    writer.writerow(["preprocess_avg_power_w", avg_preprocess_pwr])
    writer.writerow(["preprocess_std_power_w", std_preprocess_pwr])
    writer.writerow(["preprocess_energy_joules", avg_preprocess_pwr * (avg_preprocess_lat / 1000.0)])
    writer.writerow([])  # blank row
    # YOLO inference metrics
    writer.writerow(["yolo_avg_latency_ms", avg_yolo_lat])
    writer.writerow(["yolo_std_latency_ms", std_yolo_lat])
    writer.writerow(["yolo_avg_power_w", avg_yolo_pwr])
    writer.writerow(["yolo_std_power_w", std_yolo_pwr])
    writer.writerow(["yolo_energy_joules", avg_yolo_pwr * (avg_yolo_lat / 1000.0)])
    writer.writerow([])  # blank row
    # Total pipeline
    writer.writerow(["total_avg_latency_ms", total_avg_lat])

print(f"✅ Summary statistics saved to: {summary_filename}")

# ============================================================
# 11. Print results to console (updated labels)
# ============================================================
print(f"\n{'='*60}")
print(f"PROFILING RESULTS (based on the last {FINAL_AVG_ITERATIONS} iterations)")
print(f"{'='*60}")
print(f"\n[Preprocessing]")
print(f"  Latency:   {avg_preprocess_lat:.3f} ± {std_preprocess_lat:.3f} ms")
print(f"  Power:     {avg_preprocess_pwr:.3f} ± {std_preprocess_pwr:.3f} W")
print(f"  Energy:    {avg_preprocess_pwr * (avg_preprocess_lat / 1000):.6f} J")

print(f"\n[YOLO Inference]")
print(f"  Latency:   {avg_yolo_lat:.3f} ± {std_yolo_lat:.3f} ms")
print(f"  Power:     {avg_yolo_pwr:.3f} ± {std_yolo_pwr:.3f} W")
print(f"  Energy:    {avg_yolo_pwr * (avg_yolo_lat / 1000):.6f} J")

print(f"\n[Total Pipeline]")
print(f"  Total Latency: {total_avg_lat:.3f} ms")
print(f"  Avg Power (overall): {(avg_preprocess_pwr + avg_yolo_pwr)/2:.3f} W (approximate)")
print(f"{'='*60}")

print(f"\n📁 CSV files saved:")
print(f"   - {raw_filename}")
print(f"   - {summary_filename}")