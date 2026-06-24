import os
import torch
import numpy as np
import time
import cv2
from ultralytics import YOLO

# --- Monsoon Power Monitor imports ---
# pip install Monsoon
import monsoon
from monsoon import sampling

# --- 1. Load the REAL image ONCE (disk I/O is excluded from the timed loop to avoid noise) ---
image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')  # Replace with your image path
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# --- 2. Initialize YOLO model on GPU ---
model = YOLO('yolov13n.pt')
model.to('cuda:0')
input_size = 640

# --- 3. Initialize Monsoon High Voltage Power Monitor ---
HVPMSerialNo = 12345  # Replace with your Monsoon's serial number

HVMON = monsoon.Monsoon()
HVMON.setup_usb(HVPMSerialNo, monsoon.USB_protocol())
HVMON.fillStatusPacket()

# Set voltage to match your Orin's supply (e.g., 12V for barrel jack, 5V for USB-C)
HVMON.setVout(12.0)  

# Create a sample engine to collect readings at 5000 Hz
HVengine = sampling.SampleEngine(HVMON)

# Optional: Save all raw data to CSV for detailed analysis
# HVengine.enableCSVOutput("full_profile.csv")

# Start collecting samples
HVengine.startSampling()

# --- 4. Warm-up and iteration parameters ---
WARMUP_ITERATIONS = 1000
TOTAL_ITERATIONS = 10000
FINAL_AVG_ITERATIONS = 9000

# --- 5. Storage for results ---
cpu_latencies = []      # Preprocessing latency (ms)
gpu_latencies = []      # Inference latency (ms)
cpu_power_readings = [] # Power during CPU phase (W)
gpu_power_readings = [] # Power during GPU phase (W)

# --- 6. Warm-up phase ---
print(f"Starting warm-up ({WARMUP_ITERATIONS} iterations)...")
for _ in range(WARMUP_ITERATIONS):
    # 6.1 CPU Preprocessing
    img_copy = img_bgr.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)          # Grayscale conversion
    gray_float = gray.astype(np.float32) / 255.0              # Normalization
    resized = cv2.resize(gray_float, (input_size, input_size)) # Dimension reduction
    # Convert to 1-channel tensor, then repeat to 3 channels (YOLO expects 3 channels)
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    input_tensor_3ch = input_tensor.repeat(1, 3, 1, 1)        # Shape: [1, 3, 640, 640]

    # 6.2 GPU Inference
    _ = model.predict(input_tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)

# --- 7. Main testing loop ---
print(f"Starting official test ({TOTAL_ITERATIONS} iterations)...")

for i in range(TOTAL_ITERATIONS):
    # ==========================================
    # PHASE 1: CPU PREPROCESSING
    # ==========================================
    cpu_start = time.perf_counter()

    # 1. Image Loading (simulated by copying the pre-loaded image into memory)
    img_copy = img_bgr.copy()
    
    # 2. Color conversion & Grayscale
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 3. Normalization (scale to [0, 1])
    gray_float = gray.astype(np.float32) / 255.0
    
    # 4. Dimension reduction (resize to fixed size)
    resized = cv2.resize(gray_float, (input_size, input_size))
    
    # Convert to GPU tensor (this is a CPU->GPU memory copy, timed here as part of prep)
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to('cuda:0')
    input_tensor_3ch = input_tensor.repeat(1, 3, 1, 1)

    cpu_end = time.perf_counter()
    cpu_latency_ms = (cpu_end - cpu_start) * 1000.0

    # Read Monsoon power IMMEDIATELY after CPU work finishes
    cpu_sample = HVengine.getLastSample()
    cpu_power_w = cpu_sample.power

    # ==========================================
    # PHASE 2: GPU INFERENCE
    # ==========================================
    gpu_start_event = torch.cuda.Event(enable_timing=True)
    gpu_end_event = torch.cuda.Event(enable_timing=True)
    gpu_start_event.record()

    # Run inference
    results = model.predict(input_tensor_3ch, imgsz=input_size, device='cuda:0', verbose=False)

    gpu_end_event.record()
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_latency_ms = gpu_start_event.elapsed_time(gpu_end_event)

    # Read Monsoon power IMMEDIATELY after GPU work finishes
    gpu_sample = HVengine.getLastSample()
    gpu_power_w = gpu_sample.power

    # --- Only record data for the last 9000 iterations ---
    if i >= (TOTAL_ITERATIONS - FINAL_AVG_ITERATIONS):
        cpu_latencies.append(cpu_latency_ms)
        gpu_latencies.append(gpu_latency_ms)
        cpu_power_readings.append(cpu_power_w)
        gpu_power_readings.append(gpu_power_w)

    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"  Completed {i+1}/{TOTAL_ITERATIONS} iterations")

# --- 8. Stop sampling and clean up ---
HVengine.stopSampling()
HVMON.setVout(0)

# --- 9. Calculate statistics ---
avg_cpu_lat = np.mean(cpu_latencies)
avg_gpu_lat = np.mean(gpu_latencies)
avg_cpu_pwr = np.mean(cpu_power_readings)
avg_gpu_pwr = np.mean(gpu_power_readings)

std_cpu_lat = np.std(cpu_latencies)
std_gpu_lat = np.std(gpu_latencies)
std_cpu_pwr = np.std(cpu_power_readings)
std_gpu_pwr = np.std(gpu_power_readings)

total_avg_lat = avg_cpu_lat + avg_gpu_lat

print(f"\n{'='*60}")
print(f"PROFILING RESULTS (based on the last {FINAL_AVG_ITERATIONS} iterations)")
print(f"{'='*60}")
print(f"\n[CPU Preprocessing]")
print(f"  Latency:   {avg_cpu_lat:.3f} ± {std_cpu_lat:.3f} ms")
print(f"  Power:     {avg_cpu_pwr:.3f} ± {std_cpu_pwr:.3f} W")
print(f"  Energy:    {avg_cpu_pwr * (avg_cpu_lat / 1000):.6f} J")

print(f"\n[GPU Inference]")
print(f"  Latency:   {avg_gpu_lat:.3f} ± {std_gpu_lat:.3f} ms")
print(f"  Power:     {avg_gpu_pwr:.3f} ± {std_gpu_pwr:.3f} W")
print(f"  Energy:    {avg_gpu_pwr * (avg_gpu_lat / 1000):.6f} J")

print(f"\n[Total Pipeline]")
print(f"  Total Latency: {total_avg_lat:.3f} ms")
print(f"  Avg Power (overall): {(avg_cpu_pwr + avg_gpu_pwr)/2:.3f} W (approximate)")
print(f"{'='*60}")