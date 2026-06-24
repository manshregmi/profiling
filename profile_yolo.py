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
# 0. Model helper
# ============================================================
def get_model_path():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir,
        "weights",
        "yolov13n.pt"
    )

    if not os.path.isfile(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    try:
        YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        return model_path

    except Exception as e:
        print("❌ YOLO model loading failed")
        print(e)
        sys.exit(1)


# ============================================================
# 1. Load image
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    script_dir,
    "test.jpg"
)

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# ============================================================
# 2. Load YOLOv13
# ============================================================

model_path = get_model_path()

device = 0 if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = YOLO(model_path)

input_size = 640


# ============================================================
# 3. Initialize Monsoon
# ============================================================

HVPMSerialNo = 12345   # CHANGE THIS

HVMON = Monsoon.Monsoon()

HVMON.setup_usb(
    HVPMSerialNo,
    Monsoon.USB_protocol()
)

HVMON.fillStatusPacket()

HVMON.setVout(12.0)


HVengine = sampleEngine.SampleEngine(HVMON)

HVengine.startSampling()



# ============================================================
# 4. Parameters
# ============================================================

WARMUP_ITERATIONS = 200
TOTAL_ITERATIONS = 10000
FINAL_AVG_ITERATIONS = 9000


preprocess_latencies = []
yolo_latencies = []

preprocess_power = []
yolo_power = []



# ============================================================
# 5. Warmup
# ============================================================

print(
    f"Warm-up ({WARMUP_ITERATIONS} iterations)..."
)


for _ in range(WARMUP_ITERATIONS):

    _ = model.predict(
        img,
        imgsz=input_size,
        device=device,
        verbose=False
    )


if torch.cuda.is_available():
    torch.cuda.synchronize()



# ============================================================
# 6. Profiling
# ============================================================

print(
    f"Profiling ({TOTAL_ITERATIONS} iterations)..."
)


for i in range(TOTAL_ITERATIONS):


    # -----------------------------
    # Preprocessing timing
    # -----------------------------

    p_start = time.perf_counter()


    resized = cv2.resize(
        img,
        (input_size, input_size)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )


    p_end = time.perf_counter()


    p_latency = (
        p_end - p_start
    ) * 1000


    p_sample = HVengine.getLastSample()

    p_power = p_sample.power



    # -----------------------------
    # YOLO timing
    # -----------------------------

    if torch.cuda.is_available():

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()



    _ = model.predict(
        rgb,
        imgsz=input_size,
        device=device,
        verbose=False
    )


    if torch.cuda.is_available():

        end_event.record()

        torch.cuda.synchronize()

        y_latency = (
            start_event.elapsed_time(
                end_event
            )
        )

    else:

        y_latency = 0



    y_sample = HVengine.getLastSample()

    y_power = y_sample.power



    # Store only final iterations

    if i >= TOTAL_ITERATIONS - FINAL_AVG_ITERATIONS:

        preprocess_latencies.append(
            p_latency
        )

        yolo_latencies.append(
            y_latency
        )

        preprocess_power.append(
            p_power
        )

        yolo_power.append(
            y_power
        )


    if (i+1) % 1000 == 0:

        print(
            f"Completed {i+1}/{TOTAL_ITERATIONS}"
        )



# ============================================================
# 7. Cleanup
# ============================================================

HVengine.stopSampling()

HVMON.setVout(0)



# ============================================================
# 8. Statistics
# ============================================================

avg_pp = np.mean(preprocess_latencies)
avg_yolo = np.mean(yolo_latencies)

std_pp = np.std(preprocess_latencies)
std_yolo = np.std(yolo_latencies)


avg_pp_power = np.mean(preprocess_power)
avg_yolo_power = np.mean(yolo_power)



timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)



# ============================================================
# 9. Save raw CSV
# ============================================================


raw_file = (
    f"profile_raw_{timestamp}.csv"
)


with open(
    raw_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "iteration",
            "preprocess_latency_ms",
            "yolo_latency_ms",
            "preprocess_power_w",
            "yolo_power_w"
        ]
    )


    for i, data in enumerate(
        zip(
            preprocess_latencies,
            yolo_latencies,
            preprocess_power,
            yolo_power
        )
    ):

        writer.writerow(
            [
                i+1,
                *data
            ]
        )


print(
    f"✅ Saved {raw_file}"
)



# ============================================================
# 10. Summary
# ============================================================


summary_file = (
    f"profile_summary_{timestamp}.csv"
)


with open(
    summary_file,
    "w",
    newline=""
) as f:


    writer = csv.writer(f)


    writer.writerow(
        [
            "metric",
            "value"
        ]
    )


    results = {

        "model":
        "yolov13n.pt",

        "iterations":
        TOTAL_ITERATIONS,

        "preprocess_avg_ms":
        avg_pp,

        "preprocess_std_ms":
        std_pp,

        "preprocess_power_W":
        avg_pp_power,

        "preprocess_energy_J":
        avg_pp_power *
        avg_pp / 1000,


        "yolo_avg_ms":
        avg_yolo,

        "yolo_std_ms":
        std_yolo,

        "yolo_power_W":
        avg_yolo_power,

        "yolo_energy_J":
        avg_yolo_power *
        avg_yolo / 1000,


        "total_latency_ms":
        avg_pp + avg_yolo
    }



    for k,v in results.items():

        writer.writerow(
            [
                k,
                v
            ]
        )


print(
    f"✅ Saved {summary_file}"
)



# ============================================================
# Final output
# ============================================================

print("\n" + "="*60)

print("YOLOv13 PROFILING RESULTS")

print("="*60)

print(
    f"Preprocess : {avg_pp:.3f} ms "
    f" {avg_pp_power:.3f} W"
)

print(
    f"YOLO       : {avg_yolo:.3f} ms "
    f" {avg_yolo_power:.3f} W"
)

print(
    f"TOTAL      : {avg_pp+avg_yolo:.3f} ms"
)

print("="*60)