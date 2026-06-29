import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
from torchvision import transforms, models
from PIL import Image

# ------------------------------
# 1. Load models
# ------------------------------
# Load YOLOv13 (using Ultralytics wrapper, then extract raw model)
detection_model = YOLO('yolov13n.pt').model.to('cuda').eval()

# Load three classifiers (choose lightweight ones for Jetson Nano)
classifier_1 = models.resnet18(pretrained=True).to('cuda').eval()
classifier_2 = models.mobilenet_v2(pretrained=True).to('cuda').eval()
classifier_3 = models.squeezenet1_0(pretrained=True).to('cuda').eval()

# For demonstration, we'll use a list of class names (ImageNet)
CLASS_NAMES = [f"Class_{i}" for i in range(1000)]

# ------------------------------
# 2. Pre‑processing functions (CPU)
# ------------------------------
def detection_preprocess_cpu(image, target_size=640):
    """
    Pre‑process for detection: resize, Gaussian blur, grayscale,
    convert back to 3 channels, normalize to [0,1].
    Returns a torch.Tensor on CPU.
    """
    resized = cv2.resize(image, (target_size, target_size))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = gray_3ch.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    return tensor

def classification_preprocess_cpu(cropped_image, target_size=224):
    """
    Pre‑process for classification: resize, blur, grayscale, 
    convert to 3‑channel, ToTensor + ImageNet normalization.
    Returns a torch.Tensor on CPU.
    """
    resized = cv2.resize(cropped_image, (target_size, target_size))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(gray_3ch)).unsqueeze(0)
    return tensor

# ------------------------------
# 3. Post‑processing functions (CPU)
# ------------------------------
def detection_postprocess_cpu(original_image, detections):
    """
    Crop objects from the original image using bounding boxes.
    detections: list of [x1, y1, x2, y2, conf, class]
    Returns a list of cropped images (numpy arrays).
    """
    crops = []
    h, w = original_image.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crops.append(original_image[y1:y2, x1:x2])
    return crops

def classification_postprocess_cpu(logits):
    """Convert logits to a class name (softmax + argmax)."""
    probs = torch.softmax(logits, dim=1)
    _, pred_idx = torch.max(probs, 1)
    return CLASS_NAMES[pred_idx.item()]

# ------------------------------
# 4. Main pipeline with hybrid timing
# ------------------------------
def run_hybrid_timed_pipeline(image_path, num_warmup=10):
    # Load input image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Warm‑up GPU (to initialise kernels) ---
    print("Warming up GPU...")
    dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = detection_model(dummy_input)
    torch.cuda.synchronize()

    # ==============================================
    # 1. Detection Stage
    # ==============================================

    # ---- Pre‑processing (CPU) ----
    t_start_cpu = time.perf_counter()
    det_input_cpu = detection_preprocess_cpu(original)
    t_end_cpu = time.perf_counter()
    det_pre_time_ms = (t_end_cpu - t_start_cpu) * 1000.0

    # ---- Move data to GPU ----
    det_input_gpu = det_input_cpu.to('cuda')

    # ---- Inference (GPU) using CUDA events ----
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        # Raw model output (may need NMS)
        outputs = detection_model(det_input_gpu)
        # Apply NMS (Ultralytics' non_max_suppression)
        dets = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45)
    end_event.record()
    torch.cuda.synchronize()          # Wait for GPU to finish
    det_inf_time_ms = start_event.elapsed_time(end_event)

    # ---- Post‑processing (CPU) ----
    t_start_cpu = time.perf_counter()
    # Convert detections from GPU tensor to CPU list
    if dets[0] is not None:
        detections = dets[0].cpu().numpy().tolist()
    else:
        detections = []
    crops = detection_postprocess_cpu(original, detections)
    t_end_cpu = time.perf_counter()
    det_post_time_ms = (t_end_cpu - t_start_cpu) * 1000.0

    # Print detection timings
    print("\n--- Detection Stage ---")
    print(f"Pre‑process (CPU)   : {det_pre_time_ms:.2f} ms")
    print(f"Inference (GPU)     : {det_inf_time_ms:.2f} ms")
    print(f"Post‑process (CPU)  : {det_post_time_ms:.2f} ms")
    print(f"Number of objects   : {len(crops)}")

    # ==============================================
    # 2. Classification Stage (per crop)
    # ==============================================
    total_class_pre_ms = 0.0
    total_class_inf_ms = 0.0
    total_class_post_ms = 0.0
    final_results = []

    for idx, crop in enumerate(crops):
        # ---- Pre‑process (CPU) ----
        t_start_cpu = time.perf_counter()
        cls_input_cpu = classification_preprocess_cpu(crop)
        t_end_cpu = time.perf_counter()
        total_class_pre_ms += (t_end_cpu - t_start_cpu) * 1000.0

        # ---- Move to GPU ----
        cls_input_gpu = cls_input_cpu.to('cuda')

        # ---- Inference on three classifiers (GPU) ----
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            out1 = classifier_1(cls_input_gpu)
            out2 = classifier_2(cls_input_gpu)
            out3 = classifier_3(cls_input_gpu)
        end_event.record()
        torch.cuda.synchronize()
        total_class_inf_ms += start_event.elapsed_time(end_event)

        # ---- Post‑process (CPU) ----
        t_start_cpu = time.perf_counter()
        label1 = classification_postprocess_cpu(out1)
        label2 = classification_postprocess_cpu(out2)
        label3 = classification_postprocess_cpu(out3)
        combined = f"{label1} | {label2} | {label3}"
        t_end_cpu = time.perf_counter()
        total_class_post_ms += (t_end_cpu - t_start_cpu) * 1000.0

        final_results.append(combined)

    # Print classification timings (totals for all crops)
    print("\n--- Classification Stage (total for all crops) ---")
    print(f"Pre‑process (CPU)   : {total_class_pre_ms:.2f} ms")
    print(f"Inference (GPU)     : {total_class_inf_ms:.2f} ms")
    print(f"Post‑process (CPU)  : {total_class_post_ms:.2f} ms")

    # ---- Overall total ----
    total_time_ms = (det_pre_time_ms + det_inf_time_ms + det_post_time_ms +
                     total_class_pre_ms + total_class_inf_ms + total_class_post_ms)
    print(f"\n=== TOTAL PIPELINE LATENCY : {total_time_ms:.2f} ms ===")

    return final_results

# ------------------------------
# 5. Run the pipeline
# ------------------------------
if __name__ == "__main__":
    results = run_hybrid_timed_pipeline("test.jpg")
    print("\nFinal cascaded labels per object:")
    for i, label in enumerate(results):
        print(f"Object {i+1}: {label}")