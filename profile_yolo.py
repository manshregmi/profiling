import time
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ------------------------------------------------------------
# YOLOv5: load from torch.hub (no ultralytics package needed)
# ------------------------------------------------------------
def load_yolov5_model(weights='yolov5s', device='cuda'):
    """
    Load YOLOv5 from torch.hub. Works with Python 3.6.
    You can replace 'yolov5s' with a custom weight path if needed.
    """
    model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True, device=device)
    model.eval()
    return model

# ------------------------------------------------------------
# Pre‑processing for YOLOv5 (takes a BGR numpy image, returns tensor)
# ------------------------------------------------------------
def yolov5_preprocess(image, target_size=640):
    """
    YOLOv5 expects a numpy array in BGR order, resized to (target_size, target_size)
    and normalized to [0, 1]. The torch.hub model will handle normalization internally,
    so we just resize and convert to tensor.
    """
    # Resize
    resized = cv2.resize(image, (target_size, target_size))
    # Convert to RGB (YOLOv5 expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # To tensor (HWC -> CHW) and scale to [0,1]
    tensor = torch.from_numpy(rgb.transpose(2,0,1)).float() / 255.0
    tensor = tensor.unsqueeze(0)  # add batch dim
    return tensor

# ------------------------------------------------------------
# Post‑processing for YOLOv5 (extract boxes, scores, classes)
# ------------------------------------------------------------
def yolov5_postprocess(pred):
    """
    pred: tensor of shape [N, 6] where each row is [x1, y1, x2, y2, conf, cls]
    Returns a list of detections in the same format as your original code.
    """
    if pred is not None and len(pred):
        # Convert to numpy
        dets = pred.cpu().numpy()
        return dets.tolist()
    return []

# ------------------------------------------------------------
# Main function (adapted)
# ------------------------------------------------------------
def run_hybrid_timed_pipeline_yolov5(image_path, num_warmup=10):
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    detection_model = load_yolov5_model('yolov5s', device='cuda')
    detection_model = detection_model.to('cuda').eval()

    # Load classifiers (unchanged)
    classifier_1 = models.resnet18(pretrained=True).to('cuda').eval()
    classifier_2 = models.mobilenet_v2(pretrained=True).to('cuda').eval()
    classifier_3 = models.squeezenet1_0(pretrained=True).to('cuda').eval()

    CLASS_NAMES = [f"Class_{i}" for i in range(1000)]

    # Warm‑up GPU
    print("Warming up GPU...")
    dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = detection_model(dummy_input)
    torch.cuda.synchronize()

    # ---------- Detection Stage ----------
    # Pre‑process (CPU)
    t_start = time.perf_counter()
    det_input_cpu = yolov5_preprocess(original)
    t_end = time.perf_counter()
    det_pre_time_ms = (t_end - t_start) * 1000.0

    # Move to GPU
    det_input_gpu = det_input_cpu.to('cuda')

    # Inference (GPU) with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        # YOLOv5 returns a pandas-style results object, but we can call the model directly
        # The model returns a tuple: (pred, ...) or a Results object.
        # To get raw predictions, we use the model's forward method.
        # Using the model's forward returns the raw tensor output before NMS.
        # But we can also use the model's __call__ which returns a Results object with detections.
        # For simplicity, we'll use the Results object and extract detections.
        results = detection_model(det_input_gpu)  # This runs inference and NMS internally
        # Get predictions as tensor of shape [N, 6] (x1,y1,x2,y2,conf,cls)
        pred = results.xyxy[0]  # predictions for the first image (batch size 1)
    end_event.record()
    torch.cuda.synchronize()
    det_inf_time_ms = start_event.elapsed_time(end_event)

    # Post‑process (CPU)
    t_start = time.perf_counter()
    detections = yolov5_postprocess(pred)
    # Convert to crops (using your existing function)
    crops = detection_postprocess_cpu(original, detections)
    t_end = time.perf_counter()
    det_post_time_ms = (t_end - t_start) * 1000.0

    print("\n--- Detection Stage ---")
    print(f"Pre‑process (CPU)   : {det_pre_time_ms:.2f} ms")
    print(f"Inference (GPU)     : {det_inf_time_ms:.2f} ms")
    print(f"Post‑process (CPU)  : {det_post_time_ms:.2f} ms")
    print(f"Number of objects   : {len(crops)}")

    # ---------- Classification Stage (unchanged) ----------
    # ... (your existing classification code remains the same)
    total_class_pre_ms = 0.0
    total_class_inf_ms = 0.0
    total_class_post_ms = 0.0
    final_results = []

    for idx, crop in enumerate(crops):
        # Pre‑process (CPU)
        t_start = time.perf_counter()
        cls_input_cpu = classification_preprocess_cpu(crop)
        t_end = time.perf_counter()
        total_class_pre_ms += (t_end - t_start) * 1000.0

        cls_input_gpu = cls_input_cpu.to('cuda')

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

        t_start = time.perf_counter()
        label1 = classification_postprocess_cpu(out1)
        label2 = classification_postprocess_cpu(out2)
        label3 = classification_postprocess_cpu(out3)
        combined = f"{label1} | {label2} | {label3}"
        t_end = time.perf_counter()
        total_class_post_ms += (t_end - t_start) * 1000.0

        final_results.append(combined)

    print("\n--- Classification Stage (total for all crops) ---")
    print(f"Pre‑process (CPU)   : {total_class_pre_ms:.2f} ms")
    print(f"Inference (GPU)     : {total_class_inf_ms:.2f} ms")
    print(f"Post‑process (CPU)  : {total_class_post_ms:.2f} ms")

    total_time_ms = (det_pre_time_ms + det_inf_time_ms + det_post_time_ms +
                     total_class_pre_ms + total_class_inf_ms + total_class_post_ms)
    print(f"\n=== TOTAL PIPELINE LATENCY : {total_time_ms:.2f} ms ===")

    return final_results

# ------------------------------
# Keep your helper functions (unchanged)
# ------------------------------
def detection_postprocess_cpu(original_image, detections):
    crops = []
    h, w = original_image.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crops.append(original_image[y1:y2, x1:x2])
    return crops

def classification_preprocess_cpu(cropped_image, target_size=224):
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

def classification_postprocess_cpu(logits):
    probs = torch.softmax(logits, dim=1)
    _, pred_idx = torch.max(probs, 1)
    return CLASS_NAMES[pred_idx.item()]

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    results = run_hybrid_timed_pipeline_yolov5("test.jpg")
    print("\nFinal cascaded labels per object:")
    for i, label in enumerate(results):
        print(f"Object {i+1}: {label}")