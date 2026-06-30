import time
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ------------------------------------------------------------
# YOLOv5 via torch.hub (no ultralytics package needed)
# ------------------------------------------------------------
def load_yolov5_model(weights='yolov5s', device='cuda'):
    """
    Load YOLOv5 from torch.hub. Works with Python 3.6.
    """
    # Ensure the hub is using the master branch (no conflicts)
    model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True, device=device)
    model.eval()
    return model

# ------------------------------------------------------------
# YOLOv5 preprocessing (BGR -> RGB, resize, normalize)
# ------------------------------------------------------------
def yolov5_preprocess(image, target_size=640):
    resized = cv2.resize(image, (target_size, target_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2,0,1)).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor

# ------------------------------------------------------------
# Detection post-processing (same as your original)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Classification functions (unchanged)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_hybrid_timed_pipeline_yolov5(image_path, num_warmup=10):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    detection_model = load_yolov5_model('yolov5s', device='cuda')
    detection_model = detection_model.to('cuda').eval()

    # Load classifiers
    classifier_1 = models.resnet18(pretrained=True).to('cuda').eval()
    classifier_2 = models.mobilenet_v2(pretrained=True).to('cuda').eval()
    classifier_3 = models.squeezenet1_0(pretrained=True).to('cuda').eval()

    global CLASS_NAMES
    CLASS_NAMES = [f"Class_{i}" for i in range(1000)]

    # Warm‑up
    print("Warming up GPU...")
    dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = detection_model(dummy_input)
    torch.cuda.synchronize()

    # ---------- Detection Stage ----------
    t_start = time.perf_counter()
    det_input_cpu = yolov5_preprocess(original)
    t_end = time.perf_counter()
    det_pre_time_ms = (t_end - t_start) * 1000.0

    det_input_gpu = det_input_cpu.to('cuda')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        # YOLOv5 via torch.hub returns a Results object with .xyxy[0] attribute
        results = detection_model(det_input_gpu)
        # Extract predictions
        if hasattr(results, 'xyxy'):
            pred = results.xyxy[0]
        else:
            # fallback: if it's a list/tuple, take first element
            pred = results[0].xyxy[0] if isinstance(results, tuple) else np.empty((0, 6))
    end_event.record()
    torch.cuda.synchronize()
    det_inf_time_ms = start_event.elapsed_time(end_event)

    t_start = time.perf_counter()
    if isinstance(pred, torch.Tensor):
        detections = pred.cpu().numpy().tolist()
    elif isinstance(pred, np.ndarray):
        detections = pred.tolist()
    else:
        detections = []
    crops = detection_postprocess_cpu(original, detections)
    t_end = time.perf_counter()
    det_post_time_ms = (t_end - t_start) * 1000.0

    print("\n--- Detection Stage ---")
    print(f"Pre‑process (CPU)   : {det_pre_time_ms:.2f} ms")
    print(f"Inference (GPU)     : {det_inf_time_ms:.2f} ms")
    print(f"Post‑process (CPU)  : {det_post_time_ms:.2f} ms")
    print(f"Number of objects   : {len(crops)}")

    # ---------- Classification Stage (unchanged) ----------
    total_class_pre_ms = 0.0
    total_class_inf_ms = 0.0
    total_class_post_ms = 0.0
    final_results = []

    for idx, crop in enumerate(crops):
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

if __name__ == "__main__":
    results = run_hybrid_timed_pipeline_yolov5("test.jpg")
    print("\nFinal cascaded labels per object:")
    for i, label in enumerate(results):
        print(f"Object {i+1}: {label}")