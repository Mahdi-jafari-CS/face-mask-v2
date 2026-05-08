"""
detector.py — Core inference engine for Face Mask Detection v2
Supports: YOLOv8n, YOLOv8s, MobileNet-SSD, RT-DETR
Improvements:
  - CLAHE contrast enhancement for low-light / low-contrast images
  - Multi-scale detection (0.7x, 1.0x, 1.3x) with NMS merging
  - Catches small / occluded faces missed by single-scale inference
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

CLASS_COLORS = {
    "with_mask":              (0,   210,  90),   # green
    "without_mask":           (220,  50,  50),   # red
    "mask_weared_incorrect":  (220, 180,   0),   # amber
}

# ── Model registry — update paths after downloading weights from Colab ─────────
MODEL_REGISTRY = {
    "yolov8n": {
        "path": "models/yolov8n_v1.pt",
        "label": "YOLOv8n (Baseline)",
        "color": "#64748b",
        "framework": "ultralytics",
    },
    "yolov8s": {
        "path": "models/yolov8s_v2.pt",
        "label": "YOLOv8s (Improved)",
        "color": "#06b6d4",
        "framework": "ultralytics",
    },
    "mobilenet": {
        "path": "models/mobilenet_ssd.pt",
        "label": "MobileNetV2-SSD",
        "color": "#f59e0b",
        "framework": "ultralytics",   # trained via ultralytics custom head
    },
    "rtdetr": {
        "path": "models/rtdetr.pt",
        "label": "RT-DETR (SOTA)",
        "color": "#a855f7",
        "framework": "ultralytics",
    },
}

# ── Lazy-loaded model cache ────────────────────────────────────────────────────
_model_cache: dict = {}


def load_model(model_key: str) -> YOLO:
    """Load and cache a YOLO/RT-DETR model by registry key."""
    if model_key not in _model_cache:
        info = MODEL_REGISTRY[model_key]
        model_path = info["path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                f"Please train and download from Colab, then copy to models/"
            )

        print(f"[detector] Loading {info['label']} from {model_path} ...")
        _model_cache[model_key] = YOLO(model_path)
        print(f"[detector] {info['label']} ready.")

    return _model_cache[model_key]


# ── Image preprocessing ─────────────────────────────────────────────────────────

def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    local contrast. Helps detect faces in low-light, backlit, or low-contrast
    images where standard models often miss faces.
    """
    # Convert to LAB color space — lightness channel L holds luminance
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on the lightness channel: clip limit 2.0, tile grid 8×8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Merge back and convert to BGR
    eq_lab = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(eq_lab, cv2.COLOR_LAB2BGR)


# ── NMS (Non-Maximum Suppression) ──────────────────────────────────────────────

def _nms(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Deduplicate overlapping bounding boxes using IoU-based NMS.
    Keeps the detection with the highest confidence per cluster.
    """
    if not detections:
        return []

    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["confidence"] for d in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


# ── Multi-scale inference ──────────────────────────────────────────────────────

_MULTI_SCALES = [0.70, 1.00, 1.35]  # small / original / large


def run_inference(image_bytes: bytes, model_key: str, conf_threshold: float = 0.25):
    """
    Run detection on raw image bytes with accuracy improvements:
      1. CLAHE contrast enhancement on the input image.
      2. Multi-scale detection (3 scales) to catch faces at different sizes.
      3. NMS deduplication to merge overlapping detections from all scales.

    Returns:
        dict with keys:
            - model_key, model_label
            - inference_time_ms
            - detections: list of {class_name, confidence, bbox [x1,y1,x2,y2]}
            - counts: {class_name: count}
            - annotated_image: base64-encoded JPEG with drawn boxes
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img_bgr.shape[:2]

    model = load_model(model_key)

    # ── Step 1: CLAHE contrast enhancement ────────────────────────────────────
    enhanced = _apply_clahe(img_bgr)

    # ── Step 2: Multi-scale detection ─────────────────────────────────────────
    t0 = time.perf_counter()
    all_detections = []

    for scale in _MULTI_SCALES:
        if scale == 1.0:
            scaled_img = enhanced
        else:
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            scaled_img = cv2.resize(enhanced, (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR)

        # Use a lower internal conf for multi-scale to capture more candidates;
        # the user's conf_threshold is applied as a final filter after NMS.
        internal_conf = max(conf_threshold * 0.6, 0.10)
        results = model(scaled_img, conf=internal_conf, verbose=False)

        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

            # Scale coordinates back to original image
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            if scale != 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale

            # Clip to image boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_orig, int(x2))
            y2 = min(h_orig, int(y2))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            all_detections.append({
                "class_name":  class_name,
                "confidence":  round(confidence, 3),
                "bbox":        [x1, y1, x2, y2],
            })

    # ── Step 3: NMS deduplication ─────────────────────────────────────────────
    merged = _nms(all_detections, iou_threshold=0.5)

    # ── Step 4: Apply user's confidence threshold ─────────────────────────────
    final = [d for d in merged if d["confidence"] >= conf_threshold]

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Counts ────────────────────────────────────────────────────────────────
    counts = {c: 0 for c in CLASSES}
    for det in final:
        if det["class_name"] in counts:
            counts[det["class_name"]] += 1

    # ── Draw annotated image ──────────────────────────────────────────────────
    annotated = _draw_boxes(img_bgr.copy(), final)
    annotated_b64 = _encode_to_base64(annotated)

    print(f"[detector] {MODEL_REGISTRY[model_key]['label']}: "
          f"{len(final)} faces ({len(all_detections)} candidates → "
          f"{len(merged)} after NMS) in {inference_ms}ms")

    return {
        "model_key":        model_key,
        "model_label":      MODEL_REGISTRY[model_key]["label"],
        "inference_time_ms": inference_ms,
        "detections":       final,
        "counts":           counts,
        "total_faces":      len(final),
        "annotated_image":  annotated_b64,
    }


def run_comparison(image_bytes: bytes, model_keys: list, conf_threshold: float = 0.25):
    """Run the same image through multiple models for side-by-side comparison."""
    results = {}
    for key in model_keys:
        try:
            results[key] = run_inference(image_bytes, key, conf_threshold)
        except FileNotFoundError as e:
            results[key] = {"error": str(e), "model_key": key,
                            "model_label": MODEL_REGISTRY[key]["label"]}
    return results


def get_available_models():
    """Return registry info for the frontend, marking which weights exist."""
    available = {}
    for key, info in MODEL_REGISTRY.items():
        available[key] = {
            **info,
            "loaded":   key in _model_cache,
            "exists":   os.path.exists(info["path"]),
        }
    return available


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes with labels on the image."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name     = det["class_name"]
        confidence     = det["confidence"]
        color          = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        label    = f"{class_name.replace('_', ' ')}  {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return img


def _encode_to_base64(img: np.ndarray) -> str:
    """Encode a BGR numpy image to a base64 data URL."""
    import base64
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
