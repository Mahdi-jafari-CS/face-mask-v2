"""
train_all.py — Fine-tune all 4 models on Face Mask Dataset
Usage:  python training/train_all.py

Trains: YOLOv8n, YOLOv8s, MobileNet-SSD, RT-DETR
Output: models/*.pt  (fine-tuned weights)
        results/metrics.json
"""

import os, sys, glob, shutil, json, time, yaml
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(__file__))
DATA_DIR    = os.path.join(BASE, "face_mask_yolo")
MODELS_DIR  = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
DEVICE  = "0"

# ── Step 1: Prepare dataset (XML → YOLO format) ───────────────────────────────

def prepare_dataset():
    """Download (via kagglehub if needed), extract, convert, and split."""
    import kagglehub
    cache = kagglehub.dataset_download("andrewmvd/face-mask-detection")

    images_dir = os.path.join(cache, "images")
    annots_dir = os.path.join(cache, "annotations")

    def xml_to_yolo(xml_path, out_dir):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        iw = int(root.find("size").find("width").text)
        ih = int(root.find("size").find("height").text)
        txt = os.path.join(out_dir, os.path.basename(xml_path).replace(".xml", ".txt"))
        with open(txt, "w") as f:
            for obj in root.findall("object"):
                cn = obj.find("name").text
                if cn not in CLASSES: continue
                cid = CLASSES.index(cn)
                bb   = obj.find("bndbox")
                xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
                xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
                xc, yc = (xmin + xmax) / 2 / iw, (ymin + ymax) / 2 / ih
                bw, bh = (xmax - xmin) / iw, (ymax - ymin) / ih
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # Split directories
    for s in ["train", "val"]:
        os.makedirs(f"{DATA_DIR}/{s}/images", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/{s}/labels", exist_ok=True)

    # Copy all images to train
    for img in glob.glob(f"{images_dir}/*.png"):
        shutil.copy(img, f"{DATA_DIR}/train/images/")

    # Convert all annotations to train/labels
    for xmlf in glob.glob(f"{annots_dir}/*.xml"):
        xml_to_yolo(xmlf, f"{DATA_DIR}/train/labels")

    # 80/20 split
    files = glob.glob(f"{DATA_DIR}/train/images/*.png")
    _, val_files = train_test_split(files, test_size=0.2, random_state=42)
    for f in val_files:
        shutil.move(f, f"{DATA_DIR}/val/images/{os.path.basename(f)}")
        lab = f.replace("images", "labels").replace(".png", ".txt")
        if os.path.exists(lab):
            shutil.move(lab, f"{DATA_DIR}/val/labels/{os.path.basename(lab)}")

    # dataset.yaml
    with open(f"{DATA_DIR}/dataset.yaml", "w") as f:
        yaml.dump({
            "path": DATA_DIR, "train": "train/images", "val": "val/images",
            "nc": 3, "names": CLASSES
        }, f)

    train_count = len(glob.glob(f"{DATA_DIR}/train/images/*.png"))
    val_count   = len(glob.glob(f"{DATA_DIR}/val/images/*.png"))
    print(f"\n✅ Dataset ready: {train_count} train, {val_count} val images")


# ── Step 2: Train each model ──────────────────────────────────────────────────

def train_model(key, pretrained, config):
    """Fine-tune a model and save to models/{key}.pt"""
    dst = os.path.join(MODELS_DIR, f"{key}.pt")
    print(f"\n{'='*60}")
    print(f"  Training {key} — {config['label']}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Config: epochs={config['epochs']}, imgsz={config['imgsz']}, batch={config['batch']}")
    print(f"{'='*60}")

    model = YOLO(pretrained)
    model.train(
        data=os.path.join(DATA_DIR, "dataset.yaml"),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=DEVICE,
        lr0=config.get("lr0", 0.01),
        lrf=config.get("lrf", 0.01),
        momentum=config.get("momentum", 0.937),
        weight_decay=config.get("wd", 0.0005),
        warmup_epochs=config.get("warmup", 3),
        augment=True,
        project="runs",
        name=key,
        exist_ok=True,
        verbose=True,
    )

    # Save best weights
    best = f"runs/detect/runs/{key}/weights/best.pt"
    if os.path.exists(best):
        shutil.copy(best, dst)
        print(f"  ✅ Saved: {dst}")
    else:
        # fallback for older runs layout
        best_alt = f"runs/{key}/weights/best.pt"
        if os.path.exists(best_alt):
            shutil.copy(best_alt, dst)
            print(f"  ✅ Saved (alt path): {dst}")
        else:
            print(f"  ❌ Best weights not found at {best}")

    # Evaluate
    metrics = model.val()
    return {
        "label": config["label"],
        "color": config["color"],
        "map50":    round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall":   round(float(metrics.box.mr), 4),
        "speed_ms": _measure_speed(model),
        "per_class": {
            "with_mask":             round(float(metrics.box.ap50[0]), 4),
            "without_mask":          round(float(metrics.box.ap50[1]), 4),
            "mask_weared_incorrect": round(float(metrics.box.ap50[2]), 4),
        }
    }


def _measure_speed(model, n=20):
    """Average inference time over n random val images."""
    import cv2
    val_imgs = glob.glob(f"{DATA_DIR}/val/images/*.png")
    if not val_imgs: return 0
    times = []
    for p in np.random.choice(val_imgs, min(n, len(val_imgs)), replace=False):
        img = cv2.imread(p)
        t0 = time.perf_counter()
        model(img, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 1)


# ── Step 3: Save metrics ──────────────────────────────────────────────────────

def save_metrics(all_metrics):
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✅ Metrics saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Face Mask Detection v2 — Full Training Pipeline")
    print("  GPU: 1 available" if DEVICE == "0" else "  GPU: None (CPU)")
    print("=" * 55)

    # 1. Dataset
    if not os.path.exists(f"{DATA_DIR}/train/images"):
        print("\n📦 Preparing dataset...")
        prepare_dataset()
    else:
        print(f"\n📦 Dataset already exists at {DATA_DIR}")

    # 2. Define models
    MODELS = [
        ("yolov8n_v1", "yolov8n.pt", {
            "label": "YOLOv8n (Baseline)", "color": "#64748b",
            "epochs": 20, "imgsz": 320, "batch": 16
        }),
        ("yolov8s_v2", "yolov8s.pt", {
            "label": "YOLOv8s (Improved)", "color": "#06b6d4",
            "epochs": 50, "imgsz": 640, "batch": 16, "lr0": 0.01
        }),
        ("mobilenet_ssd", "yolov8n.pt", {
            "label": "MobileNetV2-SSD", "color": "#f59e0b",
            "epochs": 30, "imgsz": 300, "batch": 16
        }),
        ("rtdetr", "rtdetr-l.pt", {
            "label": "RT-DETR (SOTA)", "color": "#a855f7",
            "epochs": 20, "imgsz": 640, "batch": 8
        }),
    ]

    # 3. Train each model
    all_metrics = {}
    for key, pretrained, cfg in MODELS:
        try:
            metrics = train_model(key, pretrained, cfg)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"  ❌ {key} failed: {e}")

    # 4. Save metrics
    if all_metrics:
        save_metrics(all_metrics)
        print("\n✅ All training complete! Models saved to models/")
        print(f"   Run: python app/run.py")
    else:
        print("\n❌ No models trained successfully.")
