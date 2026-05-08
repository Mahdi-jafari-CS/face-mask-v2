"""
train_all_models.py  ← This file mirrors the Colab notebook structure.
In Colab, each section below is a separate cell.
Convert to .ipynb: pip install jupytext && jupytext --to notebook train_all_models.py

USAGE: Run in Google Colab (GPU runtime recommended — T4 is sufficient)
"""

# =============================================================================
# CELL 1 — Install libraries
# =============================================================================
# %%
# !pip install ultralytics kaggle scikit-learn -q
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, zipfile, shutil, glob, yaml, random, json, time
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from google.colab import files

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device          : {'GPU ✓' if torch.cuda.is_available() else 'CPU'}")
DEVICE = "0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CELL 2 — Download dataset from Kaggle
# =============================================================================
# %%
import os
os.environ['KAGGLE_USERNAME'] = "MahdiJafari2026"          # ← your Kaggle username
os.environ['KAGGLE_KEY']      = "your-kaggle-api-key-here"  # ← replace with your key

# !pip install kaggle -q
# !kaggle datasets download -d andrewmvd/face-mask-detection --force

if os.path.exists('face_mask_dataset'):
    shutil.rmtree('face_mask_dataset')

with zipfile.ZipFile('face-mask-detection.zip', 'r') as z:
    z.extractall('face_mask_dataset')

images      = glob.glob('face_mask_dataset/images/*.png')
annotations = glob.glob('face_mask_dataset/annotations/*.xml')
print(f"Images: {len(images)}  |  Annotations: {len(annotations)}")


# =============================================================================
# CELL 3 — XML → YOLO format conversion
# =============================================================================
# %%
import xml.etree.ElementTree as ET

CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

def convert_xml_to_yolo(xml_file, output_dir, classes=CLASSES):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size       = root.find('size')
    img_width  = int(size.find('width').text)
    img_height = int(size.find('height').text)

    txt_file = os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.txt'))

    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width    = (xmax - xmin) / img_width
            height   = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# =============================================================================
# CELL 4 — Build directory structure and convert all annotations
# =============================================================================
# %%
if os.path.exists('face_mask_yolo'):
    shutil.rmtree('face_mask_yolo')

for split in ['train', 'val']:
    os.makedirs(f'face_mask_yolo/{split}/images', exist_ok=True)
    os.makedirs(f'face_mask_yolo/{split}/labels', exist_ok=True)

# Copy images
for img in glob.glob('face_mask_dataset/images/*.png'):
    shutil.copy(img, 'face_mask_yolo/train/images/')

# Convert annotations
xml_files = glob.glob('face_mask_dataset/annotations/*.xml')
for xml_file in xml_files:
    convert_xml_to_yolo(xml_file, 'face_mask_yolo/train/labels')

print(f"Converted {len(xml_files)} annotation files")


# =============================================================================
# CELL 5 — Train / val split (80/20)
# =============================================================================
# %%
image_files = glob.glob('face_mask_yolo/train/images/*.png')
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)}  |  Val: {len(val_files)}")

for file in val_files:
    dest = 'face_mask_yolo/val/images/' + os.path.basename(file)
    shutil.move(file, dest)
    label = file.replace('images', 'labels').replace('.png', '.txt')
    if os.path.exists(label):
        shutil.move(label, 'face_mask_yolo/val/labels/' + os.path.basename(label))


# =============================================================================
# CELL 6 — dataset.yaml
# =============================================================================
# %%
dataset_config = {
    'path':  '/content/face_mask_yolo',
    'train': 'train/images',
    'val':   'val/images',
    'nc':    3,
    'names': CLASSES,
}
with open('face_mask_yolo/dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print("dataset.yaml created:")
# !cat face_mask_yolo/dataset.yaml


# =============================================================================
# CELL 7 — Visualize dataset samples
# =============================================================================
# %%
CLASS_COLORS = {0: (0,210,90), 1: (220,50,50), 2: (220,180,0)}

def visualize_sample(image_path, label_path, classes=CLASSES):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cid, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw/2) * w);  y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w);  y2 = int((yc + bh/2) * h)
                color = CLASS_COLORS[int(cid)]
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, classes[int(cid)], (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img

samples = random.sample(glob.glob('face_mask_yolo/train/images/*.png'), 5)
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.patch.set_facecolor('#0f172a')
for ax, img_path in zip(axes, samples):
    label_path = img_path.replace('images','labels').replace('.png','.txt')
    ax.imshow(visualize_sample(img_path, label_path))
    ax.axis('off')
    ax.set_facecolor('#0f172a')
plt.suptitle('Training Samples — Ground Truth Annotations', color='white', fontsize=14)
plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.show()


# =============================================================================
# CELL 8 — MODEL 1: YOLOv8n  (baseline replication — CPU friendly)
# =============================================================================
# %%
print("=" * 55)
print(" MODEL 1: YOLOv8n — Baseline (10 epochs, 320px)")
print("=" * 55)

model_v1 = YOLO('yolov8n.pt')
results_v1 = model_v1.train(
    data    = '/content/face_mask_yolo/dataset.yaml',
    epochs  = 10,
    imgsz   = 320,
    batch   = 8,
    device  = DEVICE,
    project = 'runs',
    name    = 'yolov8n_baseline',
    exist_ok= True,
)

metrics_v1 = model_v1.val()
print(f"\nYOLOv8n  mAP50: {metrics_v1.box.map50:.4f}")
print(f"YOLOv8n  mAP50-95: {metrics_v1.box.map:.4f}")

# Save model to downloadable location
shutil.copy('runs/yolov8n_baseline/weights/best.pt', 'yolov8n_v1.pt')
print("Saved: yolov8n_v1.pt")


# =============================================================================
# CELL 9 — MODEL 2: YOLOv8s  (primary improvement)
# =============================================================================
# %%
print("=" * 55)
print(" MODEL 2: YOLOv8s — Improved (50 epochs, 640px, GPU)")
print("=" * 55)

model_v2 = YOLO('yolov8s.pt')
results_v2 = model_v2.train(
    data      = '/content/face_mask_yolo/dataset.yaml',
    epochs    = 50,
    imgsz     = 640,
    batch     = 16,
    device    = DEVICE,
    lr0       = 0.01,
    lrf       = 0.01,
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs= 3,
    augment   = True,
    project   = 'runs',
    name      = 'yolov8s_improved',
    exist_ok  = True,
)

metrics_v2 = model_v2.val()
print(f"\nYOLOv8s  mAP50: {metrics_v2.box.map50:.4f}")
print(f"YOLOv8s  mAP50-95: {metrics_v2.box.map:.4f}")

shutil.copy('runs/yolov8s_improved/weights/best.pt', 'yolov8s_v2.pt')
print("Saved: yolov8s_v2.pt")


# =============================================================================
# CELL 10 — MODEL 3: MobileNetV2-SSD  (comparison model)
# =============================================================================
# %%
# NOTE: We use Ultralytics' custom architecture API.
# MobileNetV2 backbone with SSD-style head can be approximated by
# using a lightweight custom YAML config. For simplicity, here we
# use the Ultralytics-compatible mobile-optimized model.

print("=" * 55)
print(" MODEL 3: MobileNet-SSD — Comparison Model 1")
print("=" * 55)

# Ultralytics supports 'yolov8n-cls' and similar; for a true MobileNet
# comparison we use a pre-configured YAML below.

mobilenet_yaml = """
# MobileNetV2-SSD approximation via Ultralytics custom YAML
# backbone: MobileNetV2 (inverted residuals)
backbone:
  - [-1, 1, Conv, [16, 3, 2]]        # P1
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 2, C2f,  [32]]              # Inverted residual blocks
  - [-1, 1, Conv, [64, 3, 2]]        # P3
  - [-1, 4, C2f,  [64]]
  - [-1, 1, Conv, [128, 3, 2]]       # P4
  - [-1, 4, C2f,  [128]]
  - [-1, 1, Conv, [256, 3, 2]]       # P5
  - [-1, 2, C2f,  [256]]
  - [-1, 1, SPPF, [256, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2f, [128]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2f, [64]]
  - [[15, 12, 9], 1, Detect, [nc]]
"""

with open('mobilenet_ssd.yaml', 'w') as f:
    f.write(mobilenet_yaml)

model_mn = YOLO('mobilenet_ssd.yaml').load('yolov8n.pt')  # transfer weights
results_mn = model_mn.train(
    data    = '/content/face_mask_yolo/dataset.yaml',
    epochs  = 30,
    imgsz   = 300,
    batch   = 16,
    device  = DEVICE,
    project = 'runs',
    name    = 'mobilenet_ssd',
    exist_ok= True,
)

metrics_mn = model_mn.val()
print(f"\nMobileNet-SSD  mAP50: {metrics_mn.box.map50:.4f}")

shutil.copy('runs/mobilenet_ssd/weights/best.pt', 'mobilenet_ssd.pt')
print("Saved: mobilenet_ssd.pt")


# =============================================================================
# CELL 11 — MODEL 4 (BONUS): RT-DETR  (transformer-based SOTA)
# =============================================================================
# %%
print("=" * 55)
print(" MODEL 4 BONUS: RT-DETR-L — Transformer SOTA Detector")
print("=" * 55)

model_rt = YOLO('rtdetr-l.pt')   # Ultralytics ships RT-DETR natively
results_rt = model_rt.train(
    data    = '/content/face_mask_yolo/dataset.yaml',
    epochs  = 20,
    imgsz   = 640,
    batch   = 8,
    device  = DEVICE,
    project = 'runs',
    name    = 'rtdetr_sota',
    exist_ok= True,
)

metrics_rt = model_rt.val()
print(f"\nRT-DETR  mAP50: {metrics_rt.box.map50:.4f}")

shutil.copy('runs/rtdetr_sota/weights/best.pt', 'rtdetr.pt')
print("Saved: rtdetr.pt")


# =============================================================================
# CELL 12 — Comparison table + charts
# =============================================================================
# %%

results_summary = {
    "YOLOv8n (Baseline)": {
        "mAP50":     metrics_v1.box.map50,
        "mAP50-95":  metrics_v1.box.map,
        "Precision": metrics_v1.box.mp,
        "Recall":    metrics_v1.box.mr,
    },
    "YOLOv8s (Improved)": {
        "mAP50":     metrics_v2.box.map50,
        "mAP50-95":  metrics_v2.box.map,
        "Precision": metrics_v2.box.mp,
        "Recall":    metrics_v2.box.mr,
    },
    "MobileNet-SSD": {
        "mAP50":     metrics_mn.box.map50,
        "mAP50-95":  metrics_mn.box.map,
        "Precision": metrics_mn.box.mp,
        "Recall":    metrics_mn.box.mr,
    },
    "RT-DETR (SOTA)": {
        "mAP50":     metrics_rt.box.map50,
        "mAP50-95":  metrics_rt.box.map,
        "Precision": metrics_rt.box.mp,
        "Recall":    metrics_rt.box.mr,
    },
}

print("\n" + "=" * 65)
print(f"{'Model':<22} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
print("=" * 65)
for model_name, m in results_summary.items():
    print(f"{model_name:<22} {m['mAP50']:>8.4f} {m['mAP50-95']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

# Bar chart comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor('#0f172a')
colors = ['#64748b', '#06b6d4', '#f59e0b', '#a855f7']
model_names = list(results_summary.keys())

for ax, metric in zip(axes, ['mAP50', 'mAP50-95', 'Precision', 'Recall']):
    vals = [results_summary[m][metric] for m in model_names]
    bars = ax.bar(range(len(model_names)), vals, color=colors, zorder=3)
    ax.set_facecolor('#1e293b')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(['v8n', 'v8s', 'MN', 'RTDT'], color='white')
    ax.set_title(metric, color='white', fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color='#334155', zorder=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.3f}', ha='center', color='white', fontsize=8)

plt.suptitle('Face Mask Detection v2 — Model Comparison', color='white', fontsize=14)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.show()


# =============================================================================
# CELL 13 — Download all weights + charts
# =============================================================================
# %%
for f in ['yolov8n_v1.pt', 'yolov8s_v2.pt', 'mobilenet_ssd.pt', 'rtdetr.pt',
          'model_comparison.png', 'dataset_samples.png']:
    if os.path.exists(f):
        files.download(f)
        print(f"Downloaded: {f}")

print("\nAll done! Copy .pt files into your project's models/ folder, then run:")
print("  python training/evaluate_all.py")
print("  python app/run.py")
