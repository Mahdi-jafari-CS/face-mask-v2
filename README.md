<div align="center">

# 🎭 Face Mask Detection v2

**A Production-Grade Computer Vision System for Real-Time Face Mask Detection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00BBFF?logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-Web%20Demo-000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-CLAHE%20%7C%20Multi--Scale-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Colab](https://img.shields.io/badge/Training-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com)

**Mahdi Jafari** · 24B032230 · KBTU · Introduction to Computer Vision · 2026

[🌐 Web Demo](#-quick-start) • [📊 Results](#-results-at-a-glance) • [🧠 Models](#-models-compared) • [📖 Documentation](#-project-structure)

---

</div>

## 📌 Overview

This project implements a **real-time face mask detection system** that compares **four distinct deep learning architectures** — from lightweight mobile-optimized models to state-of-the-art transformer-based detectors. Built as the final project for KBTU's *Introduction to Computer Vision* course, it features a complete pipeline including data preparation, model training, cross-model evaluation, and an interactive web demo with drag-and-drop image upload.

The system detects three classes:
- ✅ **with_mask** — Properly worn face mask
- ❌ **without_mask** — No face mask detected
- ⚠️ **mask_weared_incorrect** — Mask worn below nose, on chin, etc.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **4 Architectures** | YOLOv8n, YOLOv8s, MobileNetV2-SSD, RT-DETR-L (Transformer) |
| **CLAHE Enhancement** | Adaptive histogram equalization for low-light / low-contrast images |
| **Multi-Scale Detection** | 3-scale inference (0.7×, 1.0×, 1.35×) with NMS merging — catches faces at any size |
| **Web UI Demo** | Flask app with drag-and-drop, live confidence sliders, side-by-side model comparison |
| **Bulk Processing** | Run detection on multiple images across multiple models simultaneously |
| **Colab Training** | One-click training notebook with automatic weight download |
| **Evaluation Suite** | Cross-model benchmarking with publication-quality comparison charts |
| **Export** | Results exportable as JSON with annotated images (base64) |

---

## 🏆 Results at a Glance

**Actual metrics from trained models on the Kaggle Face Mask Detection validation set:**

| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Speed | Size |
|:----:|-------|:-----:|:--------:|:---------:|:------:|:----:|:----:|
| 🥇 | **YOLOv8s (Improved)** | **0.8852** | **0.6086** | **0.9409** | **0.8115** | 30 ms | 22 MB |
| 🥈 | **RT-DETR-L (SOTA)** | 0.8274 | 0.5732 | 0.9139 | 0.7435 | 60 ms | 67 MB |
| 🥉 | MobileNetV2-SSD | 0.7287 | 0.4678 | 0.8381 | 0.6519 | 8 ms | 3.4 MB |
| 4 | YOLOv8n (Baseline) | 0.6962 | 0.4389 | 0.7138 | 0.6829 | 18 ms | 6 MB |

### Per-Class Performance (mAP50)

| Model | with_mask | without_mask | mask_weared_incorrect |
|-------|:---------:|:------------:|:---------------------:|
| **YOLOv8s** | **0.9678** | **0.8924** | **0.7954** |
| RT-DETR-L | 0.9629 | 0.8829 | 0.6364 |
| MobileNet-SSD | 0.9033 | 0.7391 | 0.5438 |
| YOLOv8n | 0.8948 | 0.6804 | 0.5133 |

> 💡 **Key Insight:** YOLOv8s achieves the best overall accuracy/speed trade-off. The `mask_weared_incorrect` class remains the hardest due to high intra-class variation (mask on chin, below nose, covering one side) and fewer training examples (~11% of data).

---

## 🖼️ Web Application Demo

The project includes a **modern web interface** built with Flask for interactive face mask detection:

- **Drag & Drop** image upload with instant feedback
- **Confidence Threshold** slider for fine-tuning detection sensitivity
- **Select any combination** of the 4 models for side-by-side comparison
- **Annotated results** with color-coded bounding boxes:
  - 🟢 Green = `with_mask`
  - 🔴 Red = `without_mask`
  - 🟡 Amber = `mask_weared_incorrect`
- **Bulk image processing** — upload multiple images at once
- **Leaderboard** with real-time metrics visualization
- **Interactive charts** showing per-class confidence scores

```bash
python app/run.py
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
face-mask-v2/
│
├── README.md                      ← This file
├── RESULTS.md                     ← Detailed evaluation criteria & expected results
├── requirements.txt               ← Python dependencies
│
├── app/                           ← Web application
│   ├── run.py                     ← Flask entry point
│   ├── detector.py                ← Core inference engine (CLAHE, multi-scale, NMS)
│   ├── templates/
│   │   └── index.html             ← Demo UI (single-page app)
│   └── static/
│       ├── css/
│       └── js/
│
├── colab/
│   └── train_all_models.py        ← Google Colab training notebook (script form)
│
├── training/
│   ├── train_all.py               ← Local training pipeline (all 4 models)
│   └── evaluate_all.py            ← Cross-model evaluation + charts
│
├── face_mask_yolo/                ← Dataset (YOLO format)
│   ├── dataset.yaml               ← Dataset configuration
│   ├── train/                     ← 682 training images + labels
│   └── val/                       ← 171 validation images + labels
│
├── models/                        ← Trained weights
│   ├── yolov8n_v1.pt              ← YOLOv8n baseline
│   ├── yolov8s_v2.pt              ← YOLOv8s improved
│   ├── mobilenet_ssd.pt           ← MobileNetV2-SSD
│   └── rtdetr.pt                  ← RT-DETR (transformer)
│
└── results/                       ← Evaluation outputs
    ├── metrics.json               ← Benchmark metrics (used by UI)
    └── plots/                     ← Comparison charts
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Mahdi-jafari-CS/face-mask-v2.git
cd face-mask-v2

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Weights

You can either:
- **Train from scratch** (see [Training](#-training) below), or
- Use the `.pt` files included in the `models/` directory

For RT-DETR, download the pretrained checkpoint:
```bash
# Ultralytics will auto-download the base weights on first use
```

---

## 🚀 Usage

### ▶️ Run the Web Demo

```bash
python app/run.py
```

Then open **http://localhost:5000** in your browser. Upload images, select which models to compare, adjust the confidence threshold, and click **"Run Detection"**.

### 🎯 Run Inference Programmatically

```python
from app.detector import run_inference

# Load your image
with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()

# Run detection with YOLOv8s
result = run_inference(image_bytes, model_key="yolov8s", conf_threshold=0.25)

print(f"Found {result['total_faces']} faces in {result['inference_time_ms']}ms")
for det in result['detections']:
    print(f"  {det['class_name']}: {det['confidence']:.3f}")
```

### 📊 Evaluate & Compare All Models

```bash
python training/evaluate_all.py
```

This outputs:
- `results/metrics.json` — Structured benchmark data
- `results/plots/overall_comparison.png` — Grouped bar chart
- `results/plots/speed_accuracy.png` — Speed vs accuracy scatter plot
- `results/plots/per_class_heatmap.png` — Per-class mAP50 heatmap

---

## 🧠 Models Compared

| Model | Architecture | Params | Epochs | Image Size | Paradigm | Purpose |
|-------|-------------|:------:|:------:|:----------:|:--------:|---------|
| **YOLOv8n** | CNN + C2f + PANet | 3.2M | 20 | 320px | Anchor-free one-stage | Baseline replication |
| **YOLOv8s** | CNN + C2f + PANet | 11.2M | 50 | 640px | Anchor-free one-stage | **Primary improvement** |
| **MobileNetV2-SSD** | Inverted Residuals + SSD | ~3.4M | 30 | 300px | Anchor-based one-stage | Lightweight comparison |
| **RT-DETR-L** | Transformer Encoder-Decoder | 32M | 20 | 640px | Transformer (no NMS) | **SOTA bonus** |

### Why These Four?

- **YOLOv8n → YOLOv8s**: Same family, different capacity — isolates the effect of model size on performance
- **MobileNetV2-SSD**: Different architecture paradigm (anchor-based, depthwise convolutions) — tests if lightweight models can compete
- **RT-DETR**: Transformer-based detector (no NMS needed, attention-based global context) — shows awareness of cutting-edge CV research

---

## 🔬 Technical Deep Dive

### 1. CLAHE Contrast Enhancement

Before inference, each image undergoes **Contrast Limited Adaptive Histogram Equalization (CLAHE)** in the LAB color space. This enhances local contrast without amplifying noise in uniform regions.

```python
# Simplified: CLAHE on lightness channel
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_eq = clahe.apply(l)
```

**Why it matters:** Standard models miss faces in low-light, backlit, or hazy images. CLAHE normalizes illumination across the image, making detection **more robust to real-world conditions**.

### 2. Multi-Scale Detection

The detector runs inference at **three scales** (0.7×, 1.0×, 1.35×) and merges results:

| Scale | Purpose |
|:-----:|---------|
| **0.7×** | Catches large faces that dominate the frame |
| **1.0×** | Standard detection at original resolution |
| **1.35×** | Catches small / distant faces missed at lower scales |

### 3. NMS Deduplication

All detections from all scales are merged using **IoU-based Non-Maximum Suppression** (threshold: 0.5). This eliminates duplicate boxes around the same face while keeping the highest-confidence prediction.

**Result:** More faces detected (especially small ones) with fewer false positives.

---

## 🗂️ Dataset

**Source:** [Kaggle — Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (andrewmvd)

- **853** images with bounding box annotations in PASCAL VOC XML format
- **3 classes** with instance distribution:
  - `with_mask`: 4,443 instances (~65%)
  - `without_mask`: 1,647 instances (~24%)
  - `mask_weared_incorrect`: 753 instances (~11%)
- **Split:** 80/20 → 682 train / 171 validation images
- **Format:** Converted from XML to YOLO `.txt` format

### Annotation Format (YOLO)

```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized to [0, 1]:

```
x_center = (xmin + xmax) / 2 / image_width
y_center = (ymin + ymax) / 2 / image_height
width    = (xmax - xmin) / image_width
height   = (ymax - ymin) / image_height
```

---

## 🏋️ Training

### Option 1: Google Colab (Recommended)

Open and run the Colab-compatible script:

```bash
# Convert to notebook and upload to Colab
pip install jupytext
jupytext --to notebook colab/train_all_models.py -o face_mask_training.ipynb
```

The notebook handles everything:
1. Installs dependencies
2. Downloads the Kaggle dataset
3. Converts XML → YOLO format
4. Trains all 4 models sequentially
5. Downloads trained `.pt` weights to your machine
6. Generates comparison charts

### Option 2: Local Training

```bash
python training/train_all.py
```

Trains all models locally (GPU recommended for YOLOv8s and RT-DETR).

---

## 📋 Requirements

```
flask
flask-cors
opencv-python
numpy
ultralytics
scikit-learn
pyyaml
kagglehub
```

Install with: `pip install -r requirements.txt`

---

## 📚 API Endpoints

The web server exposes a REST API for integration with external applications:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available models and their status |
| `/api/detect` | POST | Run detection with a single model |
| `/api/compare` | POST | Run comparison across multiple models |
| `/api/compare_bulk` | POST | Bulk detection on multiple images |
| `/api/metrics` | GET | Retrieve benchmark metrics |

---

## 📊 Grading Rubric Addressed

| Criteria | Implementation |
|----------|---------------|
| **Model improvement** | YOLOv8s → YOLOv8n: **+27% mAP50** improvement |
| **Model complexity** | 4 architectures across 3 paradigms (CNN, MobileNet, Transformer) |
| **≥2 model comparison** | YOLOv8n vs YOLOv8s vs MobileNet vs RT-DETR |
| **UI for upload + results** | Flask web app with drag-and-drop, side-by-side comparison |
| **Report quality** | This README + RESULTS.md with detailed analysis |
| **Bonus: SOTA model** | RT-DETR (transformer-based detector, no NMS needed) |
| **Bonus: Deployment** | Deployable via `python app/run.py` |
| **Bonus: Unseen data** | Custom image upload in web UI |

---

## 📄 License

This project is developed for educational purposes as part of the **Introduction to Computer Vision** course at **Kazakh-British Technical University (KBTU)**.

---

<div align="center">

**Made with 🎭 by Mahdi Jafari** · 24B032230 · KBTU · 2026

[![GitHub](https://img.shields.io/badge/GitHub-Mahdi--jafari--CS-181717?logo=github&logoColor=white)](https://github.com/Mahdi-jafari-CS/face-mask-v2)

</div>
