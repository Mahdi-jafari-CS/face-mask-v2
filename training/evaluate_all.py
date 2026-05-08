"""
evaluate_all.py — Cross-model benchmarking and comparison chart export
Run from project root: python training/evaluate_all.py

Outputs:
  results/metrics.json     ← Used by the web UI
  results/plots/           ← Comparison charts (PNG)
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────

MODELS = {
    "yolov8n":  {"path": "models/yolov8n_v1.pt", "label": "YOLOv8n\n(Baseline)"},
    "yolov8s":  {"path": "models/yolov8s_v2.pt", "label": "YOLOv8s\n(Improved)"},
    "mobilenet": {"path": "models/mobilenet_ssd.pt", "label": "MobileNet\nSSD"},
    "rtdetr":   {"path": "models/rtdetr.pt",      "label": "RT-DETR\n(SOTA)"},
}

COLORS   = ["#64748b", "#06b6d4", "#f59e0b", "#a855f7"]
DATA_YAML = "face_mask_yolo/dataset.yaml"
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Evaluate each model ────────────────────────────────────────────────────────

def evaluate_model(model_key: str, model_info: dict) -> dict:
    model_path = model_info["path"]
    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_key}: weights not found at {model_path}")
        return None

    print(f"\n  Evaluating {model_key} ...")
    model   = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, verbose=False)

    # Speed test (average over 20 random val inferences)
    import glob, random, cv2
    val_images = glob.glob("face_mask_yolo/val/images/*.png")[:20]
    times = []
    for img_path in val_images:
        img = cv2.imread(img_path)
        t0  = time.perf_counter()
        model(img, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    speed_ms = round(float(np.mean(times)), 1)

    return {
        "label":      model_info["label"].replace("\n", " "),
        "map50":      round(float(metrics.box.map50), 4),
        "map50_95":   round(float(metrics.box.map),   4),
        "precision":  round(float(metrics.box.mp),    4),
        "recall":     round(float(metrics.box.mr),    4),
        "speed_ms":   speed_ms,
        # Per-class mAP50
        "per_class": {
            "with_mask":            round(float(metrics.box.ap50[0]), 4),
            "without_mask":         round(float(metrics.box.ap50[1]), 4),
            "mask_weared_incorrect": round(float(metrics.box.ap50[2]), 4),
        }
    }


# ── Chart: grouped bar — overall metrics ──────────────────────────────────────

def plot_overall_comparison(metrics: dict):
    keys   = list(metrics.keys())
    labels = [metrics[k]["label"].replace("\n", " ") for k in keys]
    colors = COLORS[:len(keys)]

    metric_names = ["mAP50", "mAP50-95", "Precision", "Recall"]
    metric_keys  = ["map50", "map50_95", "precision", "recall"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax, m_name, m_key in zip(axes, metric_names, metric_keys):
        vals = [metrics[k][m_key] for k in keys]
        bars = ax.bar(labels, vals, color=colors, edgecolor="none", zorder=3)
        ax.set_facecolor("#1e293b")
        ax.set_title(m_name, color="white", fontsize=13, pad=10)
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color="#334155", zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=8)

    plt.suptitle("Model Comparison — Face Mask Detection v2",
                 color="white", fontsize=15, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "overall_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Saved: {out}")


# ── Chart: speed vs accuracy scatter ──────────────────────────────────────────

def plot_speed_accuracy(metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for (key, data), color in zip(metrics.items(), COLORS):
        ax.scatter(data["speed_ms"], data["map50"],
                   color=color, s=200, zorder=5, edgecolors="white", lw=0.8)
        ax.annotate(data["label"], (data["speed_ms"], data["map50"]),
                    textcoords="offset points", xytext=(10, 5),
                    color="white", fontsize=9)

    ax.set_xlabel("Inference Speed (ms per image)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("mAP50",                          color="#94a3b8", fontsize=11)
    ax.set_title("Speed vs Accuracy Trade-off",     color="white",   fontsize=13)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.yaxis.grid(True, color="#334155", zorder=0)
    ax.xaxis.grid(True, color="#334155", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "speed_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Saved: {out}")


# ── Chart: per-class heatmap ───────────────────────────────────────────────────

def plot_per_class(metrics: dict):
    keys        = list(metrics.keys())
    labels      = [metrics[k]["label"] for k in keys]
    class_names = ["with mask", "without mask", "mask incorrect"]
    data        = np.array([
        [metrics[k]["per_class"]["with_mask"],
         metrics[k]["per_class"]["without_mask"],
         metrics[k]["per_class"]["mask_weared_incorrect"]]
        for k in keys
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    im = ax.imshow(data, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(class_names)));  ax.set_xticklabels(class_names, color="white")
    ax.set_yticks(range(len(labels)));       ax.set_yticklabels(labels, color="white")
    ax.set_title("Per-Class mAP50 Heatmap", color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)

    for i in range(len(keys)):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    color="black" if data[i,j] > 0.75 else "white", fontsize=10)

    plt.colorbar(im, ax=ax, label="mAP50").ax.yaxis.label.set_color("white")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "per_class_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Face Mask Detection v2 — Cross-Model Evaluation")
    print("=" * 55)

    all_metrics = {}
    for key, info in MODELS.items():
        result = evaluate_model(key, info)
        if result:
            all_metrics[key] = result
            # Add color for UI
            all_metrics[key]["color"] = COLORS[list(MODELS.keys()).index(key)]

    if not all_metrics:
        print("\n[!] No model weights found. Train first, then run this script.")
        sys.exit(1)

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved: {out_json}")

    # Plots
    print("\n  Generating comparison charts ...")
    plot_overall_comparison(all_metrics)
    plot_speed_accuracy(all_metrics)
    plot_per_class(all_metrics)

    print("\n  Done! Run `python app/run.py` to view in the web UI.")
