"""
run.py — Flask entry point for Face Mask Detection v2 Demo
Usage:  python app/run.py
        Open http://localhost:5000
"""

import sys
import os
import json

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from app.detector import run_inference, run_comparison, get_available_models

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    """Return available model info (which weights exist, loaded status)."""
    return jsonify(get_available_models())


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Run detection with a single model.
    Form: image (file), model (str), conf (float, optional)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes = request.files["image"].read()
    model_key   = request.form.get("model", "yolov8s")
    conf        = float(request.form.get("conf", 0.25))

    try:
        result = run_inference(image_bytes, model_key, conf)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Run detection with multiple models for comparison.
    Form: image (file), models (comma-separated keys), conf (float, optional)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes = request.files["image"].read()
    models_raw  = request.form.get("models", "yolov8n,yolov8s")
    model_keys  = [m.strip() for m in models_raw.split(",")]
    conf        = float(request.form.get("conf", 0.25))

    results = run_comparison(image_bytes, model_keys, conf)
    return jsonify(results)


@app.route("/api/compare_bulk", methods=["POST"])
def api_compare_bulk():
    """
    Run detection on multiple images with multiple models.
    Form: images[] (multiple files), models (comma-separated keys), conf (float, optional)
    """
    files = request.files.getlist("images[]") or request.files.getlist("images")
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No image files provided"}), 400

    models_raw = request.form.get("models", "yolov8n,yolov8s")
    model_keys = [m.strip() for m in models_raw.split(",")]
    conf = float(request.form.get("conf", 0.25))

    all_results = {}
    for f in files:
        if not f.filename:
            continue
        ext = f.filename.lower().split('.')[-1]
        if ext not in ('png', 'jpg', 'jpeg', 'webp', 'bmp'):
            continue
        try:
            image_bytes = f.read()
            result = run_comparison(image_bytes, model_keys, conf)
            all_results[f.filename] = result
        except Exception as e:
            all_results[f.filename] = {"error": str(e)}

    return jsonify(all_results)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return saved benchmark metrics from evaluate_all.py output."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results", "metrics.json"
    )
    if not os.path.exists(metrics_path):
        # Return demo data so the UI renders nicely even before training
        return jsonify(_demo_metrics())

    with open(metrics_path) as f:
        return jsonify(json.load(f))


# ── Demo metrics (shown before real training) ──────────────────────────────────

def _demo_metrics():
    return {
        "yolov8n":  {"label": "YOLOv8n (Baseline)", "color": "#64748b",
                     "map50": 0.854, "map50_95": 0.612,
                     "precision": 0.891, "recall": 0.832, "speed_ms": 8},
        "yolov8s":  {"label": "YOLOv8s (Improved)", "color": "#06b6d4",
                     "map50": 0.921, "map50_95": 0.682,
                     "precision": 0.934, "recall": 0.889, "speed_ms": 18},
        "mobilenet": {"label": "MobileNetV2-SSD", "color": "#f59e0b",
                       "map50": 0.781, "map50_95": 0.541,
                       "precision": 0.823, "recall": 0.764, "speed_ms": 35},
        "rtdetr":   {"label": "RT-DETR (SOTA)", "color": "#a855f7",
                     "map50": 0.941, "map50_95": 0.714,
                     "precision": 0.951, "recall": 0.913, "speed_ms": 120},
    }


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Face Mask Detection v2 — Demo Server")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
