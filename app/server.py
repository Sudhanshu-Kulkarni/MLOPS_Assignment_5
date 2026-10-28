"""Flask web app for CIFAR-10 image classification (Docker + AWS EC2 deployment).

Endpoints:
  GET  /          — HTML upload form
  GET  /health    — JSON health check (verifies model loads)
  POST /predict   — accepts image file, returns JSON prediction
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from src.utils import class_names_cifar10, load_json, load_params

app = Flask(__name__)

# Lazy-loaded globals — populated on first request
_selector = None
_model    = None
_history  = None
_params   = None


def _load_artifacts() -> None:
    global _selector, _model, _history, _params
    _params  = load_params()
    _history = load_json(ROOT / "models" / "training_history.json")

    selector_path = ROOT / "data" / "features" / "selector.joblib"
    _selector = joblib.load(selector_path) if selector_path.exists() else None

    backend    = str(_history.get("backend", "tensorflow")).lower()
    model_path = ROOT / "models" / _history.get("model_path", "classifier/model.keras")

    if backend == "sklearn":
        _model = joblib.load(model_path)
    else:
        from tensorflow.keras.models import load_model
        _model = load_model(str(model_path))


def _apply_selector(selector, x_flat: np.ndarray) -> np.ndarray:
    if selector is None:
        return x_flat
    if isinstance(selector, tuple) and selector[0] == "pipeline":
        _, var, kbest = selector
        return kbest.transform(var.transform(x_flat))
    return selector.transform(x_flat)


def _preprocess(file_bytes: bytes, image_size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def classify(file_bytes: bytes) -> dict:
    if _model is None:
        _load_artifacts()

    image_size = int(_params["data_processing"]["image_size"])
    image      = _preprocess(file_bytes, image_size)           # (H, W, 3)
    batch      = image[np.newaxis, ...]                        # (1, H, W, 3)
    flat       = batch.reshape(1, -1)
    features   = _apply_selector(_selector, flat)

    backend      = str(_history.get("backend", "tensorflow")).lower()
    architecture = str(_history.get("architecture", "cnn")).lower()
    names        = class_names_cifar10()

    if backend == "sklearn" or architecture == "mlp":
        pred_idx = int(_model.predict(features)[0])
        proba    = None
    else:
        proba    = _model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(proba))

    result = {
        "predicted_class": names[pred_idx],
        "class_index":     pred_idx,
        "backend":         backend,
        "architecture":    architecture,
    }
    if proba is not None:
        result["confidence"] = round(float(proba[pred_idx]) * 100, 1)
        result["top3"] = [
            {"class": names[i], "confidence": round(float(proba[i]) * 100, 1)}
            for i in np.argsort(proba)[::-1][:3]
        ]
    return result


# ── HTML template ────────────────────────────────────────────────────────────

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CIFAR-10 Classifier — DVC + Docker + AWS</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 680px; margin: 2rem auto; padding: 0 1rem; }
    h1   { color: #1a73e8; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; background: #fafafa; }
    input[type=file] { margin: 0.5rem 0; }
    button { background: #1a73e8; color: white; border: none; padding: 0.5rem 1.5rem;
             border-radius: 4px; cursor: pointer; font-size: 1rem; }
    button:hover { background: #1558b0; }
    .classes { font-size: 0.85rem; color: #555; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>CIFAR-10 Image Classifier</h1>
  <p>Deployed with <strong>DVC + Docker + AWS EC2 + GitHub Actions CI/CD</strong></p>

  <div class="card">
    <h2>Upload an image</h2>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required /><br>
      <button type="submit">Classify</button>
    </form>
  </div>

  <div class="classes">
    <strong>CIFAR-10 classes:</strong>
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  </div>

  <p><a href="/health">Health check</a></p>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Prediction Result</title>
  <style>
    body  { font-family: Arial, sans-serif; max-width: 680px; margin: 2rem auto; padding: 0 1rem; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; background: #fafafa; }
    .label{ font-size: 2rem; color: #1a73e8; font-weight: bold; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    td, th{ border: 1px solid #ddd; padding: 0.4rem 0.8rem; }
    th    { background: #f0f0f0; }
    a     { color: #1a73e8; }
  </style>
</head>
<body>
  <h1>Prediction Result</h1>
  <div class="card">
    <div class="label">{{ result.predicted_class }}</div>
    {% if result.confidence is defined %}
    <p>Confidence: <strong>{{ result.confidence }}%</strong></p>
    <h3>Top 3</h3>
    <table>
      <tr><th>Class</th><th>Confidence</th></tr>
      {% for item in result.top3 %}
      <tr><td>{{ item.class }}</td><td>{{ item.confidence }}%</td></tr>
      {% endfor %}
    </table>
    {% endif %}
    <p>Backend: {{ result.backend }} | Architecture: {{ result.architecture }}</p>
  </div>
  <p><a href="/">← Upload another image</a></p>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return render_template_string(HOME_HTML)


@app.get("/health")
def health():
    try:
        _load_artifacts()
        return jsonify({"status": "ok", "message": "Model loaded successfully",
                        "backend": _history.get("backend"),
                        "architecture": _history.get("architecture")})
    except Exception as exc:
        return jsonify({"status": "degraded", "message": str(exc)}), 500


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        result = classify(f.read())
        # Return JSON for API calls, HTML for browser form submissions
        if request.accept_mimetypes.best == "application/json":
            return jsonify(result)
        return render_template_string(RESULT_HTML, result=result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
