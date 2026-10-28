"""Inference module — load a trained model and predict a single image.

This module is not a DVC pipeline stage but a standalone utility that can
be imported or run directly:

    python -m src.predict path/to/image.png

It loads the trained model from models/classifier/ and prints the top-3
predicted CIFAR-10 class labels with their confidence scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from src.utils import ROOT, class_names_cifar10, load_json


def preprocess(image_path: Path, image_size: int = 32) -> np.ndarray:
    """Load an image, resize to image_size x image_size, normalize to [0,1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]          # (1, H, W, 3)


def predict_top3(image_path: Path, image_size: int = 32) -> list[tuple[str, float]]:
    """Return top-3 (class_name, confidence) predictions for an image file."""
    names   = class_names_cifar10()
    history = load_json(ROOT / "models" / "training_history.json")
    backend = str(history.get("backend", "tensorflow")).lower()
    arch    = str(history.get("architecture", "cnn")).lower()

    x = preprocess(image_path, image_size)

    if backend == "sklearn":
        import joblib
        from sklearn.feature_selection import SelectKBest, VarianceThreshold
        model    = joblib.load(ROOT / "models" / history.get("model_path", "classifier/model.joblib"))
        selector_path = ROOT / "data" / "features" / "selector.joblib"
        selector = joblib.load(selector_path) if selector_path.exists() else None
        x_flat = x.reshape(1, -1)
        if selector is not None:
            if isinstance(selector, tuple) and selector[0] == "pipeline":
                _, var, kbest = selector
                x_flat = kbest.transform(var.transform(x_flat))
            else:
                x_flat = selector.transform(x_flat)
        proba = model.predict_proba(x_flat)[0]
    else:
        from tensorflow.keras.models import load_model
        model = load_model(ROOT / "models" / history.get("model_path", "classifier/model.keras"))
        if arch == "mlp":
            import joblib
            selector_path = ROOT / "data" / "features" / "selector.joblib"
            if selector_path.exists():
                selector = joblib.load(selector_path)
                x_flat = x.reshape(1, -1)
                if isinstance(selector, tuple) and selector[0] == "pipeline":
                    _, var, kbest = selector
                    x_flat = kbest.transform(var.transform(x_flat))
                else:
                    x_flat = selector.transform(x_flat)
                proba = model.predict(x_flat, verbose=0)[0]
            else:
                proba = model.predict(x.reshape(1, -1), verbose=0)[0]
        else:
            proba = model.predict(x, verbose=0)[0]

    top3_idx = np.argsort(proba)[::-1][:3]
    return [(names[i], float(proba[i])) for i in top3_idx]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)
    results = predict_top3(image_path)
    print(f"\nTop-3 predictions for: {image_path.name}")
    for rank, (label, conf) in enumerate(results, 1):
        print(f"  {rank}. {label:<15s}  {conf*100:5.1f}%")


if __name__ == "__main__":
    main()
