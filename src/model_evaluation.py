"""Stage 5 — evaluate the trained model and write reports."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score, recall_score)

from src.utils import ROOT, class_names_cifar10, ensure_dir, load_json, load_params, save_json, set_seed


def _predict(history, x_images, x_features):
    backend      = str(history.get("backend", "tensorflow")).lower()
    architecture = str(history.get("architecture", "cnn")).lower()
    if backend == "sklearn":
        model = joblib.load(ROOT / "models" / history.get("model_path", "classifier/model.joblib"))
        return model.predict(x_features)
    from tensorflow.keras.models import load_model
    model = load_model(ROOT / "models" / history.get("model_path", "classifier/model.keras"))
    x = x_features if architecture == "mlp" else x_images
    return np.argmax(model.predict(x, verbose=0), axis=1)


def _save_confusion_plot(matrix, names, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — CIFAR-10")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    cfg     = params["model_evaluation"]
    names   = class_names_cifar10()
    average = str(cfg.get("average", "macro"))

    feature_dir  = ROOT / "data" / "features"
    reports_dir  = ensure_dir(ROOT / "reports")
    history      = load_json(ROOT / "models" / "training_history.json")

    test   = np.load(feature_dir / "test.npz")
    y_true = test["labels"]
    y_pred = _predict(history, test["images"], test["features"])

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "architecture": history.get("architecture"),
        "backend":      history.get("backend"),
        "test_samples": int(len(y_true)),
        "average":      average,
    }
    save_json(metrics, reports_dir / "metrics.json")
    (reports_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=names, zero_division=0), encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["actual", "predicted"])
        for a, p in zip(y_true, y_pred):
            w.writerow([names[int(a)], names[int(p)]])

    if bool(cfg.get("save_plots", True)):
        _save_confusion_plot(matrix, names, reports_dir / "confusion_matrix.png")

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Reports written to {reports_dir}")


if __name__ == "__main__":
    main()
