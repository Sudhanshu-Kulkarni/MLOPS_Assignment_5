"""Automated tests for the CIFAR-10 image classification pipeline.

Mirrors the reference assignment test structure:
  - UtilsTests          — config, seeds, JSON round-trip
  - DataCollectionTests — synthetic fallback shapes, PNG writing, manifest
  - PipelineIntegrationTests — end-to-end run using synthetic data (no download)

Run locally:
    python test.py

Executed automatically by GitHub Actions CI on every push to main/master.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collection import _save_split, _write_manifest, collect_synthetic
from src.utils import (
    PARAMS_PATH,
    class_names_cifar10,
    ensure_dir,
    load_json,
    load_params,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class UtilsTests(unittest.TestCase):

    def test_load_params_has_pipeline_sections(self) -> None:
        """All 6 top-level pipeline sections must be present in params.yaml."""
        params = load_params()
        for key in ("base", "data_collection", "data_processing",
                    "feature_selection", "model_training", "model_evaluation"):
            self.assertIn(key, params, msg=f"Missing section: {key}")

    def test_base_section_values(self) -> None:
        """base section must have random_seed and num_classes=10."""
        base = load_params()["base"]
        self.assertIn("random_seed", base)
        self.assertEqual(int(base["num_classes"]), 10)

    def test_class_names_cifar10_count_and_unique(self) -> None:
        names = class_names_cifar10()
        self.assertEqual(len(names), 10)
        self.assertEqual(len(set(names)), 10)

    def test_class_names_known_entries(self) -> None:
        names = class_names_cifar10()
        for expected in ("airplane", "automobile", "bird", "cat", "deer",
                         "dog", "frog", "horse", "ship", "truck"):
            self.assertIn(expected, names)

    def test_set_seed_is_idempotent(self) -> None:
        set_seed(42)
        set_seed(42)

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            payload = {"accuracy": 0.87, "classes": ["cat", "dog"], "n": 100}
            save_json(payload, path)
            self.assertEqual(load_json(path), payload)

    def test_ensure_dir_creates_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "c"
            ensure_dir(nested)
            self.assertTrue(nested.is_dir())

    def test_params_model_training_has_required_keys(self) -> None:
        """model_training section must have all required hyperparameter keys."""
        mt = load_params()["model_training"]
        for key in ("learning_rate", "batch_size", "epochs", "optimizer",
                    "dropout_rate", "dense_units", "conv_filters",
                    "early_stopping_patience", "l2_regularization"):
            self.assertIn(key, mt, msg=f"Missing key: {key}")


# ---------------------------------------------------------------------------
# Data collection tests
# ---------------------------------------------------------------------------

class DataCollectionTests(unittest.TestCase):

    def test_collect_synthetic_train_shape(self) -> None:
        names = class_names_cifar10()
        x_tr, y_tr, x_te, y_te = collect_synthetic(120, names, image_size=32)
        self.assertEqual(x_tr.shape, (120, 32, 32, 3))
        self.assertEqual(len(y_tr), 120)

    def test_collect_synthetic_test_shape(self) -> None:
        names = class_names_cifar10()
        x_tr, y_tr, x_te, y_te = collect_synthetic(120, names, image_size=32)
        self.assertEqual(x_te.ndim, 4)
        self.assertEqual(x_te.shape[1:], (32, 32, 3))

    def test_collect_synthetic_label_range(self) -> None:
        names = class_names_cifar10()
        _, y_tr, _, y_te = collect_synthetic(200, names, image_size=32)
        self.assertTrue(np.all(y_tr >= 0) and np.all(y_tr < len(names)))
        self.assertTrue(np.all(y_te >= 0) and np.all(y_te < len(names)))

    def test_collect_synthetic_pixel_range(self) -> None:
        names = class_names_cifar10()
        x_tr, _, _, _ = collect_synthetic(50, names, image_size=32)
        self.assertGreaterEqual(int(x_tr.min()), 0)
        self.assertLessEqual(int(x_tr.max()), 255)

    def test_save_split_writes_correct_count(self) -> None:
        names = class_names_cifar10()
        x_tr, y_tr, _, _ = collect_synthetic(40, names, image_size=32)
        with tempfile.TemporaryDirectory() as tmp:
            count = _save_split(x_tr, y_tr, Path(tmp) / "train", names)
            self.assertEqual(count, 40)

    def test_save_split_writes_png_files(self) -> None:
        names = class_names_cifar10()
        x_tr, y_tr, _, _ = collect_synthetic(40, names, image_size=32)
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "train"
            _save_split(x_tr, y_tr, split_dir, names)
            pngs = list(split_dir.rglob("*.png"))
            self.assertEqual(len(pngs), 40)

    def test_write_manifest_creates_csv(self) -> None:
        names = class_names_cifar10()
        x_tr, y_tr, x_te, y_te = collect_synthetic(60, names, image_size=32)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _save_split(x_tr, y_tr, raw_dir / "train", names)
            _save_split(x_te, y_te, raw_dir / "test",  names)
            _write_manifest(raw_dir, names)
            manifest = raw_dir / "manifest.csv"
            self.assertTrue(manifest.exists())
            lines = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "filepath,split,label")
            # header + all images
            self.assertEqual(len(lines) - 1, len(x_tr) + len(x_te))


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------

class PipelineIntegrationTests(unittest.TestCase):
    """Run a fast end-to-end pipeline using synthetic data — no CIFAR-10 download needed."""

    def test_full_pipeline_produces_metrics(self) -> None:
        original_params = PARAMS_PATH.read_text(encoding="utf-8")
        backup = PARAMS_PATH.with_suffix(".yaml.bak")
        shutil.copy2(PARAMS_PATH, backup)
        try:
            params = yaml.safe_load(original_params)
            # Force synthetic so CI never tries to download CIFAR-10
            params["data_collection"]["dataset_name"] = "synthetic"
            params["data_collection"]["max_samples"]  = 300
            params["model_training"]["epochs"]        = 2
            params["model_training"]["batch_size"]    = 16
            PARAMS_PATH.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

            stages = [
                [sys.executable, "-m", "src.data_collection"],
                [sys.executable, "-m", "src.data_processing"],
                [sys.executable, "-m", "src.feature_selection"],
                [sys.executable, "-m", "src.model_training"],
                [sys.executable, "-m", "src.model_evaluation"],
            ]
            for cmd in stages:
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(
                    result.returncode, 0,
                    msg=f"Stage failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}",
                )

            # Check metrics.json exists and is valid
            metrics_path = ROOT / "reports" / "metrics.json"
            self.assertTrue(metrics_path.is_file(), "reports/metrics.json not created")
            metrics = load_json(metrics_path)
            for key in ("accuracy", "precision", "recall", "f1"):
                self.assertIn(key, metrics, msg=f"Missing metric: {key}")
            self.assertGreaterEqual(float(metrics["accuracy"]), 0.0)
            self.assertLessEqual(float(metrics["accuracy"]), 1.0)

            # Check confusion matrix CSV exists
            cm_csv = ROOT / "reports" / "confusion_matrix.csv"
            self.assertTrue(cm_csv.is_file(), "confusion_matrix.csv not created")

            # Check collection_meta confirms synthetic source
            meta = load_json(ROOT / "data" / "raw" / "collection_meta.json")
            self.assertEqual(meta["source"], "synthetic")
            self.assertEqual(len(meta["classes"]), 10)

        finally:
            PARAMS_PATH.write_text(original_params, encoding="utf-8")
            backup.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class ModelTests(unittest.TestCase):

    def test_build_cnn_output_shape(self) -> None:
        """CNN must output (batch, 10) probabilities."""
        from src.model_training import build_cnn
        cfg = {
            "conv_filters": [8, 16],
            "dense_units": 32,
            "dropout_rate": 0.0,
            "l2_regularization": 0.0001,
        }
        model = build_cnn((32, 32, 3), cfg, 10)
        dummy = np.zeros((4, 32, 32, 3), dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        self.assertEqual(out.shape, (4, 10))

    def test_build_cnn_probabilities_sum_to_one(self) -> None:
        """Each row of softmax output must sum to ~1.0."""
        from src.model_training import build_cnn
        cfg = {"conv_filters": [8], "dense_units": 16,
               "dropout_rate": 0.0, "l2_regularization": 0.0001}
        model = build_cnn((32, 32, 3), cfg, 10)
        dummy = np.random.rand(5, 32, 32, 3).astype(np.float32)
        out   = model.predict(dummy, verbose=0)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5), atol=1e-5)

    def test_predict_preprocess_shape(self) -> None:
        """preprocess() must return (1, image_size, image_size, 3)."""
        from src.predict import preprocess
        from PIL import Image as PILImage
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            PILImage.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            ).save(img_path)
            arr = preprocess(img_path, image_size=32)
            self.assertEqual(arr.shape, (1, 32, 32, 3))
            self.assertGreaterEqual(float(arr.min()), 0.0)
            self.assertLessEqual(float(arr.max()), 1.0)


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------

class FlaskAppTests(unittest.TestCase):

    def setUp(self) -> None:
        from app.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_returns_200(self) -> None:
        """GET / must return 200 with upload form."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"CIFAR-10", resp.data)

    def test_health_endpoint_exists(self) -> None:
        """GET /health must return JSON with a status key."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("status", data)

    def test_predict_no_file_returns_400(self) -> None:
        """POST /predict without a file must return 400."""
        resp = self.client.post("/predict", data={})
        self.assertEqual(resp.status_code, 400)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in (UtilsTests, DataCollectionTests, ModelTests, FlaskAppTests, PipelineIntegrationTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
