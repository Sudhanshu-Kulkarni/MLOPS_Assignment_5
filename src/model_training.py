"""Stage 4 — train a CNN (TensorFlow) or MLP (TensorFlow / scikit-learn)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.utils import ROOT, load_params, save_json, set_seed


def _artifact_dir() -> Path:
    d = ROOT / "models" / "classifier"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _has_tensorflow() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _tf_optimizer(name: str, lr: float):
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop
    return {"adam": Adam(lr), "sgd": SGD(lr, momentum=0.9), "rmsprop": RMSprop(lr)}[name]


def build_cnn(image_shape: tuple, cfg: dict, num_classes: int):
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                          Dropout, Flatten, Input, MaxPooling2D)
    l2 = regularizers.l2(float(cfg["l2_regularization"]))
    dr = float(cfg["dropout_rate"])
    model = Sequential([Input(shape=image_shape)])
    for f in cfg["conv_filters"]:
        model.add(Conv2D(int(f), 3, activation="relu", padding="same", kernel_regularizer=l2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(int(cfg["dense_units"]), activation="relu", kernel_regularizer=l2))
    model.add(Dropout(dr))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def build_mlp(n_features: int, cfg: dict, num_classes: int):
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import Dense, Dropout, Input
    l2 = regularizers.l2(float(cfg["l2_regularization"]))
    dr = float(cfg["dropout_rate"])
    units = int(cfg["dense_units"])
    return Sequential([
        Input(shape=(n_features,)),
        Dense(units, activation="relu", kernel_regularizer=l2),
        Dropout(dr),
        Dense(max(units // 2, 16), activation="relu", kernel_regularizer=l2),
        Dropout(dr),
        Dense(num_classes, activation="softmax"),
    ])


def _train_tensorflow(cfg, architecture, train, val, num_classes):
    from tensorflow.keras.callbacks import EarlyStopping
    if architecture == "mlp":
        x_tr, x_va = train["features"], val["features"]
        model = build_mlp(x_tr.shape[1], cfg, num_classes)
    else:
        x_tr, x_va = train["images"], val["images"]
        model = build_cnn(x_tr.shape[1:], cfg, num_classes)
    y_tr, y_va = train["labels"], val["labels"]
    model.compile(
        optimizer=_tf_optimizer(str(cfg["optimizer"]).lower(), float(cfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_va, y_va),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        callbacks=[EarlyStopping(monitor="val_accuracy",
                                 patience=int(cfg["early_stopping_patience"]),
                                 restore_best_weights=True)],
        verbose=2,
    )
    artifact_dir = _artifact_dir()
    model.save(artifact_dir / "model.keras")
    payload = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    payload.update({"architecture": architecture, "backend": "tensorflow",
                    "epochs_ran": len(history.history["loss"]),
                    "model_path": "classifier/model.keras"})
    save_json({"backend": "tensorflow", "model_file": "model.keras"}, artifact_dir / "format.json")
    return payload


def _train_sklearn(cfg, train, val):
    x_tr, y_tr = train["features"], train["labels"]
    x_va, y_va = val["features"],   val["labels"]
    solver = "adam" if str(cfg["optimizer"]).lower() in {"adam", "rmsprop"} else "sgd"
    model = MLPClassifier(
        hidden_layer_sizes=(int(cfg["dense_units"]), max(int(cfg["dense_units"]) // 2, 16)),
        activation="relu", solver=solver,
        learning_rate_init=float(cfg["learning_rate"]),
        batch_size=int(cfg["batch_size"]),
        max_iter=int(cfg["epochs"]) * 20,
        early_stopping=True, n_iter_no_change=int(cfg["early_stopping_patience"]),
        random_state=42, verbose=False,
    )
    model.fit(x_tr, y_tr)
    artifact_dir = _artifact_dir()
    joblib.dump(model, artifact_dir / "model.joblib")
    save_json({"backend": "sklearn", "model_file": "model.joblib"}, artifact_dir / "format.json")
    print(f"sklearn MLP — train={model.score(x_tr,y_tr):.4f}  val={model.score(x_va,y_va):.4f}")
    return {
        "architecture": "mlp", "backend": "sklearn",
        "accuracy": [float(model.score(x_tr, y_tr))],
        "val_accuracy": [float(model.score(x_va, y_va))],
        "epochs_ran": int(getattr(model, "n_iter_", 0)),
        "model_path": "classifier/model.joblib",
    }


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    cfg = params["model_training"]
    num_classes = int(params["base"]["num_classes"])
    feature_dir = ROOT / "data" / "features"
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    train = np.load(feature_dir / "train.npz")
    val   = np.load(feature_dir / "val.npz")
    architecture = str(cfg["architecture"]).lower()

    history = _train_tensorflow(cfg, architecture, train, val, num_classes) \
        if _has_tensorflow() else _train_sklearn(cfg, train, val)

    save_json(history, ROOT / "models" / "training_history.json")
    print(f"Saved training artifacts to {ROOT / 'models'}")


if __name__ == "__main__":
    main()
