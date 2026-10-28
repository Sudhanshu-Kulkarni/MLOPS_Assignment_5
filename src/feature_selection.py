"""Stage 3 — reduce flattened image features before training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif

from src.utils import ROOT, ensure_dir, load_params, save_json, set_seed

SCORE_FUNCS = {"f_classif": f_classif, "mutual_info": mutual_info_classif}


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    p = np.load(path)
    return p["images"], p["labels"]


def _fit_selector(x_flat, y, cfg):
    method = str(cfg["method"]).lower()
    if method == "variance":
        sel = VarianceThreshold(threshold=float(cfg["variance_threshold"]))
        sel.fit(x_flat)
        return sel, "variance"
    if method == "pca":
        n = min(int(cfg["pca_components"]), x_flat.shape[1], x_flat.shape[0])
        sel = PCA(n_components=n, random_state=42)
        sel.fit(x_flat)
        return sel, "pca"
    score_func = SCORE_FUNCS.get(str(cfg.get("score_func", "f_classif")), f_classif)
    var = VarianceThreshold(threshold=float(cfg["variance_threshold"]))
    x_var = var.fit_transform(x_flat)
    k = min(int(cfg["k_best"]), x_var.shape[1])
    kbest = SelectKBest(score_func=score_func, k=k)
    kbest.fit(x_var, y)
    return ("pipeline", var, kbest), "selectkbest"


def _transform(selector, x_flat):
    if isinstance(selector, tuple) and selector[0] == "pipeline":
        _, var, kbest = selector
        return kbest.transform(var.transform(x_flat))
    return selector.transform(x_flat)


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    cfg = params["feature_selection"]

    processed_dir = ROOT / "data" / "processed"
    out_dir = ensure_dir(ROOT / "data" / "features")

    x_train, y_train = _load_npz(processed_dir / "train.npz")
    x_val,   y_val   = _load_npz(processed_dir / "val.npz")
    x_test,  y_test  = _load_npz(processed_dir / "test.npz")

    x_tr_f = x_train.reshape(len(x_train), -1)
    x_va_f = x_val.reshape(len(x_val), -1)
    x_te_f = x_test.reshape(len(x_test), -1)

    selector, method = _fit_selector(x_tr_f, y_train, cfg)
    x_tr_s = _transform(selector, x_tr_f)
    x_va_s = _transform(selector, x_va_f)
    x_te_s = _transform(selector, x_te_f)

    np.savez_compressed(out_dir / "train.npz", images=x_train, labels=y_train, features=x_tr_s)
    np.savez_compressed(out_dir / "val.npz",   images=x_val,   labels=y_val,   features=x_va_s)
    np.savez_compressed(out_dir / "test.npz",  images=x_test,  labels=y_test,  features=x_te_s)
    joblib.dump(selector, out_dir / "selector.joblib")
    save_json({
        "method":               method,
        "original_feature_dim": int(x_tr_f.shape[1]),
        "selected_feature_dim": int(x_tr_s.shape[1]),
        "train_samples":        int(len(y_train)),
    }, out_dir / "feature_meta.json")
    print(f"Selected {x_tr_s.shape[1]} / {x_tr_f.shape[1]} features using {method} -> {out_dir}")


if __name__ == "__main__":
    main()
