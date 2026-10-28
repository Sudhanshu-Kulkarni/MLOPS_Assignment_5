"""Stage 2 — resize, normalize, split, and augment images."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from src.utils import ROOT, class_names_cifar10, ensure_dir, load_params, save_json, set_seed


def _load_split(split_dir: Path, names: list[str], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for class_idx, class_name in enumerate(names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            img = Image.open(image_path).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(class_idx)
    if not images:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return np.stack(images), np.array(labels, dtype=np.int64)


def _augment_one(image: np.ndarray, aug: dict, rng: np.random.Generator) -> np.ndarray:
    pil = Image.fromarray(image)
    rotation = float(aug.get("rotation_range", 0))
    if rotation:
        pil = pil.rotate(float(rng.uniform(-rotation, rotation)), resample=Image.BILINEAR, fillcolor=(0,0,0))
    if bool(aug.get("horizontal_flip", False)) and rng.random() < 0.5:
        pil = ImageOps.mirror(pil)
    zoom = float(aug.get("zoom_range", 0))
    if zoom:
        factor = 1.0 + float(rng.uniform(-zoom, zoom))
        w, h = pil.size
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
        zoomed = pil.resize((nw, nh), Image.BILINEAR)
        if factor >= 1:
            pil = zoomed.crop(((nw-w)//2, (nh-h)//2, (nw-w)//2+w, (nh-h)//2+h))
        else:
            canvas = Image.new("RGB", (w, h))
            canvas.paste(zoomed, ((w-nw)//2, (h-nh)//2))
            pil = canvas
    ws = float(aug.get("width_shift_range", 0))
    hs = float(aug.get("height_shift_range", 0))
    if ws or hs:
        dx = int(ws * pil.size[0] * float(rng.uniform(-1, 1)))
        dy = int(hs * pil.size[1] * float(rng.uniform(-1, 1)))
        pil = pil.transform(pil.size, Image.AFFINE, (1,0,dx,0,1,dy), resample=Image.BILINEAR, fillcolor=(0,0,0))
    brightness = float(aug.get("brightness_delta", 0))
    if brightness:
        pil = ImageEnhance.Brightness(pil).enhance(1.0 + float(rng.uniform(-brightness, brightness)))
    return np.asarray(pil, dtype=np.uint8)


def _augment_set(images, labels, aug, rng):
    extra = np.stack([_augment_one(img, aug, rng) for img in images])
    return np.concatenate([images, extra]), np.concatenate([labels, labels])


def main() -> None:
    params = load_params()
    seed   = int(params["base"]["random_seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    cfg        = params["data_processing"]
    names      = class_names_cifar10()
    image_size = int(cfg["image_size"])
    raw_dir    = ROOT / params["data_collection"]["output_dir"]
    out_dir    = ensure_dir(ROOT / "data" / "processed")

    x_train, y_train = _load_split(raw_dir / "train", names, image_size)
    x_test,  y_test  = _load_split(raw_dir / "test",  names, image_size)

    val_split = float(cfg["validation_split"])
    idx = rng.permutation(len(x_train))
    val_n = max(1, int(len(idx) * val_split))
    x_val,   y_val   = x_train[idx[:val_n]], y_train[idx[:val_n]]
    x_train, y_train = x_train[idx[val_n:]], y_train[idx[val_n:]]

    aug = cfg.get("augmentation", {})
    if bool(aug.get("enabled", False)):
        x_train, y_train = _augment_set(x_train, y_train, aug, rng)

    norm = bool(cfg.get("normalize", True))
    scale = 255.0 if norm else 1.0
    x_train_f = x_train.astype(np.float32) / scale
    x_val_f   = x_val.astype(np.float32)   / scale
    x_test_f  = x_test.astype(np.float32)  / scale

    np.savez_compressed(out_dir / "train.npz", images=x_train_f, labels=y_train)
    np.savez_compressed(out_dir / "val.npz",   images=x_val_f,   labels=y_val)
    np.savez_compressed(out_dir / "test.npz",  images=x_test_f,  labels=y_test)
    save_json({
        "image_size":    image_size,
        "train_samples": int(len(y_train)),
        "val_samples":   int(len(y_val)),
        "test_samples":  int(len(y_test)),
        "normalized":    norm,
        "augmented":     bool(aug.get("enabled", False)),
        "classes":       names,
    }, out_dir / "processing_meta.json")
    print(f"Processed splits written to {out_dir}")


if __name__ == "__main__":
    main()
