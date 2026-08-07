"""
anchor_cross_domain.py

Compute AnchorScore (zero-shot CLIP accuracy) on diverse domain datasets:
- EuroSAT (satellite)    10 classes, RGB 64x64 → 224x224
- PathMNIST (medical)    9 classes, 28x28 grayscale → 224x224 RGB
- BloodMNIST (medical)   8 classes
- TissueMNIST (medical)  8 classes

Usage:
  python anchor_cross_domain.py [--datasets EuroSAT,PathMNIST,...]
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent

# EuroSAT directory (download from https://github.com/phelber/EuroSAT)
EUROSAT_DIR = Path(os.environ.get("EUROSAT_DIR", str(PROJ / "data" / "eurosat_rgb" / "2750")))
# MedMNIST cache directory (auto-downloaded by medmnist package)
MEDMNIST_ROOT = Path(os.environ.get("MEDMNIST_ROOT", str(Path.home() / ".medmnist")))

# EuroSAT classes (alphabetically sorted by directory)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

DATASET_CFG = {
    "EuroSAT": {
        "type": "folder",
        "path": EUROSAT_DIR,
        "classes": EUROSAT_CLASSES,
    },
    "PathMNIST": {
        "type": "medmnist",
        "class_name": "PathMNIST",
        "download": True,
        "size": 28,
    },
    "BloodMNIST": {
        "type": "medmnist",
        "class_name": "BloodMNIST",
        "download": True,
        "size": 28,
    },
    "TissueMNIST": {
        "type": "medmnist",
        "class_name": "TissueMNIST",
        "download": True,
        "size": 28,
    },
}

PROMPT_TEMPLATES = [
    "a satellite image of {}.",
    "an aerial photograph showing {}.",
    "a remote sensing scene of {}.",
]

MED_PROMPTS = [
    "a microscopic image of {}.",
    "a histopathology slide showing {}.",
    "a medical image of {}.",
]


def load_models():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model

    model, tokenizer, transform = load_clip_model()
    logger.info("Loaded ViT-L-14 laion2B-s32B-b82K")
    return model, tokenizer, transform


def load_medmnist(cfg):
    """Load medmnist dataset; returns (images, labels, class_count, class_names)."""
    import numpy as np
    import medmnist
    from medmnist import INFO

    cls_name = cfg["class_name"]
    info = INFO[cls_name.lower()]
    n_classes = len(info["label"])
    class_names = list(info["label"].values())
    download = cfg.get("download", True)

    logger.info(f"Loading {cls_name} ({n_classes} classes)...")
    DataClass = getattr(medmnist, cls_name)
    train_set = DataClass(split="train", download=download, root=str(MEDMNIST_ROOT))
    test_set = DataClass(split="test", download=download, root=str(MEDMNIST_ROOT))

    # Use test split
    images = np.asarray(test_set.imgs)  # (N, H, W) or (N, H, W, C)
    labels = np.asarray(test_set.labels).squeeze()  # (N,)

    # Handle grayscale → RGB
    if images.ndim == 3:
        images = images[..., None].repeat(3, axis=-1)

    logger.info(f"  {cls_name}: {len(images)} test images, shape {images.shape}")
    return images, labels, n_classes, class_names


def compute_medmnist(model, tokenizer, transform, device, batch_size, ds_name, cfg, model_dtype):
    images, labels, n_classes, class_names = load_medmnist(cfg)

    prompts = []
    for cname in class_names:
        for t in MED_PROMPTS:
            prompts.append(t.format(cname))
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(n_classes, -1, text_feats.shape[-1]).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    cls_correct = {i: 0 for i in range(n_classes)}
    cls_total = {i: 0 for i in range(n_classes)}
    labels = torch.from_numpy(labels).to(device)

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_pil = [Image.fromarray(img).convert("RGB") for img in batch_imgs]
        batch_tensor = torch.stack([transform(img) for img in batch_pil]).to(device).to(model_dtype)
        batch_labels = labels[i : i + batch_size]

        img_feats = model.encode_image(batch_tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logits = (100.0 * img_feats) @ text_feats.T
        preds = logits.argmax(dim=1)

        for cid in range(n_classes):
            mask = batch_labels == cid
            cls_correct[cid] += (preds[mask] == cid).sum().item()
            cls_total[cid] += mask.sum().item()

    return _summarize(class_names, cls_correct, cls_total, n_classes, ds_name)


def compute_folder_dataset(model, tokenizer, transform, device, batch_size, ds_name, cfg, model_dtype):
    base = cfg["path"]
    classes = cfg["classes"]
    n_classes = len(classes)

    prompts = []
    for cls in classes:
        for t in PROMPT_TEMPLATES:
            prompts.append(t.format(cls))
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(n_classes, -1, text_feats.shape[-1]).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    cls_correct = {i: 0 for i in range(n_classes)}
    cls_total = {i: 0 for i in range(n_classes)}

    for ci, cls_name in enumerate(classes):
        cls_dir = base / cls_name
        if not cls_dir.exists():
            continue
        paths = sorted([p for p in cls_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        logger.info(f"  {cls_name}: {len(paths)} images")

        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            tensors = []
            valid = []
            for p in batch:
                try:
                    tensors.append(transform(Image.open(p).convert("RGB")))
                    valid.append(p)
                except Exception as e:
                    logger.warning(f"    skipping {p.name}: {e}")
            if not tensors:
                continue
            batch = valid
            images = torch.stack(tensors).to(device).to(model_dtype)

            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            logits = (100.0 * img_feats) @ text_feats.T
            preds = logits.argmax(dim=1)
            cls_correct[ci] += (preds == ci).sum().item()
            cls_total[ci] += len(batch)

    return _summarize(classes, cls_correct, cls_total, n_classes, ds_name)


def _summarize(classes, cls_correct, cls_total, n_classes, ds_name):
    per_class = {}
    tot_correct = 0
    tot_count = 0
    for cid, cls_name in enumerate(classes):
        n = cls_total[cid]
        acc = cls_correct[cid] / n * 100 if n else 0.0
        per_class[cls_name] = {"n": int(n), "acc": round(acc, 2)}
        tot_correct += cls_correct[cid]
        tot_count += n
        logger.info(f"    {cls_name:28s}  n={n:5d}  {acc:6.2f}%")
    overall = round(tot_correct / tot_count * 100, 2) if tot_count else 0.0
    logger.info(f"  Overall: {overall}% ({tot_correct}/{tot_count})")
    return {
        "overall_acc": overall, "per_class_acc": per_class,
        "total": int(tot_count), "correct": int(tot_correct),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eurosat-dir", type=str, default=None,
                        help="EuroSAT RGB root directory")
    parser.add_argument("--medmnist-root", type=str, default=None,
                        help="MedMNIST cache directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results/02_robustness/anchor_score_cross_domain)")
    args = parser.parse_args()

    # Override paths if provided
    global EUROSAT_DIR, MEDMNIST_ROOT
    if args.eurosat_dir:
        EUROSAT_DIR = Path(args.eurosat_dir)
    if args.medmnist_root:
        MEDMNIST_ROOT = Path(args.medmnist_root)

    # Limit datasets if requested
    selected = set(args.datasets.split(",")) if args.datasets else None
    cfg = {k: v for k, v in DATASET_CFG.items() if selected is None or k in selected}

    out_dir = Path(args.out or str(PROJ / "results" / "02_robustness" / "anchor_score_cross_domain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, transform = load_models()
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    model_dtype = next(model.parameters()).dtype

    results = {}
    for ds_name, ds_cfg in cfg.items():
        logger.info(f"\n{'='*60}\n{ds_name}")
        t0 = time.time()

        try:
            if ds_cfg["type"] == "folder":
                res = compute_folder_dataset(
                    model, tokenizer, transform, device,
                    args.batch_size, ds_name, ds_cfg, model_dtype,
                )
            elif ds_cfg["type"] == "medmnist":
                res = compute_medmnist(
                    model, tokenizer, transform, device,
                    args.batch_size, ds_name, ds_cfg, model_dtype,
                )
            results[ds_name] = res
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

        logger.info(f"  [{time.time()-t0:.0f}s]")

    out_path = out_dir / "anchor_scores.json"
    # Merge with existing results (so separate runs don't overwrite each other)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
