"""
cross_domain_probe.py

Validate that AnchorScore (CLIP zero-shot) correlates with supervised
accuracy on cross-domain data, using a linear probe on CLIP features.

Logic (for the paper):
  On SCB5: AnchorScore correlates with MLLM accuracy (ρ=0.705)
  On cross-domain: AnchorScore correlates with supervised CLIP accuracy
  ∴ AnchorScore likely predicts MLLM difficulty across domains.

Datasets: EuroSAT (folder), BloodMNIST/TissueMNIST/PathMNIST (medmnist)
"""

import json
import logging
import os
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(os.environ.get("FEATURE_CACHE", str(PROJ / "data" / "_cache" / "cross_probe")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# EuroSAT config (download from https://github.com/phelber/EuroSAT)
EUROSAT_DIR = Path(os.environ.get("EUROSAT_DIR", str(PROJ / "data" / "eurosat_rgb" / "2750")))
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
EUROSAT_PROMPTS = {c: c for c in EUROSAT_CLASSES}  # class name = prompt

# MedMNIST config (no train/val split in medmnist, use test split)
MEDMNIST_CFG = {
    "bloodmnist": {"classes": [
        "lymphocyte", "monocyte", "neutrophil", "eosinophil",
        "basophil", "erythroblast", "myeloblast", "monoblast",
    ]},
    "tissuemnist": {"classes": [
        "kidney connective tissue", "prostate connective tissue",
        "lung connective tissue", "breast connective tissue",
        "colon connective tissue", "ovary connective tissue",
        "endometrium connective tissue", "duodenum connective tissue",
    ]},
    "pathmnist": {"classes": [
        "adipose tissue", "breast tissue", "liver tissue",
        "squamous cell carcinoma", "cervix tissue", "kidney tissue",
        "bladder tissue", "stomach tissue", "prostate tissue",
    ]},
}


def load_clip(tag="ViT-L-14"):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    model, tokenizer, transform = load_clip_model(tag)
    return model, tokenizer, transform


@torch.no_grad()
def compute_zeroshot_anchor(model, tokenizer, transform, device, images, classes, prompt_template="a satellite image of {}"):
    """Compute per-class zero-shot accuracy (AnchorScore)."""
    prompts = [prompt_template.format(c) for c in classes]
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = F.normalize(text_feats, dim=-1)

    model_dtype = next(model.parameters()).dtype
    images_tensor = torch.stack([
        transform(Image.fromarray(img).convert("RGB"))
        for img in images
    ]).to(device).to(model_dtype)

    img_feats = model.encode_image(images_tensor)
    img_feats = F.normalize(img_feats, dim=-1)

    logits = (100.0 * img_feats) @ text_feats.T
    return logits.argmax(dim=1)


@torch.no_grad()
def extract_features(model, transform, device, images, bs=64):
    model_dtype = next(model.parameters()).dtype
    all_feats = []
    for i in range(0, len(images), bs):
        batch = images[i:i+bs]
        tensor = torch.stack([
            transform(Image.fromarray(img).convert("RGB"))
            for img in batch
        ]).to(device).to(model_dtype)
        feats = model.encode_image(tensor)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def train_probe(train_feats, train_labels, val_feats, val_labels, n_classes):
    train_feats = train_feats.float().cuda()
    val_feats = val_feats.float().cuda()
    train_labels = train_labels.cuda()
    val_labels = val_labels.cuda()

    d = train_feats.shape[1]
    clf = nn.Linear(d, n_classes).cuda()
    opt = torch.optim.AdamW(clf.parameters(), lr=0.01, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    for epoch in range(500):
        clf.train()
        loss = F.cross_entropy(clf(train_feats), train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        clf.eval()
        with torch.no_grad():
            acc = (clf(val_feats).argmax(1) == val_labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = clf.state_dict()

    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        preds = clf(val_feats).argmax(1)
    per_class = {}
    for c in range(n_classes):
        mask = val_labels == c
        n = mask.sum().item()
        correct = (preds[mask] == c).sum().item()
        per_class[c] = {"n": n, "acc": round(correct / n * 100, 2) if n else 0.0}

    overall = round(sum(p["acc"] * p["n"] for p in per_class.values()) /
                    sum(p["n"] for p in per_class.values()), 2)
    return overall, per_class


def run_eurosat(model, tokenizer, transform, device, args):
    """EuroSAT: folder-based, 10 classes, 27000 images."""
    logger.info("\n" + "="*50)
    logger.info("EuroSAT")

    # Load images from folders
    class_images = {}
    class_labels = {}
    for cid, cls_name in enumerate(EUROSAT_CLASSES):
        img_dir = EUROSAT_DIR / cls_name
        imgs = []
        for f in sorted(img_dir.iterdir())[:args.max_per_class]:
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                imgs.append(np.array(Image.open(f).convert("RGB")))
        class_images[cls_name] = imgs
        class_labels[cls_name] = np.full(len(imgs), cid)
        logger.info(f"  {cls_name:25s}: {len(imgs)} images")

    # Split into train/val (80/20)
    rng = np.random.RandomState(42)
    train_x, train_y = [], []
    val_x, val_y = [], []
    for cls_name in EUROSAT_CLASSES:
        imgs = class_images[cls_name]
        idx = np.arange(len(imgs))
        rng.shuffle(idx)
        n_train = int(len(imgs) * 0.8)
        for i in idx[:n_train]:
            train_x.append(imgs[i])
            train_y.append(class_labels[cls_name][i])
        for i in idx[n_train:]:
            val_x.append(imgs[i])
            val_y.append(class_labels[cls_name][i])

    # Zero-shot AnchorScore on val
    preds = compute_zeroshot_anchor(model, tokenizer, transform, device, val_x, EUROSAT_CLASSES)
    val_labels_t = torch.tensor(val_y, device=device)
    per_class_anchor = {}
    for c in range(len(EUROSAT_CLASSES)):
        mask = val_labels_t == c
        n = mask.sum().item()
        correct = (preds[mask] == c).sum().item()
        per_class_anchor[EUROSAT_CLASSES[c]] = round(correct / n * 100, 2) if n else 0.0
    overall_anchor = round(sum(per_class_anchor.values()) / len(EUROSAT_CLASSES), 2)
    logger.info(f"  Zero-shot Anchor: overall={overall_anchor}%")

    # Extract features for supervised probe
    logger.info("  Extracting features...")
    train_feats = extract_features(model, transform, device, train_x)
    val_feats = extract_features(model, transform, device, val_x)

    # Train probe
    logger.info("  Training linear probe...")
    probe_acc, probe_per_class = train_probe(
        train_feats, torch.tensor(train_y),
        val_feats, torch.tensor(val_y),
        len(EUROSAT_CLASSES)
    )
    anchor_vals = [per_class_anchor[c] for c in EUROSAT_CLASSES]
    probe_vals = [probe_per_class[i]["acc"] for i in range(len(EUROSAT_CLASSES))]
    r_s, p_s = spearmanr(anchor_vals, probe_vals)
    logger.info(f"  Supervised probe: overall={probe_acc}%")
    logger.info(f"  Anchor vs Probe: ρ={r_s:.4f}, p={p_s:.4f}")

    for i, c in enumerate(EUROSAT_CLASSES):
        logger.info(f"    {c:25s}  Anchor={per_class_anchor[c]:.1f}%  Probe={probe_per_class[i]['acc']:.1f}%")

    return {
        "dataset": "EuroSAT",
        "classes": EUROSAT_CLASSES,
        "overall_anchor": overall_anchor,
        "overall_probe": probe_acc,
        "spearman": round(r_s, 4),
        "spearman_p": round(p_s, 6),
        "per_class": {
            c: {"anchor": per_class_anchor[c], "probe": probe_per_class[i]["acc"]}
            for i, c in enumerate(EUROSAT_CLASSES)
        },
    }


def run_medmnist(model, tokenizer, transform, device, ds_name, cfg, args):
    """MedMNIST: loaded via medmnist library."""
    import medmnist
    from medmnist import INFO

    logger.info(f"\n{'='*50}")
    logger.info(f"{ds_name}")

    info = INFO[ds_name]
    DataClass = getattr(medmnist, info["python_class"])
    data = DataClass(split="test", download=True)

    images = data.imgs  # shape (N, 28, 28, 3) for medmnist
    labels = data.labels.flatten()
    classes = cfg["classes"]
    n_classes = len(classes)

    # Limit per class
    rng = np.random.RandomState(42)
    all_idx = []
    for c in range(n_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_want = min(len(c_idx), args.max_per_class)
        all_idx.extend(c_idx[:n_want])
    all_idx = np.array(all_idx)
    images = images[all_idx]
    labels = labels[all_idx]

    # Split train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(images)), test_size=0.2, random_state=42, stratify=labels
    )
    train_x, train_y = list(images[train_idx]), labels[train_idx]
    val_x, val_y = list(images[val_idx]), labels[val_idx]

    logger.info(f"  Train: {len(train_x)}, Val: {len(val_x)}")

    # Zero-shot
    preds = compute_zeroshot_anchor(model, tokenizer, transform, device, val_x, classes)
    val_labels_t = torch.tensor(val_y, device=device)
    per_class_anchor = {}
    for c in range(n_classes):
        mask = val_labels_t == c
        n = mask.sum().item()
        correct = (preds[mask] == c).sum().item()
        per_class_anchor[c] = round(correct / n * 100, 2) if n else 0.0
    overall_anchor = round(sum(per_class_anchor.values()) / n_classes, 2)
    logger.info(f"  Zero-shot Anchor: overall={overall_anchor}%")

    # Features
    logger.info("  Extracting features...")
    train_feats = extract_features(model, transform, device, train_x)
    val_feats = extract_features(model, transform, device, val_x)

    # Probe
    logger.info("  Training linear probe...")
    probe_acc, probe_per_class = train_probe(
        train_feats, torch.tensor(train_y),
        val_feats, torch.tensor(val_y),
        n_classes
    )
    anchor_vals = [per_class_anchor[i] for i in range(n_classes)]
    probe_vals = [probe_per_class[i]["acc"] for i in range(n_classes)]
    r_s, p_s = spearmanr(anchor_vals, probe_vals)
    logger.info(f"  Supervised probe: overall={probe_acc}%")
    logger.info(f"  Anchor vs Probe: ρ={r_s:.4f}, p={p_s:.4f}")

    for i, c in enumerate(classes):
        logger.info(f"    {c:30s}  Anchor={per_class_anchor[i]:.1f}%  Probe={probe_per_class[i]['acc']:.1f}%")

    return {
        "dataset": ds_name,
        "classes": classes,
        "overall_anchor": overall_anchor,
        "overall_probe": probe_acc,
        "spearman": round(r_s, 4),
        "spearman_p": round(p_s, 6) if p_s > 1e-10 else 0.0,
        "per_class": {
            c: {"anchor": per_class_anchor[i], "probe": probe_per_class[i]["acc"]}
            for i, c in enumerate(classes)
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="eurosat,bloodmnist")
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Max images per class (for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eurosat-dir", type=str, default=None,
                        help="EuroSAT RGB root directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results/02_robustness/cross_domain_probe)")
    args = parser.parse_args()

    # Override EUROSAT_DIR if provided
    global EUROSAT_DIR
    if args.eurosat_dir:
        EUROSAT_DIR = Path(args.eurosat_dir)
    out_dir = Path(args.out or str(PROJ / "results" / "02_robustness" / "cross_domain_probe"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, transform = load_clip()
    model = model.to(device).eval().half()
    logger.info(f"Loaded ViT-L/14 on {device}")

    datasets = [d.strip() for d in args.datasets.split(",")]
    results_list = []

    if "eurosat" in datasets:
        res = run_eurosat(model, tokenizer, transform, device, args)
        results_list.append(res)
        # Clean GPU
        del res
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for ds_name in datasets:
        if ds_name in MEDMNIST_CFG:
            res = run_medmnist(model, tokenizer, transform, device, ds_name, MEDMNIST_CFG[ds_name], args)
            results_list.append(res)
            del res
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Summary
    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN PROBE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':20s} {'Anchor%':>8s} {'Probe%':>8s} {'ρ(Anchor,Probe)':>16s}")
    for res in results_list:
        print(f"  {res['dataset']:20s} {res['overall_anchor']:>7.2f}% {res['overall_probe']:>7.2f}%"
              f"  {res['spearman']:>+.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
