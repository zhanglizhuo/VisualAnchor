"""
dinov2_anchor_scb5.py

Non-CLIP baseline: DINOv2 kNN per-class accuracy as an alternative
"AnchorScore". Tests whether the AnchorScore-MLLM correlation is
specific to CLIP's vision-language alignment or a generic property
of any strong visual classifier (rules out a generic-classifier explanation).

Uses DINOv2 ViT-B/14 (self-supervised, no text encoder) + kNN (k=5)
with the train split as labeled reference, evaluated on val.

Output format matches anchor_scores.json for direct comparison.

Usage:
  python dinov2_anchor_scb5.py [--k 5] [--batch_size 64]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import read_label

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
SERVER_DATA = Path(os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))

DATASET_CFG = {
    "TeacherBehavior": {
        "dir": "SCB5_TeacherBehavior",
        "subdir": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
        ],
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": "SCB5-Handrise-Read-write-2024-9-17",
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
    },
}

DINOV2_TRANSFORM = None


def get_transform():
    global DINOV2_TRANSFORM
    if DINOV2_TRANSFORM is None:
        from torchvision import transforms
        DINOV2_TRANSFORM = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    return DINOV2_TRANSFORM


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    logger.info("Loaded DINOv2 ViT-B/14")
    return model


def collect_samples(cfg, split):
    base = SERVER_DATA / cfg["dir"]
    sub = cfg["subdir"]
    img_dir = (base / sub / "images" / split if sub else base / "images" / split)
    lbl_dir = (base / sub / "labels" / split if sub else base / "labels" / split)

    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        cid = read_label(lbl_path)
        if cid is None or cid >= len(cfg["classes"]):
            continue
        samples.append((str(img_path), cid))
    return samples


@torch.no_grad()
def extract_features(model, transform, device, samples, batch_size=64):
    model_dtype = next(model.parameters()).dtype
    feats, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p, _ in batch
        ]).to(device).to(model_dtype)
        batch_labels = [c for _, c in batch]
        f = model(images)
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu().float())
        labels.extend(batch_labels)
    return torch.cat(feats), torch.tensor(labels)


@torch.no_grad()
def knn_classify(train_feats, train_labels, val_feats, k, device):
    """kNN classification using cosine similarity."""
    train_feats = train_feats.to(device)
    val_feats = val_feats.to(device)
    train_labels = train_labels.to(device)
    n_classes = int(train_labels.max().item()) + 1
    preds = []

    bs = 256
    for i in range(0, len(val_feats), bs):
        batch = val_feats[i: i + bs]
        sim = batch @ train_feats.T  # cosine since normalized
        topk_sim, topk_idx = sim.topk(k, dim=1)
        neighbor_labels = train_labels[topk_idx].cpu()  # (batch, k)
        for row in neighbor_labels:
            votes = torch.bincount(row, minlength=n_classes)
            preds.append(int(votes.argmax().item()))

    return torch.tensor(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    out_dir = Path(args.out or str(PROJ / "results" / "01_core/anchor_score_scb5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = load_dinov2(device)
    transform = get_transform()

    results = {}
    for ds_name, cfg in DATASET_CFG.items():
        logger.info(f"\n{'='*60}\n{ds_name}")
        classes = cfg["classes"]
        n_classes = len(classes)

        train_samples = collect_samples(cfg, "train")
        val_samples = collect_samples(cfg, "val")
        logger.info(f"  train: {len(train_samples)}, val: {len(val_samples)}")

        if not train_samples or not val_samples:
            logger.warning(f"  Skipping {ds_name}: no data")
            continue

        logger.info("  Extracting DINOv2 features (train)...")
        train_feats, train_labels = extract_features(
            model, transform, device, train_samples, args.batch_size
        )
        logger.info("  Extracting DINOv2 features (val)...")
        val_feats, val_labels = extract_features(
            model, transform, device, val_samples, args.batch_size
        )

        logger.info(f"  kNN classification (k={args.k})...")
        preds = knn_classify(train_feats, train_labels, val_feats, args.k, device)

        per_class = {}
        tot_correct = 0
        tot_count = 0
        for cid, cls_name in enumerate(classes):
            mask = val_labels == cid
            n = mask.sum().item()
            correct = (preds[mask] == cid).sum().item() if n else 0
            acc = correct / n * 100 if n else 0.0
            per_class[cls_name] = {"n": int(n), "acc": round(acc, 2)}
            tot_correct += correct
            tot_count += n
            logger.info(f"    {cls_name:28s}  n={n:5d}  {acc:6.2f}%")

        overall = round(tot_correct / tot_count * 100, 2)
        logger.info(f"  Overall DINOv2 kNN: {overall}% ({tot_correct}/{tot_count})")
        results[ds_name] = {
            "overall_acc": overall, "per_class_acc": per_class,
            "total": int(tot_count), "correct": int(tot_correct),
            "model": "dinov2_vitb14", "k": args.k,
        }

    out_path = out_dir / "dinov2_anchor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  DINOv2 ANCHORSCORE (kNN, k={args.k})")
    print(f"{'='*60}")
    print(f"  {'Dataset':22s}  {'Overall':>8s}")
    for ds_name, r in results.items():
        print(f"  {ds_name:22s}  {r['overall_acc']:>7.2f}%")


if __name__ == "__main__":
    main()
