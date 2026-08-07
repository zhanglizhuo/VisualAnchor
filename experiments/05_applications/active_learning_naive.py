"""
active_learning_naive.py

Simulate active learning: compare data selection strategies for CLIP linear probing (ViT-B/32).

Strategies:
  1. Uniform (random): equal samples per class
  2. Anchor-weight: samples ∝ 1/AnchorScore (more data for hard classes)
  3. Anchor-proportional: samples ∝ AnchorScore (more data for easy classes, negative control)

Pipeline:
  - Load ViT-B/32 (fastest), extract vision features for ALL train+val
  - For each dataset, subsample train set to budget with different strategies
  - Train linear probe (single fc layer), evaluate on val
"""

import argparse
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label, load_clip_model

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
        "budget_ratio": 0.2,
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": "SCB5-Handrise-Read-write-2024-9-17",
        "classes": ["hand-raising", "read", "write"],
        "budget_ratio": 0.2,
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
        "budget_ratio": 0.2,
    },
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]

CACHE_DIR = Path(os.environ.get("FEATURE_CACHE", str(PROJ / "data" / "_cache" / "features")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Model loading ──

def load_clip(tag="ViT-B-32"):
    return load_clip_model(tag, "openai")


# ── Feature extraction (cached) ──

def extract_features(model, transform, device, split, ds_cfg):
    """Extract CLIP vision features for all images in a split."""
    cache_key = f"v1_b32_{split}_{ds_cfg['dir']}_{ds_cfg.get('subdir', 'none')}.pkl"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Loaded cached {split} features ({len(data['feats'])} samples)")
        return data

    base = SERVER_DATA / ds_cfg["dir"]
    sub = ds_cfg["subdir"]
    img_dir = (base / sub / "images" / split if sub else base / "images" / split)
    lbl_dir = (base / sub / "labels" / split if sub else base / "labels" / split)
    classes = ds_cfg["classes"]

    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        cid = read_label(lbl_path)
        if cid is None or cid >= len(classes):
            continue
        samples.append((img_path, cid))

    logger.info(f"  Extracting {split} features for {len(samples)} images...")
    feats, labels = [], []
    model_dtype = next(model.parameters()).dtype
    batch_size = 64

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p, _ in batch
        ]).to(device).to(model_dtype)
        batch_labels = [c for _, c in batch]

        with torch.no_grad():
            batch_feats = model.encode_image(images)
            batch_feats = F.normalize(batch_feats, dim=-1)

        feats.append(batch_feats.cpu())
        labels.extend(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.tensor(labels)

    data = {"feats": feats, "labels": labels, "class_names": classes}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Saved {split} features to cache ({len(feats)} samples)")
    return data


# ── Linear probe ──

def train_probe(train_feats, train_labels, val_feats, val_labels, n_classes, n_epochs=500):
    d = train_feats.shape[1]
    train_feats = train_feats.float()
    val_feats = val_feats.float()
    classifier = nn.Linear(d, n_classes).to(train_feats.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(n_epochs):
        classifier.train()
        logits = classifier(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            preds = classifier(val_feats).argmax(dim=1)
            acc = (preds == val_labels).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = classifier.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        preds = classifier(val_feats).argmax(dim=1)
        correct_counts = {}
        total_counts = {}
        for c in range(n_classes):
            mask = val_labels == c
            total_counts[c] = mask.sum().item()
            correct_counts[c] = (preds[mask] == c).sum().item()

    per_class_acc = {
        c: {
            "n": total_counts[c],
            "acc": round(correct_counts[c] / total_counts[c] * 100, 2) if total_counts[c] else 0.0,
        }
        for c in range(n_classes)
    }
    overall_acc = round(sum(correct_counts.values()) / sum(total_counts.values()) * 100, 2)

    return overall_acc, per_class_acc


def sample_data(feats, labels, n_per_class, seed):
    """Sample n_per_class[c] examples from each class."""
    rng = np.random.RandomState(seed)
    n_classes = len(n_per_class)
    indices = []
    for c in range(n_classes):
        c_mask = (labels == c).nonzero(as_tuple=True)[0]
        n_avail = len(c_mask)
        n_want = min(n_per_class[c], n_avail)
        if n_want < n_per_class[c]:
            logger.warning(f"  Class {c}: want {n_per_class[c]} but only {n_avail} available")
        chosen = c_mask[rng.choice(n_avail, size=n_want, replace=False)]
        indices.append(chosen)
    idx = torch.cat(indices)
    return feats[idx], labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-path", type=str, default=None,
                        help="Path to backbone_results.json (default: results/02_robustness/multi_backbone/backbone_results.json)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results/05_applications/active_learning)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="SCB5 data root (default: $SCB5_DATA_ROOT or data/scb5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reproduce", type=int, default=1,
                        help="Number of seeds to repeat for robustness")
    parser.add_argument("--budget-ratio", type=float, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Override SERVER_DATA and CACHE_DIR
    global SERVER_DATA, CACHE_DIR
    if args.data_root:
        SERVER_DATA = Path(args.data_root)
    if args.out:
        CACHE_DIR = Path(args.out) / "_cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Default anchor path: project-relative
    anchor_path = Path(args.anchor_path or str(PROJ / "results" / "02_robustness" / "multi_backbone" / "backbone_results.json"))
    out_dir = Path(args.out or str(PROJ / "results" / "05_applications" / "active_learning"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, transform = load_clip()
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    logger.info("Loaded ViT-B/32")

    # Load AnchorScore per class from anchor_scb5 results
    if not anchor_path.exists():
        logger.error(f"AnchorScore file not found: {anchor_path}")
        return
    with open(anchor_path) as f:
        backbone_data = json.load(f)
    laion_l14 = backbone_data.get("laion_l14")
    if laion_l14 is None:
        laion_l14 = backbone_data.get("openai_b32", {})
    anchor_scores = laion_l14.get("results", {})

    strategies = ["uniform", "anchor_weight", "anchor_proportional"]
    all_results = {}

    for ds_name, cfg in DATASET_CFG.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        classes = cfg["classes"]
        n_classes = len(classes)
        budget_ratio = args.budget_ratio or cfg["budget_ratio"]

        # Load features
        train_data = extract_features(model, transform, device, "train", cfg)
        val_data = extract_features(model, transform, device, "val", cfg)

        train_feats, train_labels = train_data["feats"], train_data["labels"]
        val_feats, val_labels = val_data["feats"].to(device), val_data["labels"].to(device)
        train_feats_dev = train_feats.to(device)
        train_labels_dev = train_labels.to(device)
        n_total = len(train_feats)

        # Get AnchorScore per class
        ds_anchor = anchor_scores.get(ds_name, {}).get("per_class_acc", {})
        anchor_vals = {}
        for i, cname in enumerate(classes):
            anchor_vals[cname] = ds_anchor.get(cname, {}).get("acc", 50.0)

        # Compute per-class budgets for each strategy
        budget = int(n_total * budget_ratio)
        eps = 1.0

        n_uniform_arr = np.full(n_classes, budget // n_classes)

        inv_weights = np.array([1.0 / (anchor_vals[c] + eps) for c in classes])
        n_anchor_weight = np.round(inv_weights / inv_weights.sum() * budget).astype(int)

        prop_weights = np.array([anchor_vals[c] + eps for c in classes])
        n_anchor_prop = np.round(prop_weights / prop_weights.sum() * budget).astype(int)

        # Fix rounding + ensure min 1 per class
        for name, arr in [("uniform", n_uniform_arr), ("anchor_weight", n_anchor_weight),
                          ("anchor_proportional", n_anchor_prop)]:
            arr[:] = np.maximum(arr, 1)
            diff = budget - arr.sum()
            if diff > 0:
                arr[:diff] += 1
            elif diff < 0:
                excess = -diff
                for i in np.argsort(arr)[::-1]:
                    if arr[i] <= 1:
                        continue
                    reduction = min(excess, arr[i] - 1)
                    arr[i] -= reduction
                    excess -= reduction
                    if excess == 0:
                        break
        for arr in [n_uniform_arr, n_anchor_weight, n_anchor_prop]:
            arr[:] = np.maximum(arr, 1)

        logger.info(f"  Total train: {n_total}, Budget: {budget} ({budget_ratio*100:.0f}%)")
        logger.info(f"  Uniform: {dict(zip(classes, n_uniform_arr.tolist()))}")
        logger.info(f"  Anchor-w: {dict(zip(classes, n_anchor_weight.tolist()))}")
        logger.info(f"  Anchor-p: {dict(zip(classes, n_anchor_prop.tolist()))}")

        ds_results = {}
        for strat_name, n_arr in [
            ("uniform", n_uniform_arr),
            ("anchor_weight", n_anchor_weight),
            ("anchor_proportional", n_anchor_prop),
        ]:
            logger.info(f"\n  --- Strategy: {strat_name} ---")
            strat_accs = []
            for seed_offset in range(args.reproduce):
                seed = args.seed + seed_offset
                s_feats, s_labels = sample_data(train_feats_dev, train_labels_dev, n_arr, seed)
                acc, per_class = train_probe(s_feats, s_labels, val_feats, val_labels, n_classes)
                strat_accs.append(acc)
                logger.info(f"    seed={seed}: {acc:.2f}%")

            mean_acc = round(float(np.mean(strat_accs)), 2)
            std_acc = round(float(np.std(strat_accs)), 2)
            logger.info(f"    => {mean_acc} ± {std_acc}%")
            ds_results[strat_name] = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "per_seed": strat_accs,
                "budget_per_class": n_arr.tolist(),
            }

        all_results[ds_name] = ds_results

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  ACTIVE LEARNING SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Dataset':22s} {'Uniform':>14s}  {'Anchor-w':>14s}  {'Anchor-p':>14s}")
    for ds_name, ds_results in all_results.items():
        u = ds_results["uniform"]["mean_acc"]
        aw = ds_results["anchor_weight"]["mean_acc"]
        ap = ds_results["anchor_proportional"]["mean_acc"]
        print(f"  {ds_name:22s}  {u:>6.2f}% ±{ds_results['uniform']['std_acc']:.2f}"
              f"  {aw:>6.2f}% ±{ds_results['anchor_weight']['std_acc']:.2f}"
              f"  {ap:>6.2f}% ±{ds_results['anchor_proportional']['std_acc']:.2f}")

    out_path = out_dir / "active_learning_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
