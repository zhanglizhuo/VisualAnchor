"""
active_learning_capped_b32.py

Capped AnchorScore-guided data selection for linear probing (ViT-B/32).

Motivation: naive 1/AnchorScore weighting over-allocates to extremely
low-score classes. Capped strategy ensures each class gets at least
50% and at most 200% of uniform allocation.

Strategies:
  uniform       — equal per class
  anchor_capped  — shift up to budget_shift% of budget from easy→hard classes
                   with per-class caps [0.5×, 2×] of uniform
  anchor_weight  — naive 1/(acc+ε) weighting (for comparison)
"""

import json
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import read_label, load_clip_model

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

CACHE_DIR = Path(os.environ.get("FEATURE_CACHE", str(PROJ / "data" / "_cache" / "features")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_clip(tag="ViT-B-32"):
    model, _, transform = load_clip_model(tag, "openai")
    return model, transform


def extract_features(model, transform, device, split, cfg, cache_subdir):
    cache_key = f"v1_b32_{split}_{cfg['dir']}_{cfg.get('subdir', 'none')}.pkl"
    cache_path = cache_subdir / cache_key
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Loaded cached {split} ({len(data['feats'])} samples)")
        return data

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
        if cid is None:
            continue
        samples.append((img_path, cid))

    model_dtype = next(model.parameters()).dtype
    feats, labels = [], []
    bs = 64
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p, _ in batch
        ]).to(device).to(model_dtype)
        batch_labels = [c for _, c in batch]
        with torch.no_grad():
            f = model.encode_image(images)
            f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
        labels.extend(batch_labels)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
    data = {"feats": feats, "labels": labels, "class_names": cfg["classes"]}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Saved {split} ({len(feats)})")
    return data


def train_probe(train_feats, train_labels, val_feats, val_labels, n_classes,
                n_epochs=1000, lr=0.01):
    train_feats = train_feats.float()
    val_feats = val_feats.float()
    d = train_feats.shape[1]
    clf = nn.Linear(d, n_classes).to(train_feats.device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(n_epochs):
        clf.train()
        logits = clf(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            acc = (clf(val_feats).argmax(1) == val_labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = clf.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 50:
                break

    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        preds = clf(val_feats).argmax(1)
    per_class = {}
    for c in range(n_classes):
        mask = val_labels == c
        n = mask.sum().item()
        correct = (preds[mask] == c).sum().item() if n else 0
        per_class[c] = {"n": n, "acc": round(correct / n * 100, 2) if n else 0.0}
    overall = round(sum(c["acc"] * c["n"] for c in per_class.values()) /
                    sum(c["n"] for c in per_class.values()), 2)
    return overall, per_class


def sample_data(feats, labels, n_per_class, seed):
    rng = np.random.RandomState(seed)
    indices = []
    for c, n_want in enumerate(n_per_class):
        c_mask = (labels == c).nonzero(as_tuple=True)[0]
        n_avail = len(c_mask)
        n = min(n_want, n_avail)
        chosen = c_mask[rng.choice(n_avail, size=n, replace=False)]
        indices.append(chosen)
    idx = torch.cat(indices)
    return feats[idx], labels[idx]


def allocate_capped(n_total, n_classes, anchor_vals, uniform_per_class,
                    budget_shift=0.3, cap_min=0.5, cap_max=2.0):
    """Capped allocation: shift budget_shift fraction of budget
    from above-median AnchorScore classes to below-median classes."""
    base = np.full(n_classes, uniform_per_class, dtype=float)
    median = np.median(list(anchor_vals.values()))
    hard = [i for i, c in enumerate(anchor_vals) if anchor_vals[c] <= median]
    easy = [i for i, c in enumerate(anchor_vals) if anchor_vals[c] > median]
    shift_amount = n_total * budget_shift / n_classes  # total shifted per class

    for i in easy:
        base[i] = max(base[i] - shift_amount, uniform_per_class * cap_min)
    for i in hard:
        base[i] = min(base[i] + shift_amount, uniform_per_class * cap_max)

    base = np.round(base).astype(int)
    diff = n_total - base.sum()
    if diff > 0:
        base[:diff] += 1
    elif diff < 0:
        for i in np.argsort(base)[::-1]:
            if base[i] <= 1:
                continue
            d = min(-diff, base[i] - 1)
            base[i] -= d
            diff += d
            if diff >= 0:
                break
    return base


def allocate_naive_weight(n_total, n_classes, anchor_vals):
    eps = 1.0
    w = np.array([1.0 / (anchor_vals[c] + eps) for c in anchor_vals])
    alloc = np.round(w / w.sum() * n_total).astype(int)
    alloc = np.maximum(alloc, 1)
    diff = n_total - alloc.sum()
    if diff > 0:
        alloc[:diff] += 1
    elif diff < 0:
        for i in np.argsort(alloc)[::-1]:
            if alloc[i] <= 1:
                continue
            d = min(-diff, alloc[i] - 1)
            alloc[i] -= d
            diff += d
            if diff >= 0:
                break
    return alloc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_ratio", type=float, default=0.2)
    parser.add_argument("--budget_shift", type=float, default=0.3)
    parser.add_argument("--reproduce", type=int, default=5)
    parser.add_argument("--anchor-path", type=str, default=None,
                        help="Path to backbone_results.json (default: results/02_robustness/multi_backbone/backbone_results.json)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="SCB5 data root (default: $SCB5_DATA_ROOT or data/scb5)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results/05_applications/active_learning)")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Override global paths
    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    anchor_path = Path(args.anchor_path or str(PROJ / "results" / "02_robustness" / "multi_backbone" / "backbone_results.json"))
    out_dir = Path(args.out or str(PROJ / "results" / "05_applications" / "active_learning"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, transform = load_clip()
    model = model.to(device).eval().half()

    with open(anchor_path) as f:
        anchor_base = json.load(f)
    # Use LAION L/14 AnchorScore as reference
    anchor_scores = anchor_base.get("laion_l14", {}).get("results", {})

    cache_subdir = CACHE_DIR
    all_results = {}
    strategies = ["uniform", "anchor_capped", "anchor_weight"]

    for ds_name, cfg in DATASET_CFG.items():
        logger.info(f"\n{'='*50}\n{ds_name}\n{'='*50}")
        classes = cfg["classes"]
        n_classes = len(classes)
        ratio = args.budget_ratio

        train = extract_features(model, transform, device, "train", cfg, cache_subdir)
        val = extract_features(model, transform, device, "val", cfg, cache_subdir)

        tfeats = train["feats"].to(device)
        tlabels = train["labels"].to(device)
        vfeats = val["feats"].to(device)
        vlabels = val["labels"].to(device)

        n_total = len(tfeats)
        budget = int(n_total * ratio)
        uniform_per_class = budget // n_classes
        remainder = budget - uniform_per_class * n_classes
        n_uniform = np.full(n_classes, uniform_per_class, dtype=int)
        n_uniform[:remainder] += 1

        ds_anchor = anchor_scores.get(ds_name, {}).get("per_class_acc", {})
        anchor_vals = {c: ds_anchor.get(c, {}).get("acc", 50.0) for c in classes}

        n_alloc = {
            "uniform": n_uniform,
            "anchor_capped": allocate_capped(budget, n_classes, anchor_vals,
                                             uniform_per_class, args.budget_shift),
            "anchor_weight": allocate_naive_weight(budget, n_classes, anchor_vals),
        }

        for sn in strategies:
            logger.info(f"  {sn}: {dict(zip(classes, n_alloc[sn]))}")

        ds_results = {}
        for sn in strategies:
            logger.info(f"\n  --- {sn} ---")
            accs = []
            for s in range(args.reproduce):
                seed = 42 + s
                sf, sl = sample_data(tfeats, tlabels, n_alloc[sn], seed)
                acc, _ = train_probe(sf, sl, vfeats, vlabels, n_classes)
                accs.append(acc)
                logger.info(f"    seed={seed}: {acc:.2f}%")
            mean = round(float(np.mean(accs)), 2)
            std = round(float(np.std(accs)), 2)
            logger.info(f"    => {mean} ± {std}%")
            ds_results[sn] = {"mean_acc": mean, "std_acc": std, "per_seed": accs}

        all_results[ds_name] = ds_results

    # Summary
    print(f"\n{'='*60}")
    print(f"  ACTIVE LEARNING RESULTS")
    print(f"{'='*60}")
    print(f"  {'Dataset':22s} {'Uniform':>14s}  {'Capped(±30%)':>14s}  {'Naive-W':>14s}")
    for ds_name, dsr in all_results.items():
        u = dsr["uniform"]
        c = dsr["anchor_capped"]
        n = dsr["anchor_weight"]
        print(f"  {ds_name:22s}  {u['mean_acc']:>6.2f}±{u['std_acc']:.2f}"
              f"  {c['mean_acc']:>6.2f}±{c['std_acc']:.2f}"
              f"  {n['mean_acc']:>6.2f}±{n['std_acc']:.2f}")

    out_path = out_dir / "active_learning_capped.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
