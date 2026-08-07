"""
active_learning_reanalysis.py

Reanalysis of active learning experiments with:
  1. Macro (balanced) accuracy as primary metric — not just micro
  2. Proportional uniform as the main baseline (not balanced uniform)
  3. 10 seeds for stability

Strategies:
  uniform              — equal per class (balanced uniform)
  proportional_uniform — proportional to class frequency in training set
  anchor_capped        — shift 30% budget from easy→hard, capped [0.5x, 2x]

Usage:
  python active_learning_reanalysis.py [--reproduce 10] [--budget_ratio 0.2]
"""

import argparse
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


def load_clip():
    model, _, transform = load_clip_model("ViT-B-32", "openai")
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
        batch = samples[i: i + bs]
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

    per_class_acc = {}
    for c in range(n_classes):
        mask = val_labels == c
        n = mask.sum().item()
        correct = (preds[mask] == c).sum().item() if n else 0
        per_class_acc[c] = correct / n * 100 if n else 0.0

    micro = float((preds == val_labels).float().mean().item() * 100)
    macro = float(np.mean(list(per_class_acc.values())))

    return round(micro, 2), round(macro, 2), {c: round(a, 2) for c, a in per_class_acc.items()}


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


def allocate_uniform(n_total, n_classes):
    per = n_total // n_classes
    alloc = np.full(n_classes, per, dtype=int)
    alloc[:n_total - alloc.sum()] += 1
    return alloc


def allocate_proportional(n_total, class_counts):
    w = np.array(class_counts, dtype=float)
    w = np.maximum(w, 1)
    w = w / w.sum()
    alloc = np.round(w * n_total).astype(int)
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


def allocate_capped(n_total, n_classes, anchor_vals, uniform_per_class,
                    budget_shift=0.3, cap_min=0.5, cap_max=2.0):
    base = np.full(n_classes, uniform_per_class, dtype=float)
    median = np.median(list(anchor_vals.values()))
    hard = [i for i, c in enumerate(anchor_vals) if anchor_vals[c] <= median]
    easy = [i for i, c in enumerate(anchor_vals) if anchor_vals[c] > median]
    shift_amount = n_total * budget_shift / n_classes

    for i in easy:
        base[i] = max(base[i] - shift_amount, uniform_per_class * cap_min)
    for i in hard:
        base[i] = min(base[i] + shift_amount, uniform_per_class * cap_max)

    base = np.round(base).astype(int)
    base = np.maximum(base, 1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_ratio", type=float, default=0.2)
    parser.add_argument("--budget_shift", type=float, default=0.3)
    parser.add_argument("--reproduce", type=int, default=10)
    parser.add_argument("--anchor-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    anchor_scores = anchor_base.get("laion_l14", {}).get("results", {})

    cache_subdir = CACHE_DIR
    strategies = ["uniform", "proportional_uniform", "anchor_capped"]
    all_results = {}

    for ds_name, cfg in DATASET_CFG.items():
        logger.info(f"\n{'='*50}\n{ds_name}\n{'='*50}")
        classes = cfg["classes"]
        n_classes = len(classes)

        train = extract_features(model, transform, device, "train", cfg, cache_subdir)
        val = extract_features(model, transform, device, "val", cfg, cache_subdir)

        tfeats = train["feats"].to(device)
        tlabels = train["labels"].to(device)
        vfeats = val["feats"].to(device)
        vlabels = val["labels"].to(device)

        n_total = len(tfeats)
        budget = int(n_total * args.budget_ratio)
        uniform_per_class = budget // n_classes

        class_counts = []
        for c in range(n_classes):
            class_counts.append(int((tlabels == c).sum().item()))

        ds_anchor = anchor_scores.get(ds_name, {}).get("per_class_acc", {})
        anchor_vals = {c: ds_anchor.get(c, {}).get("acc", 50.0) for c in classes}

        n_alloc = {
            "uniform": allocate_uniform(budget, n_classes),
            "proportional_uniform": allocate_proportional(budget, class_counts),
            "anchor_capped": allocate_capped(budget, n_classes, anchor_vals,
                                              uniform_per_class, args.budget_shift),
        }

        for sn in strategies:
            logger.info(f"  {sn}: {dict(zip(classes, n_alloc[sn]))}")

        ds_results = {}
        for sn in strategies:
            logger.info(f"\n  --- {sn} ---")
            micros, macros = [], []
            per_seed_detail = []
            for s in range(args.reproduce):
                seed = 42 + s
                sf, sl = sample_data(tfeats, tlabels, n_alloc[sn], seed)
                micro, macro, pca = train_probe(sf, sl, vfeats, vlabels, n_classes)
                micros.append(micro)
                macros.append(macro)
                per_seed_detail.append({"seed": seed, "micro": micro, "macro": macro, "per_class": pca})
                logger.info(f"    seed={seed}: micro={micro:.2f}% macro={macro:.2f}%")

            ds_results[sn] = {
                "micro_mean": round(float(np.mean(micros)), 2),
                "micro_std": round(float(np.std(micros)), 2),
                "macro_mean": round(float(np.mean(macros)), 2),
                "macro_std": round(float(np.std(macros)), 2),
                "per_seed": per_seed_detail,
            }
            logger.info(f"    => micro={ds_results[sn]['micro_mean']}±{ds_results[sn]['micro_std']}%  "
                        f"macro={ds_results[sn]['macro_mean']}±{ds_results[sn]['macro_std']}%")

        all_results[ds_name] = ds_results

    print(f"\n{'='*70}")
    print(f"  ACTIVE LEARNING REANALYSIS (Micro / Macro Accuracy)")
    print(f"{'='*70}")
    print(f"  {'Dataset':22s}  {'Uniform':>16s}  {'Proportional':>16s}  {'AnchorCap':>16s}")
    for ds_name, dsr in all_results.items():
        u = dsr["uniform"]
        p = dsr["proportional_uniform"]
        c = dsr["anchor_capped"]
        print(f"  {ds_name:22s}  {u['micro_mean']:>6.1f}/{u['macro_mean']:<6.1f}  "
              f"{p['micro_mean']:>6.1f}/{p['macro_mean']:<6.1f}  "
              f"{c['micro_mean']:>6.1f}/{c['macro_mean']:<6.1f}")

    print(f"\n  Delta vs Proportional (macro accuracy):")
    for ds_name, dsr in all_results.items():
        p_macro = dsr["proportional_uniform"]["macro_mean"]
        for sn in ["uniform", "anchor_capped"]:
            delta = dsr[sn]["macro_mean"] - p_macro
            print(f"    {ds_name:22s}  {sn:22s}: {delta:+.2f} pp")

    out_path = out_dir / "reanalysis_macro.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
