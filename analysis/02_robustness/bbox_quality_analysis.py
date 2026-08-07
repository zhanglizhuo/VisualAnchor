"""
Compute per-class bbox quality statistics (area ratio, aspect ratio)
and correlate with AnchorScore accuracy drop (full-image vs bbox).

Usage:
    python bbox_quality_analysis.py --data-root /path/to/scb5 --anchor-results anchor_scores.json --anchor-bbox-results anchor_scores_bbox.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
        "subdir": "SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17",
        "classes": ["BowHead", "TurnHead"],
    },
}


def compute_bbox_stats(data_root):
    data_root = Path(data_root)
    stats = {}  # dataset -> class_name -> {area_ratios: [], aspect_ratios: []}

    for ds_name, cfg in DATASET_CFG.items():
        base = data_root / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" if sub else base / "images") / "val"
        lbl_dir = (base / sub / "labels" if sub else base / "labels") / "val"

        if not img_dir.exists():
            logger.warning(f"  image dir not found: {img_dir}")
            continue

        classes = cfg["classes"]
        ds_stats = {c: {"area_ratios": [], "aspect_ratios": []} for c in classes}

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Read image size
            with Image.open(img_path) as tmp:
                img_w, img_h = tmp.size

            # Read YOLO bbox
            with open(lbl_path) as f:
                line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            if cid >= len(classes):
                continue
            cls_name = classes[cid]
            x_c, y_c, w_norm, h_norm = map(float, parts[1:5])

            # bbox area ratio (normalized to [0,1])
            area_ratio = w_norm * h_norm
            # bbox aspect ratio (pixel-space)
            bbox_w = w_norm * img_w
            bbox_h = h_norm * img_h
            aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0

            ds_stats[cls_name]["area_ratios"].append(area_ratio)
            ds_stats[cls_name]["aspect_ratios"].append(aspect_ratio)

        stats[ds_name] = ds_stats
        n_total = sum(len(v["area_ratios"]) for v in ds_stats.values())
        logger.info(f"  {ds_name}: {n_total} images, {len(classes)} classes")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--anchor-results", required=True)
    parser.add_argument("--anchor-bbox-results", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Load anchor scores
    with open(args.anchor_results) as f:
        full_anchor = json.load(f)
    with open(args.anchor_bbox_results) as f:
        bbox_anchor = json.load(f)

    # Compute bbox stats
    logger.info("Computing bbox quality statistics...")
    stats = compute_bbox_stats(args.data_root)

    # Per-class analysis
    logger.info("\n=== Per-class bbox quality vs accuracy drop ===\n")
    rows = []
    for ds_name, ds_stats in stats.items():
        for cls_name, cls_stats in ds_stats.items():
            area_ratios = cls_stats["area_ratios"]
            aspect_ratios = cls_stats["aspect_ratios"]
            if len(area_ratios) == 0:
                continue

            full_acc = full_anchor[ds_name]["per_class_acc"][cls_name]["acc"]
            bb_acc = bbox_anchor[ds_name]["per_class_acc"][cls_name]["acc"]
            acc_drop = full_acc - bb_acc

            rows.append({
                "dataset": ds_name,
                "class": cls_name,
                "n": len(area_ratios),
                "mean_area_ratio": float(np.mean(area_ratios)),
                "median_area_ratio": float(np.median(area_ratios)),
                "std_area_ratio": float(np.std(area_ratios)),
                "mean_aspect_ratio": float(np.mean(aspect_ratios)),
                "median_aspect_ratio": float(np.median(aspect_ratios)),
                "full_acc": full_acc,
                "bbox_acc": bb_acc,
                "acc_drop": acc_drop,
            })

    # Print table
    print(f"{'Dataset':20s} {'Class':25s} {'n':>5s} {'AreaRatio':>9s} {'AspectR':>7s} {'Full':>7s} {'Bbox':>7s} {'Drop':>7s}")
    print("-" * 85)
    for r in rows:
        print(f"{r['dataset']:20s} {r['class']:25s} {r['n']:5d} {r['mean_area_ratio']:9.4f} {r['mean_aspect_ratio']:7.2f} {r['full_acc']:6.2f}% {r['bbox_acc']:6.2f}% {r['acc_drop']:+6.2f}pp")

    # Correlations: area_ratio vs acc_drop
    area_ratios = [r["mean_area_ratio"] for r in rows]
    acc_drops = [r["acc_drop"] for r in rows]

    if len(rows) >= 3:
        sr, sp = spearmanr(area_ratios, acc_drops)
        pr, pp = pearsonr(area_ratios, acc_drops)
        logger.info(f"\n=== Area ratio vs accuracy drop ===")
        logger.info(f"Spearman ρ = {sr:.4f} (p={sp:.4e})")
        logger.info(f"Pearson  r = {pr:.4f} (p={pp:.4e})")

        # Also: area_ratio vs bbox_acc
        bbox_accs = [r["bbox_acc"] for r in rows]
        sr2, sp2 = spearmanr(area_ratios, bbox_accs)
        logger.info(f"\n=== Area ratio vs bbox AnchorScore ===")
        logger.info(f"Spearman ρ = {sr2:.4f} (p={sp2:.4e})")

        # Also: area_ratio vs full_acc
        full_accs = [r["full_acc"] for r in rows]
        sr3, sp3 = spearmanr(area_ratios, full_accs)
        logger.info(f"\n=== Area ratio vs full-image AnchorScore ===")
        logger.info(f"Spearman ρ = {sr3:.4f} (p={sp3:.4e})")

    out_path = args.out
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
