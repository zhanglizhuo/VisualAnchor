"""
vl_baseline_correlate.py

Compute Spearman correlation between a VL model's per-class zero-shot
accuracy and the canonical MLLM mean accuracy (6 core MLLMs from
pooled_class_level_results.json), mirroring the AnchorScore analysis.

This script loads a JSON file in the anchor_score format (same structure
as anchor_scb5.py output) and computes the class-level Spearman rho
against the canonical MLLM accuracy from pooled_class_level_results.json.

Usage:
  python vl_baseline_correlate.py \
    --vl-results results/01_core/anchor_score_scb5/siglip_anchor.json \
    --out results/01_core/anchor_score_scb5/siglip_correlation.json

  python vl_baseline_correlate.py \
    --vl-results results/01_core/anchor_score_scb5/blip2_anchor.json \
    --out results/01_core/anchor_score_scb5/blip2_correlation.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent


def load_vl_accuracy(vl_path):
    """Load per-class VL accuracy from an anchor_score format JSON."""
    with open(vl_path) as f:
        data = json.load(f)
    per_class = {}
    for ds_name, ds_data in data.items():
        for cls_name, cls_info in ds_data["per_class_acc"].items():
            per_class[cls_name] = cls_info["acc"]
    return per_class


def load_mllm_accuracy():
    """Load per-class MLLM mean accuracy from canonical source (6 core MLLMs)."""
    path = PROJ / "results" / "01_core" / "correlation" / "pooled_class_level_results.json"
    with open(path) as f:
        data = json.load(f)["data"]
    return {entry["class"]: entry["mllm_mean"] for entry in data if entry["domain"].startswith("SCB5")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vl-results", type=str, required=True,
                        help="Path to VL model results JSON (anchor_score format)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for correlation results")
    args = parser.parse_args()

    vl_path = Path(args.vl_results)
    if not vl_path.exists():
        logger.error(f"VL results not found: {vl_path}")
        return

    vl_accs = load_vl_accuracy(vl_path)
    mllm_accs = load_mllm_accuracy()

    common_classes = [c for c in vl_accs if c in mllm_accs]
    vl_vals = [vl_accs[c] for c in common_classes]
    mllm_vals = [mllm_accs[c] for c in common_classes]

    logger.info(f"Common classes: {len(common_classes)}")
    for c, v, m in sorted(zip(common_classes, vl_vals, mllm_vals)):
        logger.info(f"  {c:28s}  VL={v:6.2f}%  MLLM={m:6.2f}%")

    rho, p = spearmanr(vl_vals, mllm_vals)
    logger.info(f"\nSpearman rho = {rho:.3f}  (p = {p:.4f}, n = {len(common_classes)})")

    results = {
        "model_label": vl_path.stem.replace("_anchor", ""),
        "model": vl_path.stem,
        "spearman_vs_mllm": {
            "rho": round(rho, 4),
            "p": round(p, 4),
            "n": len(common_classes),
        },
        "per_class_vl": vl_accs,
        "per_class_mllm": {c: round(mllm_accs[c], 1) for c in common_classes},
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {out_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
