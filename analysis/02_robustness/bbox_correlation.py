#!/usr/bin/env python3
"""
Compute the full-frame vs bbox-aligned AnchorScore correlation with MLLM accuracy.

Reads:
  - results/01_core/paper_data/mllm_raw.json  (canonical 6-model MLLM data)
  - results/01_core/anchor_score_scb5/anchor_scores.json  (full-frame CLIP)
  - results/01_core/anchor_score_scb5_bbox/anchor_scores_bbox.json  (bbox-cropped CLIP)

Outputs:
  - results/02_robustness/input_repr/input_repr_results.json

Usage:
  python analysis/02_robustness/bbox_correlation.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "results"

MLLM_FILE = RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json"
FULL_ANCHOR_FILE = RESULTS_DIR / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
BBOX_ANCHOR_FILE = RESULTS_DIR / "01_core" / "anchor_score_scb5_bbox" / "anchor_scores_bbox.json"
OUT_FILE = RESULTS_DIR / "02_robustness" / "input_repr" / "input_repr_results.json"

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_paired_data(mllm_data, anchor_src):
    """Build aligned arrays of (anchor_score, mllm_accuracy) pairs."""
    anchor_vals, mllm_vals = [], []
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        ds_acc = anchor_src[ds_name]["per_class_acc"]
        for cname in CLASS_MAP[ds_name]:
            a_val = ds_acc[cname]["acc"]
            for mname, mdata in mllm_data.get(ds_name, {}).items():
                if mname.startswith("_"):
                    continue
                if cname in mdata:
                    anchor_vals.append(a_val)
                    mllm_vals.append(mdata[cname])
    return np.array(anchor_vals), np.array(mllm_vals)


def compute_bootstrap_ci(anchor_vals, mllm_vals, n_bootstrap=5000, seed=42):
    """Point-level bootstrap CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(anchor_vals)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = spearmanr(anchor_vals[idx], mllm_vals[idx]).statistic
    return np.percentile(boot, [2.5, 97.5])


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    mllm = load_json(MLLM_FILE)
    full_anchor = load_json(FULL_ANCHOR_FILE)
    bbox_anchor = load_json(BBOX_ANCHOR_FILE)

    full_a, all_m = build_paired_data(mllm, full_anchor)
    bbox_a, _ = build_paired_data(mllm, bbox_anchor)

    sp_full = spearmanr(full_a, all_m)
    sp_bbox = spearmanr(bbox_a, all_m)
    delta = sp_full.statistic - sp_bbox.statistic

    ci_full = compute_bootstrap_ci(full_a, all_m)
    ci_bbox = compute_bootstrap_ci(bbox_a, all_m)

    # Bootstrap CI for delta (paired difference)
    rng = np.random.default_rng(42)
    n = len(all_m)
    boot_delta = np.zeros(5000)
    for i in range(5000):
        idx = rng.integers(0, n, size=n)
        rf = spearmanr(full_a[idx], all_m[idx]).statistic
        rb = spearmanr(bbox_a[idx], all_m[idx]).statistic
        boot_delta[i] = rf - rb
    ci_delta = np.percentile(boot_delta, [2.5, 97.5])

    results = {
        "n_points": int(n),
        "full_frame": {
            "spearman_rho": round(sp_full.statistic, 4),
            "p_value": sp_full.pvalue,
            "ci_95": [round(float(ci_full[0]), 4), round(float(ci_full[1]), 4)],
        },
        "bbox_aligned": {
            "spearman_rho": round(sp_bbox.statistic, 4),
            "p_value": sp_bbox.pvalue,
            "ci_95": [round(float(ci_bbox[0]), 4), round(float(ci_bbox[1]), 4)],
        },
        "delta": {
            "value": round(delta, 4),
            "ci_95": [round(float(ci_delta[0]), 4), round(float(ci_delta[1]), 4)],
            "attenuation_pct": round(delta / sp_full.statistic * 100, 1),
        },
        "method": "Point-level bootstrap, B=5000, seed=42",
        "input_files": {
            "mllm": str(MLLM_FILE),
            "full_anchor": str(FULL_ANCHOR_FILE),
            "bbox_anchor": str(BBOX_ANCHOR_FILE),
        },
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Input representation correlation (n={results['n_points']}):")
    print(f"  Full-frame:  ρ={results['full_frame']['spearman_rho']:.3f}  "
          f"95% CI {results['full_frame']['ci_95']}  "
          f"p={results['full_frame']['p_value']:.1e}")
    print(f"  Bbox-aligned: ρ={results['bbox_aligned']['spearman_rho']:.3f}  "
          f"95% CI {results['bbox_aligned']['ci_95']}  "
          f"p={results['bbox_aligned']['p_value']:.1e}")
    print(f"  Δρ: {results['delta']['value']:.3f}  "
          f"95% CI {results['delta']['ci_95']}  "
          f"(attenuation {results['delta']['attenuation_pct']:.1f}%)")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
