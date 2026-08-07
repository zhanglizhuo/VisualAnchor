#!/usr/bin/env python3
"""
Verify that the class-level headline ρ=0.769 (n=13) is unchanged after adding
Qwen3.6-35B's TeacherBehavior data.

Reads:
  - results/01_core/paper_data/mllm_raw.json (canonical 6-model MLLM data)
  - results/01_core/anchor_score_scb5/anchor_scores.json (LAION ViT-L/14)
  - results/02_robustness/multi_backbone/backbone_results.json (all backbones)

Outputs:
  - results/01_core/correlation/class_level_comparison.json

Usage:
  python analysis/01_core/verify_class_level_rho.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "results"

MLLM_FILE = RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json"
ANCHOR_FILE = RESULTS_DIR / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
BACKBONE_FILE = RESULTS_DIR / "02_robustness" / "multi_backbone" / "backbone_results.json"
OUT_FILE = RESULTS_DIR / "01_core" / "correlation" / "class_level_comparison.json"

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

BACKBONE_KEYS = {
    "laion_l14": "LAION ViT-L/14",
    "openai_l14": "OpenAI ViT-L/14",
    "openai_b32": "OpenAI ViT-B/32",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_class_level(anchor_src, mllm_data):
    """Compute class-level (n=13) anchor and mean MLLM accuracy arrays."""
    class_anchor, class_mllm = [], []
    class_names = []
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        ds_acc = anchor_src[ds_name]["per_class_acc"]
        for cname in CLASS_MAP[ds_name]:
            vals = [
                mdata[cname]
                for mname, mdata in mllm_data.get(ds_name, {}).items()
                if not mname.startswith("_") and cname in mdata
            ]
            if vals:
                class_anchor.append(ds_acc[cname]["acc"])
                class_mllm.append(np.mean(vals))
                class_names.append(f"{ds_name}/{cname}")
    return np.array(class_anchor), np.array(class_mllm), class_names


def compute(mllm_data, anchor_src, label):
    class_anchor, class_mllm, names = build_class_level(anchor_src, mllm_data)
    sp = spearmanr(class_anchor, class_mllm)
    print(f"\n{label}:")
    print(f"  n = {len(class_anchor)}")
    print(f"  Spearman ρ = {sp.statistic:.4f}")
    print(f"  p = {sp.pvalue:.4f}")
    return {
        "n": len(class_anchor),
        "spearman_rho": round(sp.statistic, 4),
        "spearman_p": round(sp.pvalue, 6),
    }


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    mllm = load_json(MLLM_FILE)
    anchor = load_json(ANCHOR_FILE)
    backbone = load_json(BACKBONE_FILE)

    # Strip metadata keys
    mllm_data = {
        ds: {k: v for k, v in models.items() if not k.startswith("_")}
        for ds, models in mllm.items() if not ds.startswith("_")
    }

    # Truncated version: remove Qwen3.6-35B from TeacherBehavior
    mllm_trunc = {ds: dict(models) for ds, models in mllm_data.items()}
    mllm_trunc["TeacherBehavior"] = {
        m: d for m, d in mllm_data["TeacherBehavior"].items() if m != "Qwen3.6-35B"
    }

    results = {}

    # --- LAION ViT-L/14 (main AnchorScore) ---
    print("=" * 60)
    print("CLASS-LEVEL CORRELATION COMPARISON")
    print("=" * 60)

    r_full = compute(mllm_data, anchor, "Full 78-point (6 models, all classes complete)")
    r_trunc = compute(mllm_trunc, anchor, "Original 70-point (Qwen3.6-35B missing TeacherBehavior)")

    print(f"\n  Δρ = {r_full['spearman_rho'] - r_trunc['spearman_rho']:+0.4f}")
    print(f"  → Class-level ρ is {'IDENTICAL' if r_full['spearman_rho'] == r_trunc['spearman_rho'] else 'CHANGED'}")

    results["laion_l14"] = {
        "description": "LAION ViT-L/14 class-level (n=13)",
        "full_78pt": r_full,
        "truncated_70pt": r_trunc,
        "delta_rho": round(r_full["spearman_rho"] - r_trunc["spearman_rho"], 4),
        "unchanged": r_full["spearman_rho"] == r_trunc["spearman_rho"],
    }

    # --- Show per-class means ---
    print("\n\nPer-class MLLM mean comparison:")
    print(f"  {'Class':30s} {'n_models(78pt)':>14s} {'MLLM_mean(78pt)':>15s} {'n_models(70pt)':>14s} {'MLLM_mean(70pt)':>15s}")
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        for cname in CLASS_MAP[ds_name]:
            vals_full = [mllm_data[ds_name][m][cname] for m in mllm_data[ds_name] if cname in mllm_data[ds_name][m]]
            vals_trunc = [mllm_trunc[ds_name][m][cname] for m in mllm_trunc[ds_name] if cname in mllm_trunc[ds_name][m]]
            if vals_full and vals_trunc:
                print(f"  {cname:30s} {len(vals_full):>14d} {np.mean(vals_full):>14.1f}% {len(vals_trunc):>14d} {np.mean(vals_trunc):>14.1f}%")

    # --- Backbone class-level comparison ---
    print("\n\n" + "=" * 60)
    print("BACKBONE CLASS-LEVEL COMPARISON")
    print("=" * 60)

    results["backbones"] = {}
    for bb_key, bb_label in BACKBONE_KEYS.items():
        bb_data = backbone[bb_key]

        r_bb_full = compute(mllm_data, bb_data["results"], f"{bb_label}: full 78pt")
        r_bb_trunc = compute(mllm_trunc, bb_data["results"], f"{bb_label}: truncated 70pt")

        print(f"  Δρ = {r_bb_full['spearman_rho'] - r_bb_trunc['spearman_rho']:+0.4f}")
        print(f"  → {'IDENTICAL' if r_bb_full['spearman_rho'] == r_bb_trunc['spearman_rho'] else 'CHANGED'}")

        results["backbones"][bb_key] = {
            "model": bb_label,
            "full_78pt": r_bb_full,
            "truncated_70pt": r_bb_trunc,
            "delta_rho": round(r_bb_full["spearman_rho"] - r_bb_trunc["spearman_rho"], 4),
            "unchanged": r_bb_full["spearman_rho"] == r_bb_trunc["spearman_rho"],
        }

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_, bool)) else float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
