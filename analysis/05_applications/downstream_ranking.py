#!/usr/bin/env python3
"""
downstream_ranking.py

Practical downstream application: use AnchorScore to prioritize which classes
need human review of MLLM outputs.

Rationale: A practitioner with limited review budget can use AnchorScore
(which requires NO MLLM inference — just CLIP zero-shot) to identify classes
where MLLM accuracy is likely lowest, and target human review accordingly.

Metrics:
  - AUC: separation between hard (below-median) and easy classes
  - Hit rate at top-K: how many truly hard classes are caught
  - Precision/Recall at K=ceil(n/3) and K=ceil(n/2)
  - Comparison against random baseline (10,000 Monte Carlo trials)

Usage:
  python analysis/05_applications/downstream_ranking.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "02_robustness" / "robustness"
ANCHOR_FILE = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
MLLM_FILE = RESULTS / "01_core" / "paper_data" / "mllm_full.json"

CLASS_MAP = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction", "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

ALL_9_MLLMS = [
    "Qwen2-VL-7B", "LLaVA-1.5-7B", "Qwen3.5-27B", "Qwen3.6-27B",
    "Qwen3.5-35B", "Qwen3.6-35B", "Gemma-3-27B", "Gemma4-31B", "Gemma4-26B",
]


def load_data():
    with open(ANCHOR_FILE) as f:
        anchor = json.load(f)
    with open(MLLM_FILE) as f:
        src = json.load(f)
    mllm = {
        ds: {m: src[ds][m] for m in ALL_9_MLLMS if m in src.get(ds, {})}
        for ds in src if not ds.startswith("_")
    }
    return anchor, mllm


def ranking_metrics(anchor_vals, mllm_vals):
    anchor_arr = np.array(anchor_vals)
    mllm_arr = np.array(mllm_vals)
    n = len(anchor_arr)

    median_mllm = np.median(mllm_arr)
    hard_labels = (mllm_arr < median_mllm).astype(int)
    n_hard = int(np.sum(hard_labels))

    if n_hard == 0 or n_hard == n:
        auc = 0.5
    else:
        auc = roc_auc_score(hard_labels, -anchor_arr)

    order = np.argsort(anchor_arr)
    per_k = {}
    for k in range(1, n + 1):
        top_k = set(order[:k])
        hard_set = set(np.where(hard_labels == 1)[0])
        n_caught = len(top_k & hard_set)
        per_k[str(k)] = {
            "precision": round(n_caught / k, 4),
            "recall": round(n_caught / n_hard, 4) if n_hard > 0 else 0,
            "n_caught": n_caught,
        }

    k_third = max(1, int(np.ceil(n / 3)))
    k_half = max(1, int(np.ceil(n / 2)))

    return {
        "n_classes": n,
        "n_hard": n_hard,
        "median_mllm": round(float(median_mllm), 2),
        "auc": round(float(auc), 4),
        "hit_rate_at_k_third": per_k.get(str(k_third), {}),
        "hit_rate_at_k_half": per_k.get(str(k_half), {}),
        "per_k": per_k,
    }


def random_baseline(n, n_hard, n_trials=10000):
    rng = np.random.default_rng(42)
    k_third = max(1, int(np.ceil(n / 3)))
    k_half = max(1, int(np.ceil(n / 2)))
    prec_third, prec_half = [], []
    for _ in range(n_trials):
        perm = rng.permutation(n)
        top_third = set(perm[:k_third])
        top_half = set(perm[:k_half])
        hard_set = set(rng.choice(n, n_hard, replace=False))
        prec_third.append(len(top_third & hard_set) / k_third)
        prec_half.append(len(top_half & hard_set) / k_half)
    return {
        "n_trials": n_trials,
        "precision_at_k_third": {
            "mean": round(float(np.mean(prec_third)), 4),
            "std": round(float(np.std(prec_third)), 4),
            "ci_95": [
                round(float(np.percentile(prec_third, 2.5)), 4),
                round(float(np.percentile(prec_third, 97.5)), 4),
            ],
        },
        "precision_at_k_half": {
            "mean": round(float(np.mean(prec_half)), 4),
            "std": round(float(np.std(prec_half)), 4),
            "ci_95": [
                round(float(np.percentile(prec_half, 2.5)), 4),
                round(float(np.percentile(prec_half, 97.5)), 4),
            ],
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor, mllm = load_data()

    all_anchor_pool, all_mllm_pool = [], []
    per_dataset = {}

    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        anchor_acc = anchor[ds_name]["per_class_acc"]
        classes = CLASS_MAP[ds_name]

        ds_anchor, ds_mllm = [], []
        for cname in classes:
            a_val = anchor_acc[cname]["acc"]
            m_vals = [mllm[ds_name][m].get(cname, np.nan) for m in mllm[ds_name]]
            m_vals = [v for v in m_vals if not np.isnan(v)]
            if m_vals:
                m_mean = float(np.mean(m_vals))
                ds_anchor.append(a_val)
                ds_mllm.append(m_mean)
                all_anchor_pool.append(a_val)
                all_mllm_pool.append(m_mean)

        rho = spearmanr(ds_anchor, ds_mllm).statistic if len(ds_anchor) >= 3 else None
        metrics = ranking_metrics(ds_anchor, ds_mllm)
        per_dataset[ds_name] = {
            "n_classes": len(ds_anchor),
            "spearman_rho": round(rho, 4) if rho else None,
            "ranking": metrics,
        }
        print(f"\n  {ds_name} (n={len(ds_anchor)}): AUC={metrics['auc']:.4f}, rho={rho:.4f}" if rho else f"\n  {ds_name} (n={len(ds_anchor)}): AUC={metrics['auc']:.4f}")

    # Pooled
    rho_pool = spearmanr(all_anchor_pool, all_mllm_pool).statistic
    pool_metrics = ranking_metrics(all_anchor_pool, all_mllm_pool)
    baseline = random_baseline(pool_metrics["n_classes"], pool_metrics["n_hard"])

    print(f"\n  POOLED (n={len(all_anchor_pool)}): AUC={pool_metrics['auc']:.4f}, rho={rho_pool:.4f}")
    print(f"  Random baseline @ K/3: prec={baseline['precision_at_k_third']['mean']:.4f} (95% CI {baseline['precision_at_k_third']['ci_95']})")
    print(f"  Random baseline @ K/2: prec={baseline['precision_at_k_half']['mean']:.4f} (95% CI {baseline['precision_at_k_half']['ci_95']})")

    output = {
        "description": "Downstream ranking: AnchorScore as MLLM review priority predictor",
        "per_dataset": per_dataset,
        "pooled": {
            "n_classes": len(all_anchor_pool),
            "spearman_rho": round(rho_pool, 4),
            "ranking": pool_metrics,
        },
        "random_baseline": baseline,
    }

    out_path = OUT_DIR / "downstream_ranking.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
