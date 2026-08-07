#!/usr/bin/env python3
"""
stanford40_ranking.py

Downstream ranking validation on Stanford40 Actions (k=40).

Mirrors analysis/05_applications/downstream_ranking.py but on Stanford40,
where n=40 classes give the ranking analysis real statistical power:

  - hard classes = below-median mean MLLM accuracy (20 of 40)
  - AUC + bootstrap 95% CI (5000 replicates, seed 42)
  - permutation test of no-ranking null (10^4 shuffles)
  - leave-one-class-out (LOCO) AUC sensitivity
  - precision/recall at k/3 review budget (ceil(40/3)=14 classes)
  - random-baseline expectation for the k/3 budget

Usage:
    python analysis/05_applications/stanford40_ranking.py
Output:
    results/05_applications/ranking/stanford40_ranking.json
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
ANCHOR_FILE = RESULTS / "02_robustness" / "stanford40" / "anchor_scores.json"
OLLAMA_FILE = RESULTS / "02_robustness" / "stanford40" / "ollama_results.json"
OUT_PATH = RESULTS / "05_applications" / "ranking" / "stanford40_ranking.json"

RNG = np.random.RandomState(42)
N_BOOT = 5000
N_PERM = 10000


def load_data():
    anchor = json.load(open(ANCHOR_FILE))["per_class_acc"]
    ollama = json.load(open(OLLAMA_FILE))["models"]
    classes = sorted(anchor.keys())
    mllm_models = list(ollama.keys())
    anchor_vals = np.array([anchor[c]["acc"] for c in classes])
    mllm_per_model = {m: np.array([ollama[m]["per_class_acc"][c]["acc"] for c in classes]) for m in mllm_models}
    mllm_mean = np.mean(np.array(list(mllm_per_model.values())), axis=0)
    return classes, anchor_vals, mllm_mean, mllm_models, mllm_per_model


def hard_labels(mllm):
    return (mllm < np.median(mllm)).astype(int)


def compute_auc(anchor, mllm):
    hard = hard_labels(mllm)
    if len(np.unique(hard)) < 2:
        return None, hard
    return roc_auc_score(hard, -anchor), hard


def bootstrap_auc(anchor, mllm, n_boot=N_BOOT):
    hard = hard_labels(mllm)
    auc_obs = roc_auc_score(hard, -anchor)
    auc_boot = []
    for _ in range(n_boot):
        idx = RNG.choice(len(anchor), len(anchor), replace=True)
        h = hard[idx]
        if len(np.unique(h)) == 2:
            try:
                auc_boot.append(roc_auc_score(h, -anchor[idx]))
            except ValueError:
                pass
    auc_boot = np.array(auc_boot)
    if len(auc_boot) < 100:
        return auc_obs, None, None
    return auc_obs, float(np.percentile(auc_boot, 2.5)), float(np.percentile(auc_boot, 97.5))


def permutation_test(anchor, mllm, n_perm=N_PERM):
    hard = hard_labels(mllm)
    auc_obs = roc_auc_score(hard, -anchor)
    count = 0
    for _ in range(n_perm):
        perm_hard = RNG.permutation(hard)
        if len(np.unique(perm_hard)) == 2:
            if roc_auc_score(perm_hard, -anchor) >= auc_obs:
                count += 1
    return auc_obs, (count + 1) / (n_perm + 1)


def loko_auc(anchor, mllm):
    hard = hard_labels(mllm)
    auc_obs = roc_auc_score(hard, -anchor)
    vals = []
    for i in range(len(anchor)):
        mask = np.ones(len(anchor), dtype=bool)
        mask[i] = False
        h = hard[mask]
        if len(np.unique(h)) == 2:
            try:
                vals.append(roc_auc_score(h, -anchor[mask]))
            except ValueError:
                pass
    vals = np.array(vals)
    return {
        "auc": round(float(auc_obs), 4),
        "loko_mean": round(float(vals.mean()), 4),
        "loko_sd": round(float(vals.std()), 4),
        "loko_min": round(float(vals.min()), 4),
        "loko_max": round(float(vals.max()), 4),
        "n_loko": int(len(vals)),
    }


def hit_rate(anchor, mllm):
    hard = hard_labels(mllm)
    n = len(anchor)
    n_hard = int(hard.sum())
    k_third = max(1, int(np.ceil(n / 3)))
    order = np.argsort(anchor)
    top_k = set(order[:k_third])
    hard_set = set(np.where(hard == 1)[0])
    n_caught = len(top_k & hard_set)
    random_expected = k_third * n_hard / n
    return {
        "k": int(k_third),
        "n_hard": int(n_hard),
        "n_caught": int(n_caught),
        "precision": round(n_caught / k_third, 4),
        "recall": round(n_caught / n_hard, 4),
        "random_expected_caught": round(float(random_expected), 4),
        "gain_vs_random": round(n_caught / random_expected, 4),
    }


def main():
    classes, anchor, mllm_mean, mllm_models, mllm_per_model = load_data()

    auc_obs, ci_low, ci_high = bootstrap_auc(anchor, mllm_mean)
    perm_auc, perm_p = permutation_test(anchor, mllm_mean)
    loko = loko_auc(anchor, mllm_mean)
    hits = hit_rate(anchor, mllm_mean)

    per_model = {}
    for m in mllm_models:
        a = anchor
        mm = mllm_per_model[m]
        hard = hard_labels(mm)
        if len(np.unique(hard)) == 2:
            per_model[m] = {
                "auc": round(float(roc_auc_score(hard, -a)), 4),
                "n_hard": int(hard.sum()),
            }
        else:
            per_model[m] = {"auc": None, "n_hard": int(hard.sum())}

    output = {
        "description": "Stanford40 (k=40) downstream ranking: AnchorScore as review priority predictor",
        "n_classes": int(len(classes)),
        "n_hard": int(hard_labels(mllm_mean).sum()),
        "median_mllm": round(float(np.median(mllm_mean)), 2),
        "auc": round(float(auc_obs), 4),
        "auc_ci_95_bootstrap_5000": [round(ci_low, 4), round(ci_high, 4)] if ci_low is not None else None,
        "permutation_p_1e4": round(float(perm_p), 6),
        "loko": loko,
        "hit_rate_at_k_third": hits,
        "per_model_auc": per_model,
        "mllm_models_used": mllm_models,
        "class_anchor_range": [round(float(anchor.min()), 2), round(float(anchor.max()), 2)],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    print(f"Stanford40 ranking (k={len(classes)}):")
    print(f"  AUC = {auc_obs:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}], B=5000)")
    print(f"  Permutation p = {perm_p:.6f} (B={N_PERM})")
    print(f"  LOCO: mean {loko['loko_mean']:.4f} (SD {loko['loko_sd']:.4f}, range {loko['loko_min']:.4f}-{loko['loko_max']:.4f})")
    print(f"  Hit rate @ k/3 ({hits['k']} of {len(classes)}): caught {hits['n_caught']}/{hits['n_hard']} "
          f"precision={hits['precision']:.3f} recall={hits['recall']:.3f} "
          f"(random expects {hits['random_expected_caught']:.1f}, gain {hits['gain_vs_random']:.2f}x)")
    print(f"  Per-model AUC: " + ", ".join(f"{m}: {v['auc']}" for m, v in per_model.items()))
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
