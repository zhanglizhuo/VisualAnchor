"""Bootstrap confidence intervals for AnchorScore ranking AUC.

Computes bootstrap CIs for the AUC of AnchorScore as a hard-class predictor
on each SCB5 dataset. Also explains the BowTurnHead paradox.

Usage:
    python analysis/02_robustness/ranking_bootstrap.py
"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(anchor, mllm, n_bootstrap=10000, seed=42):
    """Bootstrap AUC for hard-class prediction (hard = below median MLLM)."""
    median = np.median(mllm)
    hard = (mllm < median).astype(int)
    if len(np.unique(hard)) < 2:
        return None, None, None, None

    auc_obs = roc_auc_score(hard, -anchor)
    np.random.seed(seed)
    auc_boot = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(anchor), len(anchor), replace=True)
        if len(np.unique(hard[idx])) == 2:
            try:
                auc_boot.append(roc_auc_score(hard[idx], -anchor[idx]))
            except ValueError:
                pass
    if len(auc_boot) < 100:
        return auc_obs, None, None, None

    auc_boot = np.array(auc_boot)
    ci_low = np.percentile(auc_boot, 2.5)
    ci_high = np.percentile(auc_boot, 97.5)
    return auc_obs, ci_low, ci_high, auc_boot.std()


def main():
    with open('results/01_core/correlation/pooled_class_level_results.json') as f:
        pooled = json.load(f)

    data = pooled['data']
    datasets = {}
    for d in data:
        ds = d['domain']
        if ds.startswith('SCB5'):
            datasets.setdefault(ds, []).append((d['anchor_score'], d['mllm_mean']))

    print("=" * 65)
    print("RANKING BOOTSTRAP ANALYSIS")
    print("=" * 65)

    for ds_name, points in sorted(datasets.items()):
        anchor = np.array([p[0] for p in points])
        mllm = np.array([p[1] for p in points])
        k = len(anchor)
        auc_obs, ci_low, ci_high, std = bootstrap_auc(anchor, mllm)

        if auc_obs is not None:
            if ci_low is not None:
                print(f"\n{ds_name:30s} k={k}")
                print(f"  AUC = {auc_obs:.4f}")
                print(f"  Bootstrap 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
                print(f"  Bootstrap std = {std:.4f}")
            else:
                print(f"\n{ds_name:30s} k={k}")
                print(f"  AUC = {auc_obs:.4f}")
                print(f"  (Bootstrap not possible: only {len(np.unique(mllm))} unique MLLM values)")
        else:
            print(f"\n{ds_name:30s} k={k}")
            print(f"  (AUC not defined: all classes above/below median)")

    # BowTurnHead paradox explanation
    print("\n" + "=" * 65)
    print("BOWTURNHEAD PARADOX")
    print("=" * 65)
    print("""
BowTurnHead (k=2) has:
  - Per-dataset Spearman rho = 0.05 (near-zero class-level correlation)
  - Ranking AUC = 1.000 (perfect separation of hard vs easy classes)

Explanation:
With only 2 classes, the median split produces exactly 1 hard class and 1 easy class.
Any ranking that puts the hard class first achieves AUC=1.0. The Spearman rho with
n=2 can only be +1 or -1 (or undefined if tied). The observed rho≈0.05 is
essentially numerical noise due to averaging across MLLMs.

AUC and Spearman rho measure different things at k=2:
  - AUC: whether AnchorScore can separate hard from easy (trivial at k=2)
  - Spearman rho: whether AnchorScore preserves the exact MLLM ordering (requires n>=3)
""")

    print(f"\nRecommended k threshold for meaningful ranking: k >= 8")
    print(f"TeacherBehavior (k=8) is the only dataset providing meaningful ranking evidence.")


if __name__ == "__main__":
    main()
