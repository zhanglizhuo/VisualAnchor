"""
validation_size_ablation.py

Ablation: How many labeled validation images per class are needed
for AnchorScore to reliably predict MLLM accuracy?

Consumes the canonical per-image CLIP prediction cache
(results/01_core/clip_per_image/per_image_predictions.json, the same
pass that produces the headline AnchorScore values), then bootstraps
at N = {5, 10, 20, 50, 100, all} images per class. For each N, computes
the class-level Spearman rho between the subsampled AnchorScore and the
canonical 6-MLLM mean accuracy over B bootstrap iterations.

At N="all" the class-level rho reproduces the headline 0.769 exactly.

Note (reproducibility): an earlier version of this ablation ran its own
CLIP pass on server-local data and produced systematically lower rhos
(rho@all=0.648) because that pass did not match the canonical AnchorScore
(e.g., BowTurnHead data-layout difference on that machine). This version
is deterministic and local: it reuses the committed canonical cache, so
it needs no GPU and no dataset download.

Usage:
  python validation_size_ablation.py [--bootstrap 200] [--seed 42]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
CLIP_CACHE = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
MLLM_RAW = RESULTS / "01_core" / "paper_data" / "mllm_raw.json"
OUT = RESULTS / "04_ablation" / "validation_size_ablation.json"

DS2CLS = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}
N_VALUES = [5, 10, 20, 50, 100, "all"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mllm = json.load(open(MLLM_RAW))
    models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
    mean_mllm = {
        (ds, cls): float(np.mean([mllm[ds][m][cls] for m in models]))
        for ds, clses in DS2CLS.items() for cls in clses
    }

    pred = json.load(open(CLIP_CACHE))
    per_image = {}
    for ds in DS2CLS:
        for it in pred[ds]["predictions"]:
            per_image.setdefault((ds, it["true_class_name"]), []).append(it["correct"])

    all_classes = [(ds, cls) for ds, clses in DS2CLS.items() for cls in clses]
    assert len(all_classes) == 13

    rng = np.random.RandomState(args.seed)
    results = {}
    for n_val in N_VALUES:
        rhos = []
        for _ in range(args.bootstrap):
            anchor_scores, mllm_accs = [], []
            for key in all_classes:
                flags = per_image[key]
                if n_val == "all":
                    sampled = flags
                else:
                    k = min(n_val, len(flags))
                    idx = rng.choice(len(flags), size=k, replace=False)
                    sampled = [flags[i] for i in idx]
                anchor_scores.append(float(np.mean(sampled)) * 100)
                mllm_accs.append(mean_mllm[key])
            rhos.append(spearmanr(anchor_scores, mllm_accs).statistic)
        r = np.array(rhos)
        results[str(n_val)] = {
            "n_bootstrap": len(r),
            "mean_rho": round(float(r.mean()), 4),
            "std_rho": round(float(r.std()), 4),
            "median_rho": round(float(np.median(r)), 4),
            "ci_lower": round(float(np.percentile(r, 2.5)), 4),
            "ci_upper": round(float(np.percentile(r, 97.5)), 4),
            "min_rho": round(float(np.min(r)), 4),
            "max_rho": round(float(np.max(r)), 4),
        }
        print(f"N={str(n_val):>4}: rho={results[str(n_val)]['mean_rho']:.3f} "
              f"± {results[str(n_val)]['std_rho']:.3f}")

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
