"""Compute pairwise Spearman ρ between all MLLMs on SCB5 class-level accuracy.

Format: {dataset: {model: {class: acc, ...}, ...}, ...}

Usage:
    python analysis/01_core/mllm_pairwise_correlation.py
"""

import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
from itertools import combinations


def main():
    base = Path(__file__).resolve().parents[2]
    mllm_path = base / "results/01_core/paper_data/mllm_full.json"
    out_path = base / "results/01_core/correlation/mllm_pairwise_correlation.json"

    with open(mllm_path) as f:
        raw = json.load(f)

    # Collect all classes and all models
    all_classes = set()
    all_models = set()
    for dataset_name, dataset in raw.items():
        if not isinstance(dataset, dict):
            continue
        for model_name, class_accs in dataset.items():
            all_models.add(model_name)
            for cls in class_accs:
                all_classes.add(cls)

    all_classes = sorted(all_classes)
    all_models = sorted(all_models)

    # Build n_classes × n_models matrix
    model_to_idx = {m: i for i, m in enumerate(all_models)}
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    matrix = np.full((len(all_classes), len(all_models)), np.nan)

    for dataset in raw.values():
        if not isinstance(dataset, dict):
            continue
        for model_name, class_accs in dataset.items():
            mi = model_to_idx[model_name]
            for cls, acc in class_accs.items():
                ci = class_to_idx[cls]
                matrix[ci, mi] = acc

    # Pairwise Spearman for all models
    pairwise_all = {}
    for m1, m2 in combinations(range(len(all_models)), 2):
        mask = ~(np.isnan(matrix[:, m1]) | np.isnan(matrix[:, m2]))
        n = int(mask.sum())
        if n >= 5:
            rho, p = spearmanr(matrix[mask, m1], matrix[mask, m2])
            key = f"{all_models[m1]} vs {all_models[m2]}"
            pairwise_all[key] = {
                "model_1": all_models[m1],
                "model_2": all_models[m2],
                "spearman_rho": round(float(rho), 4),
                "p": float(p),
                "n": n,
            }

    # Core-6 only (the paper's main MLLM set)
    core6 = sorted(["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B",
                    "Gemma4-31B", "Gemma4-26B"])
    idx_core = [model_to_idx[m] for m in core6 if m in model_to_idx]

    pairwise_core = {}
    for i, j in combinations(range(len(idx_core)), 2):
        mi, mj = idx_core[i], idx_core[j]
        mask = ~(np.isnan(matrix[:, mi]) | np.isnan(matrix[:, mj]))
        n = int(mask.sum())
        if n >= 5:
            rho, p = spearmanr(matrix[mask, mi], matrix[mask, mj])
            key = f"{all_models[mi]} vs {all_models[mj]}"
            pairwise_core[key] = {
                "model_1": all_models[mi],
                "model_2": all_models[mj],
                "spearman_rho": round(float(rho), 4),
                "p": float(p),
                "n": n,
            }

    rhos_core = np.array([v["spearman_rho"] for v in pairwise_core.values()])

    # By family
    qwen_pairs = {k: v for k, v in pairwise_core.items()
                  if "Qwen" in k and "Gemma" not in k}
    gemma_pairs = {k: v for k, v in pairwise_core.items()
                   if "Gemma" in k and "Qwen" not in k}
    cross_pairs = {k: v for k, v in pairwise_core.items()
                   if ("Qwen" in k and "Gemma" in k)}

    output = {
        "description": "MLLM pairwise Spearman ρ on SCB5 class-level accuracy (13 classes)",
        "n_models": len(all_models),
        "n_classes": len(all_classes),
        "model_names": all_models,
        "pairwise_core6": pairwise_core,
        "summary_core6": {
            "mean_rho": round(float(np.mean(rhos_core)), 4),
            "min_rho": round(float(np.min(rhos_core)), 4),
            "max_rho": round(float(np.max(rhos_core)), 4),
            "std_rho": round(float(np.std(rhos_core)), 4),
            "n_pairs": len(rhos_core),
        },
        "summary_by_family": {
            "qwen_qwen_within": {
                "description": "Qwen-3.5 vs Qwen-3.6 same-size comparisons",
                "mean_rho": round(float(np.mean([v["spearman_rho"] for v in qwen_pairs.values()])), 4),
                "n_pairs": len(qwen_pairs),
            },
            "gemma_gemma_within": {
                "description": "Gemma4-26B vs Gemma4-31B",
                "mean_rho": round(float(np.mean([v["spearman_rho"] for v in gemma_pairs.values()])), 4),
                "n_pairs": len(gemma_pairs),
            },
            "qwen_gemma_cross": {
                "description": "Any Qwen vs any Gemma",
                "mean_rho": round(float(np.mean([v["spearman_rho"] for v in cross_pairs.values()])), 4),
                "n_pairs": len(cross_pairs),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    print(f"\nCore-6 MLLM pairwise ρ on SCB5 (13 classes):")
    print(f"  Mean: {output['summary_core6']['mean_rho']:.4f}")
    print(f"  Min:  {output['summary_core6']['min_rho']:.4f}")
    print(f"  Max:  {output['summary_core6']['max_rho']:.4f}")
    print(f"  Std:  {output['summary_core6']['std_rho']:.4f}")

    print(f"\n  Within-family:")
    for k, v in output["summary_by_family"].items():
        print(f"    {v['description']:40s}  mean ρ={v['mean_rho']:.4f}  (n={v['n_pairs']} pairs)")

    print(f"\n  All pairwise:")
    for k, v in pairwise_core.items():
        print(f"    {k:45s}  ρ={v['spearman_rho']:.4f}, p={v['p']:.4f}, n={v['n']}")


if __name__ == "__main__":
    main()
