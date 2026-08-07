"""Compute AnchorScore-MLLM correlation on Stanford40.

Reads existing per-class accuracy data, computes Spearman ρ,
and saves evidence JSON with bootstrap CI.

Usage:
    python analysis/02_robustness/stanford40_correlation.py
"""

import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path


def load_anchor_scores(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_class_acc"]


def load_ollama_scores(path):
    with open(path) as f:
        data = json.load(f)
    models = {}
    for model_name, model_data in data["models"].items():
        models[model_name] = {
            cls: v["acc"]
            for cls, v in model_data["per_class_acc"].items()
        }
    return models


def load_llava_scores(path):
    with open(path) as f:
        data = json.load(f)
    return {
        cls: v["acc"]
        for cls, v in data["per_class_acc"].items()
    }


def compute_spearman(anchor, mllm_mean):
    classes = sorted(set(anchor.keys()) & set(mllm_mean.keys()))
    x = np.array([anchor[c]["acc"] for c in classes])
    y = np.array([mllm_mean[c] for c in classes])
    rho, p = spearmanr(x, y)
    return rho, p, classes


def bootstrap_ci(x, y, n_iter=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        rho, _ = spearmanr(x[idx], y[idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    base = Path(__file__).resolve().parents[2]
    anchor_path = base / "results/02_robustness/stanford40/anchor_scores.json"
    ollama_path = base / "results/02_robustness/stanford40/ollama_results.json"
    llava_path = base / "results/02_robustness/stanford40/llava_results.json"
    out_path = base / "results/02_robustness/stanford40/stanford40_correlation.json"

    anchor = load_anchor_scores(anchor_path)
    ollama = load_ollama_scores(ollama_path)
    llava_data = load_llava_scores(llava_path)

    # Per-model Spearman
    per_model = {}
    model_names = list(ollama.keys()) + ["LLaVA-1.5-7B"]
    all_mllm_data = {**ollama, "LLaVA-1.5-7B": llava_data}

    for m in model_names:
        classes = sorted(set(anchor.keys()) & set(all_mllm_data[m].keys()))
        x = np.array([anchor[c]["acc"] for c in classes])
        y = np.array([all_mllm_data[m][c] for c in classes])
        rho, p = spearmanr(x, y)
        per_model[m] = {"spearman_rho": round(float(rho), 4), "p": float(p), "n": len(classes)}

    # Multi-model mean (5 Ollama models, matching original commit 16e88c5)
    ollama_5 = ["qwen3.5:27b", "qwen3.5:35b-a3b", "qwen3.6:27b", "qwen3.6:35b-a3b", "gemma4:26b"]
    classes = sorted(set(anchor.keys()) & set(ollama[ollama_5[0]].keys()))
    mean_acc_5 = np.mean([[ollama[m][c] for m in ollama_5] for c in classes], axis=1)
    x = np.array([anchor[c]["acc"] for c in classes])
    rho_5, p_5 = spearmanr(x, mean_acc_5)
    ci_5 = bootstrap_ci(x, mean_acc_5)

    result_5 = {
        "models": ollama_5,
        "notes": "5 Ollama MLLMs (gemma4:31b excluded due to multi-GPU cold-start overhead)",
        "spearman_rho": round(float(rho_5), 4),
        "p": float(p_5),
        "n": len(classes),
        "bootstrap_95_ci": [round(ci_5[0], 4), round(ci_5[1], 4)],
        "bootstrap_n_iter": 10000,
    }

    # Multi-model mean (all 6 Ollama MLLMs, incl. gemma4:31b)
    ollama_6 = ollama_5 + ["gemma4:31b"]
    mean_acc_6o = np.mean([[ollama[m][c] for m in ollama_6] for c in classes], axis=1)
    rho_6o, p_6o = spearmanr(x, mean_acc_6o)
    ci_6o = bootstrap_ci(x, mean_acc_6o)

    result_6_ollama = {
        "models": ollama_6,
        "notes": "All 6 Ollama MLLMs (gemma4:31b now completed, negligible Δ vs 5-model)",
        "spearman_rho": round(float(rho_6o), 4),
        "p": float(p_6o),
        "n": len(classes),
        "bootstrap_95_ci": [round(ci_6o[0], 4), round(ci_6o[1], 4)],
        "bootstrap_n_iter": 10000,
    }

    # Multi-model mean (all 6 MLLMs: 5 Ollama + LLaVA-1.5-7B)
    all_6 = ollama_5 + ["LLaVA-1.5-7B"]
    mean_acc_6 = np.mean([[all_mllm_data[m][c] for m in all_6] for c in classes], axis=1)
    rho_6, p_6 = spearmanr(x, mean_acc_6)
    ci_6 = bootstrap_ci(x, mean_acc_6)

    result_6 = {
        "models": all_6,
        "notes": "All 6 MLLMs (5 Ollama + LLaVA-1.5-7B)",
        "spearman_rho": round(float(rho_6), 4),
        "p": float(p_6),
        "n": len(classes),
        "bootstrap_95_ci": [round(ci_6[0], 4), round(ci_6[1], 4)],
        "bootstrap_n_iter": 10000,
    }

    output = {
        "description": "Stanford40 AnchorScore-MLLM correlation",
        "anchor_source": str(anchor_path),
        "mllm_source": str(ollama_path),
        "per_model": per_model,
        "multi_model_mean_5_ollama": result_5,
        "multi_model_mean_6_ollama": result_6_ollama,
        "multi_model_mean_6_mllms": result_6,
        "canonical": "6 Ollama MLLMs (includes gemma4:31b)",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    print(f"\n5-Ollama mean: ρ={result_5['spearman_rho']:.4f}, p={result_5['p']:.4f}, "
          f"95% CI [{result_5['bootstrap_95_ci'][0]:.4f}, {result_5['bootstrap_95_ci'][1]:.4f}]")
    print(f"6-MLLM mean:  ρ={result_6['spearman_rho']:.4f}, p={result_6['p']:.4f}, "
          f"95% CI [{result_6['bootstrap_95_ci'][0]:.4f}, {result_6['bootstrap_95_ci'][1]:.4f}]")


if __name__ == "__main__":
    main()
