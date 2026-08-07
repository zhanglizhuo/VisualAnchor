"""
Bootstrap confidence intervals for prompt optimization per-class improvements.

Usage:
    python analysis/04_ablation/prompt_opt_bootstrap.py

Output:
    results/04_ablation/prompt_optimization/prompt_opt_bootstrap.json
"""

import json
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
RNG = np.random.RandomState(42)
B = 10000


def load_prompt_results(path):
    """Load per-class prompt results from ollama results JSON."""
    data = json.load(open(path))
    items = []
    for domain, domain_data in data.items():
        if isinstance(domain_data, dict):
            for cls, cls_data in domain_data.items():
                if isinstance(cls_data, dict) and "standard_acc" in cls_data:
                    items.append(cls_data)
    return items


def bootstrap_per_class(items, B=B):
    """Bootstrap CI for per-class standard, enhanced, and delta."""
    results = {}
    for item in items:
        cls = item.get("anchor_score", None)
        # We don't have per-image results, so we simulate from accuracy + count
        n_std = int(item.get("n_standard", item.get("n", 20)))
        n_enh = int(item.get("n_enhanced", item.get("n", 20)))
        acc_std = item["standard_acc"] / 100.0
        acc_enh = item["enhanced_acc"] / 100.0

        # Simulate individual binary outcomes matching the observed accuracy
        n_correct_std = int(round(acc_std * n_std))
        n_correct_enh = int(round(acc_enh * n_enh))

        bootstrap_deltas = []
        bootstrap_std = []
        bootstrap_enh = []
        for _ in range(B):
            idx_std = RNG.choice(n_std, n_std, replace=True)
            idx_enh = RNG.choice(n_enh, n_enh, replace=True)
            outcomes_std = np.zeros(n_std)
            outcomes_enh = np.zeros(n_enh)
            outcomes_std[:n_correct_std] = 1
            outcomes_enh[:n_correct_enh] = 1
            bs_std = outcomes_std[idx_std].mean()
            bs_enh = outcomes_enh[idx_enh].mean()
            bootstrap_std.append(bs_std)
            bootstrap_enh.append(bs_enh)
            bootstrap_deltas.append(bs_enh - bs_std)

        ci_std = np.percentile(bootstrap_std, [2.5, 97.5])
        ci_enh = np.percentile(bootstrap_enh, [2.5, 97.5])
        ci_delta = np.percentile(bootstrap_deltas, [2.5, 97.5])

        results[item.get("name", item.get("class", "unknown"))] = {
            "n_standard": n_std,
            "n_enhanced": n_enh,
            "standard_acc": round(acc_std * 100, 1),
            "enhanced_acc": round(acc_enh * 100, 1),
            "standard_ci_95": [round(ci_std[0] * 100, 1), round(ci_std[1] * 100, 1)],
            "enhanced_ci_95": [round(ci_enh[0] * 100, 1), round(ci_enh[1] * 100, 1)],
            "delta_pp": round((acc_enh - acc_std) * 100, 1),
            "delta_ci_95": [round(ci_delta[0] * 100, 1), round(ci_delta[1] * 100, 1)],
        }
    return results


def main():
    # Load both models
    q35_data = load_prompt_results(PROJ / "results/04_ablation/prompt_optimization/ollama_qwen35_results.json")
    q36_data = load_prompt_results(PROJ / "results/04_ablation/prompt_optimization/ollama_qwen36_results.json")

    # Dynamically extract class names from results (not hardcoded — adapts to which
    # classes pass the anchor_score > 50 filter in run_prompt_opt_simple.py)
    q35_raw = json.load(open(PROJ / "results/04_ablation/prompt_optimization/ollama_qwen35_results.json"))
    q36_raw = json.load(open(PROJ / "results/04_ablation/prompt_optimization/ollama_qwen36_results.json"))

    def extract_named_items(raw):
        """Extract all classes that have actual results (not skipped)."""
        items = {}
        for domain in raw:
            for cls_name, cls_data in raw[domain].items():
                if isinstance(cls_data, dict) and "standard_acc" in cls_data:
                    items[cls_name] = cls_data
        return items

    q35_items = extract_named_items(q35_raw)
    q36_items = extract_named_items(q36_raw)

    # Use the union of classes that appear in BOTH models' results
    all_cls = sorted(set(q35_items.keys()) & set(q36_items.keys()))

    results = {"qwen35": {}, "qwen36": {}}

    for cls_name in all_cls:
        for mllm_key, items in [("qwen35", q35_items), ("qwen36", q36_items)]:
            if cls_name not in items:
                continue
            item = items[cls_name]
            n_std = int(item.get("n_standard", 20))
            n_enh = int(item.get("n_enhanced", 20))
            acc_std = item["standard_acc"] / 100.0
            acc_enh = item["enhanced_acc"] / 100.0

            n_correct_std = int(round(acc_std * n_std))
            n_correct_enh = int(round(acc_enh * n_enh))

            bs_deltas = []
            for _ in range(B):
                outcomes_std = np.zeros(n_std)
                outcomes_enh = np.zeros(n_enh)
                outcomes_std[:n_correct_std] = 1
                outcomes_enh[:n_correct_enh] = 1
                bs_deltas.append(outcomes_enh[RNG.choice(n_enh, n_enh, replace=True)].mean() -
                                 outcomes_std[RNG.choice(n_std, n_std, replace=True)].mean())

            ci = np.percentile(bs_deltas, [2.5, 97.5])
            results[mllm_key][cls_name] = {
                "delta_pp": round((acc_enh - acc_std) * 100, 1),
                "delta_ci_95": [round(ci[0] * 100, 1), round(ci[1] * 100, 1)],
                "n_std": n_std,
                "n_enh": n_enh,
            }

    out_path = PROJ / "results" / "04_ablation" / "prompt_optimization" / "prompt_opt_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"Classes: {all_cls}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
