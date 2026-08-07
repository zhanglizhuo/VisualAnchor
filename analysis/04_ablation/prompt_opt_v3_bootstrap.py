#!/usr/bin/env python3
"""
prompt_opt_v3_bootstrap.py

Paired bootstrap analysis of the three-condition prompt optimization run
(prompt_optimization_v3.py):

  standard  : baseline prompt
  enhanced  : negation-style hint (v1)
  direct    : direct visual-distinction description (v2)

For each class x MLLM pair, resample images jointly across conditions
(B = 20,000 paired resamples, seed 42) and report:

  - per-condition accuracy
  - delta(enhanced - standard), delta(direct - standard), delta(direct - enhanced)
  - 95% CI and significance (CI excludes zero)

Usage:
    python analysis/04_ablation/prompt_opt_v3_bootstrap.py
Output:
    results/04_ablation/prompt_optimization/prompt_opt_v3_bootstrap.json
"""

import json
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJ / "results" / "04_ablation" / "prompt_optimization"
RNG = np.random.RandomState(42)
B = 20000


def aligned_correct(results):
    """Return (image_order, correct_array) aligned across conditions."""
    by_image = {}
    for r in results:
        by_image[r["image"]] = r["is_correct"]
    return by_image


def paired_delta(a_by_img, b_by_img, images):
    """Paired per-image correct arrays for two conditions (0/1)."""
    a = np.array([1.0 if a_by_img.get(im, False) else 0.0 for im in images])
    b = np.array([1.0 if b_by_img.get(im, False) else 0.0 for im in images])
    return a, b


def main():
    files = {
        "qwen35": OUT_DIR / "ollama_qwen35_results_v3.json",
        "qwen36": OUT_DIR / "ollama_qwen36_results_v3.json",
    }
    classes = ["blackBoard", "answer", "stand", "read", "BowHead"]

    out = {"n_bootstrap": B, "per_class_model": {}, "summary": {}}
    per_model_direct = {"qwen35": [], "qwen36": []}
    per_model_enhanced = {"qwen35": [], "qwen36": []}
    all_direct = []
    all_enhanced = []

    for model, fpath in files.items():
        data = json.load(open(fpath))
        for cls in classes:
            item = data[cls]
            conds = {}
            for cond in ["standard", "enhanced", "direct"]:
                conds[cond] = aligned_correct(item[f"{cond}_results"])

            images = sorted(set.intersection(*[set(v.keys()) for v in conds.values()]))
            if len(images) == 0:
                print(f"WARN: {model}/{cls}: no aligned images")
                continue

            arr = {c: np.array([1.0 if conds[c][im] else 0.0 for im in images]) for c in conds}

            def stats(a, b):
                obs_a, obs_b = a.mean(), b.mean()
                obs_delta = obs_b - obs_a
                deltas = []
                for _ in range(B):
                    idx = RNG.randint(0, len(images), len(images))
                    deltas.append(b[idx].mean() - a[idx].mean())
                deltas = np.array(deltas)
                ci = np.percentile(deltas, [2.5, 97.5])
                return {
                    "n_images": len(images),
                    "acc_a": round(obs_a * 100, 1),
                    "acc_b": round(obs_b * 100, 1),
                    "delta_pp": round(obs_delta * 100, 1),
                    "delta_ci_95": [round(ci[0] * 100, 1), round(ci[1] * 100, 1)],
                    "significant": bool(bool(ci[0] > 0) or bool(ci[1] < 0)),
                }

            res = {
                "n_images": len(images),
                "standard_acc": round(arr["standard"].mean() * 100, 1),
                "enhanced_acc": round(arr["enhanced"].mean() * 100, 1),
                "direct_acc": round(arr["direct"].mean() * 100, 1),
                "enhanced_vs_standard": stats(arr["standard"], arr["enhanced"]),
                "direct_vs_standard": stats(arr["standard"], arr["direct"]),
                "direct_vs_enhanced": stats(arr["enhanced"], arr["direct"]),
            }
            out["per_class_model"][f"{model}/{cls}"] = res

            d = res["direct_vs_standard"]["delta_pp"]
            e = res["enhanced_vs_standard"]["delta_pp"]
            all_direct.append(d)
            all_enhanced.append(e)
            per_model_direct[model].append(d)
            per_model_enhanced[model].append(e)

            print(f"{model:6s} {cls:12s}: std={res['standard_acc']:5.1f} "
                  f"enh={res['enhanced_acc']:5.1f} (d={e:+5.1f}{'*' if res['enhanced_vs_standard']['significant'] else ' '}) "
                  f"dir={res['direct_acc']:5.1f} (d={d:+5.1f}{'*' if res['direct_vs_standard']['significant'] else ' '})")

    out["summary"] = {
        "combined_mean_enhanced_delta_pp": round(float(np.mean(all_enhanced)), 1),
        "combined_mean_direct_delta_pp": round(float(np.mean(all_direct)), 1),
        "qwen35_mean_enhanced_delta_pp": round(float(np.mean(per_model_enhanced["qwen35"])), 1),
        "qwen35_mean_direct_delta_pp": round(float(np.mean(per_model_direct["qwen35"])), 1),
        "qwen36_mean_enhanced_delta_pp": round(float(np.mean(per_model_enhanced["qwen36"])), 1),
        "qwen36_mean_direct_delta_pp": round(float(np.mean(per_model_direct["qwen36"])), 1),
        "n_significant_enhanced": int(sum(1 for m in ["qwen35", "qwen36"] for c in classes
                                           if out["per_class_model"][f"{m}/{c}"]["enhanced_vs_standard"]["significant"])),
        "n_significant_direct": int(sum(1 for m in ["qwen35", "qwen36"] for c in classes
                                         if out["per_class_model"][f"{m}/{c}"]["direct_vs_standard"]["significant"])),
        "n_significant_direct_vs_enhanced": int(sum(1 for m in ["qwen35", "qwen36"] for c in classes
                                                     if out["per_class_model"][f"{m}/{c}"]["direct_vs_enhanced"]["significant"])),
    }

    with open(OUT_DIR / "prompt_opt_v3_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSummary:", json.dumps(out["summary"], indent=2))
    print(f"Saved: {OUT_DIR / 'prompt_opt_v3_bootstrap.json'}")


if __name__ == "__main__":
    main()
