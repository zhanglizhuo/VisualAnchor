#!/usr/bin/env python3
"""
hybrid_annotation_stanford40.py

CLIP+MLLM hybrid annotation simulation on Stanford40.
Uses existing per-class AnchorScores and Ollama MLLM results.
Class-level simulation — no GPU needed.
"""

import os, json, sys
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent

ANCHOR_PATH = PROJ / "results" / "02_robustness" / "stanford40" / "anchor_scores.json"
OLLAMA_PATH = PROJ / "results" / "02_robustness" / "stanford40" / "ollama_results.json"
OUTPUT_DIR = PROJ / "results" / "05_applications" / "hybrid_stanford40"

MLLM_COST_RATIO = 100.0


def load_data():
    anchor = json.load(open(ANCHOR_PATH))
    ollama = json.load(open(OLLAMA_PATH))
    return anchor, ollama


def get_best_mllm(ollama):
    """Find best single Ollama model by overall accuracy."""
    best_name = None
    best_acc = -1.0
    for mname, mdata in ollama["models"].items():
        acc = mdata["overall_acc"]
        if acc > best_acc:
            best_acc = acc
            best_name = mname
    return best_name, best_acc


def get_best_mllm_per_class(ollama, best_mllm_name):
    """Get per-class accuracy of the best single MLLM."""
    mdata = ollama["models"][best_mllm_name]
    per_class = mdata["per_class_acc"]
    return {cls: info["acc"] / 100.0 for cls, info in per_class.items()}


def get_all_mllm_per_class(ollama):
    """Get per-class accuracy for every MLLM model."""
    result = {}
    for mname, mdata in ollama["models"].items():
        result[mname] = {
            cls: info["acc"] / 100.0
            for cls, info in mdata["per_class_acc"].items()
        }
    return result


def simulate_hybrid(anchor, mllm_per_class, classes, threshold, mllm_cost_ratio):
    per_class_data = anchor["per_class_acc"]
    n_total = anchor["total"]
    cost_total = 0.0
    correct_total = 0.0
    per_class_detail = {}

    for cls in classes:
        info = per_class_data.get(cls, {})
        n = info.get("n", 0)
        anchor_acc = info.get("acc", 0) / 100.0
        mllm_acc = mllm_per_class.get(cls, 0.0)

        if anchor_acc * 100 >= threshold:
            cost = 1.0 * n
            correct = n * anchor_acc
            use_type = "clip"
        else:
            cost = mllm_cost_ratio * n
            correct = n * mllm_acc
            use_type = "mllm"

        cost_total += cost
        correct_total += correct

        per_class_detail[cls] = {
            "n": n,
            "use": use_type,
            "anchor_acc": round(anchor_acc * 100, 2),
            "mllm_acc": round(mllm_acc * 100, 2),
            "correct": round(correct, 1),
            "cost": round(cost, 1),
        }

    overall_acc = 100.0 * correct_total / n_total if n_total > 0 else 0.0
    all_mllm_cost = n_total * mllm_cost_ratio
    cost_savings = (
        100.0 * (all_mllm_cost - cost_total) / all_mllm_cost
        if all_mllm_cost > 0
        else 0
    )

    return {
        "threshold": threshold,
        "accuracy": round(overall_acc, 2),
        "cost": round(cost_total, 1),
        "all_mllm_cost": all_mllm_cost,
        "cost_savings_pct": round(cost_savings, 1),
        "n_total": n_total,
        "per_class": per_class_detail,
    }


def compute_fixed_mllm_range(anchor, all_mllm_accs, classes, threshold, mllm_cost_ratio):
    """Compute hybrid gain for every MLLM individually at a given threshold."""
    clip_acc = anchor["overall_acc"]
    gains = []
    for mname, mllm_per_class in all_mllm_accs.items():
        r = simulate_hybrid(anchor, mllm_per_class, classes, threshold, mllm_cost_ratio)
        gains.append(r["accuracy"] - clip_acc)
    return min(gains), max(gains)


def main():
    anchor, ollama = load_data()
    classes = sorted(anchor["per_class_acc"].keys())
    best_mllm_name, best_mllm_overall = get_best_mllm(ollama)
    mllm_per_class = get_best_mllm_per_class(ollama, best_mllm_name)
    all_mllm_accs = get_all_mllm_per_class(ollama)

    print(f"Stanford40 hybrid annotation simulation")
    print(f"  Classes: {len(classes)}")
    print(f"  Total images: {anchor['total']}")
    print(f"  CLIP overall: {anchor['overall_acc']:.1f}%")
    print(f"  Best MLLM: {best_mllm_name} ({best_mllm_overall:.1f}%)")
    print(f"  MLLM cost ratio: {MLLM_COST_RATIO:.0f}x")
    print()

    thresholds = list(range(0, 101, 5))
    all_results = {}

    for t in thresholds:
        all_results[t] = simulate_hybrid(
            anchor, mllm_per_class, classes, t, MLLM_COST_RATIO
        )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hybrid_simulation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")
    print()

    # Summary
    clip_acc = anchor["overall_acc"]
    print(f"{'Thr':>5}   {'Acc':>7}   {'Gain':>7}   {'Saving':>7}   {'Cost':>8}")
    for t in thresholds:
        r = all_results[t]
        gain = r["accuracy"] - clip_acc
        print(
            f"{t:>5d}   {r['accuracy']:>6.1f}%   {gain:>+6.1f}pp   "
            f"{r['cost_savings_pct']:>6.0f}%   {r['cost']:>8.0f}"
        )

    # Pareto analysis
    print(f"\nPareto-optimal thresholds:")
    clipped_acc = [
        (
            t,
            all_results[t]["accuracy"],
            all_results[t]["cost_savings_pct"],
            all_results[t]["cost"],
        )
        for t in thresholds
        if all_results[t]["accuracy"] > clip_acc and all_results[t]["cost_savings_pct"] > 0
    ]
    pareto = []
    for t, acc, sav, cost in clipped_acc:
        dominated = False
        for t2, acc2, sav2, cost2 in clipped_acc:
            if t2 != t and acc2 >= acc and sav2 >= sav and (acc2 > acc or sav2 > sav):
                dominated = True
                break
        if not dominated:
            pareto.append((t, acc, sav))

    for t, acc, sav in pareto:
        gain = acc - clip_acc
        print(f"  thr={t:3d}: acc={acc:5.1f}% (+{gain:.1f}pp), saves {sav:.0f}% cost")

    # Fixed-MLLM range for representative thresholds
    print(f"\nFixed-MLLM range for selected thresholds:")
    for t in [80, 90, 95]:
        mn, mx = compute_fixed_mllm_range(anchor, all_mllm_accs, classes, t, MLLM_COST_RATIO)
        r = all_results[t]
        print(f"  thr={t:3d}: best={r['accuracy']:.1f}%, gain={r['accuracy']-clip_acc:+.1f}pp, "
              f"fixed-MLLM range=[{mn:+.1f}, {mx:+.1f}]pp, saves {r['cost_savings_pct']:.0f}%")

    # Low AnchorScore classes
    low_classes = [
        (c, anchor["per_class_acc"][c]["acc"], mllm_per_class.get(c, 0) * 100)
        for c in classes
        if anchor["per_class_acc"][c]["acc"] < 60
    ]
    if low_classes:
        print(f"\nLow AnchorScore classes (<60%):")
        for c, a_acc, m_acc in sorted(low_classes, key=lambda x: x[1]):
            print(f"  {c:30s}: CLIP={a_acc:5.1f}%  MLLM={m_acc:5.1f}%")


if __name__ == "__main__":
    main()
