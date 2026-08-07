#!/usr/bin/env python3
"""
hybrid_annotation.py

Simulate CLIP+MLLM hybrid annotation: use CLIP (cheap) for high AnchorScore classes,
fall back to MLLM (expensive) for low AnchorScore classes.

Uses per-class AnchorScores from the paper's reference anchor_scores.json.
No per-image data needed — class-level simulation.

Usage:
  python experiments/05_applications/hybrid_annotation.py
  python experiments/05_applications/hybrid_annotation.py --anchor-scores results/01_core/anchor_score_scb5/anchor_scores.json
"""

import os, json, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent.parent

DATASET_CFG_HYBRID = {
    "TeacherBehavior": {
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
        ],
    },
    "HandriseReadWrite": {
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "classes": ["BowHead", "TurnHead"],
    },
}


def get_mllm_names(mllm_data, ds_name):
    if ds_name not in mllm_data:
        return []
    ds_entry = mllm_data[ds_name]
    if not isinstance(ds_entry, dict):
        return []
    return [k for k in ds_entry if isinstance(ds_entry[k], dict)]


def simulate_hybrid(anchor_scores, mllm_data, threshold, mllm_cost_ratio):
    """
    Class-level hybrid simulation.

    For each class c:
      - If AnchorScore[c] >= threshold: use CLIP (cost=1/img, accuracy=AnchorScore[c])
      - Else: use best single MLLM (cost=mllm_cost_ratio/img, accuracy=MLLM_acc[c])

    Uses the BEST SINGLE MLLM (by overall weighted accuracy) for all fallback classes.
    """
    results = {}
    for ds_name in DATASET_CFG_HYBRID:
        if ds_name not in anchor_scores or ds_name not in mllm_data:
            continue

        classes = DATASET_CFG_HYBRID[ds_name]["classes"]
        ds = anchor_scores[ds_name]
        per_class = ds.get("per_class_acc", {})

        # Per-class counts and CLIP accuracy from reference
        cls_n = {}
        cls_anchor = {}
        for c in classes:
            info = per_class.get(c, {})
            cls_n[c] = info.get("n", 0)
            cls_anchor[c] = info.get("acc", 0) / 100.0

        n_total = sum(cls_n.values())
        if n_total == 0:
            continue

        # Find best single MLLM by weighted overall accuracy
        available = get_mllm_names(mllm_data, ds_name)
        best_mllm_name = None
        best_mllm_overall = -1.0
        mllm_cls_acc = {c: 0.0 for c in classes}

        for mname in available:
            mds = mllm_data[ds_name].get(mname, {})
            weighted = 0.0
            total_w = 0
            for c in classes:
                if c in mds:
                    weighted += mds[c] / 100.0 * cls_n[c]
                    total_w += cls_n[c]
            if total_w > 0:
                overall = weighted / total_w
                if overall > best_mllm_overall:
                    best_mllm_overall = overall
                    best_mllm_name = mname
                    for c in classes:
                        mllm_cls_acc[c] = max(0.0, mds.get(c, 0) / 100.0)

        # Simulate
        cost_total = 0.0
        correct_total = 0.0
        per_class_result = {}

        for c in classes:
            n = cls_n[c]
            if n == 0:
                per_class_result[c] = {"n": 0, "use": "none", "acc": 0}
                continue

            anchor_acc = cls_anchor[c]

            if anchor_acc * 100 >= threshold:
                cost = 1.0 * n
                correct = n * anchor_acc
                use_mllm = False
                mllm_acc_val = None
            else:
                cost = mllm_cost_ratio * n
                correct = n * mllm_cls_acc[c]
                use_mllm = True
                mllm_acc_val = mllm_cls_acc[c] * 100

            cost_total += cost
            correct_total += correct
            per_class_result[c] = {
                "n": n,
                "use": "mllm" if use_mllm else "clip",
                "anchor_acc": round(anchor_acc * 100, 2),
                "mllm_acc": round(mllm_acc_val, 2) if use_mllm else None,
                "best_mllm": best_mllm_name if use_mllm else None,
                "correct": round(correct, 1),
                "cost": round(cost, 1),
            }

        acc = 100.0 * correct_total / n_total
        all_mllm_cost = n_total * mllm_cost_ratio
        cost_savings = 100.0 * (all_mllm_cost - cost_total) / all_mllm_cost if all_mllm_cost > 0 else 0

        results[ds_name] = {
            "threshold": threshold,
            "accuracy": round(acc, 2),
            "cost": round(cost_total, 1),
            "all_mllm_cost": all_mllm_cost,
            "cost_savings_pct": round(cost_savings, 1),
            "n_total": n_total,
            "best_mllm": best_mllm_name,
            "best_mllm_overall_acc": round(best_mllm_overall * 100, 2),
            "per_class": per_class_result,
        }

    return results


def compute_baselines(anchor_scores, mllm_data):
    """Compute all-CLIP and all-MLLM baselines from reference data."""
    baselines = {}
    for ds_name in DATASET_CFG_HYBRID:
        if ds_name not in anchor_scores or ds_name not in mllm_data:
            continue

        ds = anchor_scores[ds_name]
        classes = DATASET_CFG_HYBRID[ds_name]["classes"]
        per_class = ds.get("per_class_acc", {})
        cls_n = {c: per_class.get(c, {}).get("n", 0) for c in classes}
        n_total = sum(cls_n.values())

        clip_acc = ds["overall_acc"]

        available = get_mllm_names(mllm_data, ds_name)
        mllm_best_acc = 0
        mllm_best_name = None
        for mname in available:
            mds = mllm_data[ds_name].get(mname, {})
            weighted = 0.0
            total_w = 0
            for c in classes:
                if c in mds:
                    weighted += mds[c] / 100.0 * cls_n[c]
                    total_w += cls_n[c]
            if total_w > 0:
                overall = weighted / total_w * 100
                if overall > mllm_best_acc:
                    mllm_best_acc = overall
                    mllm_best_name = mname

        baselines[ds_name] = {
            "clip_acc": round(clip_acc, 2),
            "mllm_best_acc": round(mllm_best_acc, 2),
            "best_mllm": mllm_best_name,
        }
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-scores", default=None)
    parser.add_argument("--mllm-accuracies", default=None)
    parser.add_argument("--mllm-cost", type=float, default=100.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or PROJ / "results" / "05_applications" / "hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference data
    ascore_path = Path(args.anchor_scores or PROJ / "results" / "01_core/anchor_score_scb5" / "anchor_scores.json")
    mllm_path = Path(args.mllm_accuracies or PROJ / "results" / "01_core" / "paper_data" / "mllm_raw.json")

    print(f"Loading AnchorScores from: {ascore_path}")
    assert ascore_path.exists(), f"Not found: {ascore_path}"
    anchor_scores = json.load(open(ascore_path))

    print(f"Loading MLLM accuracies from: {mllm_path}")
    mllm_data = json.load(open(mllm_path))
    assert mllm_data, f"Not found: {mllm_path}"

    # Baselines
    baselines = compute_baselines(anchor_scores, mllm_data)
    print("\n=== Baselines (paper reference values) ===")
    for ds, bl in baselines.items():
        print(f"  {ds}: CLIP (AnchorScore)={bl['clip_acc']:.1f}%, Best MLLM={bl['mllm_best_acc']:.1f}% ({bl['best_mllm']})")

    # Sweep
    thresholds = list(range(0, 101, 5))
    all_results = {}
    for t in thresholds:
        all_results[t] = simulate_hybrid(anchor_scores, mllm_data, t, args.mllm_cost)

    out_path = output_dir / "hybrid_simulation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Pareto analysis
    print("\n=== Pareto-optimal thresholds ===")
    for ds_name in DATASET_CFG_HYBRID:
        if ds_name not in baselines:
            continue
        clip_acc = baselines[ds_name]["clip_acc"]
        mllm_acc = baselines[ds_name]["mllm_best_acc"]

        pareto = []
        for t in thresholds:
            r = all_results.get(t, {}).get(ds_name)
            if r and r["accuracy"] > clip_acc and r["cost_savings_pct"] > 0:
                pareto.append((t, r["accuracy"], r["cost_savings_pct"]))

        print(f"  {ds_name} (CLIP={clip_acc:.1f}%, MLLM={mllm_acc:.1f}%):")
        if pareto:
            for t, acc, sav in pareto:
                gain = acc - clip_acc
                print(f"    thr={t:3d}: acc={acc:5.1f}% (+{gain:.1f}pp), saves {sav:.0f}% cost")
        else:
            print(f"    No hybrid threshold improves over CLIP")

    # Summary table
    print("\n=== Full sweep ===")
    ds_names = list(DATASET_CFG_HYBRID.keys())
    header = f"{'Thr':>5}"
    for ds in ds_names:
        if ds in baselines:
            header += f"  {ds[:8]:>9} {ds[:8]+'_sv':>9}"
    print(header)
    for t in thresholds:
        row = f"{t:>5}"
        for ds in ds_names:
            if ds in baselines and ds in all_results.get(t, {}):
                r = all_results[t][ds]
                row += f"  {r['accuracy']:>8.1f}% {r['cost_savings_pct']:>8.0f}%"
            else:
                row += f"  {'N/A':>8} {'N/A':>8}"
        print(row)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, ds_name in enumerate(ds_names):
        if ds_name not in baselines:
            continue
        ax = axes[idx]
        xs, ys = [], []
        for t in thresholds:
            r = all_results.get(t, {}).get(ds_name)
            if r:
                xs.append(r["cost_savings_pct"])
                ys.append(r["accuracy"])

        ax.plot(xs, ys, "b-", linewidth=2, label="Hybrid")
        clip_val = baselines[ds_name]["clip_acc"]
        mllm_val = baselines[ds_name]["mllm_best_acc"]
        ax.axhline(y=clip_val, color="gray", linestyle="--",
                   label=f"CLIP ({clip_val:.0f}%)")
        ax.axhline(y=mllm_val, color="green", linestyle="--",
                   label=f"MLLM ({mllm_val:.0f}%)")
        ax.set_xlabel("MLLM Cost Saved (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(ds_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "hybrid_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
