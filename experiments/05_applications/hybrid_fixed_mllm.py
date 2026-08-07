#!/usr/bin/env python3
"""
hybrid_fixed_mllm.py

R3-1 revision: Simulate hybrid annotation with EACH individual MLLM as the
fixed fallback (instead of oracle-best). This produces a realistic deployment
table showing what gain a practitioner gets when they commit to one MLLM
without knowing a priori which is best.

For each dataset and each available MLLM:
  - Classes with AnchorScore >= tau use CLIP
  - All other classes use THAT FIXED MLLM (not the oracle best)

Reports accuracy, gain over CLIP-alone, and MLLM cost saved at the
Pareto-optimal threshold from the original hybrid experiment.

Usage:
  python experiments/05_applications/hybrid_fixed_mllm.py
"""

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent

DATASET_CFG = {
    "TeacherBehavior": {
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
        ],
        "tau_star": 45,
    },
    "HandriseReadWrite": {
        "classes": ["hand-raising", "read", "write"],
        "tau_star": 45,
    },
    "BowTurnHead": {
        "classes": ["BowHead", "TurnHead"],
        "tau_star": 55,
    },
}

MLLM_COST_RATIO = 100.0


def load_data():
    ascore_path = PROJ / "results" / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
    mllm_path = PROJ / "results" / "01_core" / "paper_data" / "mllm_raw.json"
    with open(ascore_path) as f:
        anchor = json.load(f)
    with open(mllm_path) as f:
        mllm = json.load(f)
    return anchor, mllm


def simulate_fixed(anchor_scores, mllm_data, ds_name, tau, fixed_mllm):
    """Run hybrid at threshold tau, using fixed_mllm for all fallback classes."""
    cfg = DATASET_CFG[ds_name]
    classes = cfg["classes"]
    per_class = anchor_scores[ds_name].get("per_class_acc", {})

    cls_n = {c: per_class.get(c, {}).get("n", 0) for c in classes}
    cls_anchor = {c: per_class.get(c, {}).get("acc", 0) / 100.0 for c in classes}
    n_total = sum(cls_n.values())
    if n_total == 0:
        return None

    mds = mllm_data[ds_name].get(fixed_mllm, {})
    if not mds:
        return None

    cost_total = 0.0
    correct_total = 0.0
    n_clip = 0
    n_mllm = 0
    for c in classes:
        n = cls_n[c]
        if n == 0:
            continue
        if cls_anchor[c] * 100 >= tau:
            cost_total += 1.0 * n
            correct_total += n * cls_anchor[c]
            n_clip += n
        else:
            mllm_acc = max(0.0, mds.get(c, 0) / 100.0)
            cost_total += MLLM_COST_RATIO * n
            correct_total += n * mllm_acc
            n_mllm += n

    acc = 100.0 * correct_total / n_total
    all_mllm_cost = n_total * MLLM_COST_RATIO
    cost_savings = 100.0 * (all_mllm_cost - cost_total) / all_mllm_cost
    return {
        "accuracy": round(acc, 2),
        "cost_savings_pct": round(cost_savings, 1),
        "n_clip": n_clip,
        "n_mllm": n_mllm,
    }


def compute_overall_acc(mllm_data, ds_name, mllm_name):
    """Weighted overall accuracy of an MLLM on a dataset."""
    cfg = DATASET_CFG[ds_name]
    classes = cfg["classes"]
    mds = mllm_data[ds_name].get(mllm_name, {})
    total_w = 0
    weighted = 0.0
    per_class = None
    # need class sizes
    return None  # filled below with anchor n


def main():
    anchor, mllm = load_data()

    results = {}
    print("=" * 80)
    print("R3-1: Fixed-MLLM Hybrid Simulation")
    print("Each row = commit to ONE MLLM for all fallback classes (no oracle).")
    print("=" * 80)

    for ds_name in DATASET_CFG:
        cfg = DATASET_CFG[ds_name]
        tau = cfg["tau_star"]
        classes = cfg["classes"]
        per_class = anchor[ds_name].get("per_class_acc", {})
        cls_n = {c: per_class.get(c, {}).get("n", 0) for c in classes}
        n_total = sum(cls_n.values())

        clip_acc = anchor[ds_name]["overall_acc"]

        available = [k for k in mllm[ds_name] if isinstance(mllm[ds_name][k], dict)]

        # Compute overall acc per MLLM (weighted by class size)
        mllm_overall = {}
        for m in available:
            mds = mllm[ds_name][m]
            w = sum(mds.get(c, 0) * cls_n.get(c, 0) for c in classes)
            tw = sum(cls_n.get(c, 0) for c in classes if c in mds)
            mllm_overall[m] = w / tw if tw > 0 else 0.0

        # Oracle = best overall MLLM
        oracle_mllm = max(mllm_overall, key=mllm_overall.get)

        print(f"\n--- {ds_name} (n_total={n_total}, CLIP={clip_acc:.1f}%, tau*={tau}) ---")
        print(f"{'MLLM':<18} {'Overall%':>8} {'Hybrid%':>8} {'Gain':>7} {'CostSaved':>10}")
        print("-" * 55)

        ds_results = {"clip_acc": clip_acc, "tau_star": tau, "rows": []}
        for m in sorted(available, key=lambda x: -mllm_overall[x]):
            ov = mllm_overall[m]
            sim = simulate_fixed(anchor, mllm, ds_name, tau, m)
            if sim is None:
                continue
            gain = sim["accuracy"] - clip_acc
            tag = " (oracle)" if m == oracle_mllm else ""
            print(f"{m:<18} {ov:>7.1f}% {sim['accuracy']:>7.1f}% {gain:>+6.1f}pp {sim['cost_savings_pct']:>9.1f}%{tag}")
            ds_results["rows"].append({
                "mllm": m,
                "overall_acc": round(ov, 2),
                "hybrid_acc": sim["accuracy"],
                "gain_pp": round(gain, 2),
                "cost_savings_pct": sim["cost_savings_pct"],
                "is_oracle": m == oracle_mllm,
            })

        # Also compute all-CLIP and all-oracle-MLLM for reference
        oracle_all = mllm_overall[oracle_mllm]
        ds_results["oracle_mllm"] = oracle_mllm
        ds_results["oracle_all_mllm_acc"] = round(oracle_all, 2)
        ds_results["clip_acc"] = clip_acc
        results[ds_name] = ds_results
        print(f"{'ALL-CLIP':<18} {clip_acc:>7.1f}% {clip_acc:>7.1f}% {0.0:>+6.1f}pp {100.0:>9.1f}% (baseline)")
        print(f"{'ALL-MLLM(oracle)':<18} {oracle_all:>7.1f}% {oracle_all:>7.1f}% {oracle_all-clip_acc:>+6.1f}pp {0.0:>9.1f}%")

    out_path = PROJ / "results" / "05_applications" / "hybrid" / "hybrid_fixed_mllm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary: what's the realistic range?
    print("\n" + "=" * 80)
    print("SUMMARY: Realistic deployment gains (non-oracle MLLMs)")
    print("=" * 80)
    for ds_name, dsr in results.items():
        clip = dsr["clip_acc"]
        non_oracle = [r for r in dsr["rows"] if not r["is_oracle"]]
        oracle = [r for r in dsr["rows"] if r["is_oracle"]][0]
        if non_oracle:
            best_non = max(non_oracle, key=lambda x: x["gain_pp"])
            worst_non = min(non_oracle, key=lambda x: x["gain_pp"])
            mean_non = sum(r["gain_pp"] for r in non_oracle) / len(non_oracle)
            print(f"\n{ds_name}:")
            print(f"  Oracle ({oracle['mllm']}):    +{oracle['gain_pp']:.1f}pp @ {oracle['cost_savings_pct']:.0f}% cost saved")
            print(f"  Best non-oracle ({best_non['mllm']}): +{best_non['gain_pp']:.1f}pp @ {best_non['cost_savings_pct']:.0f}% cost saved")
            print(f"  Worst non-oracle ({worst_non['mllm']}): +{worst_non['gain_pp']:.1f}pp @ {worst_non['cost_savings_pct']:.0f}% cost saved")
            print(f"  Mean non-oracle:            +{mean_non:.1f}pp")


if __name__ == "__main__":
    main()
