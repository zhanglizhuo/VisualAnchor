#!/usr/bin/env python3
"""
class_level_correlation.py

Compare AnchorScore (CLIP zero-shot accuracy) with MLLM annotation accuracy
from the paper across all SCB5 datasets and classes.

Usage:
  python class_level_correlation.py
  python class_level_correlation.py --plot
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

ANCHOR_FILE = RESULTS_DIR / "01_core/anchor_score_scb5" / "anchor_scores.json"

# MLLM per-class accuracy from single source of truth
with open(RESULTS_DIR / "01_core" / "paper_data" / "mllm_full.json") as f:
    _MLLM_SRC = json.load(f)
MODELS_OF_INTEREST = ["Qwen2-VL-7B", "LLaVA-1.5-7B", "Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B", "Gemma-3-27B", "Gemma4-31B", "Gemma4-26B"]
MLLM_DATA = {
    ds: {m: _MLLM_SRC[ds][m] for m in MODELS_OF_INTEREST if m in _MLLM_SRC.get(ds, {})}
    for ds in _MLLM_SRC if not ds.startswith("_")
}

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction",
        "blackboard-writing", "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}


def load_anchor_scores():
    with open(ANCHOR_FILE) as f:
        return json.load(f)


def correlate(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": int(len(x)), "spearman": None, "pearson": None, "p_s": None, "p_p": None}
    sp = spearmanr(x, y)
    pr = pearsonr(x, y)
    return {
        "n": int(len(x)),
        "spearman": round(sp.statistic, 4),
        "p_s": round(sp.pvalue, 4),
        "pearson": round(pr.statistic, 4),
        "p_p": round(pr.pvalue, 4),
    }


def run_lomo(anchor, mllm_data):
    """Leave-one-MLLM-out cross-validation on class-level rho."""
    all_models = sorted({m for ds in mllm_data.values() for m in ds})
    print(f"\n{'='*70}")
    print("  LEAVE-ONE-MLLM-OUT (LOMO) — class-level rho")
    print(f"{'='*70}")

    class_dict = {}
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        for cname in CLASS_MAP[ds_name]:
            a_val = anchor[ds_name]["per_class_acc"][cname]["acc"]
            m_vals = {m: mllm_data[ds_name][m].get(cname, np.nan)
                      for m in mllm_data[ds_name]}
            m_vals = {k: v for k, v in m_vals.items() if not np.isnan(v)}
            if m_vals:
                class_dict[f"{ds_name}/{cname}"] = {"anchor": a_val, "mllm_vals": m_vals}

    classes = list(class_dict.keys())
    results = []
    for held_out in all_models:
        remaining_anchor, remaining_mllm = [], []
        for ckey, cdata in class_dict.items():
            if held_out in cdata["mllm_vals"]:
                other_vals = [v for m, v in cdata["mllm_vals"].items() if m != held_out]
                if other_vals:
                    remaining_anchor.append(cdata["anchor"])
                    remaining_mllm.append(np.mean(other_vals))
            else:
                remaining_anchor.append(cdata["anchor"])
                remaining_mllm.append(np.mean(list(cdata["mllm_vals"].values())))
        if len(remaining_anchor) < 3:
            continue
        r, p = spearmanr(remaining_anchor, remaining_mllm)
        results.append({"held_out": held_out, "rho": round(r, 4), "p": round(p, 4), "n": len(remaining_anchor)})
        sig = " **" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  Exclude {held_out:20s}  rho={r:.4f}  p={p:.4f}{sig}  n={len(remaining_anchor)}")

    rhos = [r["rho"] for r in results]
    summary = {
        "n_mllms": len(all_models),
        "n_classes": len(classes),
        "full_rho": 0.7692,
        "rho_min": round(min(rhos), 4),
        "rho_max": round(max(rhos), 4),
        "rho_mean": round(float(np.mean(rhos)), 4),
        "rho_std": round(float(np.std(rhos)), 4),
        "per_model": results,
    }
    print(f"\n  Summary: min={summary['rho_min']:.4f}, max={summary['rho_max']:.4f}, mean={summary['rho_mean']:.4f}, std={summary['rho_std']:.4f}")

    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "02_robustness" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leave_one_mllm_out_9mllm.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_dir / 'leave_one_mllm_out_9mllm.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--lomo", action="store_true", help="Run leave-one-MLLM-out CV")
    args = parser.parse_args()

    if args.lomo:
        anchor = load_anchor_scores()
        run_lomo(anchor, MLLM_DATA)
        return

    anchor = load_anchor_scores()

    results = {}
    all_anchor_vals, all_mllm_vals = [], []
    all_anchor_ds, all_mllm_ds = [], []

    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        classes = CLASS_MAP[ds_name]
        anchor_acc = [anchor[ds_name]["per_class_acc"][c]["acc"] for c in classes]

        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")
        print(f"  {'Class':28s} {'Anchor':>8s} | ", end="")
        for m in MLLM_DATA[ds_name]:
            print(f"{m:>14s}", end="")
        print()

        ds_corrs = {}
        for mname, mdata in MLLM_DATA[ds_name].items():
            mllm_acc = [mdata.get(c, np.nan) for c in classes]
            valid = [(a, m) for a, m in zip(anchor_acc, mllm_acc) if not np.isnan(m)]
            for a, m in valid:
                all_anchor_vals.append(a)
                all_mllm_vals.append(m)
                all_anchor_ds.append(ds_name)
                all_mllm_ds.append(mname)

            corr = correlate(np.array([a for a, _ in valid]), np.array([m for _, m in valid]))
            ds_corrs[mname] = corr

        for ci, cname in enumerate(classes):
            print(f"  {cname:28s} {anchor_acc[ci]:>7.2f}% | ", end="")
            for mname, mdata in MLLM_DATA[ds_name].items():
                v = mdata.get(cname, np.nan)
                if not np.isnan(v):
                    print(f"{v:>12.2f}%", end="  ")
                else:
                    print(f"{'N/A':>12s}", end="  ")
            print()

        print("\n  Correlations (per-model):")
        for mname, corr in ds_corrs.items():
            s, p = corr["spearman"], corr["p_s"]
            sig = " **" if p is not None and p < 0.01 else " *" if p is not None and p < 0.05 else ""
            print(f"    {mname:20s}  r_s={s:.3f}  p={p:.4f}{sig}" if s else f"    {mname:20s}  n={corr['n']} insufficient")

        results[ds_name] = {"per_model": ds_corrs}

    # Across all datasets, all models
    print(f"\n{'='*70}")
    print("  ALL DATASETS × ALL MODELS")
    print(f"{'='*70}")
    print(f"  N = {len(all_anchor_vals)} class-model pairs")
    global_corr = correlate(np.array(all_anchor_vals), np.array(all_mllm_vals))
    print(f"  Spearman: r_s={global_corr['spearman']:.4f}, p={global_corr['p_s']:.4f}")
    print(f"  Pearson:  r_p={global_corr['pearson']:.4f}, p={global_corr['p_p']:.4f}")

    results["global"] = global_corr
    results["n_pairs"] = len(all_anchor_vals)

    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "01_core" / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "correlation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'correlation_summary.json'}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 12})

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        colors = {"TeacherBehavior": "#e74c3c", "HandriseReadWrite": "#3498db", "BowTurnHead": "#2ecc71"}
        models_list = list(MLLM_DATA["TeacherBehavior"].keys())

        for idx, (ds_name, classes) in enumerate(CLASS_MAP.items()):
            ax = axes[idx]
            anchor_acc = [anchor[ds_name]["per_class_acc"][c]["acc"] for c in classes]
            for mname in models_list:
                mllm_acc = [MLLM_DATA[ds_name][mname].get(c, np.nan) for c in classes]
                valid = [(a, m) for a, m in zip(anchor_acc, mllm_acc) if not np.isnan(m)]
                if valid:
                    ax.scatter([a for a, _ in valid], [m for _, m in valid],
                              alpha=0.5, s=20, color=colors[ds_name])

            ax.set_xlabel("AnchorScore (CLIP zero-shot %)")
            ax.set_ylabel("MLLM Annotation Accuracy (%)")
            ax.set_title(ds_name)
            ax.grid(True, alpha=0.3)

            # Add regression line
            anchor_vals = np.array([anchor[ds_name]["per_class_acc"][c]["acc"] for c in classes])
            all_m = []
            for mname in models_list:
                for c in classes:
                    v = MLLM_DATA[ds_name][mname].get(c, np.nan)
                    if not np.isnan(v):
                        all_m.append(v)
            all_m = np.array(all_m)
            if len(anchor_vals) == len(all_m):
                z = np.polyfit(anchor_vals, all_m, 1)
                p = np.poly1d(z)
                x_line = np.linspace(anchor_vals.min(), anchor_vals.max(), 50)
                ax.plot(x_line, p(x_line), "--", color=colors[ds_name], alpha=0.7)

        plt.tight_layout()
        out_path = out_dir / "correlation_scatter.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
