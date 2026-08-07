#!/usr/bin/env python3
"""
comprehensive_correlation.py

Full correlation analysis between AnchorScore (CLIP zero-shot acc) and
MLLM annotation accuracy. Includes:
1. Per-class correlation on SCB5 datasets vs Qwen3.5-27B, Qwen3.6-27B, etc.
2. Cross-domain AnchorScore summary (EuroSAT, BloodMNIST, TissueMNIST)
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
ANCHOR_SCB5 = RESULTS_DIR / "01_core/anchor_score_scb5" / "anchor_scores.json"
ANCHOR_CROSS = RESULTS_DIR / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"
OUT_DIR = RESULTS_DIR / "01_core" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MLLM per-class data from single source of truth
with open(RESULTS_DIR / "01_core" / "paper_data" / "mllm_full.json") as f:
    _MLLM_SRC = json.load(f)
MODELS_OF_INTEREST = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B", "Gemma-3-27B", "Gemma4-31B", "Gemma4-26B"]
MLLM_DATA = {
    ds: {m: _MLLM_SRC[ds][m] for m in MODELS_OF_INTEREST if m in _MLLM_SRC.get(ds, {})}
    for ds in _MLLM_SRC if not ds.startswith("_")
}

CROSS_DOMAIN_STANDARD = {}
with open(ANCHOR_CROSS) as f:
    cross_raw = json.load(f)
for ds_name, ds_data in cross_raw.items():
    n_cls = len(ds_data.get("per_class_acc", {}))
    CROSS_DOMAIN_STANDARD[f"{ds_name} ({n_cls}cls)"] = {
        "AnchorScore": ds_data.get("overall_acc", 0),
        "n": ds_data.get("total", 0),
    }


def load():
    with open(ANCHOR_SCB5) as f:
        scb5 = json.load(f)
    with open(ANCHOR_CROSS) as f:
        cross = json.load(f)
    return scb5, cross


def main():
    scb5, cross = load()

    # ── SCB5 per-class correlation ──
    all_anchor, all_mllm = [], []
    fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 5))
    ds_order = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]
    colors_ds = {"TeacherBehavior": "#e74c3c", "HandriseReadWrite": "#3498db", "BowTurnHead": "#2ecc71"}

    for idx, ds_name in enumerate(ds_order):
        ax = axes_c[idx]
        anchor_acc = scb5[ds_name]["per_class_acc"]
        class_list = list(anchor_acc.keys())

        ds_anchor_vals, ds_mllm_vals = [], []
        for cname in class_list:
            a_val = anchor_acc[cname]["acc"]
            for mname, mdata in MLLM_DATA[ds_name].items():
                if cname in mdata:
                    m_val = mdata[cname]
                    ds_anchor_vals.append(a_val)
                    ds_mllm_vals.append(m_val)
                    all_anchor.append(a_val)
                    all_mllm.append(m_val)

        ax.scatter(ds_anchor_vals, ds_mllm_vals, alpha=0.4, s=15, color=colors_ds[ds_name])
        ax.set_xlabel("AnchorScore (%)")
        ax.set_ylabel("MLLM Accuracy (%)")
        ax.set_title(ds_name)
        ax.grid(True, alpha=0.3)

        if ds_name == "TeacherBehavior":
            r_s, p_s = spearmanr(ds_anchor_vals, ds_mllm_vals)
            ax.text(0.05, 0.95, f"$\\rho$={r_s:.3f} (p={p_s:.4f})", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

    plt.tight_layout()
    fig_c.savefig(OUT_DIR / "scb5_correlation.png", bbox_inches="tight")
    print(f"Saved scb5_correlation.png")

    # ── Global SCB5 correlation ──
    r_s, p_s = spearmanr(all_anchor, all_mllm)
    r_p, p_p = pearsonr(all_anchor, all_mllm)
    print(f"\nSCB5 GLOBAL ({len(all_anchor)} points):")
    print(f"  Spearman: r_s={r_s:.4f}, p={p_s:.2e}")
    print(f"  Pearson:  r_p={r_p:.4f}, p={p_p:.2e}")

    # ── Cross-domain AnchorScore summary ──
    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN ANCHORSCORE")
    print(f"{'='*60}")
    for ds_name, info in CROSS_DOMAIN_STANDARD.items():
        print(f"  {ds_name:40s}  {info['AnchorScore']:5.1f}% (n={info['n']})")
    print(f"  {'SCB5 (weighted avg)':40s}  {sum(scb5[d]['correct'] for d in scb5)/sum(scb5[d]['total'] for d in scb5)*100:.1f}% (n={sum(scb5[d]['total'] for d in scb5)})")

    # ── Summary table for paper ──
    rows = []
    for ds_name, info in [("EuroSAT", None), ("BloodMNIST", None), ("TissueMNIST", None)]:
        if ds_name in scb5:
            d = scb5[ds_name]
            rows.append((ds_name, "SCB5", d["total"], d["overall_acc"]))
        elif ds_name in cross:
            d = cross[ds_name]
            rows.append((ds_name, "Cross-domain", d["total"], d["overall_acc"]))
    for ds_name in scb5:
        if ds_name not in [r[0] for r in rows]:
            d = scb5[ds_name]
            rows.append((ds_name, "SCB5", d["total"], d["overall_acc"]))

    print(f"\n{'='*60}")
    print(f"  COMPLETE RESULTS TABLE")
    print(f"{'='*60}")
    print(f"  {'Dataset':28s} {'Type':14s} {'Samples':>8s}  {'Anchor%':>8s}")
    for name, typ, n, acc in rows:
        print(f"  {name:28s} {typ:14s} {n:>8d}  {acc:>7.2f}%")

    # ── Per-class analysis for TeacherBehavior ──
    print(f"\n{'='*60}")
    print(f"  TEACHERBEHAVIOR PER-CLASS (avg across {len(MLLM_DATA['TeacherBehavior'])} MLLMs)")
    print(f"{'='*60}")
    tb = scb5["TeacherBehavior"]["per_class_acc"]
    classes = list(tb.keys())
    mllm_avgs = {}
    for cname in classes:
        vals = [m[cname] for m in MLLM_DATA["TeacherBehavior"].values() if cname in m]
        mllm_avgs[cname] = np.mean(vals)

    # Sort by AnchorScore
    sorted_classes = sorted(classes, key=lambda c: tb[c]["acc"])
    print(f"  {'Class':28s} {'Anchor%':>8s}  {'MLLM avg%':>10s}  {'Delta':>8s}")
    for cname in sorted_classes:
        a = tb[cname]["acc"]
        m = mllm_avgs[cname]
        print(f"  {cname:28s} {a:>7.2f}%  {m:>9.2f}%  {m-a:>+7.2f}%")

    # ── Save everything ──
    out = {
        "scb5_global": {
            "n": len(all_anchor),
            "spearman_rho": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 6),
        },
        "cross_domain": CROSS_DOMAIN_STANDARD,
        "per_class_teacher_behavior": {
            c: {"anchor": tb[c]["acc"], "mllm_avg": round(mllm_avgs[c], 2)}
            for c in classes
        },
    }
    with open(OUT_DIR / "comprehensive_results.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"\nSaved to {OUT_DIR / 'comprehensive_results.json'}")


if __name__ == "__main__":
    main()
