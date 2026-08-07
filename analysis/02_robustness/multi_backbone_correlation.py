#!/usr/bin/env python3
"""
multi_backbone_correlation.py

For each CLIP backbone, compute the Spearman/Pearson correlation between
AnchorScore and MLLM annotation accuracy across all SCB5 classes.

This tells us whether the anchoring effect is model-specific or general.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
BACKBONE_FILE = RESULTS_DIR / "02_robustness" / "multi_backbone" / "backbone_results.json"
OUT_DIR = RESULTS_DIR / "02_robustness" / "multi_backbone"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MLLM per-class data from paper's canonical 6-model source
with open(RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json") as f:
    _MLLM_SRC = json.load(f)
MLLM_DATA = {
    ds: {k: v for k, v in models.items() if not k.startswith("_")}
    for ds, models in _MLLM_SRC.items() if not ds.startswith("_")
}

BACKBONE_LABELS = {
    "laion_l14": "ViT-L/14 (LAION-2B)",
    "openai_l14": "ViT-L/14 (OpenAI)",
    "openai_b32": "ViT-B/32 (OpenAI)",
}

BACKBONE_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
DS_COLORS = {"TeacherBehavior": "#e74c3c", "HandriseReadWrite": "#3498db", "BowTurnHead": "#2ecc71"}
DS_ORDER = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def main():
    with open(BACKBONE_FILE) as f:
        backbone_data = json.load(f)

    results = {}
    fig, axes = plt.subplots(len(backbone_data), 3, figsize=(15, 4 * len(backbone_data)))

    for row, (tag, info) in enumerate(backbone_data.items()):
        label = BACKBONE_LABELS.get(tag, tag)
        model_variant = f"{info['pretrained']} {info['model_name']}"
        print(f"\n{'='*60}")
        print(f"Backbone: {tag}  ({model_variant})")
        print(f"{'='*60}")

        all_anchor, all_mllm = [], []

        for col, ds_name in enumerate(DS_ORDER):
            ds_acc = info["results"][ds_name]["per_class_acc"]
            ds_anchor, ds_mllm = [], []

            for cname, acc_data in ds_acc.items():
                a_val = acc_data["acc"]
                for mname, mdata in MLLM_DATA[ds_name].items():
                    if cname in mdata:
                        m_val = mdata[cname]
                        ds_anchor.append(a_val)
                        ds_mllm.append(m_val)
                        all_anchor.append(a_val)
                        all_mllm.append(m_val)

            # Per-dataset scatter
            ax = axes[row, col] if len(backbone_data) > 1 else axes[col]
            ax.scatter(ds_anchor, ds_mllm, alpha=0.4, s=15, color=DS_COLORS[ds_name])
            ax.set_xlabel("AnchorScore (%)")
            ax.set_ylabel("MLLM Accuracy (%)")
            ax.set_title(f"{ds_name}")
            ax.grid(True, alpha=0.3)

            # Per-dataset Spearman
            if len(ds_anchor) > 3:
                r_s_ds, p_s_ds = spearmanr(ds_anchor, ds_mllm)
                ax.text(0.05, 0.95, f"ρ={r_s_ds:.3f} (p={p_s_ds:.4f})",
                        transform=ax.transAxes, fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", alpha=0.8))

        # Global correlation
        all_anchor = np.array(all_anchor)
        all_mllm = np.array(all_mllm)
        r_s, p_s = spearmanr(all_anchor, all_mllm)
        r_p, p_p = pearsonr(all_anchor, all_mllm)

        print(f"  Points: {len(all_anchor)}")
        print(f"  Spearman: ρ={r_s:.4f}, p={p_s:.2e}")
        print(f"  Pearson:  r={r_p:.4f}, p={p_p:.2e}")

        results[tag] = {
            "model_name": info["model_name"],
            "pretrained": info["pretrained"],
            "n_points": len(all_anchor),
            "spearman_rho": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 6),
        }

        # Row label
        if len(backbone_data) > 1:
            fig.text(0.01, 1 - (row + 0.5) / len(backbone_data),
                     label, ha="left", va="center", fontsize=11, fontweight="bold",
                     rotation=90)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "multi_backbone_correlation.png", bbox_inches="tight",
                pad_inches=0.5)
    print(f"\nSaved: {OUT_DIR / 'multi_backbone_correlation.png'}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Multi-Backbone Correlation Comparison")
    print(f"{'='*60}")
    print(f"  {'Backbone':22s} {'Variant':30s} {'Points':>7s}  {'ρ':>7s}  {'ρ-p':>8s}  {'r':>7s}  {'r-p':>8s}")
    for tag, r in results.items():
        print(f"  {tag:22s} {r['pretrained']+' '+r['model_name']:30s} "
              f"{r['n_points']:>7d}  {r['spearman_rho']:>7.4f}  {r['spearman_p']:>8.2e}  "
              f"{r['pearson_r']:>7.4f}  {r['pearson_p']:>8.2e}")

    # Save
    with open(OUT_DIR / "backbone_correlation_results.json", "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nSaved: {OUT_DIR / 'backbone_correlation_results.json'}")


if __name__ == "__main__":
    main()
