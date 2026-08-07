#!/usr/bin/env python3
"""
Cross-domain MLLM correlation analysis.
Correlates AnchorScore (CLIP zero-shot) with MLLM accuracy across
EuroSAT, PathMNIST, BloodMNIST, TissueMNIST for 3 MLLMs:
  LLaVA-1.5-7B, Qwen2.5-VL-7B, Qwen2-VL-7B
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ── Load AnchorScore (per-class, cross-domain) ──
with open(RESULTS_DIR / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json") as f:
    ANCHOR = json.load(f)

# ── Load MLLM results ──
MLLM_FILES = {
    "LLaVA-1.5-7B": "cross_domain_mllm/llava15b_results.json",
    "LLaVA-NeXT-Mistral-7B": "cross_domain_mllm/llavanext_results.json",
    "Qwen2.5-VL-7B": "cross_domain_mllm/qwen25vl7b_results.json",
    "Qwen2-VL-7B": "cross_domain_mllm/qwen2vl7b_results.json",
}

MLLM = {}
for mname, mpath in MLLM_FILES.items():
    p = RESULTS_DIR / mpath
    if p.exists():
        with open(p) as f:
            MLLM[mname] = json.load(f)
        print(f"  Loaded {mname}: {len(MLLM[mname])} datasets")
    else:
        print(f"  WARNING: {p} not found")

# ── Class name mapping (AnchorScore → MLLM) ──
CLASS_MAP = {
    "EuroSAT": {},  # same names
    "PathMNIST": {
        "adipose": "Adipose tissue",
        "background": "Background",
        "debris": "Debris",
        "lymphocytes": "Lymphocytes",
        "mucus": "Mucus",
        "smooth muscle": "Smooth muscle",
        "normal colon mucosa": "Normal colon mucosa",
        "cancer-associated stroma": "Cancer-associated stroma",
        "colorectal adenocarcinoma epithelium": "Tumor epithelium",
    },
    "BloodMNIST": {
        "basophil": "Basophil",
        "eosinophil": "Eosinophil",
        "erythroblast": "Erythroblast",
        "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "Immature granulocyte",
        "lymphocyte": "Lymphocyte",
        "monocyte": "Monocyte",
        "neutrophil": "Neutrophil",
        "platelet": "Platelet",
    },
    "TissueMNIST": {
        "Collecting Duct, Connecting Tubule": "Collecting Duct",  # merged → partial
        "Distal Convoluted Tubule": "Distal Convoluted Tubule",
        "Glomerular endothelial cells": "Glomerular Endothelial Cells",
        "Interstitial endothelial cells": "Interstitial Endothelial Cells",
        "Leukocytes": "Leukocytes",
        "Podocytes": "Podocytes",
        "Proximal Tubule Segments": "Proximal Tubule Segments",
    },
}

def match_class(anchor_name, mllm_results):
    """Find best matching MLLM class name for an AnchorScore class name."""
    ds_order = ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]
    for ds_name in ds_order:
        if ds_name not in mllm_results:
            continue
        for mllm_cls in mllm_results[ds_name].get("accuracy", {}):
            if anchor_name.lower() == mllm_cls.lower():
                return ds_name, mllm_cls
    return None, None

# ── Build (AnchorScore, MLLM) pairs per dataset per MLLM ──
all_anchor, all_mllm = [], []
per_mllm = {}
per_dataset = {}

for mname, mdata in MLLM.items():
    pairs = []
    for ds_name in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
        if ds_name not in ANCHOR or ds_name not in mdata:
            continue
        anchor_ds = ANCHOR[ds_name]["per_class_acc"]
        mllm_acc = mdata[ds_name]["accuracy"]
        cmap = CLASS_MAP.get(ds_name, {})

        for a_cls, a_info in anchor_ds.items():
            a_val = a_info["acc"]
            # Map class name
            m_cls = cmap.get(a_cls, a_cls)
            # Try exact match
            if m_cls in mllm_acc:
                m_val = mllm_acc[m_cls]
            else:
                # Try case-insensitive partial match
                found = False
                for mk in mllm_acc:
                    if mk.lower().replace(" ", "") == a_cls.lower().replace(" ", ""):
                        m_val = mllm_acc[mk]
                        found = True
                        break
                if not found:
                    # For TissueMNIST Collecting Duct, Connecting Tubule
                    if ds_name == "TissueMNIST" and "Collecting Duct" in a_cls:
                        m_val = max(mllm_acc.get("Collecting Duct", 0), mllm_acc.get("Connecting Tubule", 0))
                        found = True
                    else:
                        continue

            pairs.append((a_val, m_val))
            all_anchor.append(a_val)
            all_mllm.append(m_val)

    per_mllm[mname] = pairs
    n_pts = len(pairs)
    if n_pts >= 4:
        r_s, p_s = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        r_p, p_p = pearsonr([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"\n{mname:20s} ({n_pts:2d} points)")
        print(f"  Spearman: ρ={r_s:.4f}, p={p_s:.4f}")
        print(f"  Pearson:  r={r_p:.4f}, p={p_p:.4f}")
    else:
        print(f"\n{mname:20s} ({n_pts:2d} points — too few)")

# ── Combined correlation ──
print(f"\n{'='*60}")
print(f"  COMBINED ({len(all_anchor)} points)")
print(f"{'='*60}")
r_s, p_s = spearmanr(all_anchor, all_mllm)
r_p, p_p = pearsonr(all_anchor, all_mllm)
print(f"  Spearman: ρ={r_s:.4f}, p={p_s:.4f}")
print(f"  Pearson:  r={r_p:.4f}, p={p_p:.4f}")

# ── Per-dataset combined ──
print(f"\n{'='*60}")
print(f"  PER-DATASET (all MLLMs pooled)")
print(f"{'='*60}")
for ds_name in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
    ds_anchor, ds_mllm = [], []
    for mname, mdata in MLLM.items():
        if ds_name not in ANCHOR or ds_name not in mdata:
            continue
        cmap = CLASS_MAP.get(ds_name, {})
        for a_cls, a_info in ANCHOR[ds_name]["per_class_acc"].items():
            m_cls = cmap.get(a_cls, a_cls)
            mllm_acc = mdata[ds_name]["accuracy"]
            if m_cls in mllm_acc:
                ds_anchor.append(a_info["acc"])
                ds_mllm.append(mllm_acc[m_cls])
    if len(ds_anchor) >= 4:
        r_s, p_s = spearmanr(ds_anchor, ds_mllm)
        r_p, p_p = pearsonr(ds_anchor, ds_mllm)
        print(f"  {ds_name:20s} ({len(ds_anchor)} pts)  Sp=ρ{r_s:.4f}(p={p_s:.4f})  Pe=r{r_p:.4f}(p={p_p:.4f})")
    else:
        print(f"  {ds_name:20s} ({len(ds_anchor)} pts — too few)")

# ── Per-MLLM per-dataset detailed table ──
print(f"\n{'='*100}")
print(f"  PER-DATASET PER-MLLM TABLE")
print(f"{'='*100}")
header = f"  {'Dataset':20s}"
for mname in MLLM:
    header += f"  {mname:20s}"
print(header)
for ds_name in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
    row = f"  {ds_name:20s}"
    for mname, mdata in MLLM.items():
        if ds_name in mdata:
            row += f"  {mdata[ds_name]['mean_accuracy']:>6.1f}%{'':>13s}"
        else:
            row += f"  {'N/A':>20s}"
    print(row)

# ── EuroSAT-only deeper breakdown ──
print(f"\n{'='*60}")
print(f"  EUROSAT CLOSE-UP (AnchorScore vs MLLM)")
print(f"{'='*60}")
ds_name = "EuroSAT"
for mname, mdata in MLLM.items():
    print(f"\n  {mname}:")
    print(f"  {'Class':22s} {'Anchor%':>8s} {'MLLM%':>8s}")
    for a_cls, a_info in sorted(ANCHOR[ds_name]["per_class_acc"].items(), key=lambda x: x[1]["acc"]):
        a_val = a_info["acc"]
        m_val = mdata[ds_name]["accuracy"].get(a_cls, None)
        if m_val is not None:
            print(f"  {a_cls:22s} {a_val:>7.2f}% {m_val:>7.2f}%")

# ── Save results ──
out = {
    "combined": {
        "n": len(all_anchor),
        "spearman_rho": round(r_s, 4), "spearman_p": round(p_s, 6),
        "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 6),
    },
    "per_mllm": {},
    "per_dataset": {},
}
for mname in MLLM:
    pairs = per_mllm[mname]
    if len(pairs) >= 4:
        vals_a = [p[0] for p in pairs]
        vals_m = [p[1] for p in pairs]
        r_s_m, p_s_m = spearmanr(vals_a, vals_m)
        r_p_m, p_p_m = pearsonr(vals_a, vals_m)
        out["per_mllm"][mname] = {
            "n": len(pairs),
            "spearman_rho": round(r_s_m, 4), "spearman_p": round(p_s_m, 6),
            "pearson_r": round(r_p_m, 4), "pearson_p": round(p_p_m, 6),
        }

OUT_DIR = RESULTS_DIR / "01_core" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "cross_domain_mllm_correlation.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_DIR / 'cross_domain_mllm_correlation.json'}")
