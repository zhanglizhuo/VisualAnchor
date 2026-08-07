#!/usr/bin/env python3
"""
pooled_class_level_correlation.py
=================================
DEPRECATED — superseded by analysis/01_core/compute_pooled_analysis.py
(the canonical generator of pooled_class_level_results.json).

Kept only for historical reference. Do NOT use for new analyses or
paper numbers; the canonical script is verified to reproduce the
committed pooled_class_level_results.json exactly.

Pooled class-level analysis: SCB5 (13 classes, 6-MLLM mean) + cross-domain
(34 classes, 4-MLLM mean: Qwen2-VL-7B, LLaVA-1.5-7B, LLaVA-NeXT-7B,
Qwen2.5-VL-7B) = 47 classes.

Cross-domain AnchorScores are read from anchor_scores.json with explicit
key mappings; TissueMNIST "Collecting Duct"/"Connecting Tubule" share one
AnchorScore entry ("Collecting Duct, Connecting Tubule") and form a single
merged point. This mirrors the logic in unified_correlation.py.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"

# === 1. SCB5 per-class data (13 classes) ===
with open(RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
    scb5_anchor = json.load(f)

with open(RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
    _MLLM_SRC = json.load(f)
MODELS_OF_INTEREST = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B", "Gemma4-31B", "Gemma4-26B"]
MLLM_DATA = {
    ds: {m: _MLLM_SRC[ds][m] for m in MODELS_OF_INTEREST if m in _MLLM_SRC.get(ds, {})}
    for ds in _MLLM_SRC if not ds.startswith("_")
}

CLASS_MAP = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction", "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

scb5_data = []  # list of (domain, class_name, anchor_score, mllm_mean_acc)
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    anchor_acc = scb5_anchor[ds_name]["per_class_acc"]
    for cname in CLASS_MAP[ds_name]:
        a_val = anchor_acc[cname]["acc"]
        mllm_vals = [mdata[cname] for mname, mdata in MLLM_DATA[ds_name].items() if cname in mdata]
        mllm_mean = np.mean(mllm_vals) if mllm_vals else 0
        scb5_data.append(("SCB5-" + ds_name, cname, a_val, mllm_mean))

print(f"SCB5: {len(scb5_data)} classes")

# === 2. Cross-domain per-class data (34 classes, 4-MLLM mean) ===
with open(RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json") as f:
    cross_anchor_full = json.load(f)

KEY_MAP = {
    "PathMNIST": {
        "Adipose tissue": "adipose",
        "Background": "background",
        "Debris": "debris",
        "Lymphocytes": "lymphocytes",
        "Mucus": "mucus",
        "Smooth muscle": "smooth muscle",
        "Normal colon mucosa": "normal colon mucosa",
        "Cancer-associated stroma": "cancer-associated stroma",
        "Tumor epithelium": "colorectal adenocarcinoma epithelium",
    },
    "BloodMNIST": {
        "Basophil": "basophil",
        "Eosinophil": "eosinophil",
        "Erythroblast": "erythroblast",
        "Immature granulocyte": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        "Lymphocyte": "lymphocyte",
        "Monocyte": "monocyte",
        "Neutrophil": "neutrophil",
        "Platelet": "platelet",
    },
    "TissueMNIST": {
        "Collecting Duct": "Collecting Duct, Connecting Tubule",
        "Distal Convoluted Tubule": "Distal Convoluted Tubule",
        "Glomerular Endothelial Cells": "Glomerular endothelial cells",
        "Interstitial Endothelial Cells": "Interstitial endothelial cells",
        "Leukocytes": "Leukocytes",
        "Podocytes": "Podocytes",
        "Proximal Tubule Segments": "Proximal Tubule Segments",
    },
}

DATASET_CLASSES = {
    "EuroSAT": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"],
    "PathMNIST": ["Adipose tissue", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma", "Tumor epithelium"],
    "BloodMNIST": ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"],
    "TissueMNIST": ["Collecting Duct", "Distal Convoluted Tubule", "Glomerular Endothelial Cells", "Interstitial Endothelial Cells", "Leukocytes", "Podocytes", "Proximal Tubule Segments"],
}

# Cross-domain MLLM files (4 models, all store {dataset: {accuracy: {class: %}}})
MLLM_FILES = [
    "qwen2vl7b_results.json",
    "llava15b_results.json",
    "llavanext_results.json",
    "qwen25vl7b_results.json",
]
mllm_data = {}
for fname in MLLM_FILES:
    fpath = RESULTS / "02_robustness" / "cross_domain_mllm" / fname
    if not fpath.exists():
        continue
    with open(fpath) as f:
        mllm_data[fname] = json.load(f)


def get_cross_anchor(ds_name, class_name):
    """Get AnchorScore from anchor_scores.json using key mapping."""
    if ds_name not in cross_anchor_full:
        return None
    pc = cross_anchor_full[ds_name]["per_class_acc"]
    json_key = KEY_MAP.get(ds_name, {}).get(class_name, class_name)
    if json_key in pc:
        return pc[json_key]["acc"]
    return None


cross_data = []
for ds_name, classes in DATASET_CLASSES.items():
    for cname in classes:
        a_val = get_cross_anchor(ds_name, cname)
        if a_val is None:
            print(f"  WARNING: no AnchorScore for {ds_name}/{cname}")
            continue

        mllm_vals = []
        for fname, mdata in mllm_data.items():
            acc = mdata.get(ds_name, {}).get("accuracy", {})
            if cname in acc:
                mllm_vals.append(acc[cname])

        if not mllm_vals:
            print(f"  WARNING: no MLLM acc for {ds_name}/{cname}")
            continue

        mllm_mean = np.mean(mllm_vals)
        anchor_key = KEY_MAP.get(ds_name, {}).get(cname, cname)
        cross_data.append((ds_name, anchor_key, a_val, mllm_mean))

print(f"Cross-domain: {len(cross_data)} classes")

# === 3. Pooled class-level correlation ===
all_data = scb5_data + cross_data
print(f"\nTotal pooled: {len(all_data)} classes")

all_anchor = [d[2] for d in all_data]
all_mllm = [d[3] for d in all_data]

rho, p = spearmanr(all_anchor, all_mllm)
print(f"\n=== Pooled class-level (n={len(all_data)}) ===")
print(f"  Spearman rho={rho:.3f}, p={p:.6f}")

r, pr = pearsonr(all_anchor, all_mllm)
print(f"  Pearson  r={r:.3f}, p={pr:.6f}")

# Per-domain breakdown
print(f"\n=== Per-domain breakdown ===")
domains = {}
for d in all_data:
    domain = d[0]
    if domain not in domains:
        domains[domain] = []
    domains[domain].append((d[2], d[3]))

for domain, vals in sorted(domains.items()):
    a = [v[0] for v in vals]
    m = [v[1] for v in vals]
    rho_d, p_d = spearmanr(a, m)
    print(f"  {domain:35s} n={len(vals):2d}  rho={rho_d:.3f}  p={p_d:.3f}")

# SCB5 only vs cross-domain only
scb5_a = [d[2] for d in scb5_data]
scb5_m = [d[3] for d in scb5_data]
rho_scb5, p_scb5 = spearmanr(scb5_a, scb5_m)
print(f"\n  SCB5 only: n={len(scb5_data)}, rho={rho_scb5:.3f}, p={p_scb5:.4f}")

cross_a = [d[2] for d in cross_data]
cross_m = [d[3] for d in cross_data]
rho_cross, p_cross = spearmanr(cross_a, cross_m)
print(f"  Cross-domain only: n={len(cross_data)}, rho={rho_cross:.3f}, p={p_cross:.4f}")

# === 4. Leave-one-dataset-out (7 domains: 3 SCB5 + 4 cross-domain) ===
print(f"\n{'='*70}")
print(f"  LEAVE-ONE-DATASET-OUT on pooled 47-class (7 domains)")
print(f"{'='*70}")
domains_list = sorted(domains.keys())
print(f"  Domains ({len(domains_list)}): {domains_list}")
lodo_pooled = {}
for holdout in domains_list:
    lo_a, lo_m = [], []
    for d in all_data:
        if d[0] == holdout:
            continue
        lo_a.append(d[2])
        lo_m.append(d[3])
    if len(lo_a) >= 3:
        lo_r, lo_p = spearmanr(lo_a, lo_m)
        lodo_pooled[f"exclude_{holdout}"] = {"n": len(lo_a), "rho": round(lo_r, 3), "p": round(lo_p, 4)}
        print(f"  Exclude {holdout:35s}  n={len(lo_a):2d}  rho={lo_r:.3f}  p={lo_p:.4f}")
lodo_rhos = [v["rho"] for v in lodo_pooled.values()]
print(f"  LODO: rho range [{min(lodo_rhos):.3f}, {max(lodo_rhos):.3f}], mean={np.mean(lodo_rhos):.3f}")

# Save
output = {
    "description": "Pooled class-level analysis: SCB5 (13) + cross-domain (34) = 47 classes. Cross-domain uses 4 MLLM avg. TissueMNIST drops Thick Ascending Limb (no MLLM match).",
    "computation_date": "2026-07-14",
    "n_mllms_cross_domain": 4,
    "cross_domain_mllms": [
        "LLaVA-1.5-7B",
        "LLaVA-NeXT-Mistral-7B",
        "Qwen2.5-VL-7B",
        "Qwen2-VL-7B"
    ],
    "pooled": {
        "n": len(all_data),
        "spearman_rho": round(rho, 3),
        "spearman_p": p,
        "pearson_r": round(r, 3),
        "pearson_p": pr,
    },
    "scb5_only": {
        "n": len(scb5_data),
        "spearman_rho": round(rho_scb5, 3),
        "spearman_p": p_scb5,
    },
    "cross_domain_only": {
        "n": len(cross_data),
        "spearman_rho": round(rho_cross, 3),
        "spearman_p": p_cross,
    },
    "per_domain": {},
    "leave_one_dataset_out": lodo_pooled,
    "leave_one_dataset_out_range": [
        round(min(lodo_rhos), 3),
        round(max(lodo_rhos), 3),
    ],
    "leave_one_dataset_out_mean": round(float(np.mean(lodo_rhos)), 3),
    "data": [{"domain": d[0], "class": d[1], "anchor_score": d[2], "mllm_mean": round(d[3], 1)} for d in all_data],
}
for domain, vals in sorted(domains.items()):
    a = [v[0] for v in vals]
    m = [v[1] for v in vals]
    rho_d, p_d = spearmanr(a, m)
    output["per_domain"][domain] = {
        "n": len(vals),
        "spearman_rho": round(rho_d, 3),
        "spearman_p": round(p_d, 4) if not np.isnan(p_d) else None,
    }

out_path = RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, allow_nan=False)
print(f"\nSaved to {out_path}")
