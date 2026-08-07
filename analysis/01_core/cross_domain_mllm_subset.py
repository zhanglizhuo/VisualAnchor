"""Cross-domain per-class correlation with 2-MLLM and 4-MLLM subsets.

Reproduces the 2-MLLM (Qwen2-VL + LLaVA-1.5) vs 4-MLLM comparison reported
in VisualAnchor.tex §4.2 (pooled class-level analysis). Uses the same class
mappings and MLLM files as analysis/01_core/unified_correlation.py.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

cross_anchor = json.load(open(RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"))

PATHMNIST_KEY_MAP = {
    "Adipose tissue": "adipose", "Background": "background", "Debris": "debris",
    "Lymphocytes": "lymphocytes", "Mucus": "mucus", "Smooth muscle": "smooth muscle",
    "Normal colon mucosa": "normal colon mucosa",
    "Cancer-associated stroma": "cancer-associated stroma",
    "Tumor epithelium": "colorectal adenocarcinoma epithelium",
}
BLOODMNIST_KEY_MAP = {
    "Basophil": "basophil", "Eosinophil": "eosinophil", "Erythroblast": "erythroblast",
    "Immature granulocyte": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
    "Lymphocyte": "lymphocyte", "Monocyte": "monocyte", "Neutrophil": "neutrophil",
    "Platelet": "platelet",
}
TISSUEMNIST_KEY_MAP = {
    "Collecting Duct": "Collecting Duct, Connecting Tubule",
    "Distal Convoluted Tubule": "Distal Convoluted Tubule",
    "Glomerular Endothelial Cells": "Glomerular endothelial cells",
    "Interstitial Endothelial Cells": "Interstitial endothelial cells",
    "Leukocytes": "Leukocytes", "Podocytes": "Podocytes",
    "Proximal Tubule Segments": "Proximal Tubule Segments",
}
KEY_MAP = {"PathMNIST": PATHMNIST_KEY_MAP, "BloodMNIST": BLOODMNIST_KEY_MAP, "TissueMNIST": TISSUEMNIST_KEY_MAP}

DATASET_CLASSES = {
    "EuroSAT": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"],
    "PathMNIST": ["Adipose tissue", "Background", "Debris", "Lymphocytes", "Mucus",
                  "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma",
                  "Tumor epithelium"],
    "BloodMNIST": ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte",
                   "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"],
    "TissueMNIST": ["Collecting Duct", "Distal Convoluted Tubule",
                    "Glomerular Endothelial Cells", "Interstitial Endothelial Cells",
                    "Leukocytes", "Podocytes", "Proximal Tubule Segments"],
}

MLLM_FILES = {
    "Qwen2-VL-7B": "qwen2vl7b_results.json",
    "LLaVA-1.5-7B": "llava15b_results.json",
    "LLaVA-NeXT-Mistral-7B": "llavanext_results.json",
    "Qwen2.5-VL-7B": "qwen25vl7b_results.json",
}

mllm_data = {name: json.load(open(RESULTS / "02_robustness" / "cross_domain_mllm" / fname))
             for name, fname in MLLM_FILES.items()}


def build_points(subset):
    """Return list of (dataset, class, anchor, mean_mllm) for the given MLLM subset."""
    pts = []
    for ds, classes in DATASET_CLASSES.items():
        for cname in classes:
            jk = KEY_MAP.get(ds, {}).get(cname, cname)
            a = cross_anchor[ds]["per_class_acc"].get(jk)
            if a is None:
                continue
            vals = []
            for name in subset:
                acc = mllm_data[name].get(ds, {}).get("accuracy", {})
                if cname in acc:
                    vals.append(acc[cname])
            if not vals:
                continue
            pts.append({"dataset": ds, "class": cname, "anchor": a["acc"],
                        "mllm_mean": float(np.mean(vals))})
    return pts


def correlate(pts):
    r = spearmanr([p["anchor"] for p in pts], [p["mllm_mean"] for p in pts])
    return {"n": len(pts), "spearman_rho": round(float(r.statistic), 4),
            "spearman_p": round(float(r.pvalue), 6)}


out = {
    "description": "Cross-domain class-level correlation by MLLM subset. "
                   "2-MLLM = Qwen2-VL-7B + LLaVA-1.5-7B (earlier two-model evaluation); "
                   "4-MLLM adds Qwen2.5-VL-7B and LLaVA-NeXT-Mistral-7B. "
                   "Reproduces unified_results.json cross_domain_class_level for 4-MLLM.",
    "two_mllm": correlate(build_points(["Qwen2-VL-7B", "LLaVA-1.5-7B"])),
    "four_mllm": correlate(build_points(list(MLLM_FILES))),
}

out_path = RESULTS / "01_core" / "correlation" / "cross_domain_mllm_subset.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
