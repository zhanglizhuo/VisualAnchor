"""Within-dataset Spearman rho between AnchorScore and 4-MLLM mean accuracy.

Computes the four within-dataset class-level correlations reported in
VisualAnchor.tex Appendix Table A3 headers (EuroSAT rho=0.286, PathMNIST
rho=0.162, BloodMNIST rho=0.019, TissueMNIST rho=0.741). Class alignment
follows analysis/01_core/cross_domain_mllm_subset.py:
- TissueMNIST merged AnchorScore class "Collecting Duct, Connecting Tubule"
  is compared against the MLLM's "Collecting Duct" accuracy (partial mapping;
  the MLLM evaluated the two sub-classes separately).
- "Thick Ascending Limb" has no MLLM counterpart and is excluded.
This yields n = 10 + 9 + 8 + 7 = 34 classes, matching the pooled
cross-domain class-level analysis (n=34).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

anchor = json.load(open(RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"))

MLLM_FILES = {
    "Qwen2-VL-7B": "qwen2vl7b_results.json",
    "LLaVA-1.5-7B": "llava15b/master_20260713_124830.json",
    "LLaVA-NeXT-Mistral-7B": "llavanext_results.json",
    "Qwen2.5-VL-7B": "qwen25vl7b_results.json",
}

TISSUE_FILES = {
    "Qwen2-VL-7B": "qwen2vl7b_results.json",
    "LLaVA-1.5-7B": "llava15b/master_20260713_130222.json",
    "LLaVA-NeXT-Mistral-7B": "llavanext_results.json",
    "Qwen2.5-VL-7B": "qwen25vl7b_results.json",
}

ALIAS = {
    "colorectal adenocarcinoma epithelium": "Tumor epithelium",
    "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "Immature granulocyte",
    "adipose": "Adipose tissue",
}


def norm(s):
    return s.strip().lower().replace("_", " ")


def acc_dict(m, ds):
    return m[ds]["accuracy"] if "accuracy" in m[ds] else m[ds]


def find(d, cls):
    c = norm(ALIAS.get(cls, cls))
    for k, v in d.items():
        if norm(k) == c:
            return v
    return None


mllm = {
    name: json.load(open(RESULTS / "02_robustness" / "cross_domain_mllm" / fname))
    for name, fname in MLLM_FILES.items()
}
tissue = {
    name: json.load(open(RESULTS / "02_robustness" / "cross_domain_mllm" / fname))
    for name, fname in TISSUE_FILES.items()
}

out = {"description": __doc__ .strip(), "per_dataset": {}}

for ds in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
    anchors, mmeans = [], []
    for cls, info in anchor[ds]["per_class_acc"].items():
        vals = []
        if ds == "TissueMNIST":
            if "Thick Ascending" in cls:
                continue
            if "Collecting Duct, Connecting" in cls:
                vals = [acc_dict(m, ds).get("Collecting Duct") for m in tissue.values()]
            else:
                vals = [find(acc_dict(m, ds), cls) for m in tissue.values()]
        else:
            vals = [find(acc_dict(m, ds), cls) for m in mllm.values()]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        anchors.append(info["acc"])
        mmeans.append(float(np.mean(vals)))
    rho, p = spearmanr(anchors, mmeans)
    out["per_dataset"][ds] = {
        "n": len(anchors),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 4),
    }
    print(f"{ds}: n={len(anchors)} rho={rho:.4f} p={p:.4f}")

out_path = RESULTS / "02_robustness" / "cross_domain_mllm" / "within_dataset_correlations.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nSaved to {out_path}")
