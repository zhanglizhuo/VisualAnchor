"""
medical_only_correlation.py

General-CLIP (LAION ViT-L/14) AnchorScore vs 4-MLLM mean accuracy on the 24
medical classes (PathMNIST 9 + BloodMNIST 8 + TissueMNIST 7; Thick Ascending
Limb excluded, no MLLM counterpart).

Distinguishes the general-CLIP medical-only signal (rho=0.214, p=0.32) from
the domain-specialized BiomedCLIP backbone control (pooled rho=0.046, p=0.83,
n=25; results/02_robustness/cross_domain_medclip/cross_domain_medclip_results.json).
"""
import json
from scipy.stats import spearmanr
from pathlib import Path
from datetime import date

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
POOLED = RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json"
OUT = RESULTS / "02_robustness" / "medical_only_correlation.json"

d = json.load(open(POOLED))
med_a, med_m = [], []
for row in d["data"]:
    if row["domain"] in ("PathMNIST", "BloodMNIST", "TissueMNIST"):
        med_a.append(row["anchor_score"])
        med_m.append(row["mllm_mean"])
rho, p = spearmanr(med_a, med_m)

out = {
    "description": "General-CLIP (LAION ViT-L/14) AnchorScore vs 4-MLLM mean accuracy on the 24 medical classes (PathMNIST 9 + BloodMNIST 8 + TissueMNIST 7, Thick Ascending Limb excluded). Non-significant: the general-CLIP signal attenuates on medical data. Contrast with the BiomedCLIP backbone control (results/02_robustness/cross_domain_medclip/): pooled rho=0.046, p=0.83, n=25.",
    "computation_date": str(date.today()),
    "n": len(med_a),
    "spearman_rho": round(rho, 3),
    "spearman_p": round(p, 3),
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"Saved {OUT}: rho={rho:.3f} p={p:.3f} n={len(med_a)}")
