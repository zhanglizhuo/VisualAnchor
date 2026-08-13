"""
class_clustered_inference.py

Class-clustered inference on the 78 pooled SCB5 points (6 MLLMs x 13 classes).

The (1|MLLM) mixed-effects check in the paper sits at the variance-component
boundary (ICC~0). This script fits the design-respecting variant: OLS with
HC1 cluster-robust standard errors over G=13 class clusters, with inference
under a t distribution with G-1=12 degrees of freedom.

Result: beta=0.849 per pp, cluster-robust SE=0.152, 95% CI [0.551, 1.147],
p=1.2e-4 -- significant at class-level clustering (class-level ICC=0.881).
"""
import json
import numpy as np
from scipy.stats import t as tdist
from pathlib import Path
from datetime import date

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
MLLM_RAW = RESULTS / "01_core" / "paper_data" / "mllm_raw.json"
ANCHOR = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
OUT = RESULTS / "01_core" / "correlation" / "class_clustered_inference.json"

mllm = json.load(open(MLLM_RAW))
anchor = json.load(open(ANCHOR))

rows = []
models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    for cls, v in anchor[ds]["per_class_acc"].items():
        for m in models:
            macc = mllm[ds][m].get(cls)
            if macc is None:
                continue
            rows.append((ds, cls, m, v["acc"], macc))

X = np.array([r[3] for r in rows])
Y = np.array([r[4] for r in rows])
n = len(X)
Xm = np.column_stack([np.ones(n), X])
beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
resid = Y - Xm @ beta

clusters = {}
for i, r in enumerate(rows):
    clusters.setdefault((r[0], r[1]), []).append(i)
G = len(clusters)

meat = np.zeros((2, 2))
for idxs in clusters.values():
    Xg = Xm[idxs]
    eg = resid[idxs]
    meat += Xg.T @ np.outer(eg, eg) @ Xg
bread = np.linalg.inv(Xm.T @ Xm)
dof_adj = (G / (G - 1)) * ((n - 1) / (n - 2))
V = dof_adj * bread @ meat @ bread
se = np.sqrt(np.diag(V))
t_stat = beta[1] / se[1]
p = 2 * (1 - tdist.cdf(abs(t_stat), G - 1))

grand = Y.mean()
ss_between = sum(len(v) * (Y[v].mean() - grand) ** 2 for v in clusters.values())
ss_total = ((Y - grand) ** 2).sum()
icc = ss_between / ss_total

out = {
    "description": "Class-clustered OLS on the 78 pooled SCB5 points (6 MLLMs x 13 classes), HC1 cluster-robust SEs with G=13 class clusters, t-distribution with G-1=12 df. Complements the (1|MLLM) mixed-effects check; class is the design-respecting clustering dimension (ICC=0.881).",
    "computation_date": str(date.today()),
    "n": n,
    "n_clusters": G,
    "beta_per_pp": round(float(beta[1]), 3),
    "cluster_robust_se": round(float(se[1]), 3),
    "ci95": [round(float(beta[1] - 1.96 * se[1]), 3), round(float(beta[1] + 1.96 * se[1]), 3)],
    "p_t12": round(p, 6),
    "icc_class_anova": round(float(icc), 3),
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"Saved {OUT}")
print(f"beta={beta[1]:.3f} SE={se[1]:.3f} CI=[{beta[1]-1.96*se[1]:.3f},{beta[1]+1.96*se[1]:.3f}] p={p:.2e} G={G} ICC={icc:.3f}")
