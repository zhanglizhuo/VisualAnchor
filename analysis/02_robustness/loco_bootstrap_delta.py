"""
Leave-one-class-out robustness, Bootstrap CI, and Delta-vs-AnchorScore analysis.

Usage:
    python analysis/02_robustness/loco_bootstrap_delta.py

Output:
    results/02_robustness/robustness/loco_bootstrap_delta.json
"""

import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
RNG = np.random.RandomState(42)
B = 10000


def load(path):
    with open(PROJ / path) as f:
        return json.load(f)


def bootstrap_ci(anchors, mllms, n_bootstrap=B):
    n = len(anchors)
    rhos = []
    for _ in range(n_bootstrap):
        idx = RNG.choice(n, n, replace=True)
        r, _ = spearmanr(anchors[idx], mllms[idx])
        rhos.append(r)
    ci = np.percentile(rhos, [2.5, 97.5])
    return round(ci[0], 4), round(ci[1], 4)


def main():
    pc = load("results/01_core/correlation/pooled_class_level_results.json")
    scb5 = [d for d in pc["data"] if d["domain"].startswith("SCB5")]
    cross = [d for d in pc["data"] if not d["domain"].startswith("SCB5")]

    classes = [d["class"] for d in scb5]
    anchors = np.array([d["anchor_score"] for d in scb5])
    mllms = np.array([d["mllm_mean"] for d in scb5])

    # 1. Full SCB5
    rho_full, p_full = spearmanr(anchors, mllms)

    # 2. LOCO
    loco = {}
    for i, cls in enumerate(classes):
        mask = np.ones(len(classes), dtype=bool)
        mask[i] = False
        r, p = spearmanr(anchors[mask], mllms[mask])
        loco[cls] = {"rho": round(r, 4), "p": round(p, 6), "n": 12}

    rho_vals = [v["rho"] for v in loco.values()]

    # 3. Bootstrap CI for SCB5
    ci_scb5 = bootstrap_ci(anchors, mllms)

    # 4. Cross-domain
    c_anchors = np.array([d["anchor_score"] for d in cross])
    c_mllms = np.array([d["mllm_mean"] for d in cross])
    rho_cross, p_cross = spearmanr(c_anchors, c_mllms)
    ci_cross = bootstrap_ci(c_anchors, c_mllms)

    # 5. Delta vs AnchorScore
    def delta_analysis(anchors, mllms, label):
        deltas = mllms - anchors
        r, p = spearmanr(anchors, deltas)
        mean_d = np.mean(deltas)
        return {"rho": round(r, 4), "p": round(p, 6), "mean_delta_pp": round(mean_d, 2)}

    result = {
        "description": "Leave-one-class-out robustness + Bootstrap CI + Delta-vs-AnchorScore",
        "full_scb5": {
            "rho": round(rho_full, 4),
            "p": round(p_full, 6),
            "n": 13,
            "bootstrap_ci_95": {"low": ci_scb5[0], "high": ci_scb5[1], "B": B},
        },
        "loco": loco,
        "loco_summary": {
            "range": [round(min(rho_vals), 4), round(max(rho_vals), 4)],
            "mean": round(float(np.mean(rho_vals)), 4),
        },
        "cross_domain": {
            "rho": round(rho_cross, 4),
            "p": round(p_cross, 6),
            "n": 34,
            "bootstrap_ci_95": {"low": ci_cross[0], "high": ci_cross[1], "B": B},
        },
        "delta_vs_anchor": {
            "pooled_47": delta_analysis(
                np.array([d["anchor_score"] for d in pc["data"]]),
                np.array([d["mllm_mean"] for d in pc["data"]]),
                "pooled_47",
            ),
            "scb5_13": delta_analysis(anchors, mllms, "scb5"),
            "cross_domain_34": delta_analysis(c_anchors, c_mllms, "cross"),
        },
    }

    out_path = PROJ / "results" / "02_robustness" / "robustness" / "loco_bootstrap_delta.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
