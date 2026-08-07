#!/usr/bin/env python3
"""Linear mixed-effects model: MLLM accuracy ~ AnchorScore + (1 | MLLM).

Addresses the concern that the 78-point pooled analysis treats 6 MLLMs as
independent replicates when they share 13 AnchorScore values and exhibit
high pairwise agreement (mean pairwise ρ=0.864).

Input: results/01_core/correlation/unified_results.json (per_class array)
Output: results/01_core/correlation/mixed_effects_model.json

Usage:
    python analysis/01_core/mixed_effects_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from statsmodels.formula.api import mixedlm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "01_core" / "correlation" / "unified_results.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "01_core" / "correlation" / "mixed_effects_model.json"


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    rows = []
    for pc in data["per_class"]:
        anchor = pc["anchor_score"]
        for mllm, acc in pc["mllm_values"].items():
            rows.append({
                "class": pc["class"],
                "dataset": pc["dataset"],
                "mllm": mllm,
                "anchor_score": anchor,
                "accuracy": acc,
            })
    return rows, data


def rows_to_dict(rows):
    return {k: [r[k] for r in rows] for k in rows[0].keys()}


def fit_mixed_model(df):
    model = mixedlm("accuracy ~ anchor_score", df, groups="mllm")
    result = model.fit()
    return result


def build_output(result, rows, unified_data):
    df = rows_to_dict(rows)
    re_var = float(result.cov_re.iloc[0, 0])
    resid_var = float(result.scale)
    icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0

    # Per-MLLM random intercepts
    re = result.random_effects
    re_vals = {k: float(v.iloc[0]) for k, v in re.items()}

    # Per-MLLM mean accuracy
    mllm_means = {}
    for mllm in set(df["mllm"]):
        idx = [i for i, m in enumerate(df["mllm"]) if m == mllm]
        mllm_means[mllm] = float(np.mean([df["accuracy"][i] for i in idx]))

    # Naive pooled Spearman
    rho, p = spearmanr(df["anchor_score"], df["accuracy"])

    ci = result.conf_int().loc["anchor_score"]

    return {
        "model": {
            "method": "REML",
            "dependent_variable": "MLLM accuracy (%)",
            "fixed_effects": {
                "intercept": float(result.fe_params["Intercept"]),
                "anchor_score": float(result.fe_params["anchor_score"]),
                "anchor_score_p": float(result.pvalues["anchor_score"]),
                "anchor_score_ci_95": [float(ci[0]), float(ci[1])],
            },
            "random_effects": {
                "mllm_intercept_var": re_var,
                "residual_var": resid_var,
            },
            "icc_mllm": icc,
            "fit": {
                "n_obs": int(result.nobs),
                "n_groups": len(result.random_effects),
                "log_likelihood": float(result.llf),
                "converged": bool(result.converged),
            },
            "software": {
                "package": "statsmodels",
                "version": mixedlm.__module__,  # just informational
            },
        },
        "per_mllm_mean_accuracy": mllm_means,
        "per_mllm_random_intercept": re_vals,
        "naive_pooled_spearman": {
            "rho": float(rho),
            "p": float(p),
            "n": int(len(df["accuracy"])),
        },
        "input_source": str(INPUT_PATH),
        "canonical_mllms": unified_data["meta"]["canonical_mllms"],
    }


def main():
    if not INPUT_PATH.exists():
        print(f"Error: input not found at {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    rows, unified_data = load_data(INPUT_PATH)
    df = rows_to_dict(rows)
    result = fit_mixed_model(df)

    output = build_output(result, rows, unified_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Mixed-effects model results saved to {OUTPUT_PATH}")
    print(f"  AnchorScore coefficient: {output['model']['fixed_effects']['anchor_score']:.4f}")
    print(f"  p-value: {output['model']['fixed_effects']['anchor_score_p']:.2e}")
    print(f"  95% CI: [{output['model']['fixed_effects']['anchor_score_ci_95'][0]:.3f}, "
          f"{output['model']['fixed_effects']['anchor_score_ci_95'][1]:.3f}]")
    print(f"  ICC(MLLM): {output['model']['icc_mllm']:.6f}")
    print(f"  Converged: {output['model']['fit']['converged']}")


if __name__ == "__main__":
    main()
