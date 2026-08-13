"""Random-effects meta-analysis of AnchorScore-MLLM correlations.

Four independent datasets:
  - SCB5 (n=13, rho=0.769)
  - SCB-LLM-202506 (n=10, rho=0.506)
  - Cross-domain per-class (n=34, rho=0.462)
  - Stanford40 (n=40, rho=0.817)

Reports fixed-effect and random-effects pooled estimates,
heterogeneity (I^2, tau^2, Q-test), and 95% prediction interval.

Usage:
    python analysis/05_applications/meta_analysis.py
"""

import json
import numpy as np
from scipy.stats import chi2
from pathlib import Path


def fisher_z(r):
    return np.arctanh(r)


def fisher_z_inv(z):
    return np.tanh(z)


def main():
    studies = [
        {"name": "SCB5", "r": 0.769, "n": 13},
        {"name": "SCB-LLM-202506", "r": 0.506, "n": 10},
        {"name": "Cross-domain per-class", "r": 0.462, "n": 34},
        {"name": "Stanford40", "r": 0.817, "n": 40},
    ]

    all_studies = [
        {"name": "SCB5", "r": 0.769, "n": 13},
        {"name": "SCB-LLM-202506", "r": 0.506, "n": 10},
        {"name": "Cross-domain per-class", "r": 0.462, "n": 34},
    ]

    k = len(studies)
    z_vals = np.array([fisher_z(s['r']) for s in studies])
    se_vals = np.array([1.0 / np.sqrt(s['n'] - 3) for s in studies])
    w_fixed = 1.0 / se_vals ** 2

    # Fixed-effect
    z_fixed = np.sum(w_fixed * z_vals) / np.sum(w_fixed)
    se_fixed = np.sqrt(1.0 / np.sum(w_fixed))
    r_fixed = fisher_z_inv(z_fixed)
    r_fixed_ci = (fisher_z_inv(z_fixed - 1.96 * se_fixed),
                  fisher_z_inv(z_fixed + 1.96 * se_fixed))

    # Heterogeneity
    Q = np.sum(w_fixed * (z_vals - z_fixed) ** 2)
    df = k - 1
    p_het = 1 - chi2.cdf(Q, df)
    I2 = max(0, (Q - df) / Q * 100)
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / C)

    # Random-effects
    w_random = 1.0 / (se_vals ** 2 + tau2)
    z_random = np.sum(w_random * z_vals) / np.sum(w_random)
    se_random = np.sqrt(1.0 / np.sum(w_random))
    r_random = fisher_z_inv(z_random)
    r_random_ci = (fisher_z_inv(z_random - 1.96 * se_random),
                   fisher_z_inv(z_random + 1.96 * se_random))

    # 95% prediction interval
    pred_sd = np.sqrt(se_random ** 2 + tau2)
    r_pred = (fisher_z_inv(z_random - 1.96 * pred_sd),
              fisher_z_inv(z_random + 1.96 * pred_sd))

    # Classroom-only (SCB5 + SCB-LLM)
    classroom = [s for s in studies if s['name'] not in ('Cross-domain per-class', 'Stanford40')]
    z_c = np.array([fisher_z(s['r']) for s in classroom])
    se_c = np.array([1.0 / np.sqrt(s['n'] - 3) for s in classroom])
    w_c = 1.0 / se_c ** 2
    z_c_pooled = np.sum(w_c * z_c) / np.sum(w_c)
    se_c_pooled = np.sqrt(1.0 / np.sum(w_c))
    r_c_pooled = fisher_z_inv(z_c_pooled)
    ci_c = (fisher_z_inv(z_c_pooled - 1.96 * se_c_pooled),
            fisher_z_inv(z_c_pooled + 1.96 * se_c_pooled))

    # All-scene (SCB5 + SCB-LLM + Stanford40)
    all_scene = [s for s in studies if s['name'] != 'Cross-domain per-class']
    z_a = np.array([fisher_z(s['r']) for s in all_scene])
    se_a = np.array([1.0 / np.sqrt(s['n'] - 3) for s in all_scene])
    w_a = 1.0 / se_a ** 2
    z_a_pooled = np.sum(w_a * z_a) / np.sum(w_a)
    se_a_pooled = np.sqrt(1.0 / np.sum(w_a))
    r_a_pooled = fisher_z_inv(z_a_pooled)
    ci_a = (fisher_z_inv(z_a_pooled - 1.96 * se_a_pooled),
            fisher_z_inv(z_a_pooled + 1.96 * se_a_pooled))
    Q_a = np.sum(w_a * (z_a - z_a_pooled) ** 2)
    I2_a = max(0, (Q_a - (len(all_scene) - 1)) / Q_a * 100) if Q_a > 0 else 0

    # Build output structure
    output = {
        "description": "Meta-analysis of AnchorScore-MLLM correlations",
        "date": "2026-07-17",
        "studies": [
            {"name": s["name"], "r": s["r"], "n": s["n"],
             "se": round(float(1.0 / np.sqrt(s["n"] - 3)), 4),
             "ci_95": [
                 round(float(fisher_z_inv(fisher_z(s["r"]) - 1.96 / np.sqrt(s["n"] - 3))), 4),
                 round(float(fisher_z_inv(fisher_z(s["r"]) + 1.96 / np.sqrt(s["n"] - 3))), 4),
             ]}
            for s in studies
        ],
        "fixed_effect": {
            "r": round(float(r_fixed), 4),
            "ci_95": [round(float(r_fixed_ci[0]), 4), round(float(r_fixed_ci[1]), 4)],
        },
        "heterogeneity": {
            "Q": round(float(Q), 4),
            "df": df,
            "p": round(float(p_het), 4),
            "I2_pct": round(float(I2), 2),
            "tau2": round(float(tau2), 4),
            "significant": bool(p_het < 0.05),
        },
        "random_effects": {
            "r": round(float(r_random), 4),
            "ci_95": [round(float(r_random_ci[0]), 4), round(float(r_random_ci[1]), 4)],
            "prediction_interval_95": [
                round(float(r_pred[0]), 4),
                round(float(r_pred[1]), 4),
            ],
        },
        "subgroup_classroom_only": {
            "description": "SCB5 + SCB-LLM-202506",
            "r": round(float(r_c_pooled), 4),
            "ci_95": [round(float(ci_c[0]), 4), round(float(ci_c[1]), 4)],
        },
        "subgroup_all_scene": {
            "description": "SCB5 + SCB-LLM-202506 + Stanford40 (excludes cross-domain)",
            "r": round(float(r_a_pooled), 4),
            "ci_95": [round(float(ci_a[0]), 4), round(float(ci_a[1]), 4)],
            "I2_pct": round(float(I2_a), 2),
            "Q": round(float(Q_a), 4),
        },
        "previous_3_study": {
            "description": "Same as prior version without Stanford40",
            "studies": [s["name"] for s in all_studies],
            "r": None,
            "ci_95": None,
        },
    }

    # Recompute 3-study version for reference
    z_old = np.array([fisher_z(s['r']) for s in all_studies])
    se_old = np.array([1.0 / np.sqrt(s['n'] - 3) for s in all_studies])
    w_old = 1.0 / se_old ** 2
    Q_old = np.sum(w_old * (z_old - np.sum(w_old * z_old) / np.sum(w_old)) ** 2)
    C_old = np.sum(w_old) - np.sum(w_old ** 2) / np.sum(w_old)
    tau2_old = max(0, (Q_old - 2) / C_old)
    w_old_random = 1.0 / (se_old ** 2 + tau2_old)
    z_old_random = np.sum(w_old_random * z_old) / np.sum(w_old_random)
    se_old_random = np.sqrt(1.0 / np.sum(w_old_random))
    r_old_random = fisher_z_inv(z_old_random)
    ci_old = (fisher_z_inv(z_old_random - 1.96 * se_old_random),
              fisher_z_inv(z_old_random + 1.96 * se_old_random))
    output["previous_3_study"]["r"] = round(float(r_old_random), 4)
    output["previous_3_study"]["ci_95"] = [round(float(ci_old[0]), 4), round(float(ci_old[1]), 4)]

    # Save
    out_path = Path(__file__).resolve().parents[2] / "results/05_applications/meta_analysis_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    # Print report
    print("=" * 65)
    print("META-ANALYSIS OF ANCHORSCORE--MLLM CORRELATIONS")
    print("=" * 65)

    print("\nIndividual studies:")
    for s in output["studies"]:
        print(f"  {s['name']:25s}  r={s['r']:.3f}, n={s['n']:2d}, "
              f"95% CI [{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}]")

    print(f"\n--- Fixed-effect ---")
    fe = output["fixed_effect"]
    print(f"Pooled r = {fe['r']:.3f}, 95% CI [{fe['ci_95'][0]:.3f}, {fe['ci_95'][1]:.3f}]")

    print(f"\n--- Heterogeneity ---")
    h = output["heterogeneity"]
    print(f"Q = {h['Q']:.2f}, df = {h['df']}, p = {h['p']:.4f}")
    print(f"I² = {h['I2_pct']:.1f}%")
    print(f"τ² = {h['tau2']:.4f}")
    print(f"{'Significant heterogeneity' if h['significant'] else 'No significant heterogeneity (p > 0.05)'}")

    print(f"\n--- Random-effects ---")
    re = output["random_effects"]
    print(f"Pooled r = {re['r']:.3f}, 95% CI [{re['ci_95'][0]:.3f}, {re['ci_95'][1]:.3f}]")
    print(f"95% Prediction interval: [{re['prediction_interval_95'][0]:.3f}, {re['prediction_interval_95'][1]:.3f}]")

    print(f"\n--- Subgroup: Classroom only ---")
    sg = output["subgroup_classroom_only"]
    print(f"  r = {sg['r']:.3f}, 95% CI [{sg['ci_95'][0]:.3f}, {sg['ci_95'][1]:.3f}]")

    print(f"\n--- Subgroup: All scene (excl. cross-domain) ---")
    sg2 = output["subgroup_all_scene"]
    print(f"  r = {sg2['r']:.3f}, 95% CI [{sg2['ci_95'][0]:.3f}, {sg2['ci_95'][1]:.3f}]")

    print(f"\n--- Previous (3-study) for reference ---")
    prev = output["previous_3_study"]
    print(f"  r = {prev['r']:.3f}, 95% CI [{prev['ci_95'][0]:.3f}, {prev['ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
