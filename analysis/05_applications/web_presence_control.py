"""Web-presence confound control for AnchorScore-MLLM correlation.

Computes Spearman partial correlation between AnchorScore and MLLM accuracy
controlling for class name word frequency (from wordfreq, based on Common Crawl).

This addresses the concern that the correlation may be driven
by shared web-scraped training data rather than shared visual competence.

Usage:
    python analysis/05_applications/web_presence_control.py

Output:
    results/01_core/correlation/web_presence_control.json — full Spearman,
    partial Spearman controlling for web frequency, Pearson partial,
    and the relationship between web frequency and each variable.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.stats import t as t_dist
from wordfreq import word_frequency
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
OUT_PATH = RESULTS / "01_core" / "correlation" / "web_presence_control.json"


def get_web_freq(name):
    """Get word frequency from wordfreq, handling multi-word names."""
    tokens = name.lower().replace('-', ' ').replace('_', ' ').split()
    freqs = []
    for t in tokens:
        f = word_frequency(t, 'en', wordlist='large')
        if f > 0:
            freqs.append(f)
    if not freqs:
        return 1e-10
    return min(freqs)


def main():
    with open(RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json") as f:
        pooled = json.load(f)

    data = pooled['data']
    classes = [d['class'] for d in data]
    anchor = np.array([d['anchor_score'] for d in data])
    mllm = np.array([d['mllm_mean'] for d in data])
    domains = [d['domain'] for d in data]

    # Compute web frequency for each class name
    web_freqs = np.array([get_web_freq(c) for c in classes])
    log_web_freqs = np.log10(web_freqs + 1e-10)

    # 1. Full Spearman (no control)
    r_full, p_full = spearmanr(anchor, mllm)
    print(f"Full Spearman:  rho = {r_full:.4f}, p = {p_full:.6f}")

    # 2. Partial Spearman controlling for web frequency
    from scipy.stats import rankdata
    rank_anchor = rankdata(anchor)
    rank_mllm = rankdata(mllm)
    rank_web = rankdata(log_web_freqs)

    X = rank_web.reshape(-1, 1)
    reg_a = LinearRegression().fit(X, rank_anchor)
    reg_m = LinearRegression().fit(X, rank_mllm)
    res_a = rank_anchor - reg_a.predict(X)
    res_m = rank_mllm - reg_m.predict(X)

    r_partial, p_partial = spearmanr(res_a, res_m)
    print(f"Partial Spearman (controlling for web freq):")
    print(f"  rho = {r_partial:.4f}, p = {p_partial:.6f}")

    # 3. Pearson partial correlation (parametric)
    r_am, _ = pearsonr(rank_anchor, rank_mllm)
    r_aw, _ = pearsonr(rank_anchor, rank_web)
    r_mw, _ = pearsonr(rank_mllm, rank_web)
    n = len(classes)
    partial_r = (r_am - r_aw * r_mw) / (
        np.sqrt(1 - r_aw ** 2) * np.sqrt(1 - r_mw ** 2)
    )
    t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r ** 2))
    partial_p = 2 * (1 - t_dist.cdf(abs(t_stat), n - 3))
    print(f"Pearson partial r = {partial_r:.4f}, p = {partial_p:.6f}")

    # 4. Web frequency vs each variable
    r_wa, p_wa = spearmanr(log_web_freqs, anchor)
    r_wm, p_wm = spearmanr(log_web_freqs, mllm)
    print(f"\nWeb freq vs Anchor: rho = {r_wa:.4f}, p = {p_wa:.6f}")
    print(f"Web freq vs MLLM:  rho = {r_wm:.4f}, p = {p_wm:.6f}")

    # 5. Per-domain analysis (classroom vs cross-domain)
    classroom_idx = [i for i, d in enumerate(domains) if d.startswith('SCB5')]
    cross_idx = [i for i, d in enumerate(domains) if not d.startswith('SCB5')]

    per_domain = {}
    for name, idx in [("Classroom only", classroom_idx), ("Cross-domain only", cross_idx)]:
        r, p = spearmanr(anchor[idx], mllm[idx])
        r_w, p_w = spearmanr(log_web_freqs[idx], anchor[idx])
        r_wm, p_wm = spearmanr(log_web_freqs[idx], mllm[idx])
        print(f"\n{name} (n={len(idx)}):")
        print(f"  Full: rho={r:.4f}, p={p:.6f}")
        print(f"  Web freq vs Anchor: rho={r_w:.4f}, p={p_w:.6f}")
        print(f"  Web freq vs MLLM: rho={r_wm:.4f}, p={p_wm:.6f}")
        per_domain[name] = {
            "n": len(idx),
            "spearman_rho": round(r, 4),
            "spearman_p": round(p, 6),
            "web_freq_vs_anchor_rho": round(r_w, 4),
            "web_freq_vs_anchor_p": round(p_w, 6),
            "web_freq_vs_mllm_rho": round(r_wm, 4),
            "web_freq_vs_mllm_p": round(p_wm, 6),
        }

    result = {
        "description": "Web-presence confound control on the pooled 47-class dataset",
        "n": n,
        "full_spearman": {"rho": round(r_full, 4), "p": round(p_full, 6)},
        "partial_spearman": {"rho": round(r_partial, 4), "p": round(p_partial, 6)},
        "pearson_partial": {"rho": round(partial_r, 4), "p": round(partial_p, 6)},
        "web_freq_vs_anchor": {"rho": round(r_wa, 4), "p": round(p_wa, 6)},
        "web_freq_vs_mllm": {"rho": round(r_wm, 4), "p": round(p_wm, 6)},
        "per_domain": per_domain,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
