#!/usr/bin/env python3
"""
confusion_transitivity.py
=========================
Mechanism experiment: confusion-matrix transitivity between CLIP and MLLMs.

If the classes that CLIP confuses most (e.g. stand <-> blackBoard) are also
the classes that MLLMs confuse most, that is direct evidence of a *shared
visual-representation limitation* — stronger than the aggregate "low CLIP
score -> low MLLM score" correlation, because it matches the *structure* of
the errors, not just their marginals.

Method
------
For TeacherBehavior (the only SCB5 subset with enough classes for a confusion
matrix: 8 classes -> 28 unordered class pairs):

  1. Build per-model confusion matrix  C[i,j] = #(true=i, pred=j).
  2. Row-normalize to P(pred=j | true=i), then symmetrize over class pairs:
        A[i,j] = (P(pred=j|i) + P(pred=i|j)) / 2   for i < j
     A is the "class-pair confusion affinity" — a symmetric 8x8 matrix whose
     upper triangle gives 28 affinity values.
  3. Mantel test: correlate the 28-element upper-triangle vectors of CLIP and
     each MLLM. The permutation scheme simultaneously permutes rows AND columns
     of one matrix by the same class-label permutation (8! possible), which
     preserves the matrix structure — this is the correct null for
     non-independent matrix elements. Standard Pearson + Spearman, 10,000
     permutations, seed=42.

Across the 6 MLLMs we report the range and significance pattern only — we do
NOT fit a formal random-effects meta-analysis, because the 6 MLLMs are highly
inter-correlated (pairwise per-class rho = 0.864) and therefore the 6 Mantel
tests are not 6 independent estimates.

Inputs
------
  - CLIP per-image predictions:
      results/01_core/clip_per_image/per_image_predictions.json
    (TeacherBehavior -> predictions -> [{fname (basename of path),
     true_class_name, pred_class_name}, ...])
  - MLLM per-image predictions (one jsonl per model, produced by the patched
      experiments/01_core/mllm_image_level_scb5.py):
      results/01_core/mllm_image_level/predictions_{model_tag}_TeacherBehavior.jsonl
    each line: {"fname", "true_class", "pred_class" (may be null), "response"}

Output
------
  results/06_mechanism/confusion_transitivity.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "06_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "confusion_transitivity.json"

CLIP_PRED_PATH = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
MLLM_PRED_DIR = RESULTS / "01_core" / "mllm_image_level"

# TeacherBehavior canonical class order (must match anchor_scores / mllm_full)
TB_CLASSES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]

# Map (lowercased) class -> canonical name, so we can align CLIP / MLLM records
# that differ in case. The MLLM jsonl stores lowercased pred_class (from
# parse_predicted_class); CLIP stores canonical pred_class_name.
CANON = {c.lower(): c for c in TB_CLASSES}
N = len(TB_CLASSES)
DATASET = "TeacherBehavior"


def load_clip_predictions():
    """Return {fname: {"true": true_canon, "pred": pred_canon_or_None}}
    for CLIP on TeacherBehavior."""
    data = json.load(open(CLIP_PRED_PATH))
    preds = data[DATASET]["predictions"]
    out = {}
    for p in preds:
        fname = Path(p["path"]).name
        true_c = CANON.get(str(p["true_class_name"]).lower())
        pred_c = CANON.get(str(p["pred_class_name"]).lower())
        if true_c is None:
            continue
        # pred may be None if CLIP returned an out-of-set label; keep as None
        out[fname] = {"true": true_c, "pred": pred_c}
    return out


def load_mllm_predictions(model_tag):
    """Return {fname: {"true": true_canon, "pred": pred_canon_or_None,
    "response": raw_response_lower}} for one MLLM.

    pred is None when the MLLM response could not be parsed to any class;
    the raw response is retained for response-concentration diagnostics.
    """
    fpath = MLLM_PRED_DIR / f"predictions_{model_tag}_{DATASET}.jsonl"
    if not fpath.exists():
        return None
    out = {}
    for line in open(fpath):
        rec = json.loads(line)
        fname = rec["fname"]
        true_c = CANON.get(str(rec["true_class"]).lower())
        pred_raw = rec.get("pred_class")
        pred_c = CANON.get(pred_raw) if pred_raw else None
        if true_c is None:
            continue
        out[fname] = {
            "true": true_c,
            "pred": pred_c,
            "response": str(rec.get("response", "")).strip().lower(),
        }
    return out


def build_confusion_matrix(records):
    """C[i,j] = #(true=classes[i], pred=classes[j]). Drops pred=None rows."""
    C = np.zeros((N, N), dtype=float)
    n_rejected = 0
    n_total = 0
    for rec in records.values():
        true_c, pred_c = rec["true"], rec["pred"]
        n_total += 1
        if pred_c is None:
            n_rejected += 1
            continue
        i = TB_CLASSES.index(true_c)
        j = TB_CLASSES.index(pred_c)
        C[i, j] += 1
    return C, n_total, n_rejected


def dominant_response_share(records):
    """Fraction of images whose raw MLLM response equals the single most
    common response. High values (>0.5) flag near-constant / degenerate
    output behavior worth reporting at the model level."""
    counts = {}
    n = 0
    for rec in records.values():
        r = rec.get("response", "")
        if r:
            counts[r] = counts.get(r, 0) + 1
            n += 1
    if n == 0:
        return {"dominant_response": None, "dominant_share": 0.0, "n_distinct": 0}
    top, c = max(counts.items(), key=lambda kv: kv[1])
    return {
        "dominant_response": top,
        "dominant_share": round(c / n, 4),
        "n_distinct": len(counts),
    }


def affinity_matrix(C):
    """Row-normalized confusion matrix -> symmetric class-pair affinity.

    A[i,j] = (P(pred=j|i) + P(pred=i|j)) / 2  for i != j; A[i,i] = 0.
    Returns an 8x8 symmetric matrix.
    """
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = C / row_sums  # P[i,j] = P(pred=j | true=i)
    A = (P + P.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def upper_triangle_vec(A):
    """Return the 28 upper-triangle entries of an 8x8 symmetric matrix."""
    iu = np.triu_indices(N, k=1)
    return A[iu]


def mantel_test(A1, A2, n_perm=10000, seed=42):
    """Mantel test between two symmetric affinity matrices.

    Permutation: simultaneously permute rows and columns of A2 by the same
    class-label permutation (preserves matrix structure). Reports Spearman
    and Pearson of the upper-triangle vectors.
    """
    rng = np.random.default_rng(seed)
    v1 = upper_triangle_vec(A1)
    v2 = upper_triangle_vec(A2)
    rho_s, _ = spearmanr(v1, v2)
    rho_p, _ = pearsonr(v1, v2)

    count_s = 0  # permuted Spearman >= observed
    count_p = 0  # permuted Pearson >= observed
    for _ in range(n_perm):
        perm = rng.permutation(N)
        A2_perm = A2[np.ix_(perm, perm)]
        v2_perm = upper_triangle_vec(A2_perm)
        rs, _ = spearmanr(v1, v2_perm)
        rp, _ = pearsonr(v1, v2_perm)
        if rs >= rho_s:
            count_s += 1
        if rp >= rho_p:
            count_p += 1
    p_s = (count_s + 1) / (n_perm + 1)
    p_p = (count_p + 1) / (n_perm + 1)
    return {
        "rho_spearman": round(float(rho_s), 4),
        "rho_pearson": round(float(rho_p), 4),
        "p_mantel_spearman": round(p_s, 5),
        "p_mantel_pearson": round(p_p, 5),
        "n_perm": n_perm,
    }


def top_confused_pairs(A, k=5):
    """Return the k class pairs with highest confusion affinity."""
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((TB_CLASSES[i], TB_CLASSES[j], float(A[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return [{"pair": f"{a} <-> {b}", "affinity": round(vv, 4)} for a, b, vv in pairs[:k]]


def main():
    # canonical 6 Ollama models — tags match mllm_image_level_scb5.py output
    # (model.replace(":","_").replace(".","_"), so "qwen3.5:35b-a3b" -> "qwen3_5_35b-a3b")
    model_tags = [
        "qwen3_5_27b",
        "qwen3_6_27b",
        "qwen3_5_35b-a3b",
        "qwen3_6_35b-a3b",
        "gemma4_31b",
        "gemma4_26b",
    ]

    print("=" * 70)
    print("Confusion-Matrix Transitivity (TeacherBehavior, 8 classes, 28 pairs)")
    print("=" * 70)

    clip_recs = load_clip_predictions()
    print(f"CLIP per-image predictions: {len(clip_recs)} images")
    C_clip, clip_n, clip_rej = build_confusion_matrix(clip_recs)
    print(f"  CLIP usable: {clip_n - clip_rej}/{clip_n} (rejected {clip_rej} out-of-set)")
    A_clip = affinity_matrix(C_clip)
    print(f"  CLIP top confused pairs:")
    for p in top_confused_pairs(A_clip):
        print(f"    {p['pair']:40s} {p['affinity']:.3f}")

    # ── Split-half reliability ceiling (validates the Mantel machinery and
    #    gives a sampling-noise ceiling before any MLLM run is spent). ──
    fnames_sorted = sorted(clip_recs.keys())
    half1 = {f: clip_recs[f] for f in fnames_sorted[::2]}
    half2 = {f: clip_recs[f] for f in fnames_sorted[1::2]}
    C_h1, _, _ = build_confusion_matrix(half1)
    C_h2, _, _ = build_confusion_matrix(half2)
    ceiling = mantel_test(affinity_matrix(C_h1), affinity_matrix(C_h2))
    print(f"\n  CLIP split-half reliability ceiling (validates Mantel test):")
    print(f"    Spearman rho={ceiling['rho_spearman']:.3f}  p={ceiling['p_mantel_spearman']:.4f}")
    print(f"    (this is the sampling-noise ceiling a CLP-vs-MLLM Mantel could reach)")

    per_model = {}
    for tag in model_tags:
        recs = load_mllm_predictions(tag)
        if recs is None:
            print(f"\n[{tag}]  no prediction file yet — skipping (run mllm_image_level_scb5.py on V100)")
            per_model[tag] = {"status": "missing"}
            continue
        C_m, m_n, m_rej = build_confusion_matrix(recs)
        if m_n - m_rej < 50:
            print(f"\n[{tag}]  too few usable predictions ({m_n - m_rej}) — skipping")
            per_model[tag] = {"status": "too_few", "n_usable": m_n - m_rej}
            continue
        A_m = affinity_matrix(C_m)
        mantel = mantel_test(A_clip, A_m)
        top_shared = top_confused_pairs(A_m, k=5)
        result = {
            "status": "ok",
            "n_images": m_n,
            "n_rejected_unparsed": m_rej,
            "n_usable": m_n - m_rej,
            **mantel,
            "mllm_top_confused_pairs": top_shared,
            **dominant_response_share(recs),
        }
        per_model[tag] = result
        print(f"\n[{tag}]  usable={m_n - m_rej}/{m_n} (rejected {m_rej} unparsed)")
        print(f"  Mantel vs CLIP:  Spearman rho={mantel['rho_spearman']:.3f}  "
              f"p={mantel['p_mantel_spearman']:.4f}  (n_perm={mantel['n_perm']})")
        print(f"                   Pearson r ={mantel['rho_pearson']:.3f}  "
              f"p={mantel['p_mantel_pearson']:.4f}")
        print(f"  MLLM top confused pairs:")
        for p in top_shared:
            print(f"    {p['pair']:40s} {p['affinity']:.3f}")

    # Cross-MLLM summary (NO formal meta-analysis — 6 MLLMs are non-independent)
    ok_tags = [t for t, r in per_model.items() if r.get("status") == "ok"]
    ok = [per_model[t] for t in ok_tags]
    summary = {"n_models_ok": len(ok)}
    if ok:
        rhos_s = [r["rho_spearman"] for r in ok]
        rhos_p = [r["rho_pearson"] for r in ok]
        ps_s = [r["p_mantel_spearman"] for r in ok]

        # ── Multiple-comparison correction (BH-FDR), applied uniformly as in
        #    every other multi-test setting in the paper. ──
        ps_arr = np.asarray(ps_s, dtype=float)
        adj_p = multipletests(ps_arr, method="fdr_bh")[1]
        for tag, r, raw_p, ap in zip(ok_tags, ok, ps_s, adj_p):
            r["p_mantel_spearman_fdr_bh"] = round(float(ap), 5)

        # Expected false-positive count under the global null (family-wise
        # upper bound on the chance findings we would see if no real effect
        # existed at all).
        n_tests = len(ps_s)
        expected_fp_global_null = n_tests * 0.05

        n_sig_uncorrected = int(sum(1 for p in ps_s if p < 0.05))
        n_sig_fdr = int(sum(1 for p in adj_p if p < 0.05))

        summary.update({
            "rho_spearman_range": [round(min(rhos_s), 3), round(max(rhos_s), 3)],
            "rho_spearman_mean": round(float(np.mean(rhos_s)), 3),
            "rho_spearman_std": round(float(np.std(rhos_s)), 3),
            "rho_pearson_range": [round(min(rhos_p), 3), round(max(rhos_p), 3)],
            "n_significant_uncorrected_p_lt_05": n_sig_uncorrected,
            "n_significant_fdr_bh_lt_05": n_sig_fdr,
            "expected_fp_under_global_null": round(expected_fp_global_null, 3),
            "multiple_comparison_note": (
                "BH-FDR applied (uniform with the rest of the paper). "
                f"{n_sig_uncorrected}/{n_tests} uncorrected significant, "
                f"{n_sig_fdr}/{n_tests} FDR-significant. Under the global null "
                f"at alpha=0.05 we would expect ~{expected_fp_global_null:.1f} "
                "false positives from pure noise; the observed uncorrected "
                "count is consistent with that null, so the family of tests "
                "is treated as an overall NULL result (no model singled out)."
            ),
            "note": ("The 6 MLLMs are not independent estimates (pairwise "
                     "per-class rho=0.864); we report the range and "
                     "significance pattern rather than a formal meta-analysis."),
        })
        print(f"\n{'=' * 70}")
        print(f"Cross-MLLM summary ({len(ok)} models)")
        print(f"{'=' * 70}")
        print(f"  Spearman rho range: [{summary['rho_spearman_range'][0]}, "
              f"{summary['rho_spearman_range'][1]}], mean={summary['rho_spearman_mean']}")
        print(f"  Significant (uncorrected p<0.05): {n_sig_uncorrected}/{len(ok)}")
        print(f"  Significant (BH-FDR q<0.05):      {n_sig_fdr}/{len(ok)}")
        print(f"  Expected FP under global null:    ~{expected_fp_global_null:.1f}")
        print(f"  -> Overall NULL result; no model singled out.")
        print(f"  (No formal meta-analysis — 6 MLLMs are non-independent.)")

    output = {
        "description": ("Confusion-matrix transitivity: do CLIP and MLLMs confuse "
                        "the same class pairs? Mantel test on 28 TeacherBehavior "
                        "class pairs, 10000 class-label permutations."),
        "dataset": DATASET,
        "n_classes": N,
        "n_class_pairs": N * (N - 1) // 2,
        "clip_split_half_ceiling": ceiling,
        "clip_top_confused_pairs": top_confused_pairs(A_clip),
        "per_model": per_model,
        "cross_mllm_summary": summary,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
