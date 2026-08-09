import json, os
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "results", "06_consensus_control")
OUT_PATH = os.path.join(OUT_DIR, "exp2_consensus_control.json")
os.makedirs(OUT_DIR, exist_ok=True)

CANONICAL = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B", "Gemma4-31B", "Gemma4-26B"]
DATASETS = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def load_anchor():
    with open(os.path.join(ROOT, "results/01_core/anchor_score_scb5/anchor_scores.json")) as f:
        d = json.load(f)
    out = {}
    for ds in DATASETS:
        out.update({k: v["acc"] for k, v in d[ds]["per_class_acc"].items()})
    return out


def load_mllm():
    with open(os.path.join(ROOT, "results/01_core/paper_data/mllm_raw.json")) as f:
        d = json.load(f)
    rows = []
    for ds in DATASETS:
        for m in CANONICAL:
            for cls, acc in d[ds][m].items():
                rows.append({"ds": ds, "class": cls, "model": m, "acc": acc})
    return rows


def partial_rho(x, y, z):
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(xr), zr])
    rx = xr - Z @ np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    anchor = load_anchor()
    rows = load_mllm()
    classes = sorted({r["class"] for r in rows})
    for row in rows:
        others = [x for x in CANONICAL if x != row["model"]]
        row["proxy"] = float(np.mean([
            r2["acc"] for r2 in rows
            if r2["class"] == row["class"] and r2["model"] in others
        ]))

    accs = [r["acc"] for r in rows]
    prox = [r["proxy"] for r in rows]
    anch = [anchor[r["class"]] for r in rows]
    class_ids = [classes.index(r["class"]) for r in rows]

    rho_acc, p_acc = spearmanr(anch, accs)
    rho_prox, p_prox = spearmanr(anch, prox)
    p_partial = partial_rho(anch, accs, prox)

    rng = np.random.default_rng(42)
    B = 10000
    boot = []
    for _ in range(B):
        picks = set(rng.choice(len(classes), size=len(classes), replace=True))
        msk = [c in picks for c in class_ids]
        aa = [anch[i] for i in range(len(rows)) if msk[i]]
        bb = [accs[i] for i in range(len(rows)) if msk[i]]
        zz = [prox[i] for i in range(len(rows)) if msk[i]]
        if len(set(aa)) < 3 or len(set(bb)) < 3 or len(set(zz)) < 3:
            continue
        boot.append(partial_rho(aa, bb, zz))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    class_mean = {c: float(np.mean([r["acc"] for r in rows if r["class"] == c])) for c in classes}
    class_proxy = {c: float(np.mean([r["proxy"] for r in rows if r["class"] == c])) for c in classes}
    rho13, p13 = spearmanr([anchor[c] for c in classes], [class_mean[c] for c in classes])
    partial13 = partial_rho([anchor[c] for c in classes], [class_mean[c] for c in classes],
                            [class_proxy[c] for c in classes])

    result = {
        "description": "Exp2 consensus control. AnchorsScore vs MLLM class accuracy, cond. on generic difficulty proxy (mean accuracy of the other 5 MLLMs on the same class).",
        "partB_flagged": "mllm_image_level jsonl (TeacherBehavior) disagree with canonical mllm_raw.json (e.g. guide 93% vs 52%) - different pipeline; excluded.",
        "partA_pooled": {
            "n": len(rows),
            "rho_anchor_vs_mllm": float(rho_acc),
            "p": float(p_acc),
            "rho_anchor_vs_loo_proxy": float(rho_prox),
            "p_proxy": float(p_prox),
            "partial_rho_given_proxy": p_partial,
            "partial_ci95_cluster_boot": ci,
        },
        "partA_class_level": {
            "n": len(classes),
            "rho_anchor_vs_mean_mllm": float(rho13),
            "p": float(p13),
            "partial_rho_given_proxy": partial13,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()