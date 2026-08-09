import json, os
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/lizhuo/Work/Mayan/VisualAnchor"
OUT = os.path.join(ROOT, "results", "06_consensus_control")
os.makedirs(OUT, exist_ok=True)

MLLM_COST_RATIO = 100.0
TAU_PRE = 50
TAU_GRID = list(range(1, 100))
DATASETS = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def load_sizes():
    d = json.load(open(f"{ROOT}/results/01_core/anchor_score_scb5/anchor_scores.json"))
    out = {}
    for ds in DATASETS:
        out.update({(ds, c): v["n"] for c, v in d[ds]["per_class_acc"].items()})
    return out


def load_mllm_mean():
    d = json.load(open(f"{ROOT}/results/01_core/paper_data/mllm_raw.json"))
    agg, cnt = {}, {}
    for ds in d:
        if isinstance(d[ds], str):
            continue
        for m in d[ds]:
            for cls, acc in d[ds][m].items():
                agg.setdefault((ds, cls), 0.0)
                cnt.setdefault((ds, cls), 0)
                agg[(ds, cls)] += acc
                cnt[(ds, cls)] += 1
    return {(ds, cls): agg[(ds, cls)] / cnt[(ds, cls)] for (ds, cls) in agg}


def hybrid_acc(classes, n, a, pc_mllm, tau):
    nc, correct, total = 0.0, 0.0, 0.0
    for c in classes:
        if n[c] == 0:
            continue
        if a[c] >= tau:
            correct += n[c] * a[c] / 100.0
            nc += n[c]
        else:
            correct += n[c] * pc_mllm.get(c, 0) / 100.0
        total += n[c]
    acc = 100.0 * correct / total if total else 0.0
    savings = 100.0 - 100.0 * (nc + (total - nc) * MLLM_COST_RATIO) / (total * MLLM_COST_RATIO) if total else 0.0
    return acc, savings


def main():
    bb = json.load(open(f"{ROOT}/results/02_robustness/multi_backbone/backbone_results.json"))
    sizes = load_sizes()
    mm = load_mllm_mean()
    mirror = {}

    for bbk in ["laion_l14", "openai_l14"]:
        label = "LAION-L/14 (paper primary)" if bbk == "laion_l14" else "OpenAI-L/14 (mirror primary)"
        d = {}
        for ds in DATASETS:
            for c, rec in bb[bbk]["results"][ds]["per_class_acc"].items():
                d[(ds, c)] = rec["acc"]
        classes_all = [k for k in d if k in mm and k[1] != 1]

        # correlation, pooled
        m_vals = [mm[k] for k in classes_all if k in mm]
        a_vals = [d[k] for k in classes_all if k in mm]
        keys = [k for k in classes_all if k in mm]
        rho, p = spearmanr([d[k] for k in keys], [mm[k] for k in keys])

        per_ds = {}
        for ds in DATASETS:
            ks = [k for k in keys if k[0] == ds]
            if len(ks) < 2:
                continue
            r2, p2 = spearmanr([d[k] for k in ks], [mm[k] for k in ks])
            per_ds[ds] = {"rho": round(float(r2), 4), "p": float(p2), "n": len(ks)}

        hy = {}
        for ds in DATASETS:
            classes = [k[1] for k in keys if k[0] == ds]
            a = {c: d[(ds, c)] for c in classes}
            n = {c: sizes[(ds, c)] for c in classes}
            pm = {c: mm[(ds, c)] for c in classes}
            grid = {tau: hybrid_acc(classes, n, a, pm, tau) for tau in TAU_GRID}
            tau_best = max(grid, key=lambda t: grid[t][0])
            acc50, sav50 = hybrid_acc(classes, n, a, pm, TAU_PRE)
            hy[ds] = {"tau_best": tau_best,
                      "acc_best": round(grid[tau_best][0], 2),
                      "sav_best": round(grid[tau_best][1], 1),
                      "acc_tau50": round(acc50, 2),
                      "sav_tau50": round(sav50, 1)}

        mirror[bbk] = {
            "label": label,
            "rho_pooled": float(rho),
            "p_pooled": float(p),
            "n_pooled": len(keys),
            "per_dataset": per_ds,
            "hybrid": hy,
        }

    out_path = os.path.join(OUT, "exp5_backbone_mirror.json")
    json.dump(mirror, open(out_path, "w"), indent=2)
    print(json.dumps(mirror, indent=1))


if __name__ == "__main__":
    main()