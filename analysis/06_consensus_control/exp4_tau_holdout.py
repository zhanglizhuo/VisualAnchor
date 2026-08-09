import json, os
import numpy as np

ROOT = "/home/lizhuo/Work/Mayan/VisualAnchor"
OUT = os.path.join(ROOT, "results", "06_consensus_control")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

MLLM_COST_RATIO = 100.0
TAU_PRE = 50
TAU_GRID = list(range(1, 100))


def load_anchor():
    with open(f"{ROOT}/results/01_core/anchor_score_scb5/anchor_scores.json") as f:
        d = json.load(f)
    out = {}
    for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        pc = d[ds]["per_class_acc"]
        out[ds] = {"acc": {k: v["acc"] for k, v in pc.items()},
                   "n": {k: v["n"] for k, v in pc.items()}}
    return out


def load_mllm():
    with open(f"{ROOT}/results/01_core/paper_data/mllm_raw.json") as f:
        d = json.load(f)
    return {ds: {m: pc for m, pc in dds.items() if isinstance(pc, dict)}
            for ds, dds in d.items() if isinstance(dds, dict)}


def hybrid_acc_fixed(classes, n, a, pc_mllm, tau):
    nc_total, correct, total = 0.0, 0.0, 0.0
    for c in classes:
        if n[c] == 0:
            continue
        if a[c] >= tau:
            correct += n[c] * a[c] / 100.0
            nc_total += n[c]
        else:
            correct += n[c] * pc_mllm.get(c, 0) / 100.0
        total += n[c]
    acc = 100.0 * correct / total if total else 0.0
    savings = 100.0 - 100.0 * (nc_total + (total - nc_total) * MLLM_COST_RATIO) / (total * MLLM_COST_RATIO) \
        if total else 0.0
    return acc, savings


def clip_only_acc(classes, n, a):
    total = sum(n[c] for c in classes)
    return 100.0 * sum(n[c] * a[c] / 100.0 for c in classes) / total if total else 0.0


def oracle_acc(classes, n, a, pc_mllm):
    total = sum(n[c] for c in classes)
    return 100.0 * sum(n[c] * max(a[c], pc_mllm.get(c, 0)) / 100.0 for c in classes) / total if total else 0.0


def random_route_acc(classes, n, a, pc_mllm, tau, iters=1000):
    n_budget_clip = sum(n[c] for c in classes if a[c] >= tau)
    n_total = sum(n[c] for c in classes)
    accs = []
    for _ in range(iters):
        pool = []
        for c in classes:
            pool += [c] * n[c]
        pool = np.array(pool)
        RNG.shuffle(pool)
        clip_imgs = set(pool[:n_budget_clip].tolist()) if n_budget_clip > 0 else set()
        correct = 0.0
        total = 0.0
        for c in classes:
            if n[c] == 0:
                continue
            n_clip = sum(1 for x in clip_imgs if x == c)
            correct += n_clip * a[c] / 100.0 + (n[c] - n_clip) * pc_mllm.get(c, 0) / 100.0
            total += n[c]
        accs.append(100.0 * correct / total)
    return float(np.mean(accs)), float(np.std(accs))


def best_tau(classes, n, a, pc_mllm):
    best, best_acc = None, -1.0
    for tau in TAU_GRID:
        acc, _ = hybrid_acc_fixed(classes, n, a, pc_mllm, tau)
        if acc > best_acc:
            best, best_acc = tau, acc
    return best, best_acc


def main():
    anchor = load_anchor()
    mllm = load_mllm()
    result = {}
    for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        classes = list(anchor[ds]["n"].keys())
        n, a = anchor[ds]["n"], {c: anchor[ds]["acc"][c] for c in classes}
        ms = [m for m in mllm[ds]]
        mean_pc = {c: float(np.mean([mllm[ds][m].get(c, 0) for m in ms])) for c in classes}

        tau_best, acc_best = best_tau(classes, n, a, mean_pc)
        acc_pre, sav_pre = hybrid_acc_fixed(classes, n, a, mean_pc, TAU_PRE)
        acc_clip = clip_only_acc(classes, n, a)
        acc_or = oracle_acc(classes, n, a, mean_pc)
        rand_mean, rand_sd = random_route_acc(classes, n, a, mean_pc, TAU_PRE)
        grid = {}
        for tau in TAU_GRID:
            acc, sav = hybrid_acc_fixed(classes, n, a, mean_pc, tau)
            grid[str(tau)] = {"hybrid_acc": round(acc, 2), "cost_savings": round(sav, 1)}

        loo = []
        for leave in classes:
            others = [c for c in classes if c != leave]
            tau_loo, _ = best_tau(others, n, a, mean_pc)
            acc_h, _ = hybrid_acc_fixed([leave], n, a, mean_pc, tau_loo)
            loo.append({"leave": leave, "tau_star": tau_loo,
                        "leave_hybrid_acc": round(acc_h, 2), "leave_anchor_acc": round(a[leave], 2)})

        result[ds] = {
            "n_classes": len(classes),
            "tau_best_raw": tau_best,
            "acc_best": round(acc_best, 2),
            "tau_pre_registered": TAU_PRE,
            "acc_pre_registered": round(acc_pre, 2),
            "savings_pre_registered": round(sav_pre, 1),
            "acc_clip_only": round(acc_clip, 2),
            "acc_oracle_per_class": round(acc_or, 2),
            "random_route_mean": round(rand_mean, 2),
            "random_route_sd": round(rand_sd, 2),
            "leave_one_class_out": loo,
            "grid": grid,
        }
        result[ds]["random_route_sd"] = round(rand_sd, 2)
    with open(f"{OUT}/exp4_tau_holdout.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()