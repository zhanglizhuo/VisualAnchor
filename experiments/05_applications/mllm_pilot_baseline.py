"""
mllm_pilot_baseline.py

MLLM-pilot baseline (reviewer demand: why CLIP instead of one cheap MLLM pass?).

Compares class-difficulty rankers at a matched label budget (~30 images/class)
against the canonical 6-MLLM mean per-class accuracy (ground truth):
  - AnchorScore (frozen CLIP, 30 imgs/class, 20 seeds): rho=0.796 +- 0.044
  - 7B MLLM pilot (LLaVA-1.5-7B, 30 imgs/class, full-frame): rho=-0.040 (no signal)
  - 26-35B MLLM pilots (30 imgs/class, full-frame): rho=0.55-0.78 (circular:
    the pilot model is part of the 6-model ground-truth mean)

Pilot per-class accuracies from the self-uncertainty cache (30 imgs/class,
full-frame, Ollama/LLaVA). AnchorScore@30 computed from cached per-image CLIP
predictions with seeded subsampling.
"""
import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
from datetime import date

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
OUT = RESULTS / "05_applications" / "mllm_pilot_baseline.json"

mllm = json.load(open(RESULTS / "01_core" / "paper_data" / "mllm_raw.json"))
pred = json.load(open(RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"))

models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
gt = {}
for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    for cls in mllm[ds][models[0]]:
        gt[cls] = float(np.mean([mllm[ds][m][cls] for m in models]))

by_class = {}
for ds in pred.values():
    for it in ds["predictions"]:
        by_class.setdefault(it["true_class_name"], []).append(it["correct"])

# AnchorScore at 30 imgs/class, 20 seeds
rhos = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    a30, g30 = [], []
    for cls, corr in by_class.items():
        idx = rng.permutation(len(corr))[:30]
        a30.append(100 * np.mean([corr[i] for i in idx]))
        g30.append(gt[cls])
    r, _ = spearmanr(a30, g30)
    rhos.append(r)

pilots = {}
pilot_files = {
    "LLaVA-1.5-7B": "llava15_7b.json",
    "Qwen3.5-27B": "ollama_qwen3_5_27b.json",
    "Gemma4-26B": "ollama_gemma4_26b.json",
    "Qwen3.6-27B": "ollama_qwen3_6_27b.json",
    "Qwen3.6-35B-A3B": "ollama_qwen3_6_35b-a3b.json",
}
for name, f in pilot_files.items():
    d = json.load(open(RESULTS / "04_ablation" / "self_uncertainty" / f))
    pilot = {}
    for ds in d.values():
        for cls, v in ds.items():
            pilot[cls] = v["acc"]
    P = [pilot[c] for c in gt]
    G = [gt[c] for c in gt]
    r, p = spearmanr(P, G)
    base = name.replace("-A3B", "").replace("-A4B", "")
    pilots[name] = {"rho": round(r, 3), "p": round(p, 4),
                    "circular": base in models,
                    "n_images_per_class": 30, "protocol": "full-frame"}

out = {
    "description": "MLLM-pilot baseline: class-difficulty rankers at matched ~30 imgs/class label budget vs canonical 6-MLLM mean per-class accuracy. AnchorScore@30 from cached per-image CLIP predictions (20 seeds). Pilots from self-uncertainty cache (full-frame, 30 imgs/class). circular=True means the pilot model is part of the 6-model ground-truth mean.",
    "computation_date": str(date.today()),
    "anchorscore_30_per_class": {"rho_mean": round(float(np.mean(rhos)), 3), "rho_sd": round(float(np.std(rhos)), 3), "seeds": 20, "protocol": "full-frame CLIP"},
    "mllm_pilots": pilots,
    "cost_note": "AnchorScore: ~390 CLIP images (minutes). 7B pilot: 390 images x ~2s. 27B pilot: 390 x ~9s (~1h). Full evaluation: 5416 x ~9s (~14h).",
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"Saved {OUT}")
print(f"AnchorScore@30: {out['anchorscore_30_per_class']}")
for k, v in pilots.items():
    print(f"  {k}: rho={v['rho']} circular={v['circular']}")
