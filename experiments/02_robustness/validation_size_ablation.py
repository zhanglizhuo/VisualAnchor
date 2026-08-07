"""
validation_size_ablation.py

Ablation: How many labeled validation images per class are needed
for AnchorScore to reliably predict MLLM accuracy?

Runs CLIP ViT-L/14 zero-shot on SCB5 val once (caching per-image
predictions), then bootstraps at N = {5, 10, 20, 50, 100, all}
images per class. For each N, computes the class-level Spearman rho
between AnchorScore and mean MLLM accuracy over B bootstrap iterations.

Usage:
  python validation_size_ablation.py [--bootstrap 200] [--batch_size 64]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label, load_clip_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
SERVER_DATA = Path(os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))

DATASET_CFG = {
    "TeacherBehavior": {
        "dir": "SCB5_TeacherBehavior",
        "subdir": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
        ],
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": "SCB5-Handrise-Read-write-2024-9-17",
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
    },
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]

N_VALUES = [5, 10, 20, 50, 100, "all"]


def load_mllm_accuracy():
    """Load per-class MLLM accuracy from mllm_raw.json (6 ollama models)."""
    with open(PROJ / "results" / "01_core" / "paper_data" / "mllm_raw.json") as f:
        raw = json.load(f)
    mllm = {}
    for ds_name, models in raw.items():
        if ds_name.startswith("_"):
            continue
        mllm[ds_name] = {}
        for model, classes in models.items():
            if model.startswith("_"):
                continue
            for cls, acc in classes.items():
                mllm[ds_name].setdefault(cls, []).append(acc)
    mean_mllm = {}
    for ds_name, cls_accs in mllm.items():
        mean_mllm[ds_name] = {cls: float(np.mean(accs)) for cls, accs in cls_accs.items()}
    return mean_mllm


@torch.no_grad()
def run_clip_inference(model, tokenizer, transform, device, model_dtype):
    """Run CLIP zero-shot on SCB5 val. Return per-image predictions."""
    per_image = {}

    for ds_name, cfg in DATASET_CFG.items():
        base = SERVER_DATA / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" if sub else base / "images") / "val"
        lbl_dir = (base / sub / "labels" if sub else base / "labels") / "val"

        if not img_dir.exists():
            logger.warning(f"  {ds_name}: val dir not found: {img_dir}")
            continue

        classes = cfg["classes"]
        num_classes = len(classes)
        prompt_texts = []
        for cls in classes:
            friendly = CLASS_NAME_MAP.get(cls, cls)
            for t in PROMPT_TEMPLATES:
                prompt_texts.append(t.format(friendly))

        text_tokens = tokenizer(prompt_texts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.view(num_classes, -1, text_feats.shape[-1]).mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            cid = read_label(lbl_path)
            if cid is None or cid >= num_classes:
                continue
            samples.append((str(img_path), cid))

        logger.info(f"  {ds_name}: {len(samples)} val images")

        ds_preds = {ci: {"correct": [], "total": 0} for ci in range(num_classes)}
        bs = 64
        for i in range(0, len(samples), bs):
            batch = samples[i: i + bs]
            images = torch.stack([
                transform(Image.open(p).convert("RGB")) for p, _ in batch
            ]).to(device).to(model_dtype)
            labels = torch.tensor([c for _, c in batch], device=device)

            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feats) @ text_feats.T
            preds = logits.argmax(dim=1)

            for j, (_, cid) in enumerate(batch):
                ds_preds[cid]["correct"].append(int(preds[j].item() == cid))
                ds_preds[cid]["total"] += 1

        per_image[ds_name] = {
            classes[ci]: {
                "correct_flags": ds_preds[ci]["correct"],
                "n_total": ds_preds[ci]["total"],
            }
            for ci in range(num_classes)
        }

    return per_image


def bootstrap_ablation(per_image, mean_mllm, n_values, n_bootstrap, seed=42):
    """For each N, bootstrap B times and compute class-level rho."""
    rng = np.random.RandomState(seed)
    results = {}

    all_classes = []
    for ds_name in DATASET_CFG:
        for cls in DATASET_CFG[ds_name]["classes"]:
            all_classes.append((ds_name, cls))

    for n_val in n_values:
        rhos = []
        for b in range(n_bootstrap):
            anchor_scores = []
            mllm_accs = []
            for ds_name, cls in all_classes:
                if ds_name not in per_image or cls not in per_image[ds_name]:
                    continue
                flags = per_image[ds_name][cls]["correct_flags"]
                n_avail = len(flags)
                if n_val == "all":
                    sampled = flags
                else:
                    k = min(n_val, n_avail)
                    idx = rng.choice(n_avail, size=k, replace=False)
                    sampled = [flags[i] for i in idx]
                acc = float(np.mean(sampled)) * 100
                anchor_scores.append(acc)
                mllm_accs.append(mean_mllm[ds_name][cls])

            if len(anchor_scores) >= 3:
                sp = spearmanr(anchor_scores, mllm_accs)
                rhos.append(sp.statistic)

        rhos = np.array(rhos)
        results[str(n_val)] = {
            "n_bootstrap": len(rhos),
            "mean_rho": round(float(np.mean(rhos)), 4),
            "std_rho": round(float(np.std(rhos)), 4),
            "median_rho": round(float(np.median(rhos)), 4),
            "ci_lower": round(float(np.percentile(rhos, 2.5)), 4),
            "ci_upper": round(float(np.percentile(rhos, 97.5)), 4),
            "min_rho": round(float(np.min(rhos)), 4),
            "max_rho": round(float(np.max(rhos)), 4),
        }
        r = results[str(n_val)]
        logger.info(
            f"  N={str(n_val):>4s}: rho={r['mean_rho']:.3f} ± {r['std_rho']:.3f} "
            f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    out_dir = Path(args.out or str(PROJ / "results" / "04_ablation" / "ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, transform = load_clip_model()
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    model_dtype = next(model.parameters()).dtype
    logger.info("Loaded ViT-L-14 laion2B-s32B-b82K")

    logger.info("Running CLIP zero-shot inference on SCB5 val...")
    per_image = run_clip_inference(model, tokenizer, transform, device, model_dtype)

    logger.info("Loading MLLM accuracy...")
    mean_mllm = load_mllm_accuracy()

    logger.info(f"\nBootstrapping (B={args.bootstrap}) at N={N_VALUES}...")
    results = bootstrap_ablation(per_image, mean_mllm, N_VALUES, args.bootstrap)

    out_path = out_dir / "validation_size_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  VALIDATION SET SIZE ABLATION")
    print(f"{'='*60}")
    print(f"  {'N/class':>8s}  {'Mean rho':>10s}  {'Std':>8s}  {'95% CI':>20s}")
    for n_val in N_VALUES:
        r = results[str(n_val)]
        print(f"  {str(n_val):>8s}  {r['mean_rho']:>10.4f}  {r['std_rho']:>8.4f}  [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")


if __name__ == "__main__":
    main()
