"""
prompt_robustness_ablation.py

Tests whether the AnchorScore-MLLM correlation is robust to prompt
template choice, or highly sensitive (as exposed by the SCB-LLM
prompt error where rho jumped from 0.147 to 0.506).

Runs 5 prompt template variations on SCB5 and computes class-level
Spearman rho for each.

Usage:
  python prompt_robustness_ablation.py [--batch_size 64]
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
from utils import CLASS_NAME_MAP, read_label

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

PROMPT_VARIANTS = {
    "original_3template": [
        "a photo of a person {} in classroom.",
        "a classroom scene showing {}.",
        "the action of {} in a school environment.",
    ],
    "minimal_photo": [
        "a photo of {}.",
    ],
    "person_no_context": [
        "a photo of a person {}.",
    ],
    "single_classroom": [
        "a photo of a person {} in classroom.",
    ],
    "classroom_only": [
        "a photo of {} in a classroom.",
    ],
    "alternative_phrasing": [
        "an image of someone {} in a school.",
        "a classroom photo depicting {}.",
        "{} happening in an educational setting.",
    ],
}


def load_mllm_accuracy():
    with open(PROJ / "results" / "01_core" / "paper_data" / "mllm_raw.json") as f:
        raw = json.load(f)
    mllm = {}
    for ds_name, models in raw.items():
        if ds_name.startswith("_"):
            continue
        for model, classes in models.items():
            if model.startswith("_"):
                continue
            for cls, acc in classes.items():
                mllm.setdefault(ds_name, {}).setdefault(cls, []).append(acc)
    return {ds: {cls: float(np.mean(accs)) for cls, accs in v.items()} for ds, v in mllm.items()}


def build_prompts(class_names, templates):
    prompts = []
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t in templates:
            prompts.append(t.format(friendly))
    return prompts


@torch.no_grad()
def compute_anchor(model, tokenizer, transform, device, model_dtype, templates, batch_size):
    results = {}
    for ds_name, cfg in DATASET_CFG.items():
        base = SERVER_DATA / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" if sub else base / "images") / "val"
        lbl_dir = (base / sub / "labels" if sub else base / "labels") / "val"

        if not img_dir.exists():
            continue

        classes = cfg["classes"]
        num_classes = len(classes)
        prompt_texts = build_prompts(classes, templates)

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
            samples.append((img_path, cid))

        cls_correct = {i: 0 for i in range(num_classes)}
        cls_total = {i: 0 for i in range(num_classes)}

        for i in range(0, len(samples), batch_size):
            batch = samples[i: i + batch_size]
            images = torch.stack([
                transform(Image.open(p).convert("RGB")) for p, _ in batch
            ]).to(device).to(model_dtype)
            labels = torch.tensor([c for _, c in batch], device=device)

            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feats) @ text_feats.T
            preds = logits.argmax(dim=1)

            for cid in range(num_classes):
                mask = labels == cid
                cls_correct[cid] += (preds[mask] == cid).sum().item()
                cls_total[cid] += mask.sum().item()

        per_class = {}
        for cid, cls_name in enumerate(classes):
            n = cls_total[cid]
            acc = cls_correct[cid] / n * 100 if n else 0.0
            per_class[cls_name] = round(acc, 2)
        results[ds_name] = per_class

    return results


def compute_class_level_rho(anchor_results, mean_mllm):
    all_classes = []
    for ds_name in DATASET_CFG:
        for cls in DATASET_CFG[ds_name]["classes"]:
            all_classes.append((ds_name, cls))

    anchor_vals, mllm_vals = [], []
    for ds_name, cls in all_classes:
        if ds_name in anchor_results and cls in anchor_results[ds_name]:
            if ds_name in mean_mllm and cls in mean_mllm[ds_name]:
                anchor_vals.append(anchor_results[ds_name][cls])
                mllm_vals.append(mean_mllm[ds_name][cls])

    sp = spearmanr(anchor_vals, mllm_vals)
    return {
        "rho": round(sp.statistic, 4),
        "p": round(sp.pvalue, 4),
        "n": len(anchor_vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    out_dir = Path(args.out or str(PROJ / "results" / "04_ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    model, tokenizer, transform = load_clip_model()
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    model_dtype = next(model.parameters()).dtype
    logger.info("Loaded ViT-L-14 laion2B-s32B-b82K (CLIP official normalization)")

    mean_mllm = load_mllm_accuracy()

    all_results = {}
    for variant_name, templates in PROMPT_VARIANTS.items():
        logger.info(f"\n{'='*50}\nPrompt variant: {variant_name} (T={len(templates)})")
        for t in templates:
            logger.info(f"  '{t}'")

        anchor = compute_anchor(model, tokenizer, transform, device, model_dtype, templates, args.batch_size)
        corr = compute_class_level_rho(anchor, mean_mllm)

        logger.info(f"  => rho={corr['rho']}, p={corr['p']}, n={corr['n']}")
        all_results[variant_name] = {
            "templates": templates,
            "n_templates": len(templates),
            "class_level_rho": corr["rho"],
            "class_level_p": corr["p"],
            "n_classes": corr["n"],
            "per_class_anchor": anchor,
        }

    rhos = [r["class_level_rho"] for r in all_results.values()]
    print(f"\n{'='*60}")
    print(f"  PROMPT ROBUSTNESS ABLATION (class-level rho, n=13)")
    print(f"{'='*60}")
    print(f"  {'Variant':25s}  {'T':>3s}  {'rho':>8s}  {'p':>8s}")
    for vn, r in all_results.items():
        print(f"  {vn:25s}  {r['n_templates']:>3d}  {r['class_level_rho']:>8.4f}  {r['class_level_p']:>8.4f}")
    print(f"\n  Mean rho: {np.mean(rhos):.4f}")
    print(f"  Std rho:  {np.std(rhos):.4f}")
    print(f"  Range:    [{min(rhos):.4f}, {max(rhos):.4f}]")
    print(f"  Swing:    {max(rhos) - min(rhos):.4f}")

    out_path = out_dir / "prompt_robustness.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
