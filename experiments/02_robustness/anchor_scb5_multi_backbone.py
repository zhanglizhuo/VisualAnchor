"""
anchor_scb5_multi_backbone.py

Run AnchorScore on SCB5 across multiple CLIP backbones to verify
the correlation is not model-specific.

Available backbones:
  - laion_l14    ViT-L-14 LAION-2B (default, 428M params)
  - openai_l14   ViT-L-14 OpenAI  (428M params)
  - openai_b32   ViT-B-32 OpenAI  (151M params)

Usage:
  python anchor_scb5_multi_backbone.py --backbones laion_l14,openai_l14,openai_b32
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label, load_clip_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent

# Model identifier → open_clip name + pretrained string mapping
MODEL_NAMES = {
    "laion_l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "openai_l14": ("ViT-L-14", "openai"),
    "openai_b32": ("ViT-B-32", "openai"),
}

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


def load_backbone(tag):
    model_name, pretrained = MODEL_NAMES[tag]
    model, tokenizer, transform = load_clip_model(model_name, pretrained)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {tag}: {model_name} ({params/1e6:.0f}M params)")
    return model, tokenizer, transform


def build_prompts(class_names, templates):
    prompts = []
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t in templates:
            prompts.append(t.format(friendly))
    return prompts


@torch.no_grad()
def compute(model, tokenizer, transform, device, batch_size, ds_cfg):
    results = {}
    for ds_name, cfg in ds_cfg.items():
        logger.info(f"\n{'='*50}\n{ds_name}")

        base = SERVER_DATA / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" if sub else base / "images")
        lbl_dir = (base / sub / "labels" if sub else base / "labels")
        val_img_dir = img_dir / "val"
        val_lbl_dir = lbl_dir / "val"

        if not val_img_dir.exists():
            logger.warning(f"  val not found: {val_img_dir}")
            continue

        classes = cfg["classes"]
        num_classes = len(classes)
        prompt_texts = build_prompts(classes, PROMPT_TEMPLATES)

        text_tokens = tokenizer(prompt_texts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.view(num_classes, -1, text_feats.shape[-1]).mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        samples = []
        for img_path in sorted(val_img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = val_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            cid = read_label(lbl_path)
            if cid is None or cid >= num_classes:
                continue
            samples.append((img_path, cid))

        logger.info(f"  {len(samples)} valid val images")

        cls_correct = {i: 0 for i in range(num_classes)}
        cls_total = {i: 0 for i in range(num_classes)}
        model_dtype = next(model.parameters()).dtype

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
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
        tot_correct, tot_count = 0, 0
        for cid, cls_name in enumerate(classes):
            n = cls_total[cid]
            acc = cls_correct[cid] / n * 100 if n else 0.0
            per_class[cls_name] = {"n": int(n), "acc": round(acc, 2)}
            tot_correct += cls_correct[cid]
            tot_count += n
            logger.info(f"    {cls_name:28s}  n={n:5d}  {acc:6.2f}%")

        overall = round(tot_correct / tot_count * 100, 2)
        logger.info(f"  Overall: {overall}% ({tot_correct}/{tot_count})")
        results[ds_name] = {
            "overall_acc": overall, "per_class_acc": per_class,
            "total": int(tot_count), "correct": int(tot_correct),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", type=str, default="laion_l14",
                        help="Comma-separated: laion_l14,openai_l14,openai_b32")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data-root", type=str, default=None,
                        help="SCB5 data root (default: $SCB5_DATA_ROOT or data/scb5)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: results/02_robustness/multi_backbone)")
    args = parser.parse_args()

    # Override SERVER_DATA in module scope
    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    out_dir = Path(args.out or str(PROJ / "results" / "02_robustness" / "multi_backbone"))
    out_dir.mkdir(parents=True, exist_ok=True)

    backbones = [b.strip() for b in args.backbones.split(",")
                 if b.strip() in MODEL_NAMES]
    if not backbones:
        logger.error(f"No valid backbones. Options: {list(MODEL_NAMES.keys())}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for tag in backbones:
        _, ptag = MODEL_NAMES[tag]
        logger.info(f"\n{'#'*60}\n# Backbone: {tag} ({ptag})\n{'#'*60}")
        t0 = time.time()

        model, tokenizer, transform = load_backbone(tag)
        model = model.to(device).eval()
        if device.type == "cuda":
            model = model.half()

        results = compute(model, tokenizer, transform, device,
                          args.batch_size, DATASET_CFG)
        all_results[tag] = {
            "model_name": MODEL_NAMES[tag][0],
            "pretrained": MODEL_NAMES[tag][1],
            "results": results,
        }
        logger.info(f"  [{time.time()-t0:.0f}s total]")

        # Free GPU memory before next backbone
        del model, tokenizer, transform
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"  Freed GPU memory")

    out_path = out_dir / "backbone_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
