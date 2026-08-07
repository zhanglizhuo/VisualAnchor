"""
anchor_prompt_ablation.py

Ablation study comparing AnchorScore with vs without CLASS_NAME_MAP prompt mapping.
Specifically tests whether the blackBoard class prompt mapping ("standing near a
blackboard or whiteboard") inflates/deflates AnchorScore relative to the literal
class name ("blackBoard").

Usage:
  python anchor_prompt_ablation.py [--batch_size 64]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))

TEACHER_BEHAVIOR_DIR = (
    DATA_ROOT
    / "SCB5_TeacherBehavior"
    / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
)

CLASSES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]


def load_models():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    model, tokenizer, transform = load_clip_model()
    logger.info("Loaded ViT-L-14 laion2B-s32B-b82K")
    return model, tokenizer, transform


def build_prompts(class_names, templates, use_mapping=True):
    prompts = []
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls) if use_mapping else cls
        for t in templates:
            prompts.append(t.format(friendly))
    return prompts


@torch.no_grad()
def compute(model, tokenizer, transform, device, batch_size, use_mapping, model_dtype):
    val_img_dir = TEACHER_BEHAVIOR_DIR / "images" / "val"
    val_lbl_dir = TEACHER_BEHAVIOR_DIR / "labels" / "val"
    num_classes = len(CLASSES)

    prompts = build_prompts(CLASSES, PROMPT_TEMPLATES, use_mapping=use_mapping)

    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(num_classes, len(PROMPT_TEMPLATES), -1).mean(dim=1)
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
    if len(samples) == 0:
        return {c: {"correct": 0, "total": 0, "acc": 0.0} for c in CLASSES}

    cls_correct = {i: 0 for i in range(num_classes)}
    cls_total = {i: 0 for i in range(num_classes)}

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

    results = {}
    for cid, cls_name in enumerate(CLASSES):
        n = cls_total[cid]
        acc = cls_correct[cid] / n * 100 if n else 0.0
        results[cls_name] = {
            "correct": int(cls_correct[cid]),
            "total": int(n),
            "acc": round(acc, 2),
        }
        logger.info(f"    {cls_name:28s}  n={n:5d}  {acc:6.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, transform = load_models()
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype
    if device.type == "cuda":
        model = model.half()
        model_dtype = next(model.parameters()).dtype

    logger.info(f"Model dtype: {model_dtype}")

    output = {
        "description": (
            "Prompt ablation: CLASS_NAME_MAP (mapped prompts) vs literal class names "
            "for blackBoard class. TeacherBehavior dataset, ViT-L-14 LAION-2B."
        ),
        "model": "ViT-L-14 laion2B-s32B-b82K",
        "prompt_templates": PROMPT_TEMPLATES,
        "results": {},
    }

    for use_mapping, label in [(True, "mapped_prompts"), (False, "literal_prompts")]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running with use_mapping={use_mapping} ({label})")
        logger.info(f"{'='*60}")
        results = compute(
            model, tokenizer, transform, device,
            args.batch_size, use_mapping=use_mapping, model_dtype=model_dtype,
        )
        output["results"][label] = results

    bb_mapped = output["results"]["mapped_prompts"]["blackBoard"]["acc"]
    bb_literal = output["results"]["literal_prompts"]["blackBoard"]["acc"]
    output["blackBoard_comparison"] = {
        "with_mapping_prompt": bb_mapped,
        "without_mapping_prompt": bb_literal,
        "delta": round(bb_mapped - bb_literal, 2),
    }
    logger.info(f"\nblackBoard comparison:")
    logger.info(f"  With mapping  (prompt='standing near a blackboard or whiteboard'): {bb_mapped}%")
    logger.info(f"  Without mapping (prompt='blackBoard'): {bb_literal}%")
    logger.info(f"  Delta: {output['blackBoard_comparison']['delta']}pp")

    out_dir = PROJ / "results" / "04_ablation" / "prompt_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blackboard_prompt_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
