"""
anchor_scb5_bbox.py

AnchorScore on bbox-cropped regions (matching MLLM evaluation protocol).

Reads YOLO labels for bbox coordinates, crops the image to the first
annotated person/region, and classifies with CLIP.  This makes the CLIP
input comparable to the llm-annotation MLLM evaluation which also uses
bbox crops instead of full images.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, load_clip_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent

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
        "subdir": "SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17",
        "classes": ["BowHead", "TurnHead"],
    },
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]


def read_label_with_bbox(path, img_width, img_height):
    """Read first bbox's class_id and pixel coords from YOLO label."""
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None, None
    parts = line.split()
    cid = int(parts[0])
    x_c, y_c, w, h = map(float, parts[1:5])
    x1 = int((x_c - w / 2) * img_width)
    y1 = int((y_c - h / 2) * img_height)
    x2 = int((x_c + w / 2) * img_width)
    y2 = int((y_c + h / 2) * img_height)
    return cid, (x1, y1, x2, y2)


def build_prompts(class_names, templates):
    prompts = []
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t in templates:
            prompts.append(t.format(friendly))
    return prompts


@torch.no_grad()
def compute_bbox(model, tokenizer, transform, device, batch_size, ds_cfg, model_dtype):
    results = {}
    for ds_name, cfg in ds_cfg.items():
        logger.info(f"\n{'='*60}\n{ds_name}")

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
            with Image.open(img_path) as tmp:
                w, h = tmp.size
            cid, bbox = read_label_with_bbox(lbl_path, w, h)
            if cid is None or cid >= num_classes:
                continue
            samples.append((img_path, cid, bbox))

        logger.info(f"  {len(samples)} valid val images (bbox-cropped)")

        cls_correct = {i: 0 for i in range(num_classes)}
        cls_total = {i: 0 for i in range(num_classes)}

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            images = []
            for img_path, _, bbox in batch:
                img = Image.open(img_path).convert("RGB").crop(bbox)
                images.append(transform(img))
            images = torch.stack(images).to(device).to(model_dtype)
            labels = torch.tensor([c for _, c, _ in batch], device=device)

            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            logits = (100.0 * img_feats) @ text_feats.T
            preds = logits.argmax(dim=1)

            for cid in range(num_classes):
                mask = labels == cid
                cls_correct[cid] += (preds[mask] == cid).sum().item()
                cls_total[cid] += mask.sum().item()

        per_class = {}
        tot_correct = 0
        tot_count = 0
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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("SCB5_DATA_ROOT") or str(PROJ / "data" / "scb5")
    out_dir = Path(args.out or str(PROJ / "results" / "01_core" / "anchor_score_scb5_bbox"))
    out_dir.mkdir(parents=True, exist_ok=True)

    global SERVER_DATA
    SERVER_DATA = Path(data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {data_root}")

    model, tokenizer, transform = load_clip_model()
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype
    if device.type == "cuda":
        model = model.half()
        model_dtype = next(model.parameters()).dtype

    cfg = DATASET_CFG
    if args.dataset:
        cfg = {k: v for k, v in cfg.items() if k == args.dataset}

    results = compute_bbox(model, tokenizer, transform, device, args.batch_size, cfg, model_dtype)

    out_path = out_dir / "anchor_scores_bbox.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
