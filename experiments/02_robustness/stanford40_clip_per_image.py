"""
Per-image CLIP predictions on Stanford40 for deployable routing.

Same protocol as anchor_stanford40.py (ViT-L-14 laion2b_s32b_b82k, FP16,
3-template averaged text features), but records per-image predictions
(true class, predicted class) instead of per-class aggregates.

Usage:
    python experiments/02_robustness/stanford40_clip_per_image.py
    # override data dir: STANFORD40_DIR=/path/to/JPEGImages python ...

Output: results/02_robustness/stanford40/per_image_predictions.json
"""

import json
import logging
import os
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("STANFORD40_DIR", PROJ / "data" / "Stanford40"))
OUT_DIR = PROJ / "results" / "02_robustness" / "stanford40"
BATCH_SIZE = 64

STANFORD40_CLASSES = [
    "applauding", "blowing bubbles", "brushing teeth", "cleaning the floor",
    "climbing", "cooking", "cutting trees", "cutting vegetables",
    "drinking", "feeding a horse", "fishing", "fixing a bike",
    "fixing a car", "gardening", "holding an umbrella", "jumping",
    "looking through a microscope", "looking through a telescope", "phoning",
    "playing guitar", "playing violin", "pouring liquid", "pushing a cart",
    "reading", "riding a bike", "riding a horse", "rowing a boat",
    "running", "shooting an arrow", "smoking", "taking photos",
    "texting message", "throwing frisby", "using a computer",
    "walking the dog", "washing dishes", "watching TV", "waving hands",
    "writing on a board", "writing on a book",
]

STANFORD40_PROMPTS = [
    "a photo of a person {}.",
    "a person {} in a photo.",
    "someone {}.",
]


def get_stanford40_data():
    images_by_class = {cls: [] for cls in STANFORD40_CLASSES}
    img_dir = DATA_DIR / "JPEGImages"
    if not img_dir.exists():
        raise FileNotFoundError(f"Stanford 40 dataset not found at {img_dir}")
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".jpg"):
            continue
        parts = fname.rsplit("_", 1)
        cls_name = parts[0].replace("_", " ")
        if cls_name in images_by_class:
            images_by_class[cls_name].append(str(img_dir / fname))
    images_by_class = {k: sorted(v) for k, v in images_by_class.items() if v}
    logger.info(f"Loaded {sum(len(v) for v in images_by_class.values())} images across {len(images_by_class)} classes")
    return images_by_class


class Stanford40Dataset(Dataset):
    def __init__(self, images_by_class, transform):
        self.samples = []
        self.class_names = sorted(images_by_class.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        for cls_name, img_paths in images_by_class.items():
            for path in img_paths:
                self.samples.append((path, self.class_to_idx[cls_name]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, Path(path).name


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    if device.type == "cuda":
        model = model.half()

    images_by_class = get_stanford40_data()
    class_names = sorted(images_by_class.keys())
    dataset = Stanford40Dataset(images_by_class, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    text_feats = []
    for prompt in STANFORD40_PROMPTS:
        texts = [prompt.format(cls) for cls in class_names]
        f = model.encode_text(tokenizer(texts).to(device))
        f = f / f.norm(dim=-1, keepdim=True)
        text_feats.append(f)
    text_feats = torch.stack(text_feats).mean(0)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    predictions = []
    n_correct = 0
    for images, labels, fnames in dataloader:
        images = images.to(device)
        if device.type == "cuda" and images.dtype == torch.float32:
            images = images.half()
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats @ text_feats.T
        preds = logits.argmax(dim=-1)
        for pred, label, fname in zip(preds, labels, fnames):
            predictions.append({
                "fname": fname,
                "true_class_name": class_names[label.item()],
                "pred_class_name": class_names[pred.item()],
            })
            n_correct += int(pred.item() == label.item())

    overall = n_correct / len(predictions) * 100
    out = {
        "description": "Per-image CLIP predictions on Stanford40 (ViT-L-14 laion2b_s32b_b82k, FP16, 3-template averaged), same protocol as anchor_stanford40.py.",
        "overall_acc": round(overall, 2),
        "n_images": len(predictions),
        "prompts": STANFORD40_PROMPTS,
        "predictions": predictions,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "per_image_predictions.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    logger.info(f"Overall acc: {overall:.2f}% over {len(predictions)} images")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
