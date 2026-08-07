"""
Compute CLIP AnchorScore on Stanford 40 Actions dataset.

Usage:
    python experiments/02_robustness/anchor_stanford40.py

Requires:
    - Stanford 40 Actions dataset downloaded to data/Stanford40/
    - CLIP ViT-L/14 LAION-2B (downloads on first run)
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ / "data" / "Stanford40"
OUT_DIR = PROJ / "results" / "02_robustness/stanford40"
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
    """Load Stanford 40 Actions dataset from local directory."""
    images_by_class = {cls: [] for cls in STANFORD40_CLASSES}

    # Stanford 40 has structure: JPEGImages/{class}_{id}.jpg
    img_dir = DATA_DIR / "JPEGImages"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Stanford 40 dataset not found at {img_dir}. "
            f"Download from http://vision.stanford.edu/Datasets/40actions.html"
            f" and extract to {DATA_DIR}"
        )

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue
        # Parse class from filename: "class_name_001.jpg"
        parts = fname.rsplit("_", 1)
        cls_name = parts[0].replace("_", " ")
        if cls_name in images_by_class:
            images_by_class[cls_name].append(str(img_dir / fname))

    # Filter out empty classes and sort
    images_by_class = {k: sorted(v) for k, v in images_by_class.items() if v}
    logger.info(f"Loaded Stanford 40: {sum(len(v) for v in images_by_class.values())} images across {len(images_by_class)} classes")
    for cls, imgs in sorted(images_by_class.items()):
        logger.info(f"  {cls}: {len(imgs)} images")
    return images_by_class


class Stanford40Dataset(Dataset):
    def __init__(self, images_by_class, transform):
        self.samples = []
        self.labels = []
        self.class_names = sorted(images_by_class.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls_name, img_paths in images_by_class.items():
            for path in img_paths:
                self.samples.append(path)
                self.labels.append(self.class_to_idx[cls_name])

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


@torch.no_grad()
def compute_anchorscore(model, tokenizer, device, dataloader, class_names):
    """Compute per-class zero-shot accuracy."""
    # Encode text prompts
    all_text_features = []
    for prompt in STANFORD40_PROMPTS:
        texts = [prompt.format(cls) for cls in class_names]
        text_tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        all_text_features.append(text_features)

    # Average across prompts
    text_features = torch.stack(all_text_features).mean(0)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Accumulate per-class predictions
    n_correct = {cls: 0 for cls in class_names}
    n_total = {cls: 0 for cls in class_names}

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if device.type == "cuda" and images.dtype == torch.float32:
            images = images.half()
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1)

        for pred, label, img in zip(preds, labels, images):
            cls_name = class_names[label.item()]
            n_total[cls_name] += 1
            if pred.item() == label.item():
                n_correct[cls_name] += 1

    per_class_acc = {}
    total_correct = 0
    total_images = 0
    for cls in class_names:
        acc = n_correct[cls] / n_total[cls] * 100 if n_total[cls] > 0 else 0.0
        per_class_acc[cls] = {"n": n_total[cls], "correct": n_correct[cls], "acc": round(acc, 2)}
        total_correct += n_correct[cls]
        total_images += n_total[cls]

    overall_acc = total_correct / total_images * 100 if total_images > 0 else 0.0
    return {"overall_acc": round(overall_acc, 2), "per_class_acc": per_class_acc, "total": total_images, "correct": total_correct}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading CLIP ViT-L/14 LAION-2B...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    model = model.to(device)

    # FP16
    if device.type == "cuda":
        model = model.half()

    # Load data
    images_by_class = get_stanford40_data()
    class_names = sorted(images_by_class.keys())
    dataset = Stanford40Dataset(images_by_class, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Compute AnchorScore
    logger.info("Computing AnchorScore...")
    results = compute_anchorscore(model, tokenizer, device, dataloader, class_names)
    logger.info(f"Overall accuracy: {results['overall_acc']:.2f}%")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "anchor_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # Print per-class
    for cls, info in sorted(results["per_class_acc"].items()):
        logger.info(f"  {cls:30s}: {info['acc']:6.2f}% ({info['correct']:3d}/{info['n']:3d})")


if __name__ == "__main__":
    main()
