"""
AnchorScore on SCB5_LLM with ORIGINAL (uncorrected) prompts.
Uses raw class directory names (e.g., "answering_questions") as prompt text,
without the CLASS_NAME_MAP normalization used in anchor_scb5_llm.py.
This quantifies the impact of prompt quality on AnchorScore-MLLM correlation.

References:
    - anchor_scb5_llm.py (corrected/fixed prompt version)
    - ensemble_correlation.json (combines this with MLLM data)
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import json
import sys
import os

PROJ = Path(__file__).resolve().parent.parent.parent
DATA = PROJ / "data" / "scb5_llm_expansion" / "val"
OUT = PROJ / "results" / "01_core" / "scb5_llm_expansion"

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_clip_model

@torch.no_grad()
def compute_clip_anchor(model, tokenizer, transform, device, classes, class_names_raw):
    n_cls = len(classes)
    n_pt = len(PROMPT_TEMPLATES)

    prompts = [p.format(name) for c in classes for name in [c.name] for p in PROMPT_TEMPLATES]

    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(n_cls, n_pt, -1).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    correct = np.zeros(n_cls, dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.float32)

    for ci, c in enumerate(classes):
        imgs = sorted(c.glob("*.[jJ][pP][gG]")) + sorted(c.glob("*.[pP][nN][gG]"))
        for img_path in imgs:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feat) @ text_feats.T
            pred = logits.argmax(dim=1).item()
            correct[ci] += (pred == ci)
            counts[ci] += 1

    return {c.name: round(float(correct[i] / max(counts[i], 1) * 100), 1) for i, c in enumerate(classes)}

def main():
    print("=" * 70)
    print("CLIP AnchorScore on SCB5-LLM-202506 with ORIGINAL (uncorrected) prompts")
    print("=" * 70)

    classes_list = sorted([
        d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_")
    ])
    print(f"\nFound {len(classes_list)} classes")
    for c in classes_list:
        print(f"  {c.name}")

    model, tokenizer, transform = load_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"\nDevice: {device}")

    original_acc = compute_clip_anchor(model, tokenizer, transform, device, classes_list, [c.name for c in classes_list])

    print(f"\nOriginal (uncorrected) prompts CLIP AnchorScore:")
    for cn, acc in sorted(original_acc.items()):
        print(f"  {cn:35s} {acc}%")

    print(f"\nPrompt verification (first template):")
    for cn in sorted(original_acc.keys()):
        p = PROMPT_TEMPLATES[0].format(cn)
        print(f"  {cn:35s} -> {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "clip_anchor_original_prompts.json"
    with open(out_path, "w") as f:
        json.dump(original_acc, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
