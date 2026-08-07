"""
AnchorScore on SCB5_LLM with ORIGINAL (uncorrected) CLASS_MAP.
The original CLASS_MAP entries included "in classroom" context, which
duplicated with the template's "in classroom", creating prompts like:
  "a photo of a person patrolling in classroom in classroom"

Paper reference (main.tex line 443):
  "initial prompts ... contain grammar artifacts and duplicated context
   that degrade CLIP's zero-shot accuracy. A corrected set of prompts
   removes the duplicated context"
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_clip_model

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]

ORIGINAL_MAP = {
    "answering_questions": "answering questions in classroom",
    "discussion": "discussing in classroom",
    "lecturing": "lecturing in classroom",
    "listening_to_lecture": "listening to a lecture in classroom",
    "patrolling": "patrolling in classroom",
    "reading_aloud": "reading aloud in classroom",
    "responding": "responding in classroom",
    "stage_interaction": "on stage interaction in classroom",
    "stage_presentation": "giving a stage presentation in classroom",
    "student_blackboard_writing": "writing on the blackboard in classroom",
}


@torch.no_grad()
def compute_clip_anchor(model, tokenizer, transform, device, classes):
    n_c = len(classes)
    n_p = len(PROMPT_TEMPLATES)

    prompts = [p.format(ORIGINAL_MAP[c.name]) for c in classes for p in PROMPT_TEMPLATES]

    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(n_c, n_p, -1).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    correct = np.zeros(n_c, dtype=np.float32)
    counts = np.zeros(n_c, dtype=np.float32)

    for ci, c in enumerate(classes):
        imgs = sorted(c.glob("*.[jJ][pP][gG]")) + sorted(c.glob("*.[pP][nN][gG]"))
        for img_path in imgs:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feat) @ text_feats.T
            pred = logits.argmax(dim=1).item()
            correct[ci] += pred == ci
            counts[ci] += 1

    return {c.name: round(float(correct[i] / max(counts[i], 1) * 100), 1) for i, c in enumerate(classes)}


def main():
    print("=== CLIP AnchorScore with ORIGINAL (uncorrected) CLASS_MAP ===")
    print("(prompts contain duplicated 'in classroom' context)")
    print()

    classes_list = sorted([
        d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_")
    ])
    print(f"Found {len(classes_list)} classes")
    for c in classes_list:
        print(f"  {c.name}")

    model, tokenizer, transform = load_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"\nDevice: {device}")

    acc = compute_clip_anchor(model, tokenizer, transform, device, classes_list)

    print("\nUncorrected (duplicated context) prompts CLIP AnchorScore:")
    for cn, v in sorted(acc.items()):
        p = PROMPT_TEMPLATES[0].format(ORIGINAL_MAP[cn])
        print(f"  {cn:35s} {v}%  (e.g., {p})")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "clip_anchor_uncorrected_prompts.json"
    with open(out_path, "w") as f:
        json.dump(acc, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
