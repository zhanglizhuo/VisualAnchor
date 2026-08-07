"""
Re-run CLIP AnchorScore on SCB5_LLM with FIXED prompts.
CLIP model downloads from Hugging Face; set HF_ENDPOINT if using a mirror.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent.parent
DATA = PROJ / "data" / "scb5_llm_expansion" / "val"
OUT = PROJ / "results" / "01_core" / "scb5_llm_expansion"

# FIXED CLASS_MAP (no grammar issues, no template doubling)
CLASS_MAP = {
    "answering_questions": "answering questions",
    "discussion": "discussing",
    "lecturing": "lecturing",
    "listening_to_lecture": "listening to a lecture",
    "patrolling": "patrolling",
    "reading_aloud": "reading aloud",
    "responding": "responding",
    "stage_interaction": "on stage interaction",
    "stage_presentation": "giving a stage presentation",
    "student_blackboard_writing": "writing on the blackboard",
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]

@torch.no_grad()
def compute_clip_anchor(model, tokenizer, transform, device, classes):
    prompts = [p.format(CLASS_MAP[c.name]) for c in classes for p in PROMPT_TEMPLATES]
    n_cls = len(classes)
    n_pt = len(PROMPT_TEMPLATES)

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
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    import json

    print("=== CLIP AnchorScore with FIXED prompts ===")
    classes_list = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_")])
    print(f"Found {len(classes_list)} classes")

    model, tokenizer, transform = load_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Compute with fixed prompts
    fixed_acc = compute_clip_anchor(model, tokenizer, transform, device, classes_list)
    print(f"\nFixed prompts CLIP AnchorScore:")
    for cn, acc in fixed_acc.items():
        print(f"  {cn:35s} {acc}%")

    # Print fixed prompts for verification
    print(f"\nPrompt verification:")
    for cn in CLASS_MAP:
        p = PROMPT_TEMPLATES[0].format(CLASS_MAP[cn])
        print(f"  {cn:35s} -> {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "clip_anchor_fixed_prompts.json"
    with open(out_path, "w") as f:
        json.dump(fixed_acc, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
