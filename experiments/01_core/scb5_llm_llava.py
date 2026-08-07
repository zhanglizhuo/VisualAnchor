"""
Run CLIP AnchorScore and LLaVA-1.5-7B inference on expanded SCB5_LLM data.
Requires a GPU with ~20GB+ VRAM.
"""

import json, time, torch, numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ / "data" / "scb5_llm_expansion"
RESULTS_DIR = PROJ / "results" / "01_core" / "scb5_llm_expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=str(Path.home() / "models" / "llava-v1.5-7b-hf"),
                    help="Path to LLaVA model (default: ~/models/llava-v1.5-7b-hf)")
parser.add_argument("--device", type=str, default=None)
ARGS = parser.parse_args()

CLASS_MAP = {
    "stage_presentation": "stage presentation on stage",
    "listening_to_lecture": "listening to lecture",
    "answering_questions": "answering questions",
    "student_blackboard_writing": "student writing on blackboard",
    "patrolling": "patrolling in classroom",
    "reading_aloud": "reading aloud",
    "discussion": "discussing in groups",
    "lecturing": "lecturing",
    "stage_interaction": "on-stage interaction",
    "responding": "responding to questions",
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]


def get_classes():
    """Return class names from data dir."""
    return sorted([d.name for d in DATA_DIR.iterdir()
                   if d.is_dir() and not d.name.startswith("_")])


@torch.no_grad()
def run_clip_anchor(model, tokenizer, transform, device, classes, prompt_templates):
    """Compute per-class AnchorScore (CLIP zero-shot accuracy)."""
    model.eval()
    prompts = [p.format(CLASS_MAP[c]) for c in classes
               for p in prompt_templates]

    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    num_classes = len(classes)
    num_prompts = len(prompt_templates)
    text_feats = text_feats.view(num_classes, num_prompts, -1).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    correct = np.zeros(num_classes, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)

    for ci, c in enumerate(classes):
        img_dir = DATA_DIR / c
        imgs = sorted(img_dir.glob("*.[jJ][pP][gG]")) + sorted(img_dir.glob("*.[pP][nN][gG]"))
        for img_path in imgs:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feat) @ text_feats.T
            pred = logits.argmax(dim=1).item()
            correct[ci] += (pred == ci)
            counts[ci] += 1

    acc = {classes[i]: round(correct[i] / max(counts[i], 1) * 100, 2)
           for i in range(num_classes)}
    return acc


@torch.no_grad()
def run_biomedclip_anchor(model, tokenizer, preprocess, device, classes):
    model.eval()
    prompt = "this is a photo of {}"
    texts = tokenizer(
        [prompt.format(CLASS_MAP[c]) for c in classes],
        context_length=256
    ).to(device)

    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = np.zeros(len(classes), dtype=np.float32)
    counts = np.zeros(len(classes), dtype=np.float32)

    for ci, c in enumerate(classes):
        img_dir = DATA_DIR / c
        imgs = sorted(img_dir.glob("*.[jJ][pP][gG]")) + sorted(img_dir.glob("*.[pP][nN][gG]"))
        for img_path in imgs:
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_feat) @ text_features.T
            pred = logits.argmax(dim=1).item()
            correct[ci] += (pred == ci)
            counts[ci] += 1

    acc = {classes[i]: round(correct[i] / max(counts[i], 1) * 100, 2)
           for i in range(len(classes))}
    return acc


def run_llava_inference(model_path, data_dir, classes, output_path):
    """Run LLaVA-1.5-7B inference using the llava library."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    device = "cuda"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device
    )

    prompt = f"{DEFAULT_IMAGE_TOKEN}\nClassify the classroom activity. Choose exactly one category: {', '.join(CLASS_MAP[c] for c in classes)}. Output only the category name."

    results = {}
    for ci, c in enumerate(classes):
        img_dir = data_dir / c
        imgs = sorted(img_dir.glob("*.[jJ][pP][gG]")) + sorted(img_dir.glob("*.[pP][nN][gG]"))
        correct = 0
        total = 0
        for img_path in imgs:
            image = Image.open(img_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(device, dtype=torch.float16)

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=32,
                    use_cache=True,
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            label = CLASS_MAP[c]
            from utils import match_class_name
            if match_class_name(outputs, label, list(CLASS_MAP.values())):
                correct += 1
            total += 1

        results[c] = {"correct": correct, "total": total, "acc": round(correct / total * 100, 1)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    classes = get_classes()
    print(f"Classes: {classes}")

    # === Step 1: General CLIP AnchorScore ===
    print("\n=== Step 1: General CLIP AnchorScore ===")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    model, tokenizer, transform = load_clip_model()
    model = model.to(device)

    clip_acc = run_clip_anchor(model, tokenizer, transform, device, classes, PROMPT_TEMPLATES)
    print(f"CLIP: {clip_acc}")

    # === Step 2: BiomedCLIP AnchorScore ===
    print("\n=== Step 2: BiomedCLIP AnchorScore ===")
    from open_clip import create_model_from_pretrained, get_tokenizer
    biomed_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    biomed_tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    biomed_model = biomed_model.to(device)

    biomed_acc = run_biomedclip_anchor(biomed_model, biomed_tokenizer, preprocess, device, classes)
    print(f"BiomedCLIP: {biomed_acc}")

    # === Step 3: LLaVA-1.5-7B Inference ===
    print("\n=== Step 3: LLaVA-1.5-7B Inference ===")
    llava_model_path = ARGS.model_path
    llava_out = RESULTS_DIR / "llava_scb5_llm_results.json"

    if llava_out.exists():
        print(f"LLaVA results already exist at {llava_out}, loading...")
        with open(llava_out) as f:
            llava_results = json.load(f)
    else:
        llava_results = run_llava_inference(llava_model_path, DATA_DIR, classes, llava_out)

    # === Step 4: Compute Correlations ===
    print("\n=== Step 4: Correlations ===")
    # New data: build arrays
    new_classes_list = [c for c in classes if c not in ["stage_interaction", "responding"]]
    clip_vals = [clip_acc[c] for c in new_classes_list]
    biomed_vals = [biomed_acc[c] for c in new_classes_list]
    llava_vals = [llava_results[c]["acc"] for c in new_classes_list]

    print(f"New classes (n={len(new_classes_list)}):")
    for i, c in enumerate(new_classes_list):
        print(f"  {c}: CLIP={clip_vals[i]}%, Biomed={biomed_vals[i]}%, LLaVA={llava_vals[i]}%")

    if len(new_classes_list) >= 4:
        r_c, p_c = spearmanr(clip_vals, llava_vals)
        r_b, p_b = spearmanr(biomed_vals, llava_vals)
        print(f"\nGeneral CLIP vs LLaVA: ρ={r_c:.3f} p={p_c:.4f}")
        print(f"BiomedCLIP vs LLaVA: ρ={r_b:.3f} p={p_b:.4f}")

    # Save all results
    all_results = {
        "classes": classes,
        "general_clip_anchor": clip_acc,
        "biomedclip_anchor": biomed_acc,
        "llava_results": llava_results,
        "new_classes_analysis": {
            "clip_vs_llava": {"rho": round(r_c, 3), "p": round(p_c, 4)} if len(new_classes_list) >= 4 else None,
            "biomed_vs_llava": {"rho": round(r_b, 3), "p": round(p_b, 4)} if len(new_classes_list) >= 4 else None,
        }
    }
    out_path = RESULTS_DIR / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
