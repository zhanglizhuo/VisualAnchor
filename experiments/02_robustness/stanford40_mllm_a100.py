"""
Run LLaVA-1.5-7B on Stanford 40 Actions (50 images/class).

Usage:
    python experiments/02_robustness/stanford40_mllm_a100.py
"""

import json, os, time, random
from pathlib import Path
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ / "data" / "Stanford40" / "JPEGImages"
OUT_DIR = PROJ / "results" / "02_robustness/stanford40"
SAMPLES_PER_CLASS = 50
BATCH_SIZE = 1

CLASS_NAMES = [
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


def load_samples():
    """Load random stratified samples."""
    class_to_files = {c: [] for c in CLASS_NAMES}
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".jpg"):
            continue
        parts = fname.rsplit("_", 1)
        cls_name = parts[0].replace("_", " ")
        if cls_name in class_to_files:
            class_to_files[cls_name].append(str(DATA_DIR / fname))

    samples = []
    for cls_name in CLASS_NAMES:
        files = class_to_files.get(cls_name, [])
        random.seed(42)
        selected = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for path in selected:
            samples.append((path, cls_name))
    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples across {len(CLASS_NAMES)} classes")
    return samples


def run_mllm(model, processor, device, samples):
    """Run LLaVA inference on all samples."""
    correct = {c: 0 for c in CLASS_NAMES}
    total = {c: 0 for c in CLASS_NAMES}
    classes_str = ", ".join(CLASS_NAMES)

    t0 = time.time()
    for i, (img_path, cls_name) in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        prompt = f"USER: <image>\nClassify this image. Choose exactly one category from: {classes_str}. Answer with only the category name.\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        resp = processor.decode(outputs[0], skip_special_tokens=True).lower()

        # Extract prediction
        if "assistant:" in resp:
            resp = resp.split("assistant:")[-1].strip().rstrip(".")
        pred = resp.strip().rstrip(".")

        # Check against class names
        is_correct = False
        for cn in CLASS_NAMES:
            if pred == cn.lower() or pred.startswith(cn.lower()[:5]):
                if cn == cls_name:
                    is_correct = True
                break

        if is_correct:
            correct[cls_name] += 1
        total[cls_name] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s elapsed, {(i+1)/elapsed:.1f} img/s")

    # Compute per-class accuracy
    per_class_acc = {}
    for cls in CLASS_NAMES:
        n = total[cls]
        acc = round(correct[cls] / n * 100, 2) if n > 0 else 0.0
        per_class_acc[cls] = {"n": n, "correct": correct[cls], "acc": acc}

    overall_acc = round(sum(correct.values()) / sum(total.values()) * 100, 2)
    return {"overall_acc": overall_acc, "per_class_acc": per_class_acc}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, revision="master"
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_name, revision="master"
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Load data
    samples = load_samples()

    # Run inference
    print(f"Running inference on {len(samples)} images...")
    results = run_mllm(model, processor, device, samples)

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "llava_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"Overall accuracy: {results['overall_acc']:.2f}%")
    for cls in CLASS_NAMES:
        info = results["per_class_acc"][cls]
        print(f"  {cls:30s}: {info['acc']:6.2f}% ({info['correct']:3d}/{info['n']:3d})")


if __name__ == "__main__":
    main()
