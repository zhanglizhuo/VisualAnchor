"""
Run core MLLMs (via Ollama) on Stanford 40 Actions (50 images/class).

Usage:
    python experiments/02_robustness/stanford40_ollama.py

Output:
    results/02_robustness/stanford40/ollama_results.json
"""

import json, os, time, random, requests
from pathlib import Path
from PIL import Image
import base64, io

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.environ.get("STANFORD40_DIR", str(PROJ / "data" / "Stanford40" / "JPEGImages"))
OUT_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJ / "results" / "02_robustness/stanford40")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_CLASS = 50
OLLAMA_HOST = "http://localhost:11434"

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

MODELS = ["qwen3.5:27b", "qwen3.5:35b-a3b", "qwen3.6:27b", "qwen3.6:35b-a3b", "gemma4:26b", "gemma4:31b"]


def load_samples():
    class_to_files = {c: [] for c in CLASS_NAMES}
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".jpg"):
            continue
        parts = fname.rsplit("_", 1)
        cls_name = parts[0].replace("_", " ")
        if cls_name in class_to_files:
            class_to_files[cls_name].append(os.path.join(DATA_DIR, fname))

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


def normalize(s):
    """Normalize prediction: lowercase, strip punctuation, remove leading/trailing whitespace."""
    return s.strip().lower().rstrip(".,!?:;'\"")


def ollama_classify(model_name, img_path, classes_str):
    """Classify image via Ollama API."""
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    prompt = f"Classify this image. Choose exactly one category from: {classes_str}. Answer with ONLY the category name, no explanation."
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0},
        "think": False,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    resp = normalize(r.json()["response"])
    return resp


def match_class(pred):
    """Match prediction to a class name. Exact match first, fallback to substring."""
    pred_lower = pred.lower()
    if not pred_lower:
        return None
    for cn in CLASS_NAMES:
        if pred_lower == cn.lower():
            return cn
    for cn in CLASS_NAMES:
        if cn.lower() in pred_lower or pred_lower in cn.lower():
            return cn
    return None


def run_model(model_name, samples):
    classes_str = ", ".join(CLASS_NAMES)
    correct = {c: 0 for c in CLASS_NAMES}
    total = {c: 0 for c in CLASS_NAMES}

    print(f"\n{'='*50}")
    print(f"Running {model_name} on {len(samples)} images...")
    t0 = time.time()
    for i, (img_path, cls_name) in enumerate(samples):
        try:
            pred = ollama_classify(model_name, img_path, classes_str)
            matched = match_class(pred)
            if matched == cls_name:
                correct[cls_name] += 1
        except Exception as e:
            print(f"  Error: {img_path}: {e}")
        total[cls_name] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s, {(i+1)/elapsed:.1f} img/s")

    elapsed = time.time() - t0
    per_class = {c: {"n": total[c], "correct": correct[c],
                      "acc": round(correct[c]/total[c]*100, 2) if total[c] > 0 else 0.0}
                 for c in CLASS_NAMES}
    overall = round(sum(correct.values()) / sum(total.values()) * 100, 2)
    print(f"  Done in {elapsed:.0f}s. Overall: {overall:.2f}%")
    return {"model": model_name, "overall_acc": overall, "per_class_acc": per_class,
            "elapsed_s": round(elapsed, 1)}


def main():
    samples = load_samples()
    results = {"description": "Stanford40 Ollama MLLM evaluation (50 img/class)",
               "samples_per_class": SAMPLES_PER_CLASS, "n_classes": len(CLASS_NAMES),
               "models": {}}

    # Resume: load existing results to skip completed models
    existing_path = OUT_DIR / "ollama_results.json"
    if existing_path.exists():
        existing = json.load(open(existing_path))
        results["models"] = existing.get("models", {})
        completed = set(results["models"].keys())
        print(f"Found existing results for: {completed}")
    else:
        completed = set()

    for model_name in MODELS:
        if model_name in completed:
            print(f"Skipping {model_name} (already completed)")
            continue
        result = run_model(model_name, samples)
        results["models"][model_name] = result

        # Save after each model (in case of crash)
        with open(OUT_DIR / "ollama_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Compute correlation with AnchorScore
    from scipy.stats import spearmanr
    anchor = json.load(open(OUT_DIR / "anchor_scores.json"))
    anchor_accs = anchor["per_class_acc"]

    for mn in MODELS:
        mllm_accs = results["models"][mn]["per_class_acc"]
        common = [c for c in CLASS_NAMES if c in anchor_accs and c in mllm_accs]
        a_vals = [anchor_accs[c]["acc"] for c in common]
        m_vals = [mllm_accs[c]["acc"] for c in common]
        rho, p = spearmanr(a_vals, m_vals)
        results["models"][mn]["correlation"] = {"spearman_rho": round(rho, 4),
                                                  "p": round(p, 6), "n": len(common)}
        print(f"\n{mn}: ρ={rho:.3f}, p={p:.4f}, n={len(common)}")

    # Multi-model mean (like main experiment)
    mean_mllm = {}
    for c in CLASS_NAMES:
        accs = [results["models"][mn]["per_class_acc"][c]["acc"] for mn in MODELS
                if results["models"][mn]["per_class_acc"][c]["n"] > 0]
        if accs:
            mean_mllm[c] = sum(accs) / len(accs)
    common = [c for c in CLASS_NAMES if c in anchor_accs and c in mean_mllm]
    a_vals = [anchor_accs[c]["acc"] for c in common]
    m_vals = [mean_mllm[c] for c in common]
    rho, p = spearmanr(a_vals, m_vals)
    results["multi_model_mean"] = {"models": MODELS, "spearman_rho": round(rho, 4),
                                    "p": round(p, 6), "n": len(common)}
    print(f"\nMulti-model mean ({len(MODELS)} models): ρ={rho:.3f}, p={p:.4f}, n={len(common)}")

    with open(OUT_DIR / "ollama_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'ollama_results.json'}")


if __name__ == "__main__":
    main()
