"""Run gemma4:31b alone on Stanford 40 with longer timeout."""
import json, os, time, random, requests, base64
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(os.environ.get("STANFORD40_DIR", PROJ / "data" / "Stanford40" / "JPEGImages"))
OUT_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJ / "results" / "02_robustness/stanford40")))
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

def load_samples():
    random.seed(42)
    class_to_files = {c: [] for c in CLASS_NAMES}
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".jpg"): continue
        parts = fname.rsplit("_", 1)
        cls_name = parts[0].replace("_", " ")
        if cls_name in class_to_files:
            class_to_files[cls_name].append(os.path.join(DATA_DIR, fname))
    samples = []
    for cls_name in CLASS_NAMES:
        files = class_to_files[cls_name]
        selected = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for path in selected:
            samples.append((path, cls_name))
    random.shuffle(samples)
    return samples

def main():
    samples = load_samples()
    classes_str = ", ".join(CLASS_NAMES)
    correct = {c: 0 for c in CLASS_NAMES}
    total = {c: 0 for c in CLASS_NAMES}

    print(f"gemma4:31b on {len(samples)} images (timeout=300s)...")
    t0 = time.time()
    for i, (path, cls) in enumerate(samples):
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": "gemma4:31b", "prompt": f"Classify this image. Choose exactly one from: {classes_str}. Answer with ONLY the category name.",
                "images": [img_b64], "stream": False, "options": {"temperature": 0}, "think": False
            }, timeout=300)
            pred = resp.json()["response"].strip().lower().rstrip(".,!?:;'\"")
            matched = next((c for c in CLASS_NAMES if pred == c.lower() or c.lower() in pred or pred in c.lower()), None)
            if matched == cls:
                correct[cls] += 1
        except Exception as e:
            print(f"  Error [{i+1}/{len(samples)}]: {e}")
        total[cls] += 1
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.0f}s, {(i+1)/(time.time()-t0):.1f} img/s")

    per_class = {c: {"n": total[c], "correct": correct[c], "acc": round(correct[c]/total[c]*100,2) if total[c] else 0} for c in CLASS_NAMES}
    overall = round(sum(correct.values())/sum(total.values())*100, 2)
    result = {"model": "gemma4:31b", "per_class_acc": per_class, "overall_acc": overall, "elapsed_s": round(time.time()-t0, 1)}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gemma4_31b_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    from scipy.stats import spearmanr
    a = json.load(open(os.path.join(OUT_DIR, "anchor_scores.json")))["per_class_acc"]
    c = [x for x in a if x in per_class]
    rho, p = spearmanr([a[x]["acc"] for x in c], [per_class[x]["acc"] for x in c])
    print(f"Done. Acc={overall:.1f}%, rho={rho:.3f}, p={p:.4f}")

if __name__ == "__main__":
    main()
