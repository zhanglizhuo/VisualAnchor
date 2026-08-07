"""
Run 4 ollama VL models on SCB5_LLM classes via chat API (think=false).
Models: qwen3.5:27b, qwen3.6:27b, qwen3.6:35b-a3b, gemma4:26b
"""
import os, json, time, base64, glob
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJ / "data" / "scb5_llm_expansion" / "val"
OUT = PROJ / "results" / "01_core" / "scb5_llm_expansion"
OUT.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"

CLASS_MAP = {
    "answering_questions": "answering_questions",
    "discussion": "discussion",
    "lecturing": "lecturing",
    "listening_to_lecture": "listening_to_lecture",
    "patrolling": "patrolling",
    "reading_aloud": "reading_aloud",
    "responding": "responding",
    "stage_interaction": "stage_interaction",
    "stage_presentation": "stage_presentation",
    "student_blackboard_writing": "student_blackboard_writing",
}

MODELS = [
    "qwen3.5:27b",
    "qwen3.6:27b",
    "qwen3.6:35b-a3b",
    "gemma4:26b",
]

PROMPT_TEMPLATE = (
    "Classify the classroom activity in this image. "
    "Choose exactly one category from the list: {classes}. "
    "Output only the category name, nothing else."
)


def get_images(class_dir):
    imgs = sorted(glob.glob(str(class_dir / "*.jpg")) + 
                  glob.glob(str(class_dir / "*.png")) + 
                  glob.glob(str(class_dir / "*.JPG")))
    return imgs


def classify_image(model, img_path, prompt, max_retries=2):
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt, 'images': [img_b64]}],
                'stream': False,
                'think': False,
                'options': {'temperature': 0, 'num_predict': 32}
            }, timeout=120)
            data = resp.json()
            content = data.get('message', {}).get('content', '').strip().lower()
            return content
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                return f"ERROR: {e}"


def run_model(model, classes_list, prompt_text):
    out_file = OUT / f"ollama_{model.replace(':', '_').replace('.', '_')}.json"
    if out_file.exists():
        print(f"  [{model}] cached, skipping")
        with open(out_file) as f:
            return json.load(f)

    results = {}
    for ci, class_dir in enumerate(classes_list):
        class_name = class_dir.name
        label = CLASS_MAP[class_name]
        imgs = get_images(class_dir)
        correct = 0
        total = 0

        print(f"  [{model}] {class_name}: {len(imgs)} images...", end=" ", flush=True)
        t0 = time.time()

        def process_img(img_path):
            content = classify_image(model, img_path, prompt_text)
            from utils import match_class_name
            all_classes = list(CLASS_MAP.values())
            return match_class_name(content, label, all_classes)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_img, img): img for img in imgs}
            for future in as_completed(futures):
                if future.result():
                    correct += 1
                total += 1

        acc = round(correct / total * 100, 1)
        elapsed = time.time() - t0
        results[class_name] = {"correct": correct, "total": total, "acc": acc}
        print(f"{acc}% ({elapsed:.0f}s)")

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    print("=== Ollama VL Inference on SCB5_LLM ===")
    classes_list = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name in CLASS_MAP])
    print(f"Found {len(classes_list)} classes")

    classes_str = ", ".join(CLASS_MAP.values())
    prompt_text = PROMPT_TEMPLATE.format(classes=classes_str)

    all_results = {}
    for model in MODELS:
        print(f"\n--- Model: {model} ---")
        t0 = time.time()
        results = run_model(model, classes_list, prompt_text)
        all_results[model] = results
        print(f"  Total time: {time.time() - t0:.0f}s")

    # Save combined
    combined_path = OUT / "ollama_all_models.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")

    # Compute ensemble mean per class
    print("\n=== Ensemble Mean (4 models) ===")
    for class_dir in classes_list:
        cn = class_dir.name
        accs = [all_results[m].get(cn, {}).get('acc', 0) for m in MODELS]
        mean_acc = sum(accs) / len(accs)
        print(f"  {cn}: mean={mean_acc:.1f}% ({', '.join(f'{a:.0f}' for a in accs)})")


if __name__ == "__main__":
    main()
