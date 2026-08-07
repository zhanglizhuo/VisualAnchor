#!/usr/bin/env python3
"""
Simple sequential MLLM inference for prompt optimization.
Runs one image at a time (no parallelism) for reliability.
Supports checkpoint resume: saved per-class, re-run skips done classes.
"""
import os, json, sys, time, random, base64, argparse
from pathlib import Path
import requests

PROJ = Path(__file__).resolve().parent.parent.parent

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = {"qwen35": "qwen3.5:27b", "qwen36": "qwen3.6:27b"}
PROMPT_SUFFIX = {"qwen35": "", "qwen36": ""}

DATASET_CFG = {
    "TeacherBehavior": {
        "dir": "SCB5_TeacherBehavior",
        "subdir": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                     "teacher", "stand", "screen", "blackBoard"],
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": None,
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
    },
}

OUT_DIR = PROJ / "results" / "04_ablation" / "prompt_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def resolve_data_root():
    env = os.environ.get("SCB5_DATA_ROOT")
    if env:
        return Path(env)
    server_path = Path("/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")
    if server_path.exists():
        return server_path
    return PROJ / "data" / "scb5"

def load_images(ds_name, cls_name, n=20, seed=42):
    cfg = DATASET_CFG[ds_name]
    data_root = resolve_data_root()
    base = data_root / cfg["dir"]
    sub = cfg["subdir"]
    if sub and (base / sub).exists():
        img_dir = base / sub / "images" / "val"
        lbl_dir = base / sub / "labels" / "val"
    else:
        img_dir = base / "images" / "val"
        lbl_dir = base / "labels" / "val"

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import read_label
    paths = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = lbl_dir / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        lbl = read_label(lbl_path)
        if lbl is not None:
            if lbl == cfg["classes"].index(cls_name):
                paths.append(str(img_dir / fname))

    random.seed(seed)
    return random.sample(paths, min(n, len(paths)))

def query(model, prompt, img_path):
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    full_prompt = prompt + PROMPT_SUFFIX.get(model, "")

    payload = {
        "model": OLLAMA_MODELS.get(model, model),
        "messages": [{"role": "user", "content": full_prompt, "images": [b64]}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 32},
        "keep_alive": "5m",
    }
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp = r.json()["message"]["content"].strip().lower()
            if resp:
                return resp
        except Exception:
            pass
        if attempt < 2:
            time.sleep(5)
    return None

def load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {}

def save_checkpoint(ckpt_path, data):
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(ckpt_path)

def run_class(model, ds_name, cls_name, pdata, images, ckpt_key):
    ckpt_path = OUT_DIR / f"ckpt_{model}_{ds_name}_{cls_name}.json"
    existing = load_checkpoint(ckpt_path)

    results_by_type = {}
    for ptype in ["standard", "enhanced"]:
        key = f"{ptype}_results"
        if key in existing:
            cls_results = existing[key]
            print(f"    {ptype}: resume ({len(cls_results)} already done)")
        else:
            cls_results = []
        prompt_text = pdata[f"{ptype}_prompt"]

        start = len(cls_results)
        for i in range(start, len(images)):
            img_path = images[i]
            t0 = time.time()
            resp = query(model, prompt_text, img_path)
            elapsed = time.time() - t0
            if resp is None:
                print(f"    [{i+1}/{len(images)}] {ptype}: TIMEOUT")
                continue

            from utils import match_class_name
            is_correct = match_class_name(resp, cls_name, DATASET_CFG[ds_name]["classes"])
            cls_results.append({
                "response": resp,
                "is_correct": is_correct,
            })
            sys.stdout.write(f"    [{i+1}/{len(images)}] {ptype}: {resp[:30]} -> {'OK' if is_correct else 'WR'} ({elapsed:.0f}s)\n")
            sys.stdout.flush()

            # checkpoint every 5 images
            if (i - start + 1) % 5 == 0:
                ckpt = {"standard_results": results_by_type.get("standard", []),
                        "enhanced_results": results_by_type.get("enhanced", [])}
                ckpt[ptype + "_results"] = cls_results
                save_checkpoint(ckpt_path, ckpt)

        results_by_type[ptype] = cls_results
        n_valid = len(cls_results)
        n_correct = sum(r["is_correct"] for r in cls_results)
        acc = 100.0 * n_correct / max(1, n_valid)
        print(f"    {ptype}: {n_valid}/{len(images)} valid, acc={acc:.1f}%")

    # save final checkpoint for this class
    ckpt = {"standard_results": results_by_type.get("standard", []),
            "enhanced_results": results_by_type.get("enhanced", [])}
    save_checkpoint(ckpt_path, ckpt)

    std = results_by_type.get("standard", [])
    enh = results_by_type.get("enhanced", [])
    return {
        "anchor_score": pdata["anchor_score"],
        "most_confused": pdata["most_confused"],
        "standard_acc": round(100.0 * sum(r["is_correct"] for r in std) / max(1, len(std)), 2),
        "enhanced_acc": round(100.0 * sum(r["is_correct"] for r in enh) / max(1, len(enh)), 2),
        "n_standard": len(std),
        "n_enhanced": len(enh),
        "standard_results": std,
        "enhanced_results": enh,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen35", choices=["qwen35", "qwen36"],
                        help="Model tag (qwen35 or qwen36)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Images per class (default 20)")
    parser.add_argument("--anchor-threshold", type=float, default=50,
                        help="Skip classes with AnchorScore above this (default 50)")
    args = parser.parse_args()
    model = args.model
    samples = args.samples
    anchor_threshold = args.anchor_threshold

    prompt_path = PROJ / "results" / "01_core" / "clip_per_image" / "enhanced_prompts.json"
    with open(prompt_path) as f:
        prompts = json.load(f)

    # load global checkpoint (which classes done)
    ckpt_global = OUT_DIR / f"ckpt_{model}_done.json"
    done_classes = set(load_checkpoint(ckpt_global).get("done_classes", []))

    results = {}
    for ds_name, ds_prompts in prompts.items():
        print(f"\n=== {ds_name} ===")
        ds_results = {}
        for cls_name, pdata in ds_prompts.items():
            # unique key for resume
            ckpt_key = f"{ds_name}/{cls_name}"

            if pdata.get("anchor_score", 100) > anchor_threshold:
                print(f"  {cls_name}: skip (acc={pdata['anchor_score']}%)")
                ds_results[cls_name] = {"skipped": True}
                continue

            if ckpt_key in done_classes:
                # load from saved checkpoint
                ckpt_path = OUT_DIR / f"ckpt_{model}_{ds_name}_{cls_name}.json"
                ckpt_data = load_checkpoint(ckpt_path)
                std = ckpt_data.get("standard_results", [])
                enh = ckpt_data.get("enhanced_results", [])
                ds_results[cls_name] = {
                    "anchor_score": pdata["anchor_score"],
                    "most_confused": pdata["most_confused"],
                    "standard_acc": round(100.0 * sum(r["is_correct"] for r in std) / max(1, len(std)), 2),
                    "enhanced_acc": round(100.0 * sum(r["is_correct"] for r in enh) / max(1, len(enh)), 2),
                    "n_standard": len(std),
                    "n_enhanced": len(enh),
                }
                print(f"  {cls_name} (acc={pdata['anchor_score']}%): resume from checkpoint ({len(std)} std, {len(enh)} enh)")
                continue

            images = load_images(ds_name, cls_name, samples)
            print(f"  {cls_name} (acc={pdata['anchor_score']}%): {len(images)} images")

            cls_result = run_class(model, ds_name, cls_name, pdata, images, ckpt_key)
            ds_results[cls_name] = cls_result

            # mark class as done
            done_classes.add(ckpt_key)
            save_checkpoint(ckpt_global, {"done_classes": sorted(done_classes)})

            # save cumulative results after each class
            results[ds_name] = ds_results
            out_path = OUT_DIR / f"ollama_{model}_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  -> cumulative saved to {out_path}")

        if ds_name not in results:
            results[ds_name] = ds_results
        else:
            results[ds_name].update(ds_results)

    out_path = OUT_DIR / f"ollama_{model}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal saved to {out_path}")

if __name__ == "__main__":
    main()
