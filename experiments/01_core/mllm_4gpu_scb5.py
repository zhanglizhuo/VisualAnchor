#!/usr/bin/env python3
"""
4-GPU parallel MLLM inference on TeacherBehavior.
Launches one Ollama server per GPU, splits 3240 images into 4 shards,
saves checkpoint every 100 images per shard, resumes on restart.
"""
import os, sys, json, time, base64, subprocess, signal, atexit, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "experiments"))
from utils import CLASS_NAME_MAP, read_label

DATA_ROOT = Path("/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")
OUT_DIR = PROJ / "results" / "01_core" / "mllm_4gpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen3.6:35b-a3b"
BASE_PORT = 11435
CLASSES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]
OLLAMA_BIN = "/usr/local/bin/ollama"

# --- Collect samples ---
DS_DIR = DATA_ROOT / "SCB5_TeacherBehavior" / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
IMG_DIR = DS_DIR / "images" / "val"
LBL_DIR = DS_DIR / "labels" / "val"

classes_str = ", ".join(CLASS_NAME_MAP.get(c, c) for c in CLASSES)
PROMPT = (
    "Classify the classroom activity in this image. "
    "Choose exactly one category from the list: {classes}. "
    "Output only the category name, nothing else."
).format(classes=classes_str)


def collect_samples():
    samples = []
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = LBL_DIR / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        lbl = read_label(lbl_path)
        if lbl is not None and 0 <= lbl < len(CLASSES):
            samples.append((str(IMG_DIR / fname), fname, lbl))
    return samples


def split_shards(samples, n_shards=4):
    """Round-robin split of samples across shards."""
    shards = [[] for _ in range(n_shards)]
    for i, s in enumerate(samples):
        shards[i % n_shards].append(s)
    return shards


# --- Ollama server management ---
servers = []


def start_ollama(gpu_id, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
    env["OLLAMA_KEEP_ALIVE"] = "5m"
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    servers.append(proc)
    print(f"  [GPU{gpu_id}] Ollama serve on port {port} (PID {proc.pid})", flush=True)
    # Wait for server to be ready
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2)
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama on GPU{gpu_id} port {port} failed to start")
    # Pre-load the model
    r = requests.post(f"http://127.0.0.1:{port}/api/generate", json={
        "model": MODEL, "prompt": "warmup", "stream": False,
        "options": {"temperature": 0, "num_predict": 1},
    }, timeout=120)
    r.raise_for_status()
    print(f"  [GPU{gpu_id}] Model loaded ({r.elapsed.total_seconds():.1f}s)", flush=True)


def cleanup():
    for proc in servers:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            try:
                proc.kill()
            except:
                pass


atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: cleanup())
signal.signal(signal.SIGINT, lambda *_: cleanup())


# --- Shard inference ---
def classify_image(port, img_path, max_retries=2):
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/api/chat", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 32},
            }, timeout=120)
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip().lower()
            return content
        except Exception as e:
            print(f"    [GPU?] Error retry {attempt+1}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)
    return ""


def infer_shard(gpu_id, port, shard, shard_id):
    """Process one shard sequentially with checkpointing."""
    ckpt_path = CKPT_DIR / f"shard_{shard_id}.json"

    # Check existing checkpoint
    done_fnames = set()
    ckpt = {}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        done_fnames = set(ckpt.get("done", []))
        print(f"  [GPU{gpu_id}] Resume: {len(done_fnames)} already done", flush=True)

    remaining = [(img_path, fname, cid) for img_path, fname, cid in shard
                 if fname not in done_fnames]
    n_remaining = len(remaining)
    n_done = len(done_fnames)
    n_total = len(shard)
    t0 = time.time()

    for idx, (img_path, fname, true_cid) in enumerate(remaining):
        true_class = CLASSES[true_cid]
        content = classify_image(port, img_path)
        from utils import match_class_name
        is_correct = match_class_name(content, true_class, CLASSES)

        ckpt.setdefault("per_class", {})
        per = ckpt["per_class"].setdefault(true_class, {"correct": 0, "total": 0})
        per["total"] += 1
        if is_correct:
            per["correct"] += 1
        ckpt.setdefault("done", [])
        ckpt["done"].append(fname)

        done_so_far = n_done + idx + 1
        if done_so_far % 100 == 0 or done_so_far == n_total:
            elapsed = time.time() - t0
            rate = done_so_far / elapsed if elapsed > 0 else 0
            print(f"  [GPU{gpu_id}] {done_so_far}/{n_total} ({rate:.2f}/s, {elapsed:.0f}s)", flush=True)
            ckpt_path.write_text(json.dumps(ckpt, indent=2))

    elapsed = time.time() - t0
    print(f"  [GPU{gpu_id}] Done {n_total} in {elapsed:.0f}s ({n_total/elapsed:.2f}/s)", flush=True)
    return ckpt


# --- Main ---
def main():
    print("=== Collecting samples ===", flush=True)
    samples = collect_samples()
    print(f"  Total: {len(samples)} images", flush=True)
    shards = split_shards(samples)
    for i, sh in enumerate(shards):
        print(f"  Shard {i}: {len(sh)} images", flush=True)

    print("\n=== Starting Ollama servers ===", flush=True)
    # Clean up: stop system runner + kill any leftover broadsense servers
    subprocess.run([OLLAMA_BIN, "stop", MODEL], capture_output=True, timeout=10)
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, timeout=5)
    time.sleep(3)

    for gpu_id in range(4):
        port = BASE_PORT + gpu_id
        start_ollama(gpu_id, port)

    print("\n=== Running inference ===", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_to_gpu = {}
        for gpu_id in range(4):
            port = BASE_PORT + gpu_id
            fut = pool.submit(infer_shard, gpu_id, port, shards[gpu_id], gpu_id)
            fut_to_gpu[fut] = gpu_id

        for fut in as_completed(fut_to_gpu):
            gpu_id = fut_to_gpu[fut]
            try:
                ckpt = fut.result()
                results[gpu_id] = ckpt
            except Exception as e:
                print(f"  [GPU{gpu_id}] FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

    print("\n=== Merging results ===", flush=True)
    merged = {"correct": 0, "total": 0, "per_class": {}}
    for gpu_id in range(4):
        ckpt = results.get(gpu_id, {})
        for cls_name, counts in ckpt.get("per_class", {}).items():
            m = merged["per_class"].setdefault(cls_name, {"correct": 0, "total": 0})
            m["correct"] += counts["correct"]
            m["total"] += counts["total"]
            merged["total"] += counts["total"]
            merged["correct"] += counts["correct"]

    overall_acc = round(100.0 * merged["correct"] / merged["total"], 2) if merged["total"] > 0 else 0
    per_class = {}
    for c in CLASSES:
        m = merged["per_class"].get(c, {"correct": 0, "total": 0})
        acc = round(100.0 * m["correct"] / m["total"], 2) if m["total"] > 0 else 0
        per_class[c] = {"correct": m["correct"], "total": m["total"], "acc": acc}
        print(f"  {c:28s}  n={m['total']:5d}  {acc:6.2f}%")

    print(f"\n  Overall: {overall_acc}% ({merged['correct']}/{merged['total']})", flush=True)

    out_path = OUT_DIR / "teacher_behavior_results.json"
    with open(out_path, "w") as f:
        json.dump({"overall_acc": overall_acc, "per_class": per_class}, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    print("\n=== Cleaning up ===", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
