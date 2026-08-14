"""
eurosat_qwen35_v100.py — M5(b) cross-domain 27B validation (EuroSAT).

Runs a canonical 26-35B MLLM (Qwen3.5-27B, Ollama) on EuroSAT under the same
protocol as the cross-domain 7B evaluations (cross_domain_llava.py /
cross_domain_qwen.py): 50 images/class, full-frame, category-name reply.

Serves 4 per-GPU Ollama instances (ports 11437-11440), round-robin shards,
JSONL checkpointing with resume.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import signal
import subprocess
import atexit
import time
import threading
from pathlib import Path

import requests
from PIL import Image

HOME = Path.home()
EUROSAT_ROOT = HOME / "works" / "lizhuo" / "datasets" / "EuroSAT_RGB" / "2750"
OUT_DIR = Path(__file__).resolve().parent / "eurosat_qwen35_out"
OLLAMA_BIN = "/usr/local/bin/ollama"
MODEL = "qwen3.5:27b"
PORTS = [11437, 11438, 11439, 11440]
N_GPU = 4
SAMPLES_PER_CLASS = 50

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def build_prompt():
    labels = ", ".join(EUROSAT_CLASSES)
    return (
        f"Classify this image. Choose exactly one category: {labels}. "
        f"Output only the category name, nothing else."
    )


def load_eurosat():
    images, labels = [], []
    for ci, cls in enumerate(EUROSAT_CLASSES):
        cls_dir = EUROSAT_ROOT / cls
        if not cls_dir.is_dir():
            raise RuntimeError(f"missing EuroSAT class dir: {cls_dir}")
        for img_path in sorted(cls_dir.glob("*.jpg")):
            images.append(img_path)
            labels.append(ci)
    return images, labels


def sample_per_class(image_paths, labels, n, seed=42):
    random.seed(seed)
    idx_per = {i: [] for i in range(len(EUROSAT_CLASSES))}
    for idx, l in enumerate(labels):
        idx_per[l].append(idx)
    sampled = {}
    for ci, indices in idx_per.items():
        k = min(n, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        sampled[EUROSAT_CLASSES[ci]] = [(image_paths[i], ci) for i in chosen]
    return sampled


def infer_pred_from_text(raw):
    text = raw.lower().strip()
    if not text:
        return None
    for idx, cls_name in enumerate(EUROSAT_CLASSES):
        tokens = [re.escape(tok) for tok in re.split(r"[-\s]+", cls_name.lower()) if tok]
        pattern = r"\b" + r"[-\s]+".join(tokens) + r"\b"
        if re.search(pattern, text):
            return idx
    return None


servers = []


def start_ollama(gpu_id, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
    env["OLLAMA_KEEP_ALIVE"] = "10m"
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    servers.append(proc)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama GPU{gpu_id} port {port} failed")
    r = requests.post(f"http://127.0.0.1:{port}/api/generate",
                      json={"model": MODEL, "prompt": "warmup", "stream": False,
                            "options": {"temperature": 0, "num_predict": 1}}, timeout=600)
    r.raise_for_status()
    print(f"[GPU{gpu_id}] port {port} ready ({r.elapsed.total_seconds():.0f}s load)", flush=True)


def cleanup():
    for p in servers:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: cleanup())
signal.signal(signal.SIGINT, lambda *_: cleanup())


def classify(port, prompt, img_path, max_retries=2):
    buf = io.BytesIO()
    Image.open(img_path).convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/api/chat", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 16},
            }, timeout=180)
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"  retry {attempt+1}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)
    return ""


def run_shard(shard_id, items, prompt):
    out_path = OUT_DIR / f"shard_{shard_id}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["f"])
                except Exception:
                    pass
    port = PORTS[shard_id % N_GPU]
    todo = [it for it in items if it["f"] not in done]
    print(f"[shard {shard_id}] {len(todo)}/{len(items)} to run (port {port})", flush=True)
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, it in enumerate(todo):
            try:
                raw = classify(port, prompt, it["path"])
                pred = infer_pred_from_text(raw)
            except Exception as e:
                pred = None
                raw = f"ERROR: {e}"
            rec = {"f": it["f"], "true": it["true"], "pred": pred,
                   "correct": (pred == it["true"]) if pred is not None else None, "raw": raw[:60]}
            f.write(json.dumps(rec) + "\n")
            if (k + 1) % 25 == 0:
                f.flush()
                el = time.time() - t0
                print(f"[shard {shard_id}] {k+1}/{len(todo)} ({el/60:.1f}m, {(k+1)/el:.2f} img/s)", flush=True)
    print(f"[shard {shard_id}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS)
    args = ap.parse_args()

    images, labels = load_eurosat()
    print(f"EuroSAT: {len(images)} images from {EUROSAT_ROOT}", flush=True)
    sampled = sample_per_class(images, labels, args.samples)

    items = []
    for cls_name, pairs in sampled.items():
        for path, ci in pairs:
            items.append({"f": f"{cls_name}/{path.name}", "true": ci, "path": str(path)})
    items.sort(key=lambda x: x["f"])
    print(f"sampled {len(items)} items ({args.samples}/class x {len(EUROSAT_CLASSES)} classes)", flush=True)

    prompt = build_prompt()
    print("PROMPT:", prompt, flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    for gpu_id, port in enumerate(PORTS):
        start_ollama(gpu_id, port)

    shards = [[] for _ in range(N_GPU)]
    for i, it in enumerate(items):
        shards[i % N_GPU].append(it)
    threads = []
    for sid, sh in enumerate(shards):
        t = threading.Thread(target=run_shard, args=(sid, sh, prompt))
        t.start()
        threads.append(t)
        time.sleep(2)
    for t in threads:
        t.join()
    print("ALL SHARDS DONE", flush=True)


if __name__ == "__main__":
    main()
