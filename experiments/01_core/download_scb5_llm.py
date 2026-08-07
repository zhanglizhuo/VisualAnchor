"""
Download SCB-LLM-202506 data from Hugging Face.

Used in the paper: 10 classes × 44-50 images = 494 images.
"""

import json, zipfile, shutil, random
from pathlib import Path
import requests

SRC = Path(__file__).resolve().parent
PROJ = SRC.parent.parent
DATA_DIR = PROJ / "data" / "scb5_llm_expansion"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_BASE = "https://huggingface.co"

# All 10 classes used in the paper
CLASSES = {
    "answering_questions": "回答问题.zip",
    "discussion": "讨论.zip",
    "lecturing": "讲授.zip",
    "listening_to_lecture": "听讲.zip",
    "patrolling": "巡视.zip",
    "reading_aloud": "朗读.zip",
    "responding": "应答.zip",
    "stage_interaction": "台上互动.zip",
    "stage_presentation": "台上展示.zip",
    "student_blackboard_writing": "学生板书.zip",
}


def download_zip(zip_name, dest):
    url = f"{HF_BASE}/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_LLM_202506/{zip_name}"
    path = dest / zip_name
    if path.exists():
        print(f"  Already cached: {zip_name}")
        return path
    print(f"  Downloading {zip_name}...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
    print(f"    Done ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def extract_and_sample(zip_path, class_name, dest_dir, n_val=50, seed=42):
    rng = random.Random(seed)
    extract_dir = dest_dir / "_extracted" / class_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith((".jpg", ".jpeg", ".png"))]
        # Filter for val/ split
        val_members = [m for m in members if "/val/" in m or "\\val\\" in m]
        if not val_members:
            val_members = members  # fallback: use all
        rng.shuffle(val_members)
        selected = val_members[:n_val]
        for m in selected:
            zf.extract(m, extract_dir)

    # Collect extracted images
    img_dir = extract_dir / "val"
    if not img_dir.exists():
        # Try to find images
        dirs = [d for d in extract_dir.rglob("*") if d.is_dir() and list(d.glob("*.jpg"))]
        if dirs:
            img_dir = dirs[0]
        else:
            img_dir = extract_dir

    images = sorted(img_dir.glob("*.[jJ][pP][gG]")) + sorted(img_dir.glob("*.[pP][nN][gG]"))
    val_dir = dest_dir / class_name
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        shutil.copy2(img, val_dir / f"{i:04d}{img.suffix}")

    n_copied = len(list(val_dir.glob("*")))
    print(f"  {class_name}: {n_copied} images in {val_dir}")
    return n_copied


def main():
    print(f"Data dir: {DATA_DIR}")
    cache_dir = DATA_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = DATA_DIR / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract all 10 classes
    class_info = {}
    all_images = 0
    for class_name, zip_name in CLASSES.items():
        zip_path = download_zip(zip_name, cache_dir)
        n = extract_and_sample(zip_path, class_name, out_dir)
        class_info[class_name] = {"zip": zip_name, "images": n}
        all_images += n

    # Save class info
    info_path = DATA_DIR / "class_info.json"
    with open(info_path, "w") as f:
        json.dump({
            "classes": sorted(CLASSES.keys()),
            "class_info": class_info,
            "total_images": all_images,
            "data_dir": str(out_dir),
        }, f, indent=2)
    print(f"\nDone! {all_images} images, {len(class_info)} classes.")
    print(f"Class info saved to {info_path}")


if __name__ == "__main__":
    main()
