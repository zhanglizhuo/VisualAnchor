"""
download_scb5.py — Download SCB5 datasets from HuggingFace

Downloads the three SCB5 sub-datasets (TeacherBehavior, HandriseReadWrite,
BowTurnHead) in YOLO format from huggingface.co/wintonYF/SCB-Dataset,
then extracts the zip files into the expected directory structure.

Usage:
    python experiments/01_core/download_scb5.py [--output_dir datasets_scb]

Environment:
    HF_ENDPOINT  — HuggingFace mirror (default: https://hf-mirror.com)

The resulting directory structure matches DATASET_CFG in the experiment scripts:
    datasets_scb/SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
    datasets_scb/SCB5_HandriseReadWrite/SCB5-Handrise-Read-write-2024-9-17/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
    datasets_scb/SCB_BowTurnHead/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
"""

import argparse
import os
import zipfile

import requests

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

DATASETS = {
    "SCB5_TeacherBehavior": {
        "repo_path": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406",
        "files": [
            "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip",
            "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406.yaml",
        ],
    },
    "SCB5_HandriseReadWrite": {
        "repo_path": "SCB5-Handrise-Read-write-2024-9-17",
        "files": [
            "SCB5-Handrise-Read-write-2024-9-17.zip",
            "SCB5-Handrise-Read-write-2024-9-17.yaml",
        ],
    },
    "SCB_BowTurnHead": {
        "repo_path": "SCB_BowTurnHead_20250509",
        "files": [
            "SCB_BowTurnHead_20250509.zip",
            "SCB_BowTurnHead_20250509.yaml",
        ],
    },
}


def download_file(url, out_path, timeout=300):
    headers = {}
    file_size = 0
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path)
        headers["Range"] = f"bytes={file_size}-"

    resp = requests.get(url, stream=True, headers=headers, timeout=(30, timeout))
    resp.raise_for_status()

    mode = "ab" if file_size > 0 else "wb"
    with open(out_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True


def extract_zip(zip_path, target_dir):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def main():
    parser = argparse.ArgumentParser(description="Download SCB5 datasets from HuggingFace")
    parser.add_argument("--output_dir", default="datasets_scb", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name, cfg in DATASETS.items():
        target_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)

        for fname in cfg["files"]:
            url = f"{HF_ENDPOINT}/datasets/wintonYF/SCB-Dataset/resolve/main/{cfg['repo_path']}/{fname}"
            out_path = os.path.join(target_dir, fname)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"  Skip {fname} (exists)")
            else:
                print(f"  Downloading {fname}...")
                download_file(url, out_path)
                print(f"  Saved to {out_path}")

            if fname.endswith(".zip"):
                print(f"  Extracting {fname}...")
                extract_zip(out_path, target_dir)

    print(f"\nDone. Data saved to {args.output_dir}/")
    print(f"Set SCB5_DATA_ROOT={os.path.abspath(args.output_dir)} before running experiments.")


if __name__ == "__main__":
    main()
