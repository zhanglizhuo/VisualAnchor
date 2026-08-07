"""
download_eurosat.py — Download EuroSAT RGB dataset

Downloads the 10-class EuroSAT satellite image dataset from the official source
or HuggingFace mirror, and extracts it into the expected directory structure.

Usage:
    python experiments/02_robustness/download_eurosat.py

Environment:
    HF_ENDPOINT  — HuggingFace mirror (default: https://hf-mirror.com)

The resulting directory structure matches load_eurosat() in experiment scripts:
    data/eurosat_rgb/2750/{AnnualCrop,Forest,...,SeaLake}/*.jpg
"""

import argparse
import os
import zipfile
from pathlib import Path

import requests

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Official source and HF mirror fallback
EUROSAT_URLS = [
    "https://madm.dfki.de/files/sentinel/EuroSAT_RGB.zip",
    f"{HF_ENDPOINT}/datasets/wintonYF/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip",
]

ZIP_FILENAME = "EuroSAT_RGB.zip"


def download_file(url, out_path, timeout=300):
    headers = {}
    file_size = 0
    if out_path.exists():
        file_size = out_path.stat().st_size
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
    parser = argparse.ArgumentParser(description="Download EuroSAT RGB dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "eurosat_rgb"),
        help="Output directory for EuroSAT data",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_FILENAME

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        downloaded = False
        for url in EUROSAT_URLS:
            try:
                print(f"Downloading from {url} ...")
                download_file(url, zip_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        if not downloaded:
            print("ERROR: All download sources failed.")
            return
        print(f"Saved to {zip_path}")
    else:
        print(f"Skip download ({ZIP_FILENAME} exists)")

    # Check if already extracted
    extracted_marker = out_dir / "2750"
    if extracted_marker.is_dir():
        print(f"Already extracted at {extracted_marker}")
    else:
        print(f"Extracting {ZIP_FILENAME} ...")
        extract_zip(zip_path, out_dir)
        print(f"Extracted to {out_dir}")

    print(f"\nDone. EuroSAT data at {out_dir / '2750'}")
    print(f"Set EUROSAT_DIR={out_dir} or place data in data/eurosat_rgb/2750/ before running experiments.")


if __name__ == "__main__":
    main()
