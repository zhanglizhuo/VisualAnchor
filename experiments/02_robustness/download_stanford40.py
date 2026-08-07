"""
download_stanford40.py — Download Stanford 40 Actions dataset

Downloads the Stanford 40 Actions dataset (40 classes, 9532 images)
from the official source.

Usage:
    python experiments/02_robustness/download_stanford40.py

The resulting directory structure matches experiment scripts:
    data/Stanford40/JPEGImages/{action}_{id}.jpg
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests

URL = "http://vision.stanford.edu/Datasets/Stanford40Actions.zip"
ZIP_FILENAME = "Stanford40Actions.zip"


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
    parser = argparse.ArgumentParser(description="Download Stanford 40 Actions dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "Stanford40"),
        help="Output directory for Stanford40 data",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_FILENAME

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"Downloading from {URL} ...")
        download_file(URL, zip_path)
        print(f"Saved to {zip_path}")
    else:
        print(f"Skip download ({ZIP_FILENAME} exists)")

    # The zip wraps contents in a Stanford40Actions/ folder
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    wrapped = extract_dir / "Stanford40Actions"
    if wrapped.is_dir():
        print(f"Already extracted at {wrapped}")
    else:
        print(f"Extracting {ZIP_FILENAME} ...")
        extract_zip(zip_path, extract_dir)
        print(f"Extracted to {wrapped}")

    # Move contents from Stanford40Actions/ to output dir
    if wrapped.is_dir():
        for item in wrapped.iterdir():
            dest = out_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"  Moved {item.name} -> {out_dir}")
            else:
                print(f"  Skip {item.name} (already exists at {dest})")

    # Cleanup extraction temp dir if empty
    if extract_dir.exists():
        remaining = list(extract_dir.iterdir())
        if not remaining:
            shutil.rmtree(extract_dir)
            print("  Cleaned up temporary extraction directory")
        else:
            print(f"  Note: {extract_dir} still contains {len(remaining)} item(s)")

    print(f"\nDone. Stanford40 data at {out_dir}")
    print(f"Set STANFORD40_DIR={out_dir / 'JPEGImages'} before running experiments.")


if __name__ == "__main__":
    main()
