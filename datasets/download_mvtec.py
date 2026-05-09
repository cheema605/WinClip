#!/usr/bin/env python3
"""Download and install MVTec-AD into the repository `datasets/` folder.

Usage:
    python download_mvtec.py [--url URL] [--dest DEST_DIR] [--force]

By default the script places the dataset at: <repo_root>/datasets/mvtec_anomaly_detection

This script uses only Python standard library and works on Windows/macOS/Linux.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request


DEFAULT_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


def download(url: str, out_path: str) -> None:
    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            sys.stdout.write(f"\rDownloading... {pct:5.1f}%")
        else:
            sys.stdout.write(f"\rDownloaded {downloaded} bytes")
        sys.stdout.flush()

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path, _hook)
    sys.stdout.write("\n")


def extract_to_temp(archive_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="mvtec_extract_")
    print(f"Extracting archive to temporary folder: {tmp_dir}")
    # tarfile will auto-detect compression when using 'r:*'
    with tarfile.open(archive_path, 'r:*') as tar:
        tar.extractall(path=tmp_dir)
    return tmp_dir


def find_mvtec_folder(extract_root: str) -> str | None:
    # Common layout: extract_root/mvtec_anomaly_detection
    cand = os.path.join(extract_root, 'mvtec_anomaly_detection')
    if os.path.isdir(cand):
        return cand

    # Otherwise find a directory that contains known category folders (e.g. carpet)
    for entry in os.listdir(extract_root):
        p = os.path.join(extract_root, entry)
        if not os.path.isdir(p):
            continue
        # heuristics: presence of 'carpet' or 'train' subfolders
        if os.path.isdir(os.path.join(p, 'carpet')) or os.path.isdir(os.path.join(p, 'train')):
            return p
    return None


def install_mvtec(extract_root: str, dest_parent: str, force: bool = False) -> str:
    mvtec_folder = find_mvtec_folder(extract_root)
    if mvtec_folder is None:
        raise RuntimeError("Could not find mvtec folder inside the extracted archive")

    final_path = os.path.join(dest_parent, 'mvtec_anomaly_detection')
    if os.path.exists(final_path):
        if not force:
            raise FileExistsError(f"Destination exists: {final_path} (use --force to overwrite)")
        print(f"Removing existing destination: {final_path}")
        shutil.rmtree(final_path)

    print(f"Moving {mvtec_folder} -> {final_path}")
    shutil.move(mvtec_folder, final_path)
    return final_path


def summarize_dataset(mvtec_dir: str) -> None:
    print('\nDataset installed at:', mvtec_dir)
    print('Summary (category : has_train has_test has_ground_truth):')
    for category in sorted(os.listdir(mvtec_dir)):
        catp = os.path.join(mvtec_dir, category)
        if not os.path.isdir(catp):
            continue
        train = os.path.isdir(os.path.join(catp, 'train'))
        test = os.path.isdir(os.path.join(catp, 'test'))
        gt = os.path.isdir(os.path.join(catp, 'ground_truth'))
        print(f"- {category} : train={train} test={test} ground_truth={gt}")


def repo_root_from_script() -> str:
    # file is located at <repo>/WinClip/datasets/download_mvtec.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', '..'))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download and install MVTec-AD for WinCLIP project")
    p.add_argument('--url', type=str, default=DEFAULT_URL, help='Archive URL')
    p.add_argument('--dest', type=str, default=None, help='Parent datasets folder (default: <repo_root>/datasets)')
    p.add_argument('--force', action='store_true', help='Overwrite existing destination if present')
    args = p.parse_args(argv)

    repo_root = repo_root_from_script()
    dest_parent = args.dest or os.path.join(repo_root, 'datasets')
    os.makedirs(dest_parent, exist_ok=True)

    tmp_archive = None
    try:
        tmp_archive = os.path.join(tempfile.gettempdir(), 'mvtec_anomaly_detection.tar.xz')
        download(args.url, tmp_archive)
        tmp_extract = extract_to_temp(tmp_archive)
        final = install_mvtec(tmp_extract, dest_parent, force=args.force)
        summarize_dataset(final)
        return 0
    except Exception as e:
        print('Error:', e)
        return 2
    finally:
        # best-effort cleanup
        try:
            if tmp_archive and os.path.exists(tmp_archive):
                os.remove(tmp_archive)
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
