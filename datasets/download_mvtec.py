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
MVTEC_CLASSES = {
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
    'screw', 'toothbrush', 'transistor', 'zipper'
}


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
    # tarfile will auto-detect compression when using 'r:*'.
    # Python 3.14 changes extraction defaults; prefer data filter when available.
    with tarfile.open(archive_path, 'r:*') as tar:
        try:
            tar.extractall(path=tmp_dir, filter='data')
        except TypeError:
            tar.extractall(path=tmp_dir)
    return tmp_dir


def _has_phase_triplet(path: str) -> bool:
    return all(os.path.isdir(os.path.join(path, d)) for d in ('train', 'test', 'ground_truth'))


def _count_valid_categories(root: str) -> int:
    count = 0
    for category in MVTEC_CLASSES:
        if _has_phase_triplet(os.path.join(root, category)):
            count += 1
    return count


def detect_layout(extract_root: str) -> tuple[str, str]:
    """Return layout kind and source path.

    kind values:
    - 'full': source contains category folders (recommended full MVTec layout)
    - 'single': source is one category folder containing train/test/ground_truth
    """
    # 1) Common top-level folder in archive
    cand = os.path.join(extract_root, 'mvtec_anomaly_detection')
    if os.path.isdir(cand):
        if _count_valid_categories(cand) > 0:
            return 'full', cand
        if _has_phase_triplet(cand):
            return 'single', cand

    # 2) Extract root itself is full layout
    if _count_valid_categories(extract_root) > 0:
        return 'full', extract_root

    # 3) Search first-level directories
    for entry in os.listdir(extract_root):
        p = os.path.join(extract_root, entry)
        if not os.path.isdir(p):
            continue
        if _count_valid_categories(p) > 0:
            return 'full', p
        if _has_phase_triplet(p):
            return 'single', p

    raise RuntimeError('Could not detect MVTec layout in extracted archive')


def _infer_single_category_name(source_path: str, explicit_name: str | None) -> str:
    if explicit_name:
        if explicit_name not in MVTEC_CLASSES:
            raise ValueError(f"Invalid single category '{explicit_name}'. Expected one of: {sorted(MVTEC_CLASSES)}")
        return explicit_name

    base = os.path.basename(os.path.normpath(source_path))
    if base in MVTEC_CLASSES:
        return base

    raise ValueError(
        'Single-category archive detected but category name could not be inferred. '
        'Pass --single-category, e.g. --single-category toothbrush'
    )


def install_mvtec(extract_root: str, dest_parent: str, force: bool = False, single_category: str | None = None) -> str:
    layout_kind, source_path = detect_layout(extract_root)
    final_path = os.path.join(dest_parent, 'mvtec_anomaly_detection')
    if os.path.exists(final_path):
        if not force:
            raise FileExistsError(f"Destination exists: {final_path} (use --force to overwrite)")
        print(f"Removing existing destination: {final_path}")
        shutil.rmtree(final_path)

    if layout_kind == 'full':
        os.makedirs(final_path, exist_ok=True)
        if single_category:
            if single_category not in MVTEC_CLASSES:
                raise ValueError(f"Invalid single category '{single_category}'")
            src_cat = os.path.join(source_path, single_category)
            if not _has_phase_triplet(src_cat):
                raise RuntimeError(f"Category '{single_category}' not found in extracted archive")
            print(f"Installing only category '{single_category}'")
            shutil.copytree(src_cat, os.path.join(final_path, single_category))
        else:
            print(f"Moving full dataset {source_path} -> {final_path}")
            # Move directory content to final path so we avoid nested folder edge cases.
            for entry in os.listdir(source_path):
                src = os.path.join(source_path, entry)
                dst = os.path.join(final_path, entry)
                shutil.move(src, dst)
    else:
        # Single-category archive: install as <final>/<category>/train|test|ground_truth
        category = _infer_single_category_name(source_path, single_category)
        os.makedirs(final_path, exist_ok=True)
        dst_cat = os.path.join(final_path, category)
        print(f"Installing single-category dataset as: {dst_cat}")
        shutil.copytree(source_path, dst_cat)

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
    p.add_argument('--single-category', type=str, default=None,
                   help='Install only one MVTec class (e.g. toothbrush)')
    args = p.parse_args(argv)

    repo_root = repo_root_from_script()
    dest_parent = args.dest or os.path.join(repo_root, 'datasets')
    os.makedirs(dest_parent, exist_ok=True)

    tmp_archive = None
    try:
        tmp_archive = os.path.join(tempfile.gettempdir(), 'mvtec_anomaly_detection.tar.xz')
        download(args.url, tmp_archive)
        tmp_extract = extract_to_temp(tmp_archive)
        final = install_mvtec(tmp_extract, dest_parent, force=args.force, single_category=args.single_category)
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
