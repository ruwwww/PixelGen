#!/usr/bin/env python3
"""Download and extract dataset from Hugging Face into a persistent host-mounted directory.

Usage examples:
  # inside container, with host mount /home/kuroko/ai/data -> /data in container
  python scripts/get_data.py --repo ruwwww/batik-processed --filename batik-256.tar.gz --out-dir /data

Options:
  --repo      HF dataset repo id (default: ruwwww/batik-processed)
  --filename  file name inside the repo (default: batik-256.tar.gz)
  --out-dir   where to extract (default: ./data)
  --force     re-download and re-extract even if target exists

Notes:
- Provide HUGGINGFACE_HUB_TOKEN via env if the dataset is private.
"""

import argparse
import os
import tarfile
from huggingface_hub import hf_hub_download


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])


def safe_extract(tar: tarfile.TarFile, path: str = "./"):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path)


def main():
    parser = argparse.ArgumentParser(description='Download and extract dataset from Hugging Face')
    parser.add_argument('--repo', type=str, default='ruwwww/batik-processed')
    parser.add_argument('--filename', type=str, default='batik-256.tar.gz')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--force', action='store_true', help='Re-download and re-extract')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    expected_marker = os.path.join(out_dir, 'batik-256')
    if os.path.exists(expected_marker) and not args.force:
        print(f"Dataset already appears extracted at {expected_marker}; use --force to re-download/re-extract.")
        return

    print(f"Downloading {args.filename} from {args.repo}...")
    path = hf_hub_download(args.repo, filename=args.filename)  # uses HUGGINGFACE_HUB_TOKEN if needed
    print(f"Downloaded to {path}; extracting to {out_dir} ...")

    if not tarfile.is_tarfile(path):
        raise RuntimeError(f"Downloaded file {path} is not a tar archive")

    with tarfile.open(path) as t:
        safe_extract(t, out_dir)

    print("Extraction complete.")
    for root, dirs, files in os.walk(out_dir):
        depth = root.replace(out_dir, '').count(os.sep)
        indent = ' ' * 2 * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth > 2:
            break


if __name__ == '__main__':
    main()
