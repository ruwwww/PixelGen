#!/usr/bin/env python3
"""Download a checkpoint file (or whole experiment folder) from a Hugging Face repo.

Usage:
  python scripts/hf_download_checkpoint.py --hf-repo username/PixelGen-Exp-S --path exp_PixelGen_S/last.ckpt --out-dir ./ckpts

  # List files in repo
  python scripts/hf_download_checkpoint.py --hf-repo username/PixelGen-Exp-S --list

Environment:
  Set HUGGINGFACE_HUB_TOKEN in env or pass --token
"""

import argparse
import os
from huggingface_hub import HfApi, hf_hub_download


def list_files(repo_id, token=None):
    api = HfApi()
    return api.list_repo_files(repo_id, repo_type='model', token=token)


def download_file(repo_id, path_in_repo, out_dir='.', token=None):
    os.makedirs(out_dir, exist_ok=True)
    print(f'Downloading {path_in_repo} from {repo_id} to {out_dir}')
    local_path = hf_hub_download(repo_id, path_in_repo, repo_type='model', use_auth_token=token)
    dest = os.path.join(out_dir, os.path.basename(path_in_repo))
    os.replace(local_path, dest)
    print('Saved to', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, required=True)
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--path', type=str, default=None, help='Path inside repo to download (e.g., exp_PixelGen_S/last.ckpt)')
    parser.add_argument('--out-dir', type=str, default='.', help='Where to save downloaded file')
    args = parser.parse_args()

    token = args.token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if args.list:
        files = list_files(args.hf_repo, token=token)
        print('Files in repo:')
        for f in files:
            print('  ', f)
    else:
        if not args.path:
            raise SystemExit('Provide --path to file inside repo to download')
        download_file(args.hf_repo, args.path, out_dir=args.out_dir, token=token)
