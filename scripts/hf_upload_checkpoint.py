#!/usr/bin/env python3
"""Upload checkpoint(s) and related config files for a single experiment to a Hugging Face repo.

Usage examples:
  # upload last.ckpt from an experiment dir, create repo if missing
  python scripts/hf_upload_checkpoint.py --exp-dir universal_pix_workdirs/exp_PixelGen_S --hf-repo username/PixelGen-Exp-S --create-repo

  # upload a specific checkpoint file and its config
  python scripts/hf_upload_checkpoint.py --ckpt universal_pix_workdirs/exp_PixelGen_S/epoch=253-step=20000.ckpt --include-config

Environment:
  Set HUGGINGFACE_HUB_TOKEN in env or pass --token

Notes:
  - The script uses huggingface_hub.Repository (git + lfs). Please ensure 'git' and 'git-lfs' are installed and configured.
"""

import argparse
import os
import shutil
import tempfile
import glob
from huggingface_hub import HfApi, Repository, create_repo, hf_hub_download


def find_checkpoint(exp_dir, pattern=None):
    if os.path.isfile(pattern or ''):
        return os.path.abspath(pattern)
    # common names
    candidates = []
    if pattern:
        candidates = glob.glob(os.path.join(exp_dir, pattern))
    else:
        candidates = glob.glob(os.path.join(exp_dir, "*.ckpt")) + glob.glob(os.path.join(exp_dir, "*.pt"))
    if not candidates:
        # check nested
        for root, _, files in os.walk(exp_dir):
            for f in files:
                if f.endswith('.ckpt') or f.endswith('.pt'):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
    # prefer last.ckpt or epoch/last
    preferred = [c for c in candidates if os.path.basename(c) in ("last.ckpt", "last.pt")]
    if preferred:
        return os.path.abspath(preferred[0])
    # otherwise pick largest file (likely full checkpoint)
    candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)
    return os.path.abspath(candidates[0])


def collect_files(exp_dir, ckpt_path, include_config=False, include_wandb=False):
    files = []
    files.append(ckpt_path)
    if include_config:
        # include config yaml(s) next to ckpt or in exp_dir
        cfgs = glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.yaml")) + glob.glob(os.path.join(exp_dir, "*.yaml"))
        files += cfgs
    if include_wandb:
        wandb_dir = os.path.join(exp_dir, "wandb")
        if os.path.exists(wandb_dir):
            # upload summary and config
            for root, _, fs in os.walk(wandb_dir):
                for f in fs:
                    if f.endswith('.json') or f.endswith('.log'):
                        files.append(os.path.join(root, f))
    # dedupe
    files = list(dict.fromkeys(files))
    return [f for f in files if os.path.exists(f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default=None, help='Experiment directory (e.g., universal_pix_workdirs/exp_PixelGen_S)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint file (overrides discovery)')
    parser.add_argument('--pattern', type=str, default=None, help='Glob pattern to find checkpoint inside exp-dir (e.g. "epoch=*ckpt")')
    parser.add_argument('--hf-repo', type=str, required=True, help='Hugging Face repo id (username/repo or org/repo)')
    parser.add_argument('--token', type=str, default=None, help='Hugging Face token (or set HUGGINGFACE_HUB_TOKEN env var)')
    parser.add_argument('--create-repo', action='store_true', help='Create the HF repo if missing')
    parser.add_argument('--private', action='store_true', help='Create the repo as private (requires create-repo)')
    parser.add_argument('--include-config', action='store_true', help='Upload config yaml files found alongside the checkpoint')
    parser.add_argument('--include-wandb', action='store_true', help='Include wandb logs (if present)')
    parser.add_argument('--commit-msg', type=str, default='Add checkpoint', help='Git commit message')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded but do not push')
    args = parser.parse_args()

    token = args.token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if token is None:
        raise RuntimeError('HUGGINGFACE_HUB_TOKEN must be set or --token provided')

    exp_dir = args.exp_dir or os.path.dirname(args.ckpt) if args.ckpt else None
    if exp_dir is None or not os.path.exists(exp_dir):
        raise RuntimeError('Provide valid --exp-dir or --ckpt path')

    # find checkpoint
    ckpt_path = args.ckpt or find_checkpoint(exp_dir, args.pattern)
    print(f"Selected checkpoint: {ckpt_path}")

    files = collect_files(exp_dir, ckpt_path, include_config=args.include_config, include_wandb=args.include_wandb)
    print('Files to upload:')
    for f in files:
        print('  ', f)

    if args.dry_run:
        print('Dry run; exiting')
        return

    api = HfApi()
    if args.create_repo:
        try:
            print(f"Creating repo {args.hf_repo} (private={args.private})")
            create_repo(args.hf_repo, token=token, private=args.private, exist_ok=True)
        except Exception as e:
            print('Create repo warning:', e)

    # clone HF repo to temp dir (git + lfs)
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Cloning repo locally (this may take a moment) ...')
        repo = Repository(tmpdir, clone_from=args.hf_repo, use_auth_token=token)
        # copy files into repo under a subdir matching experiment name
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        target_dir = os.path.join(tmpdir, exp_name)
        os.makedirs(target_dir, exist_ok=True)
        for p in files:
            shutil.copy2(p, target_dir)

        # save a small metadata file
        meta_path = os.path.join(target_dir, 'upload_metadata.txt')
        with open(meta_path, 'w') as fh:
            fh.write(f"uploaded_from={os.getcwd()}\n")
            fh.write(f"source_ckpt={ckpt_path}\n")
        repo.git_add(pattern='.')
        repo.git_commit(args.commit_msg)
        print('Committed changes locally')
        print('Pushing to Hugging Face...')
        repo.git_push()

    print('Upload complete')


if __name__ == '__main__':
    main()
