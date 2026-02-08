#!/usr/bin/env python3
"""Sync checkpoints to a single Hugging Face repo with per-experiment / per-step layout.

Layout on HF repo:
  <exp_name>/step_<step>/last.ckpt
  <exp_name>/step_<step>/*.yaml

Commands:
  upload    --exp-dir /path/to/exp --exp-name EXP --hf-repo user/batik-pixelgen-exp [--interval 50000]
  list-steps --hf-repo user/batik-pixelgen-exp --exp-name EXP
  download  --hf-repo user/batik-pixelgen-exp --exp-name EXP --step 50000 --out-dir ./ckpts

Usage examples:
  # upload all checkpoints at 50k increments from an experiment dir
  python scripts/hf_sync.py upload --exp-dir universal_pix_workdirs/exp_PixelGen_S --exp-name PixelGen_S --hf-repo ruwwww/batik-pixelgen-exp --interval 50000

  # list available steps for an experiment
  python scripts/hf_sync.py list-steps --hf-repo ruwwww/batik-pixelgen-exp --exp-name PixelGen_S

  # download a step
  python scripts/hf_sync.py download --hf-repo ruwwww/batik-pixelgen-exp --exp-name PixelGen_S --step 50000 --out-dir ./ckpts

Environment:
  Set HUGGINGFACE_HUB_TOKEN in env or pass --token
"""

import argparse
import os
import re
import shutil
import tempfile
import glob
from huggingface_hub import HfApi, Repository, create_repo, hf_hub_download

STEP_RE = re.compile(r"step[=_-]?(\d+)")


def find_checkpoints_for_interval(exp_dir, interval):
    # find candidate ckpt files and pick those where step % interval == 0
    candidates = []
    for root, _, files in os.walk(exp_dir):
        for f in files:
            if f.endswith('.ckpt') or f.endswith('.pt'):
                full = os.path.join(root, f)
                # try to extract step from filename
                m = STEP_RE.search(f)
                step = None
                if m:
                    step = int(m.group(1))
                else:
                    # try parent folder name
                    m2 = STEP_RE.search(os.path.basename(root))
                    if m2:
                        step = int(m2.group(1))
                candidates.append((full, step))
    selected = []
    for path, step in candidates:
        if step is None:
            continue
        if step % interval == 0:
            selected.append((step, path))
    selected.sort()
    return selected


def hf_path_for(exp_name, step, filename):
    return f"{exp_name}/step_{step}/{filename}"


def upload_files_to_repo(repo_id, files, target_prefix, token):
    # clone repo to temp dir and copy files under target_prefix
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(tmpdir, clone_from=repo_id, use_auth_token=token)
        target_dir = os.path.join(tmpdir, target_prefix)
        os.makedirs(target_dir, exist_ok=True)
        for p in files:
            shutil.copy2(p, target_dir)
        # metadata
        meta = os.path.join(target_dir, 'uploaded_by.txt')
        with open(meta, 'w') as fh:
            fh.write(f'source_dir={os.getcwd()}\n')
        repo.git_add(pattern='.')
        repo.git_commit('Add checkpoint(s)')
        repo.git_push()


def upload(exp_dir, exp_name, hf_repo, token, interval=50000, include_config=False, dry_run=False, create_repo_flag=False):
    token = token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    api = HfApi()
    if create_repo_flag:
        create_repo(hf_repo, token=token, exist_ok=True)

    selected = find_checkpoints_for_interval(exp_dir, interval)
    if not selected:
        print(f"No checkpoints found in {exp_dir} matching interval {interval}")
        return

    for step, ckpt_path in selected:
        print(f"Found step {step}: {ckpt_path}")
        # collect files: checkpoint + optionally nearby config files
        files = [ckpt_path]
        if include_config:
            cfgs = glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.yaml")) + glob.glob(os.path.join(exp_dir, "*.yaml"))
            files += cfgs
        target_prefix = f"{exp_name}/step_{step}"
        # check remote existence
        remote_files = api.list_repo_files(hf_repo, token=token)
        exists = any(p.startswith(target_prefix + '/') for p in remote_files)
        if exists:
            print(f"Remote step {step} already exists in {hf_repo}; skipping")
            continue
        print(f"Uploading files for step {step} to {hf_repo}/{target_prefix} ...")
        if dry_run:
            for f in files:
                print('  ', f)
            continue
        upload_files_to_repo(hf_repo, files, target_prefix, token)
        print('Upload finished')


def list_steps(hf_repo, exp_name, token):
    api = HfApi()
    files = api.list_repo_files(hf_repo, token=token)
    steps = set()
    prefix = f"{exp_name}/step_"
    for p in files:
        if p.startswith(prefix):
            rest = p[len(prefix):]
            m = re.match(r"(\d+)/(.*)", rest)
            if m:
                steps.add(int(m.group(1)))
    for s in sorted(steps):
        print(s)
    if not steps:
        print('No steps found for experiment', exp_name)


def download_step(hf_repo, exp_name, step, out_dir, token):
    token = token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    os.makedirs(out_dir, exist_ok=True)
    # list remote files first
    api = HfApi()
    files = api.list_repo_files(hf_repo, token=token)
    prefix = f"{exp_name}/step_{step}/"
    matches = [p for p in files if p.startswith(prefix)]
    if not matches:
        raise SystemExit(f'No files found at {prefix} in {hf_repo}')
    for p in matches:
        print('Downloading', p)
        local = hf_hub_download(hf_repo, p, repo_type='model', use_auth_token=token)
        dest = os.path.join(out_dir, os.path.basename(p))
        os.replace(local, dest)
        print('Saved to', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_up = sub.add_parser('upload')
    p_up.add_argument('--exp-dir', required=True)
    p_up.add_argument('--exp-name', required=True)
    p_up.add_argument('--hf-repo', required=True)
    p_up.add_argument('--token', default=None)
    p_up.add_argument('--interval', type=int, default=50000)
    p_up.add_argument('--include-config', action='store_true')
    p_up.add_argument('--dry-run', action='store_true')
    p_up.add_argument('--create-repo', action='store_true')

    p_ls = sub.add_parser('list-steps')
    p_ls.add_argument('--hf-repo', required=True)
    p_ls.add_argument('--exp-name', required=True)
    p_ls.add_argument('--token', default=None)

    p_dl = sub.add_parser('download')
    p_dl.add_argument('--hf-repo', required=True)
    p_dl.add_argument('--exp-name', required=True)
    p_dl.add_argument('--step', type=int, required=True)
    p_dl.add_argument('--out-dir', default='./ckpts')
    p_dl.add_argument('--token', default=None)

    args = parser.parse_args()
    if args.cmd == 'upload':
        upload(args.exp_dir, args.exp_name, args.hf_repo, args.token, args.interval, args.include_config, args.dry_run, args.create_repo)
    elif args.cmd == 'list-steps':
        list_steps(args.hf_repo, args.exp_name, args.token)
    elif args.cmd == 'download':
        download_step(args.hf_repo, args.exp_name, args.step, args.out_dir, args.token)
    else:
        parser.print_help()
