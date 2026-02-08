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
import glob
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

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


def upload_many_files_to_repo(repo_id, mapping, token):
    """Upload multiple groups of files to a repo in a single commit using HfApi.
    Does NOT clone the repository, so it avoids downloading LFS files.

    mapping: dict target_prefix -> list of file paths
    """
    api = HfApi(token=token)
    operations = []
    
    print(f"Preparing batch upload to {repo_id}...")
    
    for target_prefix, files in mapping.items():
        # Add files
        for p in files:
            path_in_repo = f"{target_prefix}/{os.path.basename(p)}"
            operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=p))
        
        # Add metadata file
        meta_path_in_repo = f"{target_prefix}/uploaded_by.txt"
        meta_content = f"source_dir={os.getcwd()}\n".encode('utf-8')
        operations.append(CommitOperationAdd(path_in_repo=meta_path_in_repo, path_or_fileobj=meta_content))

    if not operations:
        print("No files to upload.")
        return

    print(f"Pushing {len(operations)} files to {repo_id} via HTTP API...")
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Add checkpoint(s) [via HfApi]"
    )
    print("Batch upload finished successfully.")


def upload_files_to_repo(repo_id, files, target_prefix, token):
    """Legacy wrapper, delegates to batch uploader."""
    upload_many_files_to_repo(repo_id, {target_prefix: files}, token)


def upload(exp_dir, exp_name, hf_repo, token, interval=50000, include_config=False, dry_run=False, create_repo_flag=False, all_files=False, include_last=False, allow_variants=False, step=None):
    token = token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    api = HfApi(token=token)
    
    # Check if repo exists, create if needed
    if create_repo_flag:
        api.create_repo(repo_id=hf_repo, exist_ok=True)
    
    # gather candidate checkpoints
    candidates = []
    for root, _, files in os.walk(exp_dir):
        for f in files:
            if f.endswith('.ckpt') or f.endswith('.pt'):
                full = os.path.join(root, f)
                m = STEP_RE.search(f)
                step_val = int(m.group(1)) if m else None
                candidates.append((step_val, full))

    # include last.ckpt specially if requested
    last_path = os.path.join(exp_dir, 'last.ckpt')
    if include_last and os.path.exists(last_path):
        # only add if not already present
        if not any(os.path.basename(p) == 'last.ckpt' for _, p in candidates):
            candidates.append((None, last_path))

    # filter by step if provided
    if step is not None:
        candidates = [(s, p) for s, p in candidates if s == step]

    # if not all_files, filter by interval
    if not all_files:
        selected = [(s, p) for s, p in candidates if s is not None and s % interval == 0]
    else:
        selected = candidates

    if not selected:
        print(f"No checkpoints found in {exp_dir} matching criteria (interval={interval}, all_files={all_files}, step={step})")
        return

    # remote files list for existence checks
    try:
        remote_files = api.list_repo_files(hf_repo, token=token)
    except Exception as e:
        print(f"Warning: could not list repo files (maybe it's empty or private/auth issue?): {e}")
        remote_files = []

    # Prepare the batch map for everything we intend to upload
    uploads_map = {}
    
    # Sort by step
    sorted_candidates = sorted(selected, key=lambda t: (t[0] if t[0] is not None else 10**12, t[1]))

    for s, ckpt_path in sorted_candidates:
        step_str = f"step_{s}" if s is not None else 'latest'
        target_prefix = f"{exp_name}/{step_str}"

        # 1. Determine candidate files for this step
        candidate_files = [ckpt_path]
        if include_config:
            cfgs = glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.yaml")) + glob.glob(os.path.join(exp_dir, "*.yaml"))
            candidate_files.extend(cfgs)
        
        # Deduplicate local list
        candidate_files = list(set(candidate_files))

        # 2. Filter files that are already on remote
        files_for_step = []
        for local_path in candidate_files:
            fname = os.path.basename(local_path)
            remote_path = f"{target_prefix}/{fname}"
            
            if remote_path in remote_files:
                # already exists
                continue
            
            # Special check: if this is the main checkpoint and variants are disallowed
            if local_path == ckpt_path and s is not None and not allow_variants:
                # Check if any OTHER checkpoint exists in this folder
                # (We already know 'remote_path' doesn't exist, otherwise we'd have hit the continue above)
                exists_variant = any(p.startswith(f"{exp_name}/step_{s}/") and (p.endswith('.ckpt') or p.endswith('.pt')) for p in remote_files)
                if exists_variant:
                    print(f"Remote step {s} already has a checkpoint and variants are not allowed; skipping {fname}")
                    continue

            files_for_step.append(local_path)

        if not files_for_step:
            continue
            
        # Add to global batch map
        if target_prefix not in uploads_map:
            uploads_map[target_prefix] = []
        
        for f in files_for_step:
            if f not in uploads_map[target_prefix]:
                uploads_map[target_prefix].append(f)
                print(f"Queued {os.path.basename(f)} for upload to {target_prefix}/")

    if not uploads_map:
        print("Nothing new to upload.")
        return

    print(f"Batch uploading {sum(len(v) for v in uploads_map.values())} files in one commit (dry_run={dry_run})")
    if dry_run:
        for tp, fls in uploads_map.items():
            print('  target:', tp)
            for f in fls:
                print('    ', f)
        return
    
    upload_many_files_to_repo(hf_repo, uploads_map, token)


def list_steps(hf_repo, exp_name, token):
    token = token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    api = HfApi(token=token)
    files = api.list_repo_files(hf_repo)
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
    api = HfApi(token=token)
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
    p_up.add_argument('--all', action='store_true', dest='all_files', help='Upload all checkpoints (ignore interval)')
    p_up.add_argument('--include-last', action='store_true', help='Also include last.ckpt if present')
    p_up.add_argument('--allow-variants', action='store_true', help='Allow uploading multiple filenames for the same numeric step')
    p_up.add_argument('--step', type=int, default=None, help='Upload only this numeric step (overrides interval)')

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
        upload(args.exp_dir,
               args.exp_name,
               args.hf_repo,
               args.token,
               args.interval,
               args.include_config,
               args.dry_run,
               args.create_repo,
               all_files=args.all_files,
               include_last=args.include_last,
               allow_variants=args.allow_variants,
               step=args.step)
    elif args.cmd == 'list-steps':
        list_steps(args.hf_repo, args.exp_name, args.token)
    elif args.cmd == 'download':
        download_step(args.hf_repo, args.exp_name, args.step, args.out_dir, args.token)
    else:
        parser.print_help()
