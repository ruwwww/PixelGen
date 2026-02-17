#!/usr/bin/env python3
"""Compute FID, IS, precision, recall for generated samples using torch-fidelity.

Scans `fid_samples/` for experiment/step folders, computes metrics against
the provided real stats NPZ (`batik-256_stats.npz`) and writes results to
`fid_results/` as JSON and a combined CSV `fid_results/summary.csv`.

Usage:
  python compute_torch_fidelity_metrics.py --samples_dir fid_samples --real_stats batik-256_stats.npz

If `torch_fidelity` is not installed or API differs, the script will print
instructions to install and how to run the CLI manually.
"""
import os
import json
import csv
import argparse
from pathlib import Path
import sys


def find_steps(samples_dir: Path):
    # samples_dir/exp_name/step_xxx/
    for exp in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
        for step in sorted(p for p in exp.iterdir() if p.is_dir()):
            yield exp.name, step.name, step


def try_import_torch_fidelity():
    try:
        from torch_fidelity import calculate_metrics
        return calculate_metrics
    except Exception:
        return None


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="fid_samples", help="Root folder containing generated samples")
    parser.add_argument("--real_stats", default="batik-256_stats.npz", help="NPZ file of real dataset statistics or path to real images")
    parser.add_argument("--out_dir", default="fid_results", help="Where to write JSON + CSV results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not set)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    real_stats = Path(args.real_stats)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not samples_dir.exists():
        raise SystemExit(f"Samples dir not found: {samples_dir}")
    if not real_stats.exists():
        raise SystemExit(f"Real stats NPZ or dir not found: {real_stats}")

    calc = try_import_torch_fidelity()
    if calc is None:
        print("torch_fidelity API not available. Install with `pip install torch-fidelity` and retry.")
        print("If you prefer using the CLI, the equivalent command is:\n  python -m torch_fidelity.calculate_metrics --input1 <real_stats.npz> --input2 <samples_folder> --fid --prc --isc --batch-size 64 --json")
        raise SystemExit(1)

    # Determine cuda usage
    use_cuda = False
    if args.device:
        use_cuda = (args.device == 'cuda')
    else:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    rows = []
    for exp_name, step_name, step_path in find_steps(samples_dir):
        print(f"Computing metrics for {exp_name}/{step_name} -> {step_path}")
        try:
            # Use Python API (handles both dataset paths and precomputed .npz)
            kwargs = dict(
                input1=str(real_stats),
                input2=str(step_path),
                batch_size=args.batch_size,
                cuda=use_cuda,
                isc=True,
                fid=True,
                prc=True,
                verbose=True,  # Enable verbose for better debugging
            )
            metrics = calc(**kwargs)
        except TypeError:
            # Fallback without batch_size (older API compatibility)
            try:
                kwargs.pop('batch_size')
                metrics = calc(**kwargs)
            except Exception as e:
                print(f"ERROR computing metrics for {step_path}: {e}")
                metrics = {'error': str(e)}
        except Exception as e:
            print(f"ERROR computing metrics for {step_path}: {e}")
            metrics = {'error': str(e)}

        # Normalize metric keys and extract fid/is/precision/recall if present
        fid = None
        is_mean = None
        is_std = None
        precision = None
        recall = None
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                kl = k.lower()
                if 'fid' in kl or 'frechet' in kl:
                    fid = float(v)
                if 'inception_score_mean' in kl:
                    is_mean = float(v)
                if 'inception_score_std' in kl:
                    is_std = float(v)
                if 'precision' in kl:
                    precision = float(v)
                if 'recall' in kl:
                    recall = float(v)

        out_json = out_dir / f"{exp_name}__{step_name}.json"
        with open(out_json, 'w') as f:
            json.dump({'exp': exp_name, 'step': step_name, 'metrics': metrics}, f, indent=2)

        rows.append({
            'experiment': exp_name,
            'step': step_name,
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
            'precision': precision,
            'recall': recall,
            'json': str(out_json)
        })

    # Write CSV
    csv_path = out_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'step', 'fid', 'is_mean', 'is_std', 'precision', 'recall', 'json'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Done. JSON results + summary CSV written to {out_dir}")


if __name__ == '__main__':
    run()
