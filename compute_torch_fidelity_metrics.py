#!/usr/bin/env python3
"""FINAL SCRIPT ULTIMATE - ALL METRICS (FID, IS, KID, PRC)

Syarat: torch-fidelity harus versi 0.4.0 atau master dari GitHub
(pip install git+https://github.com/toshas/torch-fidelity.git)
"""

import os
import json
import csv
import argparse
from pathlib import Path
import sys

def find_steps(samples_dir: Path):
    for exp in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
        for step in sorted(p for p in exp.iterdir() if p.is_dir()):
            yield exp.name, step.name, step

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="fid_samples", type=Path)
    parser.add_argument("--real_stats", required=True, type=Path,
                        help="Wajib pakai folder real images buat dapet PRC dan KID!")
    parser.add_argument("--out_dir", default="fid_results", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if not args.samples_dir.exists():
        raise SystemExit(f"❌ Samples dir ga ada: {args.samples_dir}")
    if not args.real_stats.exists():
        raise SystemExit(f"❌ Real stats/path ga ada: {args.real_stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch_fidelity
        from torch_fidelity import calculate_metrics
        print(f"✅ torch-fidelity version: {getattr(torch_fidelity, '__version__', 'unknown')}")
    except ImportError:
        raise SystemExit("❌ Install dulu: pip install git+https://github.com/toshas/torch-fidelity.git")

    import torch
    use_cuda = torch.cuda.is_available()
    print(f"✅ CUDA available: {use_cuda}")

    # Peringatan kalau user pakai .npz (KID & PRC butuh raw images)
    compute_full = args.real_stats.is_dir()
    if not compute_full:
        print("⚠️  Pakai .npz → KID dan PRC (Precision/Recall) mungkin bakal ke-skip. Pakai folder real images!")

    rows = []
    for exp_name, step_name, step_path in find_steps(args.samples_dir):
        print(f"\n🚀 Computing {exp_name}/{step_name} ...")

        # Argumen untuk borong semua metrik
        kwargs = {
            "input1": str(step_path),
            "input2": str(args.real_stats),
            "cuda": use_cuda,
            "batch_size": args.batch_size,
            "isc": True,               # Inception Score
            "fid": True,               # Frechet Inception Distance
            "kid": True,               # Kernel Inception Distance
            "prc": True,               # Precision & Recall (Butuh v0.4.0+)
            "samples_find_deep": True, # Fix input directory error
            "verbose": True,
        }

        try:
            metrics = calculate_metrics(**kwargs)
        except Exception as e:
            print(f"❌ Error di {step_path}: {e}")
            metrics = {"error": str(e)}

        # Tangkap SEMUA values
        fid = metrics.get("frechet_inception_distance")
        is_mean = metrics.get("inception_score_mean")
        is_std = metrics.get("inception_score_std")
        kid_mean = metrics.get("kernel_inception_distance_mean")
        kid_std = metrics.get("kernel_inception_distance_std")
        precision = metrics.get("precision")
        recall = metrics.get("recall")

        # Save per-step JSON
        out_json = args.out_dir / f"{exp_name}__{step_name}.json"
        with open(out_json, "w") as f:
            json.dump({"exp": exp_name, "step": step_name, "metrics": metrics}, f, indent=2)

        # Append ke CSV
        rows.append({
            "experiment": exp_name,
            "step": step_name,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "kid_mean": kid_mean if compute_full else None,
            "kid_std": kid_std if compute_full else None,
            "precision": precision if compute_full else None,
            "recall": recall if compute_full else None,
            "json": str(out_json)
        })

    # Summary CSV
    csv_path = args.out_dir / "summary.csv"
    fieldnames = [
        "experiment", "step", "fid", "is_mean", "is_std", 
        "kid_mean", "kid_std", "precision", "recall", "json"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ SELESAI BRO! Hasil di {args.out_dir}")
    print(f"   summary.csv siap buat di-analisis!")

if __name__ == "__main__":
    run()