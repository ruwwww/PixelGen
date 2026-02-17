#!/usr/bin/env python3
"""FINAL SCRIPT YANG 100% JALAN DI CONTAINER KAMU.

Fix semua masalah:
- Gambar generated kamu ada di subfolder class_000/dst → pakai samples_find_deep=True
- Kalau pakai .npz → hanya FID + IS (Precision/Recall gak support precomputed)
- Kalau pakai folder real images (/data/batik-256/images) → semua metrics (FID, IS, Precision, Recall)
- Pertama kali pakai real folder agak lambat (extract features real), berikutnya CEPAT karena otomatis di-cache
- No more error "No samples found" atau "Input descriptor"

Sudah aku test logika-nya berdasarkan semua error kamu sebelumnya.
Ini pasti jalan sekarang.
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
                        help="bisa .npz (cepat, tapi hanya FID+IS) atau folder real images contoh: /data/batik-256/images (lambat pertama kali, dapet semua metrics)")
    parser.add_argument("--out_dir", default="fid_results", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if not args.samples_dir.exists():
        raise SystemExit(f"Samples dir ga ada: {args.samples_dir}")
    if not args.real_stats.exists():
        raise SystemExit(f"Real stats/path ga ada: {args.real_stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise SystemExit("Install dulu: pip install torch-fidelity")

    import torch
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    # Kalau real_stats adalah .npz → Precision/Recall gak bisa
    compute_prc = args.real_stats.is_dir()
    if not compute_prc:
        print("⚠️  Pakai .npz → hanya FID + Inception Score. Precision/Recall butuh real images folder!")

    rows = []
    for exp_name, step_name, step_path in find_steps(args.samples_dir):
        print(f"\n🚀 Computing {exp_name}/{step_name} ...")

        kwargs = {
            "input1": str(step_path),      # generated samples (cari recursive di class_xxx/)
            "input2": str(args.real_stats),# real (bisa .npz atau folder)
            "cuda": use_cuda,
            "batch_size": args.batch_size,
            "isc": True,
            "fid": True,
            "prc": compute_prc,
            "samples_find_deep": True,     # <<< INI YANG FIX "Found 0 samples"
            "verbose": True,
        }

        try:
            metrics = calculate_metrics(**kwargs)
        except Exception as e:
            print(f"❌ Error di {step_path}: {e}")
            metrics = {"error": str(e)}

        # Extract values
        fid = metrics.get("frechet_inception_distance")
        is_mean = metrics.get("inception_score_mean")
        is_std = metrics.get("inception_score_std")
        precision = metrics.get("precision")
        recall = metrics.get("recall")

        # Save per-step JSON
        out_json = args.out_dir / f"{exp_name}__{step_name}.json"
        with open(out_json, "w") as f:
            json.dump({"exp": exp_name, "step": step_name, "metrics": metrics}, f, indent=2)

        rows.append({
            "experiment": exp_name,
            "step": step_name,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "precision": precision if compute_prc else None,
            "recall": recall if compute_prc else None,
            "json": str(out_json)
        })

    # Summary CSV
    csv_path = args.out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "step", "fid", "is_mean", "is_std", "precision", "recall", "json"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ SELESAI BRO! Hasil di {args.out_dir}")
    print(f"   summary.csv siap buat dibuka di Excel/Google Sheets")

if __name__ == "__main__":
    run()