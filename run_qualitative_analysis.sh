#!/bin/bash
# run_qualitative_analysis.sh
# Usage: ./run_qualitative_analysis.sh <CHECKPOINT_PATH> <CONFIG_PATH> <TRAINING_DATA_DIR> <GENERATED_SAMPLES_DIR>

set -e

CKPT=$1
CONFIG=$2
TRAIN_DIR=$3
GEN_DIR=$4

if [ -z "$CKPT" ] || [ -z "$CONFIG" ] || [ -z "$TRAIN_DIR" ] || [ -z "$GEN_DIR" ]; then
    echo "Usage: $0 <CHECKPOINT_PATH> <CONFIG_PATH> <TRAINING_DATA_DIR> <GENERATED_SAMPLES_DIR>"
    echo "Example: $0 checkpoints/last.ckpt configs_c2i/PixelGen_XL.yaml data/train_images outdir/samples"
    exit 1
fi

# Ensure output directories exist
mkdir -p qualitative_analysis/texture_fidelity
mkdir -p qualitative_analysis/structural_coherence
mkdir -p qualitative_analysis/nearest_neighbors

echo "Installing dependencies..."
pip install scikit-learn matplotlib diffusers torchvision

echo "=================================================="
echo "Running Texture Fidelity Analysis (VAE Reconstruction)..."
echo "=================================================="
# We use the training directory to check how well the VAE reconstructs real images
python sample_diagnostic_sweep.py texture \
    --image_folder "$TRAIN_DIR" \
    --out_dir "qualitative_analysis/texture_fidelity" \
    --num_images 8

echo "=================================================="
echo "Running Structural Coherence Analysis (Attention Maps)..."
echo "=================================================="
python sample_diagnostic_sweep.py attention \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --out_dir "qualitative_analysis/structural_coherence"

echo "=================================================="
echo "Running Nearest Neighbor Analysis (Novelty Check)..."
echo "=================================================="
python sample_diagnostic_sweep.py nn \
    --gen_dir "$GEN_DIR" \
    --train_dir "$TRAIN_DIR" \
    --out_dir "qualitative_analysis/nearest_neighbors"

echo "=================================================="
echo "Analysis Complete. Results in qualitative_analysis/"
echo "=================================================="
