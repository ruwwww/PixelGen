#!/bin/bash
# Generate FID samples for all checkpoints at 100k step intervals
# Usage: bash generate_fid_all_steps.sh <exp_name> [output_base_dir] [num_steps]
# Example: bash generate_fid_all_steps.sh jit_b_pixel_bs64_repa_4_lpips_pdino_adamw fid_samples 100

set -e

# Configuration
EXP_NAME="${1:-jit_b_pixel_bs64_repa_4_lpips_pdino_adamw}"
OUTPUT_BASE="${2:-fid_samples/${EXP_NAME}}"
NUM_STEPS="${3:-}"  # Optional: override sampling steps (NFE)
EXP_DIR="universal_pix_workdirs/exp_${EXP_NAME}"

# FID generation parameters
NUM_CLASSES=20
SAMPLES_PER_CLASS=1000
BATCH_SIZE=50
SEED=42
NUM_STEPS="${NUM_STEPS:-}"  # Optional: override sampling steps

# Checkpoint steps to generate (every 100k)
STEPS=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

echo "=============================================="
echo "FID Sample Generation Script"
echo "=============================================="
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint dir: ${EXP_DIR}"
echo "Output dir: ${OUTPUT_BASE}"
if [ -n "$NUM_STEPS" ]; then
  echo "Sampling steps (NFE): ${NUM_STEPS}"
else
  echo "Sampling steps: default (from config)"
fi
echo "Steps to process: ${STEPS[@]}"
echo "=============================================="
echo ""

# Check if experiment directory exists
if [ ! -d "${EXP_DIR}" ]; then
  echo "Error: Experiment directory not found: ${EXP_DIR}"
  exit 1
fi

# Create output base directory
mkdir -p "${OUTPUT_BASE}"

# Track statistics
TOTAL_STEPS=${#STEPS[@]}
COMPLETED=0
SKIPPED=0
FAILED=0

# Process each checkpoint
for step in "${STEPS[@]}"; do
  echo ""
  echo "================================================"
  echo "Processing step ${step}... (${COMPLETED}/${TOTAL_STEPS} completed)"
  echo "================================================"
  
  # Find the checkpoint file for this step
  CKPT=$(find "${EXP_DIR}" -name "epoch=*-step=${step}.ckpt" | head -n1)
  
  if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "⚠ WARNING: Checkpoint not found for step ${step}"
    ((SKIPPED++))
    continue
  fi
  
  echo "Found checkpoint: ${CKPT}"
  
  # Output directory for this step
  STEP_OUT="${OUTPUT_BASE}/step_${step}"
  
  # Check if already generated
  if [ -d "$STEP_OUT" ]; then
    NUM_EXISTING=$(find "$STEP_OUT" -name "*.png" 2>/dev/null | wc -l)
    EXPECTED=$((NUM_CLASSES * SAMPLES_PER_CLASS))
    
    if [ "$NUM_EXISTING" -eq "$EXPECTED" ]; then
      echo "✓ Already generated (found ${NUM_EXISTING} images)"
      ((COMPLETED++))
      continue
    else
      echo "Found incomplete generation (${NUM_EXISTING}/${EXPECTED} images), regenerating..."
      rm -rf "$STEP_OUT"
    fi
  fi
  
  # Run generation
  echo "Generating ${SAMPLES_PER_CLASS} samples per class for ${NUM_CLASSES} classes..."
  
# Build command with optional num_steps
    CMD="python3 generate_fid_samples.py \
      --ckpt \"$CKPT\" \
      --out \"$STEP_OUT\" \
      --num_classes \"$NUM_CLASSES\" \
      --samples_per_class \"$SAMPLES_PER_CLASS\" \
      --batch_size \"$BATCH_SIZE\" \
      --seed \"$SEED\""
    
    if [ -n "$NUM_STEPS" ]; then
      CMD="$CMD --num_steps \"$NUM_STEPS\""
    fi
    
    if eval "$CMD"; then
    
    # Verify output
    NUM_GENERATED=$(find "$STEP_OUT" -name "*.png" 2>/dev/null | wc -l)
    echo "✓ Generation complete! Generated ${NUM_GENERATED} images"
    ((COMPLETED++))
  else
    echo "✗ FAILED: Generation failed for step ${step}"
    ((FAILED++))
  fi
done

# Summary
echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Total steps: ${TOTAL_STEPS}"
echo "Completed: ${COMPLETED}"
echo "Skipped: ${SKIPPED}"
echo "Failed: ${FAILED}"
echo ""
echo "Output directory: ${OUTPUT_BASE}"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
  exit 1
fi
