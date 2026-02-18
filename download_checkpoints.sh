#!/bin/bash
# Download checkpoints from HuggingFace for specific experiment and steps
# Usage: bash download_checkpoints.sh <exp_name> <hf_repo> [output_dir] [steps...]
# Example: bash download_checkpoints.sh jit_b_pixel_bs64_adamw ruwwww/batik-pixelgen-exp ./universal_pix_workdirs/jit_b_pixel_bs64_adamw 100000 200000 300000 400000 500000

set -e

# Configuration
EXP_NAME="${1:-jit_b_pixel_bs64_adamw}"
HF_REPO="${2:-ruwwww/batik-pixelgen-exp}"
OUTPUT_DIR="${3:-./universal_pix_workdirs/${EXP_NAME}}"

# Default steps if not provided
if [ $# -gt 3 ]; then
    # Use provided steps
    STEPS=("${@:4}")
else
    # Default: 100k to 500k at 100k intervals
    STEPS=(100000 200000 300000 400000 500000)
fi

echo "=============================================="
echo "Download Checkpoints from HuggingFace"
echo "=============================================="
echo "Experiment: ${EXP_NAME}"
echo "HF Repo: ${HF_REPO}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Steps to download: ${STEPS[@]}"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Track statistics
TOTAL_STEPS=${#STEPS[@]}
DOWNLOADED=0
FAILED=0

# Download each checkpoint
for step in "${STEPS[@]}"; do
  echo ""
  echo "================================================"
  echo "Downloading step ${step}... (${DOWNLOADED}/${TOTAL_STEPS} downloaded)"
  echo "================================================"

  # Run download
  if python3 scripts/hf_sync.py download \
    --hf-repo "${HF_REPO}" \
    --exp-name "${EXP_NAME}" \
    --step "${step}" \
    --out-dir "${OUTPUT_DIR}"; then

    echo "✓ Successfully downloaded step ${step}"
    ((DOWNLOADED++))
  else
    echo "✗ FAILED to download step ${step}"
    ((FAILED++))
  fi
done

# Summary
echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo "Total steps: ${TOTAL_STEPS}"
echo "Downloaded: ${DOWNLOADED}"
echo "Failed: ${FAILED}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
  echo "⚠ Some downloads failed. Check the output above for details."
  exit 1
else
  echo "✅ All downloads completed successfully!"
fi