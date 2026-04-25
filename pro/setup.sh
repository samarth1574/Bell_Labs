#!/usr/bin/env bash
# ============================================================
# setup.sh — Turn-Key Reproducibility Script
# High-Density Object Segmentation with Soft-NMS (Phase 3)
# ============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs all dependencies
#   3. Generates the synthetic dataset
#   4. Prepares SKU-110K annotations (25% subset)
#   5. Runs unit tests
#   6. Prints next-step commands for training
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  High-Density Object Segmentation — Setup Script"
echo "============================================================"
echo ""

# ---- 1. Virtual Environment ----
if [ ! -d "venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/6] Virtual environment already exists."
fi

echo "[1/6] Activating virtual environment..."
source venv/bin/activate

# ---- 2. Install Dependencies ----
echo "[2/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q
echo "       ✓ Dependencies installed."

# ---- 3. Generate Synthetic Dataset ----
echo "[3/6] Generating synthetic dataset (500 images)..."
if [ -f "data/synthetic/annotations.json" ]; then
    echo "       ✓ Synthetic dataset already exists, skipping."
else
    python -m src.synthetic_generator
    echo "       ✓ Synthetic dataset generated."
fi

# ---- 4. Prepare SKU-110K (25% subset) ----
echo "[4/6] Preparing SKU-110K annotations (25% subset)..."
if [ -f "data/sku110k/sku110k.yaml" ]; then
    echo "       ✓ SKU-110K already prepared, skipping."
    echo "       (Run with --force to regenerate: python scripts/download_sku110k.py --download annotations --fraction 0.25 --force)"
else
    python scripts/download_sku110k.py \
        --download annotations \
        --fraction 0.25 \
        --root data/sku110k
    echo "       ✓ SKU-110K annotations prepared (25% subset)."
fi

# ---- 5. Run Unit Tests ----
echo "[5/6] Running unit tests..."
echo ""
echo "  --- Soft-NMS Tests ---"
python -m src.models.soft_nms
echo ""
echo "  --- Evaluation Metrics Tests ---"
python -m src.evaluation.metrics --verbose
echo ""
echo "       ✓ All tests passed."

# ---- 6. Summary ----
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  # Activate the environment"
echo "  source venv/bin/activate"
echo ""
echo "  # Train YOLO11 on SKU-110K (25% subset)"
echo "  python scripts/train_dl_sku110k.py --mode train \\"
echo "    --config configs/phase3_hybrid_yolo11.yaml"
echo ""
echo "  # Validate trained weights"
echo "  python scripts/train_dl_sku110k.py --mode val \\"
echo "    --weights runs/phase3/yolo11_sku110k/weights/best.pt"
echo ""
echo "  # Run Phase 3 ablations"
echo "  python -m src.models.hybrid_sku_detector \\"
echo "    --weights runs/phase3/yolo11_sku110k/weights/best.pt \\"
echo "    --mode all --limit 200"
echo ""
echo "  # Run baseline experiments on synthetic data"
echo "  python experiments/run_baselines.py --verbose"
echo ""
echo "============================================================"
