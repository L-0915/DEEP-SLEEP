#!/bin/bash
# DeepSleep Full Pipeline — Pretrain → SFT → DPO
#
# Usage:
#   bash scripts/run/run_all.sh
#
# This runs the complete three-stage training pipeline.
# Each stage loads from the previous stage's output.

set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=========================================="
echo " DeepSleep Full Training Pipeline"
echo "=========================================="

STAGE=${1:-all}

run_pretrain() {
    echo ""
    echo ">>> Stage 1/3: Pretrain <<<"
    bash scripts/run/run_pretrain.sh
    echo ">>> Pretrain done. <<<"
}

run_sft() {
    echo ""
    echo ">>> Stage 2/3: SFT <<<"
    FROM_WEIGHT=out/pretrain/final bash scripts/run/run_sft.sh
    echo ">>> SFT done. <<<"
}

run_dpo() {
    echo ""
    echo ">>> Stage 3/3: DPO <<<"
    SFT_CKPT=out/sft/final bash scripts/run/run_dpo.sh
    echo ">>> DPO done. <<<"
}

case $STAGE in
    pretrain) run_pretrain ;;
    sft)      run_sft ;;
    dpo)      run_dpo ;;
    all)
        run_pretrain
        run_sft
        run_dpo
        ;;
    *)
        echo "Usage: bash scripts/run/run_all.sh [pretrain|sft|dpo|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " All done!"
echo "=========================================="
