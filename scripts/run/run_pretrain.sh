#!/bin/bash
# DeepSleep Pretrain — Bilingual (zh+en) from HuggingFace
#
# Quick start:
#   bash scripts/run/run_pretrain.sh
#
# Resume from checkpoint:
#   RESUME=out/pretrain/checkpoint-10000 bash scripts/run/run_pretrain.sh
#
# Logs: out/pretrain_train.log
# TensorBoard: tensorboard --logdir out/pretrain/runs

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- HuggingFace Mirror ----------
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$(pwd)/.hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---------- All config from YAML (overrides here if needed) ----------
CONFIG=${CONFIG:-configs/pretrain.yaml}
TOKENIZER=${TOKENIZER:-checkpoints/tokenizer}
RESUME=${RESUME:-}

# ---------- Run ----------
mkdir -p out/pretrain

CMD="python trainer/train_pretrain.py --config $CONFIG --tokenizer_path $TOKENIZER"
[ -n "$RESUME" ] && CMD="$CMD --resume_from_checkpoint $RESUME"

echo "========================================"
echo " DeepSleep Pretrain"
echo "========================================"
echo " Config:    $CONFIG"
echo " Tokenizer: $TOKENIZER"
echo " Resume:    ${RESUME:-none}"
echo " Log:       out/pretrain_train.log"
echo "========================================"

$CMD 2>&1 | tee out/pretrain_train.log
