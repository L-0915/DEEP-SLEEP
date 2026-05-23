#!/bin/bash
# DeepSleep DPO (Direct Preference Optimization)
#
# Quick start:
#   bash scripts/run/run_dpo.sh
#
# Custom SFT checkpoint:
#   SFT_CKPT=out/sft/final bash scripts/run/run_dpo.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Overrides ----------
CONFIG=${CONFIG:-configs/dpo.yaml}
TOKENIZER=${TOKENIZER:-checkpoints/tokenizer}
DATA=${DATA:-data/dpo/xiaoxi_dpo.jsonl}
SFT_CKPT=${SFT_CKPT:-out/sft/final}
OUTPUT=${OUTPUT:-out/dpo}
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-4}
LR=${LR:-5e-7}
BETA=${BETA:-0.1}
SEQ_LEN=${SEQ_LEN:-2048}

# ---------- Command ----------
CMD="python trainer/train_dpo.py \
  --config $CONFIG \
  --data_path $DATA \
  --tokenizer_path $TOKENIZER \
  --sft_checkpoint $SFT_CKPT \
  --save_dir $OUTPUT \
  --epochs $EPOCHS \
  --batch_size $BATCH \
  --learning_rate $LR \
  --dpo_beta $BETA \
  --max_seq_len $SEQ_LEN"

echo "========================================"
echo " DeepSleep DPO"
echo "========================================"
echo " Config:  $CONFIG"
echo " Data:    $DATA"
echo " From:    $SFT_CKPT"
echo " Output:  $OUTPUT"
echo " Beta:    $BETA"
echo "========================================"

eval $CMD
