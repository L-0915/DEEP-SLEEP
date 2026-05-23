#!/bin/bash
# DeepSleep SFT (Supervised Fine-Tuning)
#
# Quick start:
#   bash scripts/run/run_sft.sh
#
# Custom pretrain checkpoint:
#   FROM_WEIGHT=out/pretrain/final bash scripts/run/run_sft.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Overrides ----------
CONFIG=${CONFIG:-configs/sft.yaml}
TOKENIZER=${TOKENIZER:-checkpoints/tokenizer}
DATA=${DATA:-data/sft/xiaoxi/xiaoxi_sft.jsonl}
FROM_WEIGHT=${FROM_WEIGHT:-out/pretrain/final}
OUTPUT=${OUTPUT:-out/sft}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-16}
LR=${LR:-1e-5}
SEQ_LEN=${SEQ_LEN:-2048}

# ---------- Command ----------
CMD="python trainer/train_sft.py \
  --config $CONFIG \
  --data_path $DATA \
  --tokenizer_path $TOKENIZER \
  --from_weight $FROM_WEIGHT \
  --save_dir $OUTPUT \
  --epochs $EPOCHS \
  --batch_size $BATCH \
  --learning_rate $LR \
  --max_seq_len $SEQ_LEN"

echo "========================================"
echo " DeepSleep SFT"
echo "========================================"
echo " Config:  $CONFIG"
echo " Data:    $DATA"
echo " From:    $FROM_WEIGHT"
echo " Output:  $OUTPUT"
echo "========================================"

eval $CMD
