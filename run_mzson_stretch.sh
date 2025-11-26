#!/bin/bash

# 기본값 설정
CONFIG_FILE=${1:-"cfg/eval_mzson.yaml"}

SPLIT=${4:-"1"}

echo "Running mzson evaluation with the following parameters:"
echo "Config: $CONFIG_FILE"
echo "Split: $SPLIT"
echo "--------------------------------------------------"

python run_mzson_nomem_stretch_jy.py \
    -cf "$CONFIG_FILE" \
    --start_ratio "$START_RATIO" \
    --end_ratio "$END_RATIO" \
    --split "$SPLIT"
