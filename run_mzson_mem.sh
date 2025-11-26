#!/bin/bash

# 기본값 설정
CONFIG_FILE=${1:-"cfg/eval_mzson.yaml"}
START_RATIO=${2:-"0.0"}
END_RATIO=${3:-"1.0"}
SPLIT=${4:-"1"}

echo "Running mzson mem evaluation with the following parameters:"
echo "Config: $CONFIG_FILE"
echo "Start Ratio: $START_RATIO"
echo "End Ratio: $END_RATIO"
echo "Split: $SPLIT"
echo "--------------------------------------------------"

python run_mzson_mem.py \
    -cf "$CONFIG_FILE" \
    --start_ratio "$START_RATIO" \
    --end_ratio "$END_RATIO" \
    --split "$SPLIT"

