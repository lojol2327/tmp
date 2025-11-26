#!/bin/bash

# 기본값 설정
CONFIG_FILE=${1:-"cfg/eval_aeqa_ours.yaml"}
START_RATIO=${2:-"0.0"}
END_RATIO=${3:-"1.0"}

echo "Running AEQA MZSON evaluation with the following parameters:"
echo "Config: $CONFIG_FILE"
echo "Start Ratio: $START_RATIO"
echo "End Ratio: $END_RATIO"
echo "--------------------------------------------------"

python run_aeqa_nomem.py \
    -cf "$CONFIG_FILE" \
    --start_ratio "$START_RATIO" \
    --end_ratio "$END_RATIO"
