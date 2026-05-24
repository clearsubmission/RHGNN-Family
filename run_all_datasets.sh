#!/bin/bash
# scripts/run_all_datasets.sh
# Reproduce all RHGNN-CA (V4) results from Table 7 & 8 of the paper.
#
# Usage:
#   bash scripts/run_all_datasets.sh              # all datasets
#   bash scripts/run_all_datasets.sh ICEWS14      # single dataset

set -e

MODEL="rhgnn_v4"
SEEDS="42 123 456"
DATASETS=("ICEWS14" "ICEWS18" "ICEWS05-15" "WIKI" "YAGO" "GDELT")

# If dataset argument provided, run only that one
if [ $# -ge 1 ]; then
    DATASETS=("$1")
fi

GPU=0

for DATASET in "${DATASETS[@]}"; do
    CONFIG="configs/$(echo $DATASET | tr '[:upper:]' '[:lower:]' | tr '-' '_').yaml"
    DATA_DIR="data/$DATASET"

    echo "======================================"
    echo "Training $MODEL on $DATASET"
    echo "======================================"

    for SEED in $SEEDS; do
        echo "--- Seed: $SEED ---"
        python scripts/train.py \
            --model $MODEL \
            --dataset $DATASET \
            --data_dir $DATA_DIR \
            --seed $SEED \
            --gpu $GPU \
            --save_path "experiments/${MODEL}_${DATASET,,}_seed${SEED}.pt"
    done

    echo "--- $DATASET complete ---"
    echo ""
done

echo "All experiments complete."
echo "Results saved to experiments/"
