#!/bin/bash
# scripts/run_ablation.sh
# Reproduce ablation study (Table 9 of the paper).
#
# Usage:
#   bash scripts/run_ablation.sh --dataset ICEWS14
#   bash scripts/run_ablation.sh --dataset YAGO

set -e

DATASET="ICEWS14"
GPU=0
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --gpu)     GPU="$2";     shift 2 ;;
        --seed)    SEED="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATA_DIR="data/$DATASET"
BASE_ARGS="--dataset $DATASET --data_dir $DATA_DIR --seed $SEED --gpu $GPU"

echo "======================================"
echo "Ablation study on $DATASET (seed=$SEED)"
echo "======================================"

# Full model (V4)
echo "[1/9] Full RHGNN-CA (V4)"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --save_path "experiments/ablation_full_${DATASET,,}.pt"

# w/o Hyperbolic geometry (replace with Euclidean)
echo "[2/9] w/o Hyperbolic"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --no_hyperbolic \
    --save_path "experiments/ablation_no_hyp_${DATASET,,}.pt"

# w/o History vocabulary (disable copy mechanism)
echo "[3/9] w/o History Vocab"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --no_history_vocab \
    --save_path "experiments/ablation_no_hist_${DATASET,,}.pt"

# w/o H-GRU (replace with simple update)
echo "[4/9] w/o H-GRU"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --no_hgru \
    --save_path "experiments/ablation_no_hgru_${DATASET,,}.pt"

# w/o ODE (disable continuous evolution)
echo "[5/9] w/o ODE"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --no_ode \
    --save_path "experiments/ablation_no_ode_${DATASET,,}.pt"

# w/o Subgraph (use 1-layer mean pooling)
echo "[6/9] w/o Subgraph"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --sgcn_layers 1 \
    --save_path "experiments/ablation_no_subgraph_${DATASET,,}.pt"

# w/o Soft Labels (smooth_label=0)
echo "[7/9] w/o Soft Labels"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --smooth_label 0.0 \
    --save_path "experiments/ablation_no_softlabel_${DATASET,,}.pt"

# w/o Contrastive loss
echo "[8/9] w/o Contrastive"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --lambda_contrast 0.0 \
    --save_path "experiments/ablation_no_contrast_${DATASET,,}.pt"

# w/o Temporal smoothness
echo "[9/9] w/o Temp. Smooth"
python scripts/train.py --model rhgnn_v4 $BASE_ARGS \
    --lambda_smooth 0.0 \
    --save_path "experiments/ablation_no_smooth_${DATASET,,}.pt"

echo ""
echo "Ablation complete. Results saved to experiments/"
