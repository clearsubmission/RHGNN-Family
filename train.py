"""
scripts/train.py — Main training entry point for RHGNN model family.

Usage:
    python scripts/train.py --model rhgnn_v4 --dataset ICEWS14 --gpu 0
    python scripts/train.py --config configs/icews14.yaml --model rhgnn_v4
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from data.dataset import TKGDataset
from utils.trainer import Trainer
from evaluation.h_mrr import HierarchicalEvaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train RHGNN model family for TKG reasoning'
    )

    # Model
    parser.add_argument('--model', type=str, default='rhgnn_v4',
        choices=['rhgnn_v1a','rhgnn_v1b','rhgnn_v2','rhgnn_v3',
                 'rhgnn_v4','rhgnn_v5',
                 'rhgnn_hlstm','rhgnn_hgru','rhgnn_fa',
                 'rhgnn_ma','rhgnn_ca','rhgnn_fh'],
        help='Model version to train')

    # Data
    parser.add_argument('--dataset', type=str, default='ICEWS14',
        choices=['ICEWS14','ICEWS18','ICEWS05-15','WIKI','YAGO','GDELT'])
    parser.add_argument('--data_dir', type=str, default='data/ICEWS14')

    # Architecture
    parser.add_argument('--dim', type=int, default=200,
        help='Embedding dimension')
    parser.add_argument('--ode_steps', type=int, default=5,
        help='Number of Euler steps for Neural ODE')
    parser.add_argument('--curvature', type=float, default=1.0,
        help='Initial global curvature (V4/V5 learn per-relation offsets)')

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3,
        help='Learning rate for RiemannianAdam')
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--lr_decay_every', type=int, default=50)
    parser.add_argument('--patience', type=int, default=60,
        help='Early stopping patience (epochs)')
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--eval_batch', type=int, default=256)

    # Loss (V4/V5)
    parser.add_argument('--smooth_label', type=float, default=0.1,
        help='Label smoothing ε for KL loss')
    parser.add_argument('--lambda_contrast', type=float, default=0.1,
        help='InfoNCE loss weight α')
    parser.add_argument('--contrast_temp', type=float, default=0.07,
        help='Contrastive temperature τ_c')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
        help='Temporal smoothness regularizer λ_s')
    parser.add_argument('--num_negatives', type=int, default=16,
        help='Number of negatives for InfoNCE')

    # Misc
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
        help='Path to YAML config (overrides CLI args)')
    parser.add_argument('--no_h_mrr', action='store_true',
        help='Skip H-MRR computation during evaluation')

    args = parser.parse_args()

    # Load config file if provided (CLI args take precedence)
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    # Auto save path
    if args.save_path is None:
        os.makedirs('experiments', exist_ok=True)
        args.save_path = f'experiments/{args.model}_{args.dataset.lower()}_seed{args.seed}.pt'

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Model: {args.model} | Dataset: {args.dataset}')

    # Load dataset
    dataset = TKGDataset(args.data_dir, args.dataset)
    print(f'Entities={dataset.num_entities} | Relations={dataset.num_relations}')
    print(f'Train={len(dataset.train_data)} | Valid={len(dataset.valid_data)} | Test={len(dataset.test_data)}')

    # Build model
    ModelClass = get_model(args.model)
    model = ModelClass(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        dim=args.dim,
        curvature=args.curvature,
        ode_steps=args.ode_steps,
        smooth_label=args.smooth_label,
        lambda_contrast=args.lambda_contrast,
        contrast_temp=args.contrast_temp,
        lambda_smooth=args.lambda_smooth,
        num_negatives=args.num_negatives,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {params/1e6:.1f}M')

    # Evaluator
    evaluator = HierarchicalEvaluator(dataset, device=device)

    # Trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        args=args,
        device=device,
    )

    # Train
    best_results = trainer.train()

    # Final test evaluation
    print('\n=== Final Test Evaluation ===')
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    test_results = evaluator.evaluate(model, split='test')

    print(f"MRR:        {test_results['mrr']:.4f}")
    print(f"H@1:        {test_results['hits@1']:.4f}")
    print(f"H@3:        {test_results['hits@3']:.4f}")
    print(f"H@10:       {test_results['hits@10']:.4f}")
    if 'h_mrr' in test_results:
        print(f"H-MRR:      {test_results['h_mrr']:.4f}")
        print(f"H-MRR/MRR:  {test_results['h_mrr_ratio']:.2f}×")
        print(f"NHC:        {test_results['nhc']:.4f}")


if __name__ == '__main__':
    main()
