"""
H-MRR: Hierarchical MRR — degree-weighted ranking metric.

Proposed in:
  "Curvature-Aware Recurrent Hyperbolic GNNs for TKG Reasoning"
  CIKM 2026 (Anonymous)

Standard MRR treats all entities equally, creating a systematic bias
against hyperbolic models that disproportionately benefit high-degree,
hierarchically central entities. H-MRR weights rankings by entity
connectivity (degree as hierarchy proxy):

    H-MRR = Σᵢ (wᵢ / ‖w‖₁) · (1/rankᵢ)
    wᵢ = deg(tᵢ) / maxⱼ deg(j)

This reveals that standard MRR underestimates RHGNN-CA's hierarchical
reasoning by 1.57–1.90× across datasets.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_h_mrr(
    rankings: torch.Tensor,
    entity_degrees: torch.Tensor,
    true_tails: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Hierarchical MRR (H-MRR).

    Args:
        rankings:        (N,) tensor — rank of each true tail entity
        entity_degrees:  (|E|,) tensor — degree of each entity in training graph
        true_tails:      (N,) tensor — indices of true tail entities (for degree lookup)
                         If None, assumes rankings already correspond to degrees.

    Returns:
        h_mrr: float — degree-weighted MRR score
    """
    if true_tails is not None:
        degrees = entity_degrees[true_tails].float()
    else:
        degrees = entity_degrees.float()

    max_deg = degrees.max().clamp(min=1.0)
    weights = degrees / max_deg                     # wᵢ = deg(tᵢ) / max_deg
    weights_norm = weights / weights.sum()          # normalize: wᵢ / ‖w‖₁

    reciprocal_ranks = 1.0 / rankings.float()
    h_mrr = (weights_norm * reciprocal_ranks).sum().item()
    return h_mrr


def compute_standard_mrr(rankings: torch.Tensor) -> float:
    """Standard filtered MRR."""
    return (1.0 / rankings.float()).mean().item()


def compute_h_mrr_ratio(
    rankings: torch.Tensor,
    entity_degrees: torch.Tensor,
    true_tails: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute standard MRR, H-MRR, and the ratio H-MRR/MRR.

    Returns:
        (mrr, h_mrr, ratio)
    """
    mrr   = compute_standard_mrr(rankings)
    h_mrr = compute_h_mrr(rankings, entity_degrees, true_tails)
    ratio = h_mrr / mrr if mrr > 0 else float('inf')
    return mrr, h_mrr, ratio


def build_entity_degrees(
    train_triples: List[Tuple],
    num_entities: int,
) -> torch.Tensor:
    """
    Build entity degree tensor from training triples.

    Args:
        train_triples:  list of (head, relation, tail, timestamp) tuples
        num_entities:   total number of entities |E|

    Returns:
        degrees: (|E|,) LongTensor — degree of each entity
    """
    degrees = torch.zeros(num_entities, dtype=torch.long)
    for h, r, t, *_ in train_triples:
        degrees[h] += 1
        degrees[t] += 1
    return degrees


class HierarchicalEvaluator:
    """
    Full evaluation suite: filtered MRR, Hits@K, H-MRR, and NHC.

    Usage:
        evaluator = HierarchicalEvaluator(dataset)
        results = evaluator.evaluate(model, split='test')
        print(results)
    """

    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.device  = device
        self.entity_degrees = build_entity_degrees(
            dataset.train_data, dataset.num_entities
        ).to(device)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        split: str = 'test',
        batch_size: int = 256,
        k_list: List[int] = [1, 3, 10],
        compute_delta: bool = False,
    ) -> Dict[str, float]:
        """
        Run full evaluation.

        Returns dict with keys:
            mrr, hits@1, hits@3, hits@10,
            h_mrr, h_mrr_ratio,
            nhc (Norm-Hierarchy Correlation)
        """
        model.eval()
        data = getattr(self.dataset, f'{split}_data')

        all_ranks     = []
        all_true_tails = []

        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start: batch_start + batch_size]
            heads, rels, tails, times = zip(*batch)
            heads = torch.tensor(heads, device=self.device)
            rels  = torch.tensor(rels,  device=self.device)
            tails = torch.tensor(tails, device=self.device)
            times = torch.tensor(times, device=self.device)

            scores = model.predict(heads, rels, times)  # (B, |E|)
            ranks  = self._filtered_rank(scores, tails, batch)
            all_ranks.append(ranks)
            all_true_tails.append(tails)

        all_ranks      = torch.cat(all_ranks)
        all_true_tails = torch.cat(all_true_tails)

        mrr, h_mrr, ratio = compute_h_mrr_ratio(
            all_ranks, self.entity_degrees, all_true_tails
        )

        results = {
            'mrr':        mrr,
            'h_mrr':      h_mrr,
            'h_mrr_ratio': ratio,
        }
        for k in k_list:
            results[f'hits@{k}'] = (all_ranks <= k).float().mean().item()

        results['nhc'] = self._norm_hierarchy_corr(model)
        return results

    def _filtered_rank(self, scores, true_tails, batch):
        """Compute filtered ranks — exclude other true answers."""
        ranks = []
        for i, (h, r, t, ts, *_) in enumerate(batch):
            s = scores[i].clone()
            true_set = self.dataset.true_tails.get((h, r, ts), set())
            for t2 in true_set:
                if t2 != t:
                    s[t2] = -1e9
            rank = (s >= s[t]).sum().item()
            ranks.append(rank)
        return torch.tensor(ranks, dtype=torch.long)

    def _norm_hierarchy_corr(self, model) -> float:
        """
        NHC: Norm-Hierarchy Correlation.
        Pearson correlation between entity embedding norm and degree.
        Positive NHC = high-degree entities correctly placed further from origin.
        """
        embs   = model.entity_embeddings.weight.detach()
        norms  = embs.norm(dim=-1).cpu().numpy()
        degrees = self.entity_degrees.cpu().numpy().astype(float)
        corr = np.corrcoef(norms, degrees)[0, 1]
        return float(corr)
