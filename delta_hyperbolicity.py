"""
δ-Hyperbolicity: Gromov's δ measures how tree-like an embedding space is.

Lower δ = more tree-like (hyperbolic) structure.
Applied to TKG embeddings to quantify hierarchical geometry capture.

Proposed in:
  "Curvature-Aware Recurrent Hyperbolic GNNs for TKG Reasoning"
  CIKM 2026 (Anonymous)

Reference:
  Gromov, M. (1987). Hyperbolic groups. Essays in group theory.
"""

import torch
import numpy as np
from itertools import combinations


def gromov_product(d_xy, d_xz, d_yz, base=0):
    """
    Gromov product (y|z)_x = 0.5 * (d(x,y) + d(x,z) - d(y,z))
    Here base is fixed as origin (0).
    """
    return 0.5 * (d_xy + d_xz - d_yz)


def compute_delta_hyp(
    embeddings: torch.Tensor,
    sample_size: int = 500,
    distance_fn: str = 'hyperbolic',
    curvature: float = 1.0,
    seed: int = 42,
) -> float:
    """
    Compute Gromov δ-hyperbolicity of embedding space.

    δ = max over all 4-tuples (x,y,z,w) of:
        max(d(x,z)+d(y,w), d(x,w)+d(y,z)) - (d(x,y)+d(z,w))
        divided by 2

    Args:
        embeddings:   (N, d) tensor of entity embeddings
        sample_size:  number of entities to sample (computational limit)
        distance_fn:  'hyperbolic' (geodesic) or 'euclidean'
        curvature:    curvature c for hyperbolic distance
        seed:         random seed for sampling

    Returns:
        delta: float — Gromov δ-hyperbolicity value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = embeddings.shape[0]
    n = min(sample_size, N)
    idx = torch.randperm(N)[:n]
    embs = embeddings[idx].detach().cpu().float()

    # Compute pairwise distances
    D = _pairwise_distances(embs, distance_fn, curvature)

    # Compute δ over random 4-tuples
    delta_max = 0.0
    indices = list(range(n))

    # Sample 4-tuples (full enumeration too expensive for large n)
    num_quads = min(50000, n * (n-1) * (n-2) * (n-3) // 24)
    quad_indices = _sample_quadruples(n, num_quads, seed)

    for x, y, z, w in quad_indices:
        s1 = D[x, z] + D[y, w]
        s2 = D[x, w] + D[y, z]
        s3 = D[x, y] + D[z, w]

        # δ for this 4-tuple
        m1, m2 = sorted([s1, s2, s3], reverse=True)[:2]
        delta = (m1 - m2) / 2.0
        if delta > delta_max:
            delta_max = delta

    return float(delta_max)


def _pairwise_distances(
    embs: torch.Tensor,
    mode: str,
    c: float,
) -> np.ndarray:
    """Compute (n, n) pairwise distance matrix."""
    n = embs.shape[0]
    D = np.zeros((n, n), dtype=np.float32)

    if mode == 'euclidean':
        diff = embs.unsqueeze(0) - embs.unsqueeze(1)  # (n, n, d)
        D = diff.norm(dim=-1).numpy()

    elif mode == 'hyperbolic':
        sqrt_c = c ** 0.5
        # d_c(x,y) = (2/√c) atanh(√c ‖-x ⊕_c y‖)
        for i in range(n):
            for j in range(i+1, n):
                x, y = embs[i], embs[j]
                mob = _mobius_add(-x, y, c)
                norm_mob = mob.norm().clamp(max=1.0 - 1e-5)
                d = (2.0 / sqrt_c) * torch.atanh(sqrt_c * norm_mob)
                D[i, j] = D[j, i] = d.item()
    else:
        raise ValueError(f"Unknown distance_fn: {mode}")

    return D


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Möbius addition x ⊕_c y."""
    x2 = (x * x).sum()
    y2 = (y * y).sum()
    xy = (x * y).sum()
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c**2 * x2 * y2
    return num / denom.clamp(min=1e-15)


def _sample_quadruples(n: int, num: int, seed: int):
    """Sample random 4-tuples of distinct indices."""
    np.random.seed(seed)
    quads = set()
    attempts = 0
    while len(quads) < num and attempts < num * 10:
        sample = tuple(sorted(np.random.choice(n, 4, replace=False)))
        quads.add(sample)
        attempts += 1
    return list(quads)


def dataset_delta_hyperbolicity(
    train_triples,
    num_entities: int,
    sample_size: int = 500,
    seed: int = 42,
) -> float:
    """
    Compute δ-hyperbolicity of the raw graph structure
    (not embeddings) using shortest-path distances.

    Used to characterize dataset hierarchy in Table 13 of the paper.
    """
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    # Build adjacency matrix
    rows, cols = [], []
    for h, r, t, *_ in train_triples:
        rows.extend([h, t])
        cols.extend([t, h])

    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(num_entities, num_entities)
    )

    # Sample nodes
    np.random.seed(seed)
    idx = np.random.choice(num_entities, min(sample_size, num_entities), replace=False)

    # Compute shortest path distances
    D_full = csg.shortest_path(adj, indices=idx, directed=False)
    D_full[D_full == np.inf] = 0  # disconnected = 0

    n = len(idx)
    D = D_full[:, idx].astype(np.float32)

    # Compute δ
    delta_max = 0.0
    quads = _sample_quadruples(n, min(50000, n**2), seed)
    for x, y, z, w in quads:
        s1 = D[x, z] + D[y, w]
        s2 = D[x, w] + D[y, z]
        s3 = D[x, y] + D[z, w]
        m1, m2 = sorted([s1, s2, s3], reverse=True)[:2]
        delta_max = max(delta_max, (m1 - m2) / 2.0)

    return float(delta_max)
