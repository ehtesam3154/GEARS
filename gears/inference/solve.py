"""
Global 2D solve from a stitched edge-distance graph.

After patchwise geometry is aggregated into a single set of weighted edge
distances over the full cell set, two stages recover 2D coordinates:

    1. Landmark Isomap initialisation -- geodesic MDS on a landmark subset,
       then trilateration of the remaining points, giving a coarse global
       layout X_init.
    2. Weighted-Huber distance-geometry refinement -- Adam optimisation of

           sum_ij w_ij * huber(|X_i - X_j| - d_ij)  +  lambda * |X - X_init|^2

       where the anchor term (lambda) pulls toward the init and decays over
       iterations, and the Huber loss makes the fit robust to outlier edges.

The refinement returns the final coordinates together with the stress
trajectory used to monitor convergence.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def landmark_isomap_init_v2(
    edges: List[Tuple[int, int]],
    distances: List[float],
    N: int,
    n_landmarks: int = 256,
    device: str = 'cuda',
    DEBUG_FLAG: bool = True,
    clip_runaway: bool = True,
) -> torch.Tensor:
    """
    Initialize 2D coordinates via Landmark Isomap.

    1. Build graph with edge distances
    2. Compute geodesic distances from landmarks (Dijkstra)
    3. Embed landmarks via classical MDS
    4. Place non-landmarks by trilateration

    Returns:
        X_init: (N, 2) initial coordinates
    """
    if DEBUG_FLAG:
        print(f"\n[ISOMAP-INIT] Landmark Isomap initialization: N={N}, landmarks={n_landmarks}")

    # Build sparse adjacency matrix
    row, col, data = [], [], []
    for (i, j), d in zip(edges, distances):
        row.extend([i, j])
        col.extend([j, i])
        data.extend([d, d])

    adj_sparse = csr_matrix((data, (row, col)), shape=(N, N))

    # Select landmarks (high-degree nodes + random)
    degrees = np.array(adj_sparse.getnnz(axis=1))
    n_landmarks = min(n_landmarks, N)

    # Top degree nodes as landmarks
    n_degree = n_landmarks // 2
    high_degree_nodes = np.argsort(degrees)[-n_degree:]

    # Random nodes for coverage
    remaining = set(range(N)) - set(high_degree_nodes)
    n_random = n_landmarks - n_degree
    random_nodes = np.random.choice(list(remaining), size=min(n_random, len(remaining)), replace=False)

    landmarks = np.concatenate([high_degree_nodes, random_nodes])
    landmarks = np.unique(landmarks)
    L = len(landmarks)

    if DEBUG_FLAG:
        print(f"[ISOMAP-INIT] Selected {L} landmarks ({n_degree} high-degree, {len(random_nodes)} random)")

    # Compute geodesic distances from landmarks
    D_geo_landmarks = dijkstra(adj_sparse, indices=landmarks, directed=False)

    # Handle disconnected nodes
    inf_mask = np.isinf(D_geo_landmarks)
    if inf_mask.any():
        n_disconnected = inf_mask.any(axis=0).sum()
        if DEBUG_FLAG:
            print(f"[ISOMAP-INIT] WARNING: {n_disconnected} nodes disconnected, using fallback distance")
        max_finite = np.nanmax(D_geo_landmarks[~inf_mask])
        D_geo_landmarks[inf_mask] = max_finite * 2

    # Classical MDS on landmarks
    D_landmarks = D_geo_landmarks[:, landmarks]  # (L, L) landmark-to-landmark

    # Double centering
    n = D_landmarks.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_landmarks ** 2) @ H

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top 2 positive eigenvalues
    pos_mask = eigvals > 1e-8
    if pos_mask.sum() < 2:
        if DEBUG_FLAG:
            print("[ISOMAP-INIT] WARNING: Less than 2 positive eigenvalues, using fallback")
        X_landmarks = np.random.randn(L, 2) * 0.1
    else:
        eigvals_pos = eigvals[pos_mask][:2]
        eigvecs_pos = eigvecs[:, pos_mask][:, :2]
        X_landmarks = eigvecs_pos @ np.diag(np.sqrt(eigvals_pos))

    if DEBUG_FLAG:
        print(f"[ISOMAP-INIT] Landmark embedding range: [{X_landmarks.min():.3f}, {X_landmarks.max():.3f}]")

    # Place non-landmarks by trilateration (least squares to landmarks)
    X_all = np.zeros((N, 2))
    X_all[landmarks] = X_landmarks

    non_landmarks = np.setdiff1d(np.arange(N), landmarks)

    for i in non_landmarks:
        # Get distances to landmarks
        d_to_landmarks = D_geo_landmarks[:, i]  # (L,)

        # Weighted least squares (weight by 1/d^2)
        weights = 1.0 / (d_to_landmarks ** 2 + 1e-8)
        weights = weights / weights.sum()

        # Initial guess: weighted average of landmark positions
        X_all[i] = (weights[:, None] * X_landmarks).sum(axis=0)

    # Refine non-landmarks with a few optimization steps
    # (Simple gradient descent on distance errors)
    X_all = _refine_trilateration(X_all, landmarks, D_geo_landmarks, n_iters=50)

    # Robust outlier clip: disconnected / poorly-trilaterated cells land far from
    # the cluster (geodesic fallback = 2x max distance) and, left in, dominate the
    # scale and drag the anchored Huber solve into a blob. Cap each cell's radius
    # from the median center at the p99 radius (direction preserved).
    if clip_runaway:
        ctr = np.median(X_all, axis=0)
        r = np.linalg.norm(X_all - ctr, axis=1)
        r_cap = np.percentile(r, 99)
        far = r > r_cap
        if far.any():
            X_all[far] = ctr + (X_all[far] - ctr) * (r_cap / (r[far][:, None] + 1e-8))
            if DEBUG_FLAG:
                print(f"[ISOMAP-INIT] clipped {int(far.sum())} runaway points to r<= {r_cap:.3f}")

    # Center and scale
    X_all = X_all - X_all.mean(axis=0)
    scale = np.sqrt((X_all ** 2).mean())
    if scale > 1e-8:
        X_all = X_all / scale

    if DEBUG_FLAG:
        print(f"[ISOMAP-INIT] Final init range: [{X_all.min():.3f}, {X_all.max():.3f}]")

    return torch.tensor(X_all, dtype=torch.float32, device=device)


def _refine_trilateration(X, landmarks, D_geo_landmarks, n_iters=50, lr=0.1):
    """Refine non-landmark positions via gradient descent on distance errors."""
    N = X.shape[0]
    L = len(landmarks)
    non_landmarks = np.setdiff1d(np.arange(N), landmarks)

    X = X.copy()

    for _ in range(n_iters):
        for i in non_landmarks:
            # Compute gradient of squared distance errors to landmarks
            diff = X[i:i+1] - X[landmarks]  # (L, 2)
            d_current = np.linalg.norm(diff, axis=1)  # (L,)
            d_target = D_geo_landmarks[:, i]  # (L,)

            # Gradient: 2 * (d_current - d_target) * (x_i - x_l) / d_current
            errors = d_current - d_target
            grad = 2 * (errors[:, None] * diff / (d_current[:, None] + 1e-8)).mean(axis=0)

            X[i] -= lr * grad

    return X


def global_distance_geometry_solve_v2(
    edges: List[Tuple[int, int]],
    distances: List[float],
    weights: List[float],
    N: int,
    X_init: torch.Tensor,
    n_iters: int = 1000,
    lr: float = 0.01,
    huber_delta: float = 0.1,
    anchor_lambda: float = 0.1,
    anchor_decay: float = 0.995,
    log_every: int = 100,
    device: str = 'cuda',
    DEBUG_FLAG: bool = True,
) -> Dict[str, Any]:
    """
    Global 2D solve via distance geometry optimization.

    Minimizes:
        sum_ij w_ij * huber(|X_i - X_j| - d_ij) + lambda * |X - X_init|^2

    Returns:
        dict with X_final, stress history, and diagnostics
    """
    if DEBUG_FLAG:
        print(f"\n[GLOBAL-SOLVE] Distance geometry optimization: N={N}, edges={len(edges)}, iters={n_iters}")

    # Convert to tensors
    edge_i = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    edge_j = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    d_target = torch.tensor(distances, dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / (w.sum() + 1e-8)  # Normalize

    # Initialize - need to enable grad even if called inside no_grad context
    X_anchor = X_init.clone().detach()
    stress_history = []

    # Use enable_grad to allow optimization even when called from no_grad context
    with torch.enable_grad():
        X = X_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=lr)
        current_anchor_lambda = anchor_lambda

        for it in range(n_iters):
            optimizer.zero_grad()

            # Compute pairwise distances for edges
            diff = X[edge_i] - X[edge_j]  # (E, 2)
            d_pred = diff.norm(dim=1)  # (E,)

            # Huber loss
            residuals = d_pred - d_target
            abs_res = residuals.abs()
            huber = torch.where(
                abs_res <= huber_delta,
                0.5 * residuals ** 2,
                huber_delta * (abs_res - 0.5 * huber_delta)
            )

            # Weighted stress
            stress = (w * huber).sum()

            # Anchor regularization (decays over time)
            anchor_loss = current_anchor_lambda * ((X - X_anchor) ** 2).mean()

            loss = stress + anchor_loss
            loss.backward()
            optimizer.step()

            # Decay anchor weight
            current_anchor_lambda *= anchor_decay

            stress_val = stress.item()
            stress_history.append(stress_val)

            if DEBUG_FLAG and (it % log_every == 0 or it == n_iters - 1):
                mean_residual = residuals.abs().mean().item()
                max_residual = residuals.abs().max().item()
                print(f"[GLOBAL-SOLVE] iter={it:4d} stress={stress_val:.6f} anchor={anchor_loss.item():.6f} "
                      f"mean_res={mean_residual:.4f} max_res={max_residual:.4f}")

    X_final = X.detach()

    # Center and compute final statistics
    X_final = X_final - X_final.mean(dim=0)

    # Final residuals
    with torch.no_grad():
        diff = X_final[edge_i] - X_final[edge_j]
        d_final = diff.norm(dim=1)
        residuals = (d_final - d_target).abs()

    diagnostics = {
        'final_stress': stress_history[-1],
        'initial_stress': stress_history[0],
        'mean_residual': residuals.mean().item(),
        'max_residual': residuals.max().item(),
        'median_residual': residuals.median().item(),
    }

    if DEBUG_FLAG:
        print(f"[GLOBAL-SOLVE] Final: stress={diagnostics['final_stress']:.6f}, "
              f"mean_res={diagnostics['mean_residual']:.4f}, max_res={diagnostics['max_residual']:.4f}")

    return {
        'X_final': X_final,
        'stress_history': stress_history,
        'diagnostics': diagnostics,
    }
