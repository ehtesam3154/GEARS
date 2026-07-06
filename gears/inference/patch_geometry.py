"""
Per-patch geometry reconstruction for the distance-first inference pipeline.

Given a small patch of encoder embeddings (a subset of dissociated cells or
spots sampled from the locality graph), this module runs the reverse
residual-diffusion sampler to recover a low-dimensional geometry V_pred for the
patch, then converts that geometry into a sparse set of pairwise distance
measurements over a within-patch kNN edge set.

Reverse sampler (residual mode):
    H       = context_encoder(Z_patch, mask)     -- per-point conditioning
    V_base  = generator(H, mask)                 -- base coordinates proposal
    R_start = sigma_start * eps                  -- residual initialised at noise
    R_0     = EDM Heun sampling of the residual over the Karras sigma schedule,
              denoised by score_net.forward_edm with two-pass self-conditioning
    V_pred  = V_base + R_0

The residual (not the absolute coordinate) is what the score network denoises,
so the geometry is composed as V_base + R at the end.
"""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from gears.models import DiffusionScoreNet


def edm_sigma_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    EDM (Karras) sigma schedule for reverse sampling.

        sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Returns:
        sigmas: (num_steps + 1,) decreasing from sigma_max to sigma_min, then 0.
        The trailing 0 drives the final exact denoising step.
    """
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1.0 / rho) +
        step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, dtype=torch.float64, device=device)])
    return t_steps.to(torch.float32)


def sample_patch_residual_diffusion_v2(
    Z_patch: torch.Tensor,        # (1, m, h) or (m, h)
    mask_patch: torch.Tensor,     # (1, m) or (m,)
    context_encoder: nn.Module,
    generator: nn.Module,
    score_net: DiffusionScoreNet,
    sigma_data: float,
    sigma_start: float,
    sigma_min: float = 0.01,
    n_steps: int = 50,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    Per-patch conditional sampling with residual diffusion.

    The residual R (not the absolute coordinate V) is diffused:
    - Initialise R_start = sigma_start * epsilon
    - Denoise in R-space over the EDM schedule
    - Compose V = V_base + R_0

    Returns:
        dict with V_final, V_base, R_final, and RMS diagnostics.
    """
    # Ensure batch dimension
    if Z_patch.dim() == 2:
        Z_patch = Z_patch.unsqueeze(0)
    if mask_patch.dim() == 1:
        mask_patch = mask_patch.unsqueeze(0)

    B, m, h = Z_patch.shape

    # Set seed for deterministic sampling if provided
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        # Conditioning
        H = context_encoder(Z_patch, mask_patch)  # (1, m, c)

        # Generator proposal
        V_base = generator(H, mask_patch)  # (1, m, D)

        # Initialize residual at sigma_start
        epsilon = torch.randn_like(V_base)
        R_t = sigma_start * epsilon  # Start from pure noise in R-space

        # EDM sigma schedule (Karras et al.)
        sigmas = edm_sigma_schedule(n_steps, sigma_min, sigma_start, rho=7.0, device=device)

        # Denoising loop in residual space
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_batch = sigma.view(1)

            # Score net predicts clean R_0 (in residual space)
            R_0_pred = _forward_edm_x0_pred(score_net, R_t, sigma_batch, H, mask_patch, sigma_data)

            # Optional CFG (usually guidance_scale=1.0 for residual mode)
            if guidance_scale != 1.0:
                H_null = torch.zeros_like(H)
                R_0_uncond = _forward_edm_x0_pred(score_net, R_t, sigma_batch, H_null, mask_patch, sigma_data)
                R_0_pred = R_0_uncond + guidance_scale * (R_0_pred - R_0_uncond)

            # Euler step
            d = (R_t - R_0_pred) / sigma.clamp(min=1e-8)
            R_euler = R_t + (sigma_next - sigma) * d

            # Heun corrector (skip if sigma_next == 0)
            if sigma_next > 1e-8:
                R_0_next = _forward_edm_x0_pred(score_net, R_euler, sigma_next.view(1), H, mask_patch, sigma_data)
                if guidance_scale != 1.0:
                    R_0_next_uncond = _forward_edm_x0_pred(score_net, R_euler, sigma_next.view(1), H_null, mask_patch, sigma_data)
                    R_0_next = R_0_next_uncond + guidance_scale * (R_0_next - R_0_next_uncond)

                d_next = (R_euler - R_0_next) / sigma_next.clamp(min=1e-8)
                R_t = R_t + 0.5 * (sigma_next - sigma) * (d + d_next)
            else:
                R_t = R_euler

        R_final = R_t

        # Compose: V = V_base + R
        V_final = V_base + R_final

        # Apply mask
        mask_f = mask_patch.unsqueeze(-1).float()
        V_final = V_final * mask_f
        V_base = V_base * mask_f
        R_final = R_final * mask_f

    # Diagnostics
    rms_V_base = V_base.pow(2).sum() / (mask_f.sum() * V_base.shape[-1] + 1e-8)
    rms_V_base = rms_V_base.sqrt().item()
    rms_R = R_final.pow(2).sum() / (mask_f.sum() * R_final.shape[-1] + 1e-8)
    rms_R = rms_R.sqrt().item()
    residual_ratio = rms_R / (rms_V_base + 1e-8)

    return {
        'V_final': V_final.squeeze(0),  # (m, D)
        'V_base': V_base.squeeze(0),
        'R_final': R_final.squeeze(0),
        'rms_V_base': rms_V_base,
        'rms_R': rms_R,
        'residual_ratio': residual_ratio,
    }


def _forward_edm_x0_pred(score_net, x_t, sigma, H, mask, sigma_data, use_self_cond=True):
    """Forward pass through the score net to get the x0 prediction with self-conditioning."""
    # Use the score_net's forward_edm method which handles preconditioning properly
    if use_self_cond:
        # Two-pass self-conditioning
        x0_pred_0 = score_net.forward_edm(
            x_t, sigma, H, mask, sigma_data,
            self_cond=None,
            center_mask=None,
        )
        x0_pred = score_net.forward_edm(
            x_t, sigma, H, mask, sigma_data,
            self_cond=x0_pred_0.detach(),
            center_mask=None,
        )
    else:
        x0_pred = score_net.forward_edm(
            x_t, sigma, H, mask, sigma_data,
            self_cond=None,
            center_mask=None,
        )
    return x0_pred


def extract_patch_distances_v2(
    V_patch: torch.Tensor,  # (m, D)
    mask_patch: torch.Tensor,  # (m,)
    k_edge: int = 20,
    centrality_weight: bool = True,
    random_edge_frac: float = 0.1,  # fraction of extra random long-range edges
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Convert a patch geometry into a set of sparse pairwise distance measurements.

    Extracts:
    1. kNN edges for local structure.
    2. Random non-kNN edges for global connectivity.

    Returns:
        dict with edges (list of (u, v)), distances, weights, and diagnostics.
    """
    m, D = V_patch.shape
    valid_mask = mask_patch.bool()
    n_valid = valid_mask.sum().item()

    if n_valid < 3:
        return {'edges': [], 'distances': [], 'weights': [], 'diagnostics': {}}

    # Get valid coordinates
    V_valid = V_patch[valid_mask]  # (n_valid, D)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # Map back to original indices

    # Compute full distance matrix for valid points
    D_mat = torch.cdist(V_valid, V_valid)  # (n_valid, n_valid)

    # Get kNN edges (directed)
    k_actual = min(k_edge, n_valid - 1)
    _, knn_indices = D_mat.topk(k_actual + 1, dim=1, largest=False)  # +1 for self
    knn_indices = knn_indices[:, 1:]  # Remove self

    # Track which edges are kNN (for avoiding duplicates with random edges)
    knn_edge_set = set()

    # Convert to edge list with distances
    edges = []
    distances = []
    weights = []

    # Compute centrality weights (downweight boundary points)
    if centrality_weight and n_valid > k_actual:
        # Mean distance to k nearest neighbors as centrality proxy
        mean_knn_dist = D_mat.topk(k_actual + 1, dim=1, largest=False)[0][:, 1:].mean(dim=1)
        median_dist = mean_knn_dist.median()
        centrality = torch.exp(-mean_knn_dist / (2 * median_dist + 1e-8))
    else:
        centrality = torch.ones(n_valid, device=device)

    # Add kNN edges
    for i in range(n_valid):
        for j_idx in range(k_actual):
            j = knn_indices[i, j_idx].item()
            if i < j:  # Avoid duplicates
                u = valid_indices[i].item()
                v = valid_indices[j].item()
                d = D_mat[i, j].item()
                w = (centrality[i] * centrality[j]).sqrt().item()

                edges.append((u, v))
                distances.append(d)
                weights.append(w)
                knn_edge_set.add((min(i, j), max(i, j)))

    # Add random longer-range edges to improve global connectivity
    n_knn_edges = len(edges)
    n_random_target = max(1, int(n_knn_edges * random_edge_frac))

    # Generate random pairs that aren't already kNN edges
    random_edges_added = 0
    max_attempts = n_random_target * 10
    attempts = 0

    while random_edges_added < n_random_target and attempts < max_attempts:
        i = random.randint(0, n_valid - 1)
        j = random.randint(0, n_valid - 1)
        if i != j:
            edge_key = (min(i, j), max(i, j))
            if edge_key not in knn_edge_set:
                u = valid_indices[i].item()
                v = valid_indices[j].item()
                d = D_mat[i, j].item()
                # Lower weight for random edges (less trusted)
                w = 0.5 * (centrality[i] * centrality[j]).sqrt().item()

                edges.append((u, v))
                distances.append(d)
                weights.append(w)
                knn_edge_set.add(edge_key)
                random_edges_added += 1
        attempts += 1

    # Planarity diagnostic (rank-2 energy)
    # B = -0.5 * J @ D^2 @ J where J is the centering matrix
    D_sq = D_mat ** 2
    J = torch.eye(n_valid, device=device) - torch.ones(n_valid, n_valid, device=device) / n_valid
    B = -0.5 * J @ D_sq @ J

    # Eigendecomposition
    eigvals = torch.linalg.eigvalsh(B)
    pos_eigvals = eigvals[eigvals > 1e-8]
    if len(pos_eigvals) >= 2:
        rank2_energy = (pos_eigvals[-1] + pos_eigvals[-2]) / (pos_eigvals.sum() + 1e-8)
    else:
        rank2_energy = 0.0

    diagnostics = {
        'n_edges': len(edges),
        'n_valid': n_valid,
        'rank2_energy': rank2_energy.item() if isinstance(rank2_energy, torch.Tensor) else rank2_energy,
        'dist_median': float(np.median(distances)) if distances else 0.0,
        'dist_max': max(distances) if distances else 0.0,
    }

    return {
        'edges': edges,
        'distances': distances,
        'weights': weights,
        'valid_indices': valid_indices.cpu().tolist(),
        'diagnostics': diagnostics,
    }
