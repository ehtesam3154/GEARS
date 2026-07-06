"""
Stage-A training objectives.

The shared encoder is trained with a VICReg self-supervised objective plus a set
of domain-alignment terms that make ST and SC embeddings indistinguishable:

    - VICReg               : invariance / variance / covariance regularization.
    - Domain adversary     : gradient-reversal + domain discriminator (see encoder.py).
    - CORAL                : match mean + covariance of ST vs SC embeddings.
    - RBF-MMD              : match ST vs SC embedding distributions (multi-kernel).
    - Local alignment      : InfoNCE over mutual-nearest-neighbor ST<->SC pairs.
    - kNN consistency      : preserve global expression neighborhoods in the embedding.

All functions operate on plain tensors so they can be composed freely in the
training loop (see train_encoder.py).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Gradient Reversal Layer (domain adversary)
# ----------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    """Identity forward; gradient is multiplied by -alpha on the backward pass."""

    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.alpha) * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(alpha))


def grl_alpha_schedule(
    epoch: int,
    warmup_epochs: int = 50,
    ramp_epochs: int = 200,
    alpha_max: float = 1.0,
) -> float:
    """
    Gradient-reversal strength schedule: 0 during warmup, linearly ramps to
    alpha_max over ramp_epochs, constant thereafter.
    """
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + ramp_epochs:
        return alpha_max * (epoch - warmup_epochs) / ramp_epochs
    return alpha_max


# ----------------------------------------------------------------------------
# VICReg
# ----------------------------------------------------------------------------
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization
    (Bardes et al., ICLR 2022; implementation adapted from facebookresearch/vicreg).

    Stage A trains on a single GPU, so the distributed embedding-gather used in
    the original multi-GPU VICReg is omitted here.
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        compute_stats_fp32: bool = True,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps
        self.compute_stats_fp32 = compute_stats_fp32

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, D = z1.shape

        # Invariance: keep the two augmented views close.
        loss_inv = F.mse_loss(z1, z2)

        if self.compute_stats_fp32:
            z1 = z1.float()
            z2 = z2.float()

        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)

        # Variance: keep per-dimension std >= gamma (hinge).
        std_z1 = torch.sqrt(z1_c.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2_c.var(dim=0) + self.eps)
        loss_var = (
            torch.mean(F.relu(self.gamma - std_z1))
            + torch.mean(F.relu(self.gamma - std_z2))
        ) / 2.0

        # Covariance: decorrelate dimensions.
        cov_z1 = (z1_c.T @ z1_c) / (z1_c.shape[0] - 1)
        cov_z2 = (z2_c.T @ z2_c) / (z2_c.shape[0] - 1)
        loss_cov = (
            off_diagonal(cov_z1).pow(2).sum() / D
            + off_diagonal(cov_z2).pow(2).sum() / D
        )

        loss = (
            self.lambda_inv * loss_inv
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
        )
        stats = {
            "inv": loss_inv.item(),
            "var": loss_var.item(),
            "cov": loss_cov.item(),
            "std_mean": (std_z1.mean().item() + std_z2.mean().item()) / 2.0,
            "std_min": min(std_z1.min().item(), std_z2.min().item()),
        }
        return loss, stats


# ----------------------------------------------------------------------------
# Domain-alignment losses
# ----------------------------------------------------------------------------
def coral_loss(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """CORAL: match the mean and covariance of two embedding distributions."""
    mu_s = z_source.mean(dim=0)
    mu_t = z_target.mean(dim=0)
    loss_mean = (mu_s - mu_t).pow(2).mean()

    z_s = z_source - mu_s
    z_t = z_target - mu_t
    cov_s = (z_s.T @ z_s) / max(z_s.shape[0] - 1, 1)
    cov_t = (z_t.T @ z_t) / max(z_t.shape[0] - 1, 1)
    loss_cov = (cov_s - cov_t).pow(2).mean()
    return loss_mean + loss_cov


def mmd_rbf_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    use_median: bool = True,
    return_sigma: bool = False,
):
    """
    Multi-kernel RBF-MMD between two sets of embeddings. The base bandwidth is
    the median pairwise distance (scaled by each entry of `sigmas`).
    """
    if x.numel() == 0 or y.numel() == 0:
        out = torch.tensor(0.0, device=x.device)
        return (out, 1.0) if return_sigma else out

    with torch.no_grad():
        if use_median:
            xy = torch.cat([x, y], dim=0)
            dists = torch.cdist(xy, xy)
            vals = dists[dists > 0]
            base_sigma = torch.median(vals).item() if vals.numel() > 0 else 1.0
        else:
            base_sigma = 1.0
        if not np.isfinite(base_sigma) or base_sigma <= 0:
            base_sigma = 1.0

    mmd = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (base_sigma * s) ** 2)
        Kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
        Kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
        Kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
        mmd = mmd + (Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    mmd = mmd / len(sigmas)
    return (mmd, base_sigma) if return_sigma else mmd


def local_alignment_loss(
    z_sc: torch.Tensor,
    z_st: torch.Tensor,
    tau_z: float = 0.1,
    bidirectional: bool = True,
    mnn_min_sim: float = 0.2,
) -> torch.Tensor:
    """
    Local alignment via mutual nearest neighbors in embedding space, trained
    with InfoNCE. Encourages SC cells and their ST counterparts (and vice versa)
    to be close relative to other candidates.
    """
    z_sc_n = F.normalize(z_sc, dim=1)
    z_st_n = F.normalize(z_st, dim=1)
    S = z_sc_n @ z_st_n.T  # (n_sc, n_st)

    nn_sc = S.argmax(dim=1)
    nn_st = S.argmax(dim=0)
    idx_sc = torch.arange(S.shape[0], device=S.device)
    mnn_mask = nn_st[nn_sc] == idx_sc
    if mnn_min_sim > 0:
        mnn_mask = mnn_mask & (S[idx_sc, nn_sc] >= mnn_min_sim)

    if mnn_mask.sum() == 0:
        return torch.tensor(0.0, device=S.device)

    pos_st = nn_sc[mnn_mask]
    pos_sc = idx_sc[mnn_mask]
    loss_sc2st = F.cross_entropy(S[pos_sc] / tau_z, pos_st)
    if not bidirectional:
        return loss_sc2st
    loss_st2sc = F.cross_entropy(S.T[pos_st] / tau_z, pos_sc)
    return 0.5 * (loss_sc2st + loss_st2sc)


def knn_consistency_loss(
    idx: torch.Tensor,
    z_batch: torch.Tensor,
    global_knn: torch.Tensor,
    z_cache: torch.Tensor,
    k: int = 15,
) -> torch.Tensor:
    """
    Encourage each cell's embedding to stay close (cosine) to the embeddings of
    its global expression-space neighbors.

    Args:
        idx:        (batch,) global indices of the batch cells.
        z_batch:    (batch, d) embeddings of the batch cells.
        global_knn: (N, K) precomputed expression-neighbor indices for all cells.
        z_cache:    (N, d) cached embeddings (refreshed periodically during training).
        k:          number of neighbors to enforce.
    """
    neighbor_idx = global_knn[idx, :k]                 # (batch, k)
    z_norm = F.normalize(z_batch, dim=1)               # (batch, d)
    z_neighbors = F.normalize(z_cache[neighbor_idx], dim=2)  # (batch, k, d)
    sim = torch.einsum("bd,bkd->bk", z_norm, z_neighbors)
    return -sim.mean()  # maximize similarity to true neighbors
