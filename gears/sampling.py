"""
Batch construction utilities for Stage-A encoder training:

    - augment_expression                   : two-view augmentation of log1p expression.
    - sample_balanced                      : sample a batch balanced across the unique
                                             values of an id tensor (e.g. ST vs SC).
    - sample_balanced_domain_and_slide     : hierarchical balancing — 50/50 ST/SC, then
                                             balanced across slides within each domain
                                             (used when SC spans multiple slides).
"""

from typing import Optional

import torch


def augment_expression(
    x: torch.Tensor,
    gene_dropout: float = 0.2,
    gauss_std: float = 0.01,
    scale_jitter: float = 0.2,
) -> torch.Tensor:
    """
    Coordinate-free augmentation of log1p expression tensors:
        1) random gene dropout,
        2) multiplicative library-size (scale) jitter in linear space,
        3) additive Gaussian noise.
    """
    device = x.device
    B, G = x.shape
    x_aug = x.clone()

    if gene_dropout > 0:
        mask = (torch.rand(B, G, device=device) > gene_dropout).to(x_aug.dtype)
        x_aug = x_aug * mask

    if scale_jitter > 0:
        x_lin = torch.expm1(x_aug)
        scale = torch.empty(B, 1, device=device).uniform_(
            1.0 - scale_jitter, 1.0 + scale_jitter
        )
        x_aug = torch.log1p(x_lin * scale)

    if gauss_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * gauss_std

    return torch.clamp(x_aug, min=-10.0, max=10.0)


def _sample_balanced_internal(
    ids: torch.Tensor, n_samples: int, device: str
) -> torch.Tensor:
    """Sample n_samples LOCAL indices balanced across the unique values of `ids`."""
    unique = torch.unique(ids)
    n_groups = len(unique)
    per_group = n_samples // n_groups
    remainder = n_samples % n_groups

    parts = []
    for i, g in enumerate(unique):
        group_idx = torch.nonzero(ids == g, as_tuple=True)[0]
        n = per_group + (1 if i < remainder else 0)
        if n <= 0:
            continue
        if len(group_idx) >= n:
            sel = group_idx[torch.randperm(len(group_idx), device=device)[:n]]
        else:  # small group: sample with replacement
            sel = group_idx[torch.randint(0, len(group_idx), (n,), device=device)]
        parts.append(sel)

    if not parts:
        return torch.randint(0, len(ids), (n_samples,), device=device)

    out = torch.cat(parts, dim=0)
    if out.numel() > n_samples:
        out = out[:n_samples]
    elif out.numel() < n_samples:
        extra = torch.randint(0, len(ids), (n_samples - out.numel(),), device=device)
        out = torch.cat([out, extra], dim=0)
    return out


def sample_balanced(
    ids: torch.Tensor, batch_size: int, device: str = "cuda"
) -> torch.Tensor:
    """
    Sample `batch_size` indices with equal representation across the unique
    values of `ids`, then shuffle. Passing domain ids (0=ST, 1=SC) yields a
    50/50 ST/SC batch.
    """
    idx = _sample_balanced_internal(ids, batch_size, device)
    return idx[torch.randperm(idx.numel(), device=device)]


def sample_balanced_domain_and_slide(
    domain_ids: torch.Tensor,
    st_slide_ids: torch.Tensor,
    sc_slide_ids: Optional[torch.Tensor],
    batch_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Hierarchical balancing over the concatenated [ST; SC] pool:
        level 1 — half the batch from ST, half from SC,
        level 2 — within ST balance across `st_slide_ids`; within SC balance
                  across `sc_slide_ids` (or uniform if None).

    Returns global indices into the concatenated pool.
    """
    st_global = torch.where(domain_ids == 0)[0]
    sc_global = torch.where(domain_ids == 1)[0]
    n_sc = sc_global.shape[0]

    n_st_sample = batch_size // 2
    n_sc_sample = batch_size - n_st_sample

    st_local = _sample_balanced_internal(st_slide_ids, n_st_sample, device)
    st_batch = st_global[st_local]

    if sc_slide_ids is not None:
        sc_local = _sample_balanced_internal(sc_slide_ids, n_sc_sample, device)
    elif n_sc >= n_sc_sample:
        sc_local = torch.randperm(n_sc, device=device)[:n_sc_sample]
    else:
        sc_local = torch.randint(0, n_sc, (n_sc_sample,), device=device)
    sc_batch = sc_global[sc_local]

    idx = torch.cat([st_batch, sc_batch], dim=0)
    return idx[torch.randperm(idx.numel(), device=device)]
