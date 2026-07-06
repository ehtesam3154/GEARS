"""Coarse geometry generator for Stage C.

Maps context features ``H`` to a coarse latent geometry proposal
``Vbase`` (B, n, D_latent) via stacked ISAB blocks and an MLP head,
mean-centering the output over the valid spots of each set.
"""

import torch
import torch.nn as nn

from .set_transformer import ISAB


class MetricSetGenerator(nn.Module):
    """Generator that produces ``V`` in R^{n x D} from context ``H``.

    Architecture: stack of ISAB blocks + MLP head.
    Output: ``V_0`` with row-mean centering over valid spots.
    """

    def __init__(
        self,
        c_dim: int = 256,
        D_latent: int = 32,
        n_heads: int = 4,
        n_blocks: int = 2,
        isab_m: int = 64,
        ln: bool = True,
    ):
        """
        Args:
            c_dim: context dimension
            D_latent: latent dimension of V
            n_heads: number of attention heads
            n_blocks: number of ISAB blocks
            isab_m: number of inducing points
            ln: use layer normalization
        """
        super().__init__()
        self.c_dim = c_dim
        self.D_latent = D_latent

        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])

        # MLP head to produce V
        self.head = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.ReLU(),
            nn.Linear(c_dim, D_latent)
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch, n, c_dim) context features
            mask: (batch, n) boolean mask

        Returns:
            V: (batch, n, D_latent) factor matrix (row-mean centered)
        """
        batch_size, n, _ = H.shape

        # Apply ISAB blocks
        X = H
        for isab in self.isab_blocks:
            X = isab(X)
            X = X * mask.unsqueeze(-1).float()

        # MLP head
        V = self.head(X)  # (batch, n, D_latent)

        # Row-mean centering over valid spots (translation neutrality)
        mask_f = mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean = (V * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        V_centered = (V - mean) * mask_f.unsqueeze(-1)

        return V_centered