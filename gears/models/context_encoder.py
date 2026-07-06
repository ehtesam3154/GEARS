"""Set-equivariant context encoder for Stage C.

Maps a set of frozen per-spot encoder embeddings ``Z_set`` to context
features ``H`` via a linear projection followed by stacked ISAB blocks.
"""

import torch
import torch.nn as nn

from .set_transformer import ISAB


class SetEncoderContext(nn.Module):
    """Permutation-equivariant context encoder using a Set Transformer.

    Takes a set of embeddings ``Z_set`` and produces context ``H``.
    Uses ISAB blocks for O(mn) complexity.
    """

    def __init__(
        self,
        h_dim: int = 128,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 3,
        isab_m: int = 64,
        ln: bool = True,
    ):
        """
        Args:
            h_dim: input embedding dimension
            c_dim: output context dimension
            n_heads: number of attention heads
            n_blocks: number of ISAB blocks
            isab_m: number of inducing points in ISAB
            ln: use layer normalization
        """
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim

        # Input projection
        self.input_proj = nn.Linear(h_dim, c_dim)

        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])

    def forward(self, Z_set: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_set: (batch, n, h_dim) set of embeddings
            mask: (batch, n) boolean mask (True = valid)

        Returns:
            H: (batch, n, c_dim) context features
        """
        batch_size, n, input_dim_actual = Z_set.shape

        if input_dim_actual != self.h_dim:
            raise ValueError(
                f"SetEncoderContext: expected input dim {self.h_dim}, "
                f"got {input_dim_actual}"
            )

        # Project to context dimension
        H = self.input_proj(Z_set)  # (batch, n, c_dim)

        for isab in self.isab_blocks:
            # ISAB expects (batch, n, dim); keep the set intact
            H = isab(H, mask=mask)
            H = H * mask.unsqueeze(-1).float()

        return H


