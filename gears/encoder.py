"""
Stage-A model definitions: the shared expression encoder and the domain
discriminator used for adversarial alignment.

Both are plain MLPs. The encoder is the only module kept after Stage A; the
discriminator is an auxiliary network used purely to drive domain invariance
during training (via a gradient-reversal adversary) and is discarded afterwards.
"""

from typing import List

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    Shared encoder f_theta mapping gene expression (ST spots or dissociated SC
    cells) into a common embedding space.

    Architecture: MLP  [n_genes] -> [512, 256, 128]
        - each non-final layer:  Linear -> LayerNorm -> ReLU -> Dropout
        - final layer:           bare Linear (no activation / norm)

    The last embedding dimension (128 by default) is the shared latent space in
    which ST and SC profiles are aligned.
    """

    def __init__(
        self,
        n_genes: int,
        n_embedding: List[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_embedding = n_embedding

        layers: List[nn.Module] = []
        prev_dim = n_genes
        for i, dim in enumerate(n_embedding):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(n_embedding) - 1:  # no activation on the last layer
                layers.append(nn.LayerNorm(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)

    @property
    def embedding_dim(self) -> int:
        return self.n_embedding[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_genes) gene expression.
        Returns:
            z: (batch, n_embedding[-1]) embeddings.
        """
        return self.encoder(x)


class SlideDiscriminator(nn.Module):
    """
    Domain discriminator used by the gradient-reversal adversary.

    Predicts which domain an embedding came from (ST vs SC). Trained to
    classify domains; the encoder is trained (through gradient reversal) to
    fool it, which suppresses domain-specific nuisance variation.

    Architecture: MLP  [input_dim] -> [hidden] -> [hidden] -> [n_domains]
    """

    def __init__(
        self,
        input_dim: int,
        n_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) embeddings.
        Returns:
            logits: (batch, n_domains) classification logits.
        """
        return self.net(x)
