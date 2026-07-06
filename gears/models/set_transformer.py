"""
Permutation-equivariant set-transformer blocks.

MAB, SAB, ISAB and PMA are the attention primitives shared by all three
Stage-C networks (the set-encoder context, the metric-set generator and the
diffusion score network). They operate on batched point sets of shape
(batch, n_points, dim) and are equivariant / invariant to the ordering of the
points within each set.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead attention block.

    Attends a set of queries to a set of key-value pairs.

    Args:
        dim_Q: Dimension of query.
        dim_K: Dimension of key.
        dim_V: Dimension of value (output).
        num_heads: Number of attention heads.
        ln: Whether to use LayerNorm.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_mask=None, attn_bias=None):
        """
        Args:
            Q: Query tensor of shape (batch_size, n_queries, dim_Q).
            K: Key-value tensor of shape (batch_size, n_keys, dim_K).
            key_mask: Optional (batch_size, n_keys) bool mask; False entries are
                excluded from the attention softmax.
            attn_bias: Optional additive attention bias of shape
                (B, H, N, N) or (B, 1, N, N) (shared across heads).

        Returns:
            Output tensor of shape (batch_size, n_queries, dim_V).
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # attention scores scaled by per-head dimension
        d_h = max(self.dim_V // self.num_heads, 1)
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(d_h)

        # additive attention bias, if provided
        if attn_bias is not None:
            # attn_bias is (B, H, N, N) or (B, 1, N, N); A is (B*H, N, N)
            B = Q.size(0)
            H = self.num_heads
            if attn_bias.size(1) == 1:  # shared across heads
                attn_bias_expanded = attn_bias.expand(B, H, -1, -1)
            else:
                attn_bias_expanded = attn_bias

            attn_bias_flat = attn_bias_expanded.transpose(0, 1).reshape(
                B * H, attn_bias_expanded.size(2), attn_bias_expanded.size(3)
            )
            A = A + attn_bias_flat

        # key mask, if provided
        if key_mask is not None:
            # key_mask is (B, N)
            B = Q.size(0)
            H = self.num_heads
            mask_expanded = key_mask.unsqueeze(1).expand(B, H, -1).reshape(
                B * H, 1, key_mask.size(1)
            )
            mask_value = -torch.finfo(A.dtype).max
            A.masked_fill_(~mask_expanded, mask_value)

        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Set attention block.

    Self-attention variant of MAB where the queries and keys are the same set.

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        num_heads: Number of attention heads.
        ln: Whether to use LayerNorm.
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None, attn_bias=None):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim_in).

        Returns:
            Output tensor of shape (batch_size, n_points, dim_out).
        """
        return self.mab(X, X, key_mask=mask, attn_bias=attn_bias)


class ISAB(nn.Module):
    """Induced set attention block.

    Reduces the quadratic self-attention cost by attending through a set of
    learned inducing points, giving O(mn) complexity with m inducing points.

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        num_heads: Number of attention heads.
        num_inds: Number of inducing points.
        ln: Whether to use LayerNorm.
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None, attn_bias=None):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim_in).

        Returns:
            Output tensor of shape (batch_size, n_points, dim_out).
        """
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_mask=mask, attn_bias=None)
        return self.mab1(X, H, key_mask=None, attn_bias=None)


class PMA(nn.Module):
    """Pooling by multihead attention.

    Aggregates a set into a fixed number of seed vectors via attention.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_seeds: Number of output seed vectors.
        ln: Whether to use LayerNorm.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim).

        Returns:
            Output tensor of shape (batch_size, num_seeds, dim).
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
