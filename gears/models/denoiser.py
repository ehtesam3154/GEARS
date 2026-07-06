"""
Stage-C denoiser: the EDM-preconditioned, set-equivariant score network.

The network predicts a clean latent geometry from a noisy one under an EDM
preconditioning wrapper

    D_theta(x, sigma) = c_skip * x + c_out * F_theta(c_in * x; c_noise, H)

where F_theta is a Set-Transformer backbone conditioned on the per-set context H.
Beyond the raw noisy coordinates the backbone consumes three geometric inductive
signals computed on the current geometry estimate: a distance-bucket attention
bias, per-node angle histograms over a kNN graph, and a self-conditioning channel
carrying the previous clean estimate. The distance buckets additionally drive an
auxiliary distogram head used for supervision during training.
"""

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from .set_transformer import SAB, ISAB


# ==============================================================================
# Geometry / EDM helpers
# ==============================================================================

def center_only(V, mask, eps=1e-8):
    """
    Center V per set (subtract the masked mean), without rescaling.
    """
    B, N, D = V.shape
    mask_expanded = mask.unsqueeze(-1)

    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (V * mask_expanded).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)

    V_centered = (V - mean) * mask_expanded
    return V_centered, mean


def pairwise_dist2(V, mask):
    """
    Squared pairwise distances.
    V: (B, N, D), mask: (B, N). Returns D2: (B, N, N) with masked pairs set to 0.
    """
    B, N, D = V.shape

    V_norm = (V ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    D2 = V_norm + V_norm.transpose(1, 2) - 2 * torch.bmm(V, V.transpose(1, 2))

    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    D2 = D2 * mask_2d

    return D2


def make_distance_bias(D2, mask, n_bins, E_bin, W, alpha_bias, bin_edges=None):
    """
    Convert squared distances into a shared-across-heads attention bias.

    Returns attn_bias: (B, 1, N, N) and bin_ids: (B, N, N) for distogram
    supervision. Bin edges are data-driven when supplied, else a fixed linear
    spacing over [0, 3.0].
    """
    B, N, _ = D2.shape
    device = D2.device

    D = torch.sqrt(D2.clamp(min=1e-8))

    if bin_edges is not None:
        edges = bin_edges.to(device)
        bin_ids = torch.bucketize(D.contiguous(), edges.contiguous()) - 1
        bin_ids = bin_ids.clamp(min=0, max=n_bins - 1)
    else:
        edges = torch.linspace(0, 3.0, n_bins, device=device)
        bin_ids = torch.searchsorted(edges, D.flatten()).reshape(B, N, N)
        bin_ids = torch.clamp(bin_ids, 0, n_bins - 1)

    bin_embeddings = E_bin[bin_ids]

    bias_raw = torch.matmul(bin_embeddings, W)

    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)

    bias = bias_raw.squeeze(-1) * mask_2d
    attn_bias = alpha_bias * bias.unsqueeze(1)

    return attn_bias, bin_ids


def knn_graph(V, mask, k=12):
    """
    Build a kNN graph on coordinates.
    Returns idx: (B, N, k) with -1 for invalid neighbors.
    """
    B, N, D = V.shape
    device = V.device

    D2 = pairwise_dist2(V, mask)

    # Mask self-connections
    D2 = D2 + torch.eye(N, device=device).unsqueeze(0) * 1e10

    # Mask invalid nodes
    invalid_mask = (~mask).unsqueeze(1).float() * 1e10
    D2 = D2 + invalid_mask

    _, idx = torch.topk(D2, k, dim=-1, largest=False)  # (B, N, k)

    valid_neighbors = mask.unsqueeze(2).expand(-1, -1, k)  # (B, N, k)
    neighbor_mask = torch.gather(mask.unsqueeze(1).expand(-1, N, -1), 2, idx)  # (B, N, k)
    idx = torch.where(valid_neighbors & neighbor_mask, idx, torch.tensor(-1, device=device))

    return idx


@lru_cache(maxsize=64)
def _triu_indices_cached(k: int, device: torch.device):
    # Upper-triangular pair indices (j > i), computed once per k.
    return torch.triu_indices(k, k, offset=1, device=device)


def angle_features(V, mask, idx, n_angle_bins: int = 8, eps: float = 1e-8):
    """
    Vectorized angle histogram per node from neighbor triangles.

    V   : (B, N, D) latent coordinates
    mask: (B, N) bool, True = valid node
    idx : (B, N, k) long, neighbor indices per node, -1 indicates missing
    n_angle_bins: number of bins for cos(theta) in [-1, 1]

    Returns angle_hist: (B, N, n_angle_bins), rows sum to 1 for valid nodes
    (0 for pads).
    """
    B, N, D = V.shape
    k = idx.shape[-1]
    device = V.device

    # Clamp negative indices so we can safely gather, but remember validity.
    idx_clamped = idx.clamp_min(0)                              # (B, N, k)
    b_ix = torch.arange(B, device=device)[:, None, None].expand(B, N, k)
    nb_mask_from_nodes = mask[b_ix, idx_clamped]               # (B, N, k)
    neighbor_valid = (idx >= 0) & nb_mask_from_nodes           # (B, N, k)

    # Gather neighbor coordinates.
    V_neighbors = V[b_ix, idx_clamped, :]                      # (B, N, k, D)

    # Centered neighbor rays U = V_j - V_i.
    V_center = V.unsqueeze(2)                                  # (B, N, 1, D)
    U = V_neighbors - V_center                                # (B, N, k, D)

    # Normalize rays to unit length.
    U_norm = torch.linalg.norm(U, dim=-1).clamp_min(eps)      # (B, N, k)
    U_unit = U / U_norm.unsqueeze(-1)                         # (B, N, k, D)

    # Zero-out invalid neighbors.
    U_unit = U_unit * neighbor_valid.unsqueeze(-1).to(U.dtype)

    # Neighbor-neighbor Gram per node: all cosines at once, (B, N, k, k).
    G = torch.matmul(U_unit, U_unit.transpose(-1, -2))

    # Select upper-triangular pairs j < k (each is an angle at the node center).
    i_idx, j_idx = _triu_indices_cached(k, device)
    cos_all = G[:, :, i_idx, j_idx]                            # (B, N, P)

    # Valid-pair mask: both neighbors must be valid.
    pair_valid = neighbor_valid[:, :, i_idx] & neighbor_valid[:, :, j_idx]  # (B, N, P)

    # Bin edges and bucketize cos(theta) in [-1, 1].
    bin_edges = torch.linspace(-1.0, 1.0, n_angle_bins + 1, device=device)
    BN = B * N
    P = cos_all.shape[-1]

    cos_flat = cos_all.reshape(BN, P)
    valid_flat = pair_valid.reshape(BN, P).to(cos_flat.dtype)

    bin_idx = torch.bucketize(cos_flat, bin_edges) - 1        # (BN, P)
    bin_idx = bin_idx.clamp_(0, n_angle_bins - 1)

    # Scatter-add into per-node histograms.
    row_offsets = (torch.arange(BN, device=device) * n_angle_bins).unsqueeze(1)  # (BN, 1)
    flat_idx = (row_offsets + bin_idx).reshape(-1)                                # (BN*P,)
    weights = valid_flat.reshape(-1)                                              # (BN*P,)

    hist_flat = torch.zeros(BN * n_angle_bins, device=device, dtype=cos_flat.dtype)
    hist_flat.index_add_(0, flat_idx, weights)
    hist = hist_flat.view(BN, n_angle_bins)

    # Normalize per node; zeros for padded rows.
    hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    hist = hist.view(B, N, n_angle_bins)

    hist = hist * mask.unsqueeze(-1).to(hist.dtype)

    return hist


def edm_precond(sigma: torch.Tensor, sigma_data: float):
    """
    EDM preconditioning scalars.

    Returns c_skip, c_out, c_in as (B, 1, 1) for broadcasting with (B, N, D),
    and c_noise as (B, 1) for the time embedding.
    """
    sigma = sigma.reshape(-1).to(torch.float32)

    sigma_3d = sigma.view(-1, 1, 1)

    c_skip = sigma_data ** 2 / (sigma_3d ** 2 + sigma_data ** 2)
    c_out = sigma_3d * sigma_data / (sigma_3d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_data ** 2 + sigma_3d ** 2).sqrt()
    c_noise = sigma.log() / 4
    c_noise = c_noise.view(-1, 1)

    return c_skip, c_out, c_in, c_noise


# ==============================================================================
# Score network
# ==============================================================================

class DiffusionScoreNet(nn.Module):
    """
    Conditional denoiser for latent geometry.

    Set-equivariant Set-Transformer backbone with Fourier time embedding, a
    distance-bucket attention bias, per-node angle features, and a
    self-conditioning channel. Wrapped by EDM preconditioning in forward_edm.
    """

    def __init__(
        self,
        D_latent: int = 32,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        isab_m: int = 64,
        time_emb_dim: int = 128,
        ln: bool = True,
        dist_bins: int = 16,
        angle_bins: int = 8,
        knn_k: int = 12,
    ):
        super().__init__()
        self.D_latent = D_latent
        self.c_dim = c_dim
        self.time_emb_dim = time_emb_dim

        self.dist_bins = dist_bins
        self.angle_bins = angle_bins
        self.knn_k = knn_k
        self.self_conditioning = True

        # Distance-bucket bias parameters (shared across heads).
        d_emb = 32
        self.E_bin = nn.Parameter(torch.randn(dist_bins, d_emb) / math.sqrt(d_emb))
        self.W_bias = nn.Parameter(torch.randn(d_emb, 1) * 0.01)
        self.alpha_bias = nn.Parameter(torch.tensor(0.1))

        # Distogram head over the distance-bucket embeddings.
        self.st_dist_head = nn.Sequential(
            nn.Linear(d_emb, 64),
            nn.ReLU(),
            nn.Linear(64, dist_bins),
        )

        # Data-driven bin edges; populated externally during training.
        self.register_buffer('st_dist_bin_edges', None)

        # Data std for EDM preconditioning; set externally during training.
        self.sigma_data = None

        # Time embedding (Fourier features -> MLP).
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, c_dim),
        )

        # Input projection over concatenated node features:
        #   [context (c_dim), time (c_dim), self-cond (D_latent),
        #    angle histogram (angle_bins), coordinates (D_latent)]
        extra_dims = D_latent + angle_bins
        self.input_proj = nn.Linear(c_dim + c_dim + extra_dims + D_latent, c_dim)
        self.bias_sab = SAB(c_dim, c_dim, n_heads, ln=ln)

        # Denoising blocks (ISAB).
        self.denoise_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])

        # Per-block FiLM conditioning from [context, time]. Output is
        # (gamma, beta) so the input width is 2 * c_dim and output 2 * c_dim.
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * c_dim, c_dim),
                nn.SiLU(),
                nn.Linear(c_dim, 2 * c_dim),
            )
            for _ in range(n_blocks)
        ])

        # Initialize FiLM output layers to zero for a stable identity start.
        for film in self.film_layers:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

        # Output head.
        self.output_head = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, D_latent),
        )

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Fourier time embedding.

        t: (batch, 1). Returns (batch, time_emb_dim).
        """
        half_dim = self.time_emb_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(self, V_t: torch.Tensor, t: torch.Tensor, H: torch.Tensor,
                mask: torch.Tensor, self_cond: torch.Tensor = None,
                attn_cached: dict = None, return_dist_aux: bool = False,
                sigma_raw: torch.Tensor = None,
                x_raw: torch.Tensor = None,
                c_in: torch.Tensor = None,
                center_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            V_t: (B, N, D_latent) noisy (preconditioned) coordinates.
            t: (B,) or (B, 1) diffusion time (c_noise under EDM).
            H: (B, N, c_dim) context.
            mask: (B, N).
            self_cond: optional (B, N, D_latent) previous clean estimate.
            attn_cached: optional dict to reuse the distance bias.
            sigma_raw: optional (B,) raw sigma from EDM (drives feature gating).
            x_raw: optional (B, N, D_latent) raw centered geometry for the bias.
            c_in: optional (B, 1, 1) preconditioning scalar for self-cond scaling.
            center_mask: optional mask used for centering (frame consistency).

        Returns:
            eps_hat: (B, N, D_latent), or (eps_hat, dist_aux) if return_dist_aux.
        """
        B, N, D = V_t.shape

        # Center in the frame of center_mask (or mask).
        mask_center = center_mask if center_mask is not None else mask
        V_in, _ = center_only(V_t, mask_center)
        if self_cond is not None:
            self_cond_canon, _ = center_only(self_cond, mask_center)
        else:
            self_cond_canon = None

        # Sigma used for smooth gating of the geometric features.
        if sigma_raw is not None:
            sigma_for_gating = sigma_raw.detach().view(-1).float()  # (B,)
        else:
            sigma_for_gating = torch.exp(4.0 * t.squeeze(-1)).detach().view(-1).float()

        # geom_gate: ~1 at low sigma, ~0 at high sigma (smooth around 0.30).
        sigma_cut = 0.30
        sigma_k = 20.0
        geom_gate = torch.sigmoid((sigma_cut - sigma_for_gating) * sigma_k)  # (B,)

        # Geometry source for the distance/angle features: raw centered input,
        # overridden by the self-conditioning estimate when available.
        if x_raw is not None:
            V_geom_for_bias = x_raw  # raw centered x_t (units match st_dist_bin_edges)
        else:
            V_geom_for_bias = V_in

        if self_cond_canon is not None:
            V_bias_geom = self_cond_canon
        else:
            V_bias_geom = V_geom_for_bias
        V_geom = V_bias_geom

        # Geometry usable when the set has at least two valid points.
        geom_ok = (mask.sum(dim=-1) >= 2)  # (B,)

        # Compose node features.
        features = [H]

        # Time embedding, expanded to all nodes.
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)
        features.append(t_emb_expanded)

        # Self-conditioning channel, scaled to the preconditioned coordinate
        # units before concatenation.
        if self_cond_canon is not None:
            if c_in is not None:
                self_cond_feat_input = self_cond_canon * c_in  # (B, 1, 1) broadcasts
            else:
                self_cond_feat_input = self_cond_canon
            sc_feat = self_cond_feat_input
            features.append(sc_feat)
        else:
            sc_feat = torch.zeros(B, N, self.D_latent, device=V_t.device)
            features.append(sc_feat)

        # Angle histogram features over the kNN graph, gated by geom_gate.
        if geom_ok.any():
            idx = knn_graph(V_geom, mask, k=self.knn_k)
            angle_feat = angle_features(V_geom, mask, idx, n_angle_bins=self.angle_bins)
            gate_angle = geom_gate.view(-1, 1, 1)
            angle_feat = angle_feat * gate_angle
            features.append(angle_feat)
        else:
            angle_feat = torch.zeros(B, N, self.angle_bins, device=V_t.device)
            features.append(angle_feat)

        # Raw noisy coordinates are always passed (never gated).
        V_coord_feat = V_in
        features.append(V_coord_feat)

        # Project concatenated features.
        X = torch.cat(features, dim=-1)
        X = self.input_proj(X)

        # Distance-bucket attention bias.
        attn_bias = None
        bin_ids = None
        bin_embeddings = None

        if attn_cached is not None and 'bias' in attn_cached:
            attn_bias = attn_cached['bias']
            bin_ids = attn_cached.get('bin_ids', None)
            bin_embeddings = attn_cached.get('bin_embeddings', None)
        elif geom_ok.any():
            D2 = pairwise_dist2(V_geom, mask)
            attn_bias, bin_ids = make_distance_bias(
                D2, mask,
                n_bins=self.dist_bins,
                E_bin=self.E_bin,
                W=self.W_bias,
                alpha_bias=self.alpha_bias,
                bin_edges=self.st_dist_bin_edges,
            )

            # Gate the bias on geometry validity only (mask-based).
            validity_gate = geom_ok.float().view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            attn_bias = attn_bias * validity_gate

            bin_embeddings = self.E_bin[bin_ids]

            if attn_cached is not None:
                attn_cached['bias'] = attn_bias
                attn_cached['bin_ids'] = bin_ids
                attn_cached['bin_embeddings'] = bin_embeddings
        else:
            attn_bias = None
            bin_ids = None
            bin_embeddings = None

        # Distance-biased self-attention, applied once.
        if attn_bias is not None:
            X = self.bias_sab(X, mask=mask, attn_bias=attn_bias)
        else:
            X = self.bias_sab(X, mask=mask, attn_bias=None)
        X = X * mask.unsqueeze(-1).float()

        # ISAB blocks with per-block FiLM conditioning.
        film_cond = torch.cat([H, t_emb_expanded], dim=-1)

        for i, isab in enumerate(self.denoise_blocks):
            X = isab(X, mask=mask, attn_bias=None)
            X = X * mask.unsqueeze(-1).float()

            gamma_beta = self.film_layers[i](film_cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            X = X * (1.0 + gamma) + beta
            X = X * mask.unsqueeze(-1).float()

        # Output head.
        eps_hat = self.output_head(X)
        eps_hat = eps_hat * mask.unsqueeze(-1).float()

        if return_dist_aux and bin_embeddings is not None:
            dist_logits = self.st_dist_head(bin_embeddings)
            return eps_hat, {'dist_logits': dist_logits, 'bin_ids': bin_ids}

        return eps_hat

    def forward_edm(
        self,
        x: torch.Tensor,           # (B, N, D) noisy input
        sigma: torch.Tensor,       # (B,) noise level
        H: torch.Tensor,           # (B, N, c_dim) context
        mask: torch.Tensor,        # (B, N)
        sigma_data: float,         # data std
        self_cond: torch.Tensor = None,
        center_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        EDM-preconditioned forward pass. Returns the denoised estimate x0_pred.

            D_theta(x, sigma) = c_skip * x + c_out * F_theta(c_in * x; c_noise, H)

        center_mask, when provided, is used for centering instead of mask so
        normal and context-dropped passes share a coordinate frame.
        """
        B, N, D = x.shape

        c_skip, c_out, c_in, c_noise = edm_precond(sigma, sigma_data)
        # c_skip, c_out, c_in: (B, 1, 1); c_noise: (B, 1)

        # Center in raw space before preconditioning so the distance bias is
        # computed in units matching st_dist_bin_edges.
        mask_for_centering = center_mask if center_mask is not None else mask
        x_c, _ = center_only(x, mask_for_centering)

        x_in = c_in * x_c

        F_x = self.forward(
            x_in,
            c_noise,
            H,
            mask,
            self_cond=self_cond,
            sigma_raw=sigma,
            x_raw=x_c,
            c_in=c_in,
            center_mask=mask_for_centering,
        )

        if isinstance(F_x, tuple):
            F_x = F_x[0]

        x0_pred = c_skip * x_c + c_out * F_x
        x0_pred = x0_pred * mask.unsqueeze(-1).float()

        return x0_pred