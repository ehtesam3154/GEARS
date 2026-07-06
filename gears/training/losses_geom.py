"""
Stage-C training objectives: EDM residual-diffusion refiner + conditional
geometry generator.

The refiner learns a residual R = V_target - V_base on top of a jointly trained
generator prior V_base, using EDM preconditioning and loss-weighting. The
geometry losses (Gram, kNN-NCA, edge, local scale) supervise the composed
coordinates V_hat = V_base + R0_hat, and a separate set of terms supervises the
generator prior V_base directly.

All geometry losses are SNR-gated: they only fire on the low-noise (high-SNR)
fraction of a batch via per-loss adaptive quantile gates, so that shape / scale
signal is not learned from heavily corrupted samples.

Everything here operates on plain tensors so the terms can be composed in a
training loop.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gears.models.denoiser import edm_precond, center_only
from .score_hi_boost import high_noise_gate, hi_boost_multiplier, fx_hi_loss


# ============================================================================
# EDM loss weight
# ============================================================================
def edm_loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """
    EDM loss weight  w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.

    Returns a (B,) tensor.
    """
    sigma = sigma.reshape(-1).to(torch.float32)
    return (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)


# ============================================================================
# Small geometry helpers
# ============================================================================
def rigid_align_apply_no_scale(V_src, V_tgt, mask, eps: float = 1e-8, enforce_rotation: bool = False):
    """
    Orthogonal Procrustes alignment (rotation/reflection only, NO scale).
    Aligns V_src into V_tgt's frame and returns the aligned V_src.

    SVD is run on CPU in float64 for robustness; no gradient flows through the
    alignment.
    """
    B, N, D = V_src.shape
    device = V_src.device
    dtype_orig = V_src.dtype

    V_aligned = V_src.clone().float()

    with torch.no_grad():
        for b in range(B):
            mb = mask[b].bool()
            n_valid = int(mb.sum().item())
            if n_valid < 2:
                continue

            X = V_src[b, mb].float()
            Y = V_tgt[b, mb].float()

            X_mean = X.mean(dim=0, keepdim=True)
            Y_mean = Y.mean(dim=0, keepdim=True)
            X_c = X - X_mean
            Y_c = Y - Y_mean

            M = X_c.T @ Y_c
            if not torch.isfinite(M).all():
                continue

            try:
                M_cpu = M.detach().cpu().double()
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(M_cpu, full_matrices=False)
                R_cpu = U_cpu @ Vh_cpu
                if enforce_rotation and torch.det(R_cpu) < 0:
                    U_cpu[:, -1] *= -1
                    R_cpu = U_cpu @ Vh_cpu
                R = R_cpu.float().to(device=device)
            except Exception:
                continue

            V_aligned[b, mb] = (X_c @ R) + Y_mean

    return V_aligned.to(dtype_orig)


def rigid_align_mse_no_scale(V_pred, V_tgt, mask, eps: float = 1e-8):
    """Rotation/reflection alignment only (no scaling), then per-set MSE."""
    B, N, D = V_pred.shape
    loss = V_pred.new_zeros(())
    cnt = 0
    for b in range(B):
        mb = mask[b]
        n = int(mb.sum().item())
        if n < 2:
            continue
        X = V_pred[b, mb].float()
        Y = V_tgt[b, mb].float()
        M = X.T @ Y
        try:
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            R = U @ Vh
        except Exception:
            continue
        Xr = X @ R
        loss = loss + (Xr - Y).pow(2).mean()
        cnt += 1
    if cnt == 0:
        return V_pred.new_tensor(0.0)
    return loss / float(cnt)


def rms_log_loss(V_pred, V_tgt, mask, eps: float = 1e-8):
    """Per-set RMS scale mismatch: (log rms_pred - log rms_tgt)^2, averaged."""
    B, N, D = V_pred.shape
    loss = V_pred.new_zeros(())
    cnt = 0
    for b in range(B):
        mb = mask[b]
        n = int(mb.sum().item())
        if n < 2:
            continue
        X = V_pred[b, mb].float()
        Y = V_tgt[b, mb].float()
        rms_x = torch.sqrt(X.pow(2).mean() + eps)
        rms_y = torch.sqrt(Y.pow(2).mean() + eps)
        loss = loss + (torch.log(rms_x) - torch.log(rms_y)).pow(2)
        cnt += 1
    if cnt == 0:
        return V_pred.new_tensor(0.0)
    return loss / float(cnt)


def variance_outside_topk(V, mask, k: int = 2, eps: float = 1e-8):
    """Mean fraction of covariance variance outside the top-k principal axes."""
    B, N, D = V.shape
    out = V.new_zeros(())
    cnt = 0
    for b in range(B):
        mb = mask[b]
        n = int(mb.sum().item())
        if n <= k:
            continue
        X = V[b, mb].float()
        X_centered = X - X.mean(dim=0, keepdim=True)
        C = (X_centered.T @ X_centered) / float(n)
        try:
            evals = torch.linalg.eigvalsh(C)
        except Exception:
            continue
        total = evals.sum().clamp_min(eps)
        outside = evals[:-k].sum()
        out = out + (outside / total)
        cnt += 1
    if cnt == 0:
        return V.new_tensor(0.0)
    return out / float(cnt)


def knn_nca_loss(V_pred: torch.Tensor, V_target: torch.Tensor, mask: torch.Tensor,
                 k: int = 15, temperature: float = 0.1, eps: float = 1e-8,
                 return_per_sample: bool = False, scale_compensate: bool = True,
                 point_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Differentiable kNN neighborhood-preservation loss (NCA-style soft retrieval).

    Target neighbors are the k nearest points in V_target; the loss maximizes the
    soft-retrieval log-probability of those same neighbors under V_pred distances.
    """
    B, N, D = V_pred.shape
    device = V_pred.device

    mask_f = mask.float()
    valid_counts = mask_f.sum(dim=1).clamp(min=1)

    diff_pred = V_pred.unsqueeze(2) - V_pred.unsqueeze(1)
    D2_pred = (diff_pred ** 2).sum(dim=-1)
    diff_tgt = V_target.unsqueeze(2) - V_target.unsqueeze(1)
    D2_tgt = (diff_tgt ** 2).sum(dim=-1)

    mask_2d = (mask_f.unsqueeze(2) * mask_f.unsqueeze(1)).bool()
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    invalid_pair = (~mask_2d) | eye

    D2_tgt_masked = D2_tgt.masked_fill(invalid_pair, 1e9)
    D2_pred_masked = D2_pred.masked_fill(invalid_pair, 1e9)

    _, knn_idx_tgt = D2_tgt_masked.topk(k, dim=-1, largest=False)

    if scale_compensate:
        with torch.no_grad():
            d2k_tgt = torch.gather(D2_tgt_masked, 2, knn_idx_tgt)[:, :, -1]
            d2k_pred = torch.gather(D2_pred_masked.detach(), 2, knn_idx_tgt)[:, :, -1]

            def masked_median(x, mf):
                x_masked = x.masked_fill(mf == 0, float('nan'))
                return torch.nanmedian(x_masked, dim=1).values

            med_tgt = masked_median(d2k_tgt, mask_f).clamp(min=eps)
            med_pred = masked_median(d2k_pred, mask_f).clamp(min=eps)
            scale = (med_pred / med_tgt).clamp(0.25, 4.0)
        temp_eff = (temperature * scale).view(B, 1, 1)
    else:
        temp_eff = temperature

    logits = -D2_pred_masked / (temp_eff + eps)
    logits = logits.masked_fill(invalid_pair, -1e9)
    logP = torch.log_softmax(logits, dim=-1)

    logP_true = torch.gather(logP, dim=2, index=knn_idx_tgt)

    mask_j = mask_f.unsqueeze(1).expand(-1, N, -1)
    neigh_valid = torch.gather(mask_j, dim=2, index=knn_idx_tgt)
    query_valid = mask_f.unsqueeze(-1)
    w = query_valid * neigh_valid

    denom = w.sum(dim=-1).clamp(min=1.0)
    loss_per_point = -(logP_true * w).sum(dim=-1) / denom

    if point_weight is not None:
        effective_weight = point_weight * mask_f
        weight_sum_per_sample = effective_weight.sum(dim=-1).clamp(min=1.0)
        loss_per_sample = (loss_per_point * effective_weight).sum(dim=-1) / weight_sum_per_sample
    else:
        loss_per_sample = (loss_per_point * mask_f).sum(dim=-1) / valid_counts

    if return_per_sample:
        return loss_per_sample
    return loss_per_sample.mean()


def knn_scale_loss(V_pred: torch.Tensor, V_target: torch.Tensor, mask: torch.Tensor,
                   knn_indices: torch.Tensor = None, k: int = 15, eps: float = 1e-8,
                   return_per_sample: bool = False, r_clip: float = 2.0) -> torch.Tensor:
    """
    Edge-wise local scale calibration: per valid kNN edge, penalize
    (log d2_pred - log d2_tgt)^2, clipped to +/- r_clip, averaged over edges.

    knn_indices: (B, N, k) precomputed neighbor indices (LOCAL or GLOBAL; the
    function auto-detects and remaps LOCAL to GLOBAL). If None, computes kNN
    on the fly from V_pred among valid points.
    """
    B, N, D = V_pred.shape
    device = V_pred.device
    per_sample_losses = []

    for b in range(B):
        mb = mask[b].bool()
        n_valid = int(mb.sum().item())
        if n_valid < k + 1:
            per_sample_losses.append(torch.tensor(0.0, device=device))
            continue

        Vp = V_pred[b]
        Vt = V_target[b]

        if knn_indices is None:
            Vp_valid = Vp[mb]
            D2 = torch.cdist(Vp_valid, Vp_valid).pow(2)
            D2.fill_diagonal_(float('inf'))
            _, knn_local = D2.topk(min(k, n_valid - 1), largest=False, dim=1)
            valid_idx = torch.where(mb)[0]
            knn_g = valid_idx[knn_local]
            src_g = valid_idx.unsqueeze(1).expand_as(knn_g)
        else:
            knn_g_full = knn_indices[b]
            valid_idx = torch.where(mb)[0]
            src_g = valid_idx.unsqueeze(1)
            knn_g = knn_g_full[mb]
            src_g = src_g.expand_as(knn_g)
            if knn_g.numel() > 0:
                knn_max = knn_g.max().item()
                if knn_max < n_valid and knn_max >= 0:
                    knn_g = valid_idx[knn_g.clamp(0, n_valid - 1)]

        valid_src = mb[src_g]
        valid_nbr = (knn_g >= 0) & (knn_g < N) & mb[knn_g.clamp(0, N - 1)]
        not_self = (knn_g != src_g)
        edge_ok = valid_src & valid_nbr & not_self

        if edge_ok.sum().item() < 8:
            per_sample_losses.append(torch.tensor(0.0, device=device))
            continue

        src_idx = src_g[edge_ok]
        nbr_idx = knn_g[edge_ok]

        d2p = (Vp[src_idx] - Vp[nbr_idx]).pow(2).sum(dim=-1).clamp_min(eps)
        d2t = (Vt[src_idx] - Vt[nbr_idx]).pow(2).sum(dim=-1).clamp_min(eps)

        r = torch.log(d2p) - torch.log(d2t)
        if r_clip is not None and r_clip > 0:
            r = r.clamp(-r_clip, r_clip)
        per_sample_losses.append((r * r).mean())

    per_sample_tensor = torch.stack(per_sample_losses)
    if return_per_sample:
        return per_sample_tensor
    return per_sample_tensor.mean()


def edge_log_ratio_loss(V_pred, V_tgt, knn_idx, mask, eps: float = 1e-8):
    """
    Per-set multiplicative edge-length error over kNN edges:
    mean over edges of (log d_pred - log d_tgt)^2. Returns (B,) per-sample.

    knn_idx: (B, N, K) long, in LOCAL [0, n_valid) indexing.
    """
    B, N, D = V_pred.shape
    K = knn_idx.shape[-1]
    device = V_pred.device
    losses = torch.zeros(B, device=device)

    for b in range(B):
        m_b = mask[b]
        n_valid = int(m_b.sum().item())
        if n_valid < 2:
            continue

        V_pred_b = V_pred[b, m_b]
        V_tgt_b = V_tgt[b, m_b]
        knn_b = knn_idx[b, m_b]

        valid_neighbors = (knn_b >= 0) & (knn_b < n_valid)
        if not valid_neighbors.any():
            continue

        i_local = torch.arange(n_valid, device=device).unsqueeze(1).expand(n_valid, K)
        edge_mask = valid_neighbors & (i_local != knn_b)
        i_edges = i_local[edge_mask]
        j_edges = knn_b[edge_mask]

        valid_j = (j_edges >= 0) & (j_edges < n_valid)
        i_edges = i_edges[valid_j]
        j_edges = j_edges[valid_j]
        if i_edges.numel() == 0:
            continue

        d2_pred = (V_pred_b[i_edges] - V_pred_b[j_edges]).pow(2).sum(dim=-1)
        d2_tgt = (V_tgt_b[i_edges] - V_tgt_b[j_edges]).pow(2).sum(dim=-1)
        d_pred = torch.sqrt(d2_pred + eps)
        d_tgt = torch.sqrt(d2_tgt + eps)

        valid_edges = (d_tgt > 1e-6)
        if not valid_edges.any():
            continue
        d_pred = d_pred[valid_edges]
        d_tgt = d_tgt[valid_edges]

        log_diff = torch.log(d_pred) - torch.log(d_tgt)
        losses[b] = log_diff.pow(2).mean()

    return losses


# ============================================================================
# SNR gating (adaptive quantile gates over noise score = -log c_skip)
# ============================================================================
class AdaptiveQuantileGate:
    """
    Maintains an adaptive threshold so that ~target_rate of samples pass, based
    on a noise score (higher = noisier). mode="low" passes the cleanest fraction
    (lowest noise); mode="high" passes the noisiest fraction.
    """

    def __init__(self, target_rate: float, mode: str = "low", warmup_steps: int = 200,
                 reservoir_size: int = 8192, update_every: int = 50, ema: float = 0.9,
                 clamp_q: tuple = (0.01, 0.99)):
        self.target_rate = float(target_rate)
        self.mode = mode
        self.warmup_steps = int(warmup_steps)
        self.reservoir_size = int(reservoir_size)
        self.update_every = int(update_every)
        self.ema = float(ema)
        self.clamp_q = clamp_q
        self._buf = []
        self._thr = None
        self._step = 0

    @property
    def thr(self):
        return self._thr

    def update(self, noise_1d: torch.Tensor):
        self._step += 1
        x = noise_1d.detach()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return
        self._buf.extend(x.to("cpu", dtype=torch.float32).flatten().tolist())
        if len(self._buf) > self.reservoir_size:
            self._buf = self._buf[-self.reservoir_size:]
        if self._step < self.warmup_steps:
            return
        if (self._step % self.update_every) != 0:
            return
        if len(self._buf) < 64:
            return
        buf = torch.tensor(self._buf, dtype=torch.float32)
        q = self.target_rate if self.mode == "low" else (1.0 - self.target_rate)
        q = min(max(q, self.clamp_q[0]), self.clamp_q[1])
        new_thr = torch.quantile(buf, q).item()
        if self._thr is None:
            self._thr = new_thr
        else:
            self._thr = self.ema * self._thr + (1.0 - self.ema) * new_thr

    def gate(self, noise: torch.Tensor, base_gate: torch.Tensor) -> torch.Tensor:
        if self._thr is None or (self._step < self.warmup_steps):
            return base_gate
        thr = torch.tensor(self._thr, device=noise.device, dtype=noise.dtype)
        pass_mask = (noise <= thr) if self.mode == "low" else (noise >= thr)
        return base_gate * pass_mask.float()


def make_geometry_gates() -> Dict[str, AdaptiveQuantileGate]:
    """Per-loss adaptive gates with their target pass-rates."""
    return {
        "gram": AdaptiveQuantileGate(target_rate=0.50, mode="low", warmup_steps=200),
        "gram_scale": AdaptiveQuantileGate(target_rate=0.60, mode="low", warmup_steps=200),
        "edge": AdaptiveQuantileGate(target_rate=0.30, mode="low", warmup_steps=200),
        "nca": AdaptiveQuantileGate(target_rate=0.40, mode="low", warmup_steps=200),
    }


def noise_score_from_sigma(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """noise = -log(c_skip), c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)."""
    sigma_flat = sigma.view(-1)
    c_skip = (sigma_data ** 2) / (sigma_flat ** 2 + sigma_data ** 2 + 1e-12)
    return -torch.log(c_skip + 1e-12)


def ensure_minimum_coverage(gate: torch.Tensor, score: torch.Tensor, min_count: int,
                            prefer_low_score: bool = True,
                            eligible_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Force at least `min_count` active samples in `gate`, filling from eligible
    non-gated samples ranked by `score` (lowest preferred if prefer_low_score).
    """
    if eligible_mask is None:
        eligible_mask = torch.ones(gate.shape[0], device=gate.device, dtype=torch.bool)

    n_active = ((gate > 0) & eligible_mask).sum().item()
    if n_active >= min_count:
        return gate

    n_needed = min_count - int(n_active)
    not_gated = (gate <= 0) & eligible_mask
    if not_gated.sum() == 0:
        return gate

    scores = score.clone()
    scores[~not_gated] = float('inf')
    if not prefer_low_score:
        scores = -scores
        scores[~not_gated] = float('inf')

    n_to_add = min(n_needed, int(not_gated.sum().item()))
    _, best_idx = scores.topk(n_to_add, largest=False)
    new_gate = gate.clone()
    new_gate[best_idx] = 1.0
    return new_gate


def build_geometry_gates(sigma: torch.Tensor, sigma_data: float, mask: torch.Tensor,
                         gates: Dict[str, AdaptiveQuantileGate],
                         cond_only: Optional[torch.Tensor] = None,
                         min_geometry_samples: int = 4) -> Dict[str, torch.Tensor]:
    """
    Update the adaptive gates and return per-loss (B,) float gate masks
    (gram / gram_scale / edge / nca).

    cond_only: (B,) 1.0 for conditioned samples (from CFG dropout); defaults to
    all-ones. Only conditioned samples are eligible; each gate additionally keeps
    only its low-noise fraction, with a minimum-coverage floor.
    """
    B = mask.shape[0]
    device = mask.device
    if cond_only is None:
        cond_only = torch.ones(B, device=device)
    base_gate = cond_only.float()

    with torch.no_grad():
        noise_score = noise_score_from_sigma(sigma, sigma_data)
        eligible = base_gate > 0.5
        if eligible.any():
            for g in gates.values():
                g.update(noise_score[eligible])

    n_valid_per_sample = mask.sum(dim=1)
    eligible_for_geo = (n_valid_per_sample >= 16) & (base_gate > 0.5)

    out = {}
    for name in ("gram", "gram_scale", "edge", "nca"):
        gm = gates[name].gate(noise_score, base_gate)
        gm = ensure_minimum_coverage(gm, noise_score, min_geometry_samples,
                                     prefer_low_score=True, eligible_mask=eligible_for_geo)
        out[name] = gm
    out["noise_score"] = noise_score
    return out


# ============================================================================
# Structure tensors shared by the geometry losses
# ============================================================================
def build_structure_tensors(V_hat: torch.Tensor, V_target: torch.Tensor, mask: torch.Tensor):
    """
    From composed prediction V_hat and target V_target build, per set:
      V_hat_centered  : centered raw prediction (scale supervision uses this)
      V_struct        : V_hat_centered / RMS(detached)  (structure, scale-free)
      V_target_struct : V_target centered / RMS         (matched normalization)
      V_target_centered
      m_bool, m_float, valid_counts
    """
    V_hat_f32 = V_hat.float()
    m_bool = mask.bool()
    m_float = mask.float().unsqueeze(-1)
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)

    mean = (V_hat_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
    V_hat_centered = (V_hat_f32 - mean) * m_float

    rms_per_sample = (V_hat_centered.pow(2) * m_float).sum(dim=(1, 2)) / valid_counts.squeeze(-1).clamp(min=1)
    rms_per_sample = rms_per_sample.sqrt().clamp(min=1e-8)
    V_struct = V_hat_centered / rms_per_sample.detach().view(-1, 1, 1)
    V_struct = V_struct * m_float

    V_target_centered, _ = center_only(V_target.float(), mask)
    V_target_centered = V_target_centered * m_float
    tgt_rms = (V_target_centered.pow(2) * m_float).sum(dim=(1, 2)) / valid_counts.squeeze(-1).clamp(min=1)
    tgt_rms = tgt_rms.sqrt().clamp(min=1e-8)
    V_target_struct = V_target_centered / tgt_rms.view(-1, 1, 1)
    V_target_struct = V_target_struct * m_float

    return {
        "V_hat_centered": V_hat_centered,
        "V_struct": V_struct,
        "V_target_struct": V_target_struct,
        "V_target_centered": V_target_centered,
        "m_bool": m_bool,
        "m_float": m_float,
        "valid_counts": valid_counts,
    }


@torch.no_grad()
def within_miniset_knn(V_target: torch.Tensor, mask: torch.Tensor, k: int = 15) -> torch.Tensor:
    """
    Per-set kNN in V_target space, restricted to the set's own valid points, so
    every returned neighbor index is guaranteed valid (100% coverage). Returns
    (B, N, k) long with -1 padding for invalid / undersized rows.
    """
    B, N, D = V_target.shape
    device = V_target.device
    knn = torch.full((B, N, k), -1, dtype=torch.long, device=device)
    for b in range(B):
        m_b = mask[b].bool()
        n_valid = int(m_b.sum().item())
        if n_valid < k + 1:
            continue
        valid_indices = torch.where(m_b)[0]
        V_b = V_target[b, valid_indices]
        D_b = torch.cdist(V_b, V_b)
        D_b.fill_diagonal_(float('inf'))
        _, knn_local = D_b.topk(k, dim=1, largest=False)
        knn[b, valid_indices] = valid_indices[knn_local]
    return knn


# ============================================================================
# (1) EDM residual denoising / score loss
# ============================================================================
def edm_residual_score_loss(
    R0_hat: torch.Tensor,
    R_target: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data_resid: float,
    mask: torch.Tensor,
    hi_sigma_compensation: bool = True,
    weight_softcap: bool = True,
    weighted_normalize: bool = True,
    cap_state: Optional[Dict] = None,
    sigma_pivot: float = 0.5,
    hi_sigma_p: float = 2.0,
    hi_sigma_mult: float = 3.0,
    sigma_cap: Optional[float] = None,
    curriculum_state: Optional[Dict] = None,
    R_t: Optional[torch.Tensor] = None,
    score_hi_gate=None,
    boost_state=None,
    hi_boost: bool = False,
    fx_hi: bool = False,
    boost_factor: float = 4.0,
    boost_ramp: int = 200,
    fx_hi_weight: float = 2.0,
    global_step: int = 0,
) -> torch.Tensor:
    """
    EDM residual score loss:

        err2_i    = mean_d (R0_hat - R_target)^2                (per node)
        err2_b    = sum_i err2_i * M_i / sum_i M_i              (per set)
        w(sigma)  = (sigma^2 + sd^2) / (sigma * sd)^2           (sd = sigma_data_resid)
        L_score   = sum_b w_b * err2_b / sum_b w_b              (weighted_normalize)

    Optional deterministic weight shaping applied on top of the base EDM weight:
      - high-sigma compensation g(sigma) = (sigma / sigma_pivot)^hi_sigma_p for
        sigma > sigma_pivot,
      - an adaptive soft-cap  w = cap * tanh(w_raw / cap), where `cap` is an
        EMA (decay 0.98) of the 95th percentile of w_raw (state kept in
        `cap_state`),
      - a flat multiplier hi_sigma_mult on samples with sigma > sigma_pivot.

    Set the corresponding flags False to recover the pure EDM-weighted loss.

    When `sigma_cap` and `curriculum_state` are supplied, the per-set squared
    error inside the cap band [0.8 * sigma_cap, sigma_cap] is accumulated into
    the curriculum state (drives the loss-plateau gate of the curriculum).
    """
    R0_hat = R0_hat.float()
    R_target = R_target.float()
    mask_fp32 = mask.float()
    sigma_flat = sigma.view(-1).float()

    err2_node = (R0_hat - R_target).pow(2).mean(dim=-1)          # (B, N)
    den = mask_fp32.sum(dim=1).clamp_min(1.0)                    # (B,)
    err2_sample = (err2_node * mask_fp32).sum(dim=1) / den       # (B,)

    w_raw = edm_loss_weight(sigma_flat, sigma_data_resid).float()  # (B,)

    if hi_sigma_compensation:
        g_sigma = torch.ones_like(sigma_flat)
        hi_mask = sigma_flat > sigma_pivot
        g_sigma[hi_mask] = (sigma_flat[hi_mask] / sigma_pivot).pow(hi_sigma_p)
        w_raw = w_raw * g_sigma

    if weight_softcap:
        if cap_state is None:
            cap_state = {}
        cap_now = torch.quantile(w_raw.detach(), 0.95)
        if cap_state.get("w_cap_ema", None) is None:
            cap_state["w_cap_ema"] = cap_now
        else:
            cap_state["w_cap_ema"] = 0.98 * cap_state["w_cap_ema"] + 0.02 * cap_now
        cap = cap_state["w_cap_ema"].clamp_min(1e-6)
        w = cap * torch.tanh(w_raw / cap)
    else:
        w = w_raw

    w_eff = w
    hi_sigma_mask = (sigma_flat > sigma_pivot)
    if hi_sigma_mask.any():
        hi_mult = torch.ones_like(w_eff)
        hi_mult[hi_sigma_mask] = hi_sigma_mult
        w_eff = w_eff * hi_mult

    # HI_BOOST / FX_HI (opt-in; ported from full_v1 EXP_SCORE_HI_BOOST / FX_HI).
    # The high-noise gate is shared by both; HI_BOOST reweights L_score, FX_HI
    # adds a direct F_x supervision term and refreshes the boost readiness ratio.
    gate_hi = None
    if (hi_boost or fx_hi) and score_hi_gate is not None:
        gate_hi = high_noise_gate(sigma_flat, sigma_data_resid, score_hi_gate)
    if hi_boost and gate_hi is not None and boost_state is not None:
        w_eff = w_eff * hi_boost_multiplier(
            gate_hi, sigma_flat, boost_state, global_step,
            boost_factor=boost_factor, ramp_steps=boost_ramp)

    if weighted_normalize:
        L_score = (w_eff * err2_sample).sum() / w_eff.sum().clamp(min=1e-8)
    else:
        L_score = (w_eff * err2_sample).mean()

    if fx_hi and gate_hi is not None and R_t is not None:
        L_fx_hi, fx_ratio = fx_hi_loss(
            R0_hat, R_target, R_t, sigma, sigma_data_resid, mask, gate_hi, w)
        if boost_state is not None and fx_ratio is not None:
            boost_state.last_fx_ratio_hi = fx_ratio
        L_score = L_score + fx_hi_weight * L_fx_hi

    # Cap-band error accumulation for the curriculum loss-plateau gate.
    if sigma_cap is not None and curriculum_state is not None:
        with torch.no_grad():
            cb_lo = 0.8 * sigma_cap
            cb_mask = (sigma_flat >= cb_lo) & (sigma_flat <= sigma_cap)
            n_cb = int(cb_mask.sum().item())
            if n_cb > 0:
                cb_err2 = err2_sample[cb_mask].mean().item()
                curriculum_state['cap_band_loss_sum'] += cb_err2 * n_cb
                curriculum_state['cap_band_loss_count'] += n_cb

    return L_score


# ============================================================================
# (2) Gram losses: scale-normalized Frobenius + log-trace scale term
# ============================================================================
def gram_losses(
    V_struct: torch.Tensor,
    V_target_struct: torch.Tensor,
    geo_gate_gram: torch.Tensor,
    geo_gate_gram_scale: torch.Tensor,
    m_bool: torch.Tensor,
    m_float: torch.Tensor,
    diag_weight: float = 0.5,
    energy_floor_frac: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L_gram       : scale-normalized Gram Frobenius loss.
        Gp = V_struct @ V_struct^T,  Gt = V_target_struct @ V_target_struct^T
        off-diagonal: per-set  mean_pair (Gp-Gt)^2 / clamp(mean_pair Gt^2, floor)
        diagonal:     per-set  sum_i (diag_p-diag_t)^2 / sum_i diag_t^2
        L_gram = <off>_gate + diag_weight * <diag>_gate

    L_gram_scale : log-trace scale term.
        log_ratio = log(tr Gp) - log(tr Gt);  L = <log_ratio^2>_gate

    Both are averaged only over gated (low-noise) sets.
    """
    Gp = V_struct @ V_struct.transpose(1, 2)          # (B, N, N)
    Gt = V_target_struct @ V_target_struct.transpose(1, 2)

    N = Gp.shape[1]
    MM = (m_bool.unsqueeze(-1) & m_bool.unsqueeze(-2)).float()
    eye = torch.eye(N, dtype=torch.bool, device=Gp.device).unsqueeze(0)
    P_off = (MM.bool() & (~eye)).float()

    # Off-diagonal, scale-normalized per set.
    diff_raw = (Gp - Gt) * P_off
    pair_cnt = P_off.sum(dim=(1, 2)).clamp_min(1.0)
    numerator = diff_raw.pow(2).sum(dim=(1, 2)) / pair_cnt
    t_energy_raw = (Gt.pow(2) * P_off).sum(dim=(1, 2)) / pair_cnt
    t_energy_median = t_energy_raw.detach().median().clamp_min(1e-12)
    denominator = t_energy_raw.clamp_min(energy_floor_frac * t_energy_median)
    per_set_relative_loss = numerator / denominator

    # Diagonal (per-point norm) matching, scale-normalized per set.
    diag_p = torch.diagonal(Gp, dim1=-2, dim2=-1)
    diag_t = torch.diagonal(Gt, dim1=-2, dim2=-1)
    m1 = m_bool.float()
    den_diag = (diag_t.pow(2) * m1).sum(dim=-1).clamp_min(1e-8)
    diag_rel = ((diag_p - diag_t).pow(2) * m1).sum(dim=-1) / den_diag

    per_set_relative_loss = torch.nan_to_num(per_set_relative_loss, nan=0.0, posinf=0.0, neginf=0.0)
    diag_rel = torch.nan_to_num(diag_rel, nan=0.0, posinf=0.0, neginf=0.0)

    gate_sum = geo_gate_gram.sum().clamp(min=1.0)
    L_gram_offdiag = (per_set_relative_loss * geo_gate_gram).sum() / gate_sum
    L_gram_diag = (diag_rel * geo_gate_gram).sum() / gate_sum
    L_gram = L_gram_offdiag + diag_weight * L_gram_diag

    # Log-trace scale term.
    tr_p = (diag_p * m1).sum(dim=1)
    tr_t = (diag_t * m1).sum(dim=1)
    log_ratio = torch.log(tr_p + 1e-8) - torch.log(tr_t + 1e-8)
    log_ratio_sq = torch.nan_to_num(log_ratio ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    gate_sum_scale = geo_gate_gram_scale.sum().clamp(min=1.0)
    L_gram_scale = (log_ratio_sq * geo_gate_gram_scale).sum() / gate_sum_scale

    return L_gram, L_gram_scale


def gram_learn_loss(
    V_hat_centered: torch.Tensor,
    V_target: torch.Tensor,
    V_t: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data_resid: float,
    mask: torch.Tensor,
    m_bool: torch.Tensor,
    m_float: torch.Tensor,
    diag_weight: float = 0.5,
    cskip_thresh: float = 0.05,
    inv_cout2_cap: float = 64.0,
    energy_floor_frac: float = 0.05,
) -> torch.Tensor:
    """
    High-sigma learned-branch Gram loss. Isolates the learned EDM branch
    V_out = x0 - c_skip * x_c and matches its Gram to the target's learned
    branch, only on very-high-noise samples (c_skip < cskip_thresh), weighted
    by (1 - c_skip)^2 and compensated by clamp(1/c_out^2, cap).
    """
    sigma_flat = sigma.view(-1).float()
    c_skip, c_out, _, _ = edm_precond(sigma_flat, sigma_data_resid)
    c_skip_1d = c_skip.view(-1)
    c_out_1d = c_out.view(-1).clamp(min=1e-6)

    V_c, _ = center_only(V_t, mask)
    V_out = (V_hat_centered - c_skip * V_c) * m_float
    V_tgt_c, _ = center_only(V_target, mask)
    V_out_tgt = (V_tgt_c - c_skip * V_c) * m_float

    sel = (c_skip_1d < cskip_thresh).float()
    inv_cout2 = (1.0 / c_out_1d.pow(2)).clamp(max=inv_cout2_cap)
    w_gl = sel * (1.0 - c_skip_1d).pow(2) * inv_cout2  # (B,) weight, currently unused

    G_pred = V_out @ V_out.transpose(1, 2)
    G_tgt = V_out_tgt @ V_out_tgt.transpose(1, 2)

    N = G_pred.shape[1]
    MM = (m_bool.unsqueeze(-1) & m_bool.unsqueeze(-2)).float()
    eye = torch.eye(N, device=G_pred.device).unsqueeze(0)
    P_off = MM * (1.0 - eye)

    pair_cnt = P_off.sum(dim=(1, 2)).clamp_min(1.0)
    num = ((G_pred - G_tgt).pow(2) * P_off).sum(dim=(1, 2)) / pair_cnt
    den = (G_tgt.pow(2) * P_off).sum(dim=(1, 2)) / pair_cnt
    den_med = den.detach().median().clamp_min(1e-12)
    den = den.clamp_min(energy_floor_frac * den_med)
    loss_off = num / den

    diag_p = torch.diagonal(G_pred, dim1=-2, dim2=-1)
    diag_t = torch.diagonal(G_tgt, dim1=-2, dim2=-1)
    m1 = m_bool.float()
    den_d = (diag_t.pow(2) * m1).sum(dim=-1).clamp_min(1e-8)
    loss_d = ((diag_p - diag_t).pow(2) * m1).sum(dim=-1) / den_d

    per_sample = torch.nan_to_num(loss_off + diag_weight * loss_d, nan=0.0, posinf=0.0, neginf=0.0)
    wsum = sel.sum().clamp_min(1.0)
    return (per_sample * sel).sum() / wsum


def out_scale_loss(
    V_hat_centered: torch.Tensor,
    V_target: torch.Tensor,
    V_t: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data_resid: float,
    mask: torch.Tensor,
    m_float: torch.Tensor,
    cskip_thresh: float = 0.25,
) -> torch.Tensor:
    """
    Learned-branch scale calibration in F_x space. On high-noise samples
    (c_skip < cskip_thresh), match the per-set RMS of the learned branch
    F_x = (x0 - c_skip * x_c) / c_out between prediction and target via a
    log-ratio penalty, weighted by (1 - c_skip)^2.
    """
    sigma_flat = sigma.view(-1).float()
    c_skip, c_out, _, _ = edm_precond(sigma_flat, sigma_data_resid)
    c_skip_1d = c_skip.view(-1)

    x_c, _ = center_only(V_t, mask)
    x0_c, _ = center_only(V_target, mask)

    V_out_pred = (V_hat_centered - c_skip * x_c) * m_float
    V_out_tgt = (x0_c - c_skip * x_c) * m_float

    hi = (c_skip_1d < cskip_thresh)
    if not hi.any():
        return V_hat_centered.new_tensor(0.0)

    c_out_hi = c_out[hi].clamp(min=1e-6)
    Fx_pred = V_out_pred[hi] / c_out_hi
    Fx_tgt = V_out_tgt[hi] / c_out_hi

    mf_hi = m_float[hi]
    denom_pts = mf_hi.sum(dim=(1, 2)).clamp(min=1.0)
    rms_pred = ((Fx_pred.pow(2) * mf_hi).sum(dim=(1, 2)) / denom_pts).sqrt()
    rms_tgt = ((Fx_tgt.pow(2) * mf_hi).sum(dim=(1, 2)) / denom_pts).sqrt()

    log_ratio_fx = torch.log(rms_pred + 1e-8) - torch.log(rms_tgt + 1e-8)
    gate_w = (1.0 - c_skip_1d[hi]).pow(2)
    return ((log_ratio_fx ** 2) * gate_w).sum() / gate_w.sum().clamp(min=1.0)


# ============================================================================
# (3) kNN neighborhood loss + edge-wise local scale penalties
# ============================================================================
def knn_nca(
    V_struct: torch.Tensor,
    V_target_struct: torch.Tensor,
    struct_mask: torch.Tensor,
    geo_gate_nca: torch.Tensor,
    tau_reference: float,
    k: int = 15,
) -> torch.Tensor:
    """
    Gated kNN-NCA neighborhood-preservation loss on the normalized structure
    tensors. `tau_reference` is the squared 15th-NN distance of the target
    coordinates (data-driven temperature).
    """
    L_per = knn_nca_loss(
        V_struct, V_target_struct, struct_mask,
        k=k, temperature=tau_reference,
        return_per_sample=True, scale_compensate=True, point_weight=None,
    )
    L_per = torch.nan_to_num(L_per, nan=0.0, posinf=0.0, neginf=0.0)
    gate_sum = geo_gate_nca.sum().clamp(min=1.0)
    return (L_per * geo_gate_nca).sum() / gate_sum


def knn_scale(
    V_hat_centered: torch.Tensor,
    V_target: torch.Tensor,
    struct_mask: torch.Tensor,
    knn_indices: torch.Tensor,
    noise_score: torch.Tensor,
    mask: torch.Tensor,
    k: int = 15,
    min_geometry_samples: int = 4,
) -> torch.Tensor:
    """
    Edge-wise local scale penalty (raw, unclamped prediction). Per-set edgewise
    log-ratio over within-miniset kNN edges, gated to sets with >= 16 valid
    points (min-coverage floored).
    """
    L_per = knn_scale_loss(
        V_hat_centered, V_target.float(), struct_mask,
        knn_indices=knn_indices, k=k, return_per_sample=True,
    )
    L_per = torch.nan_to_num(L_per, nan=0.0, posinf=0.0, neginf=0.0)

    n_valid_per_sample = mask.sum(dim=1)
    scale_gate = (n_valid_per_sample >= 16).float()
    scale_gate = ensure_minimum_coverage(
        scale_gate, noise_score, min_count=min_geometry_samples,
        prefer_low_score=False, eligible_mask=(n_valid_per_sample >= 16),
    )
    gate_sum = scale_gate.sum().clamp(min=1.0)
    return (L_per * scale_gate).sum() / gate_sum


def edge_loss(
    V_struct: torch.Tensor,
    V_target_struct: torch.Tensor,
    struct_mask: torch.Tensor,
    knn_indices: torch.Tensor,
    geo_gate_edge: torch.Tensor,
) -> torch.Tensor:
    """Gated multiplicative edge-length loss over within-miniset kNN edges."""
    L_per = edge_log_ratio_loss(
        V_pred=V_struct, V_tgt=V_target_struct, knn_idx=knn_indices, mask=struct_mask,
    )
    L_per = torch.nan_to_num(L_per, nan=0.0, posinf=0.0, neginf=0.0)
    gate_sum = geo_gate_edge.sum().clamp(min=1.0)
    return (L_per * geo_gate_edge).sum() / gate_sum


def subspace_loss(V_struct: torch.Tensor, mask: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Low-rank subspace penalty: variance outside the top-k principal axes."""
    return variance_outside_topk(V_struct, mask, k=k)


# ============================================================================
# (4) Generator supervision (on the prior V_base = generator(H))
# ============================================================================
def generator_supervision(
    V_gen: torch.Tensor,
    V_target: torch.Tensor,
    V_target_struct: torch.Tensor,
    struct_mask: torch.Tensor,
    knn_indices: torch.Tensor,
    mask: torch.Tensor,
    gen_scale_local_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Supervise the generator prior V_gen directly:

      L_gen_align : orthogonal-Procrustes MSE (no scale) between normalized
                    generator structure and normalized target structure.
      L_gen_gram  : masked Gram MSE between the two normalized structures
                    (all valid pairs, diagonal included).
      L_gen_scale : rms_log_loss(V_gen_centered, V_target) global scale term
                    + gen_scale_local_weight * edge-wise local scale
                    (knn_scale_loss on within-miniset kNN).
    """
    V_gen_f32 = V_gen.float()
    m_float = mask.float().unsqueeze(-1)
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)

    mean_Vgen = (V_gen_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
    V_gen_centered = (V_gen_f32 - mean_Vgen) * m_float

    gen_rms = (V_gen_centered.pow(2) * m_float).sum(dim=(1, 2)) / valid_counts.squeeze(-1).clamp(min=1)
    gen_rms = gen_rms.sqrt().clamp(min=1e-8)
    V_gen_struct = V_gen_centered / gen_rms.detach().view(-1, 1, 1)
    V_gen_struct = V_gen_struct * m_float

    L_gen_align = rigid_align_mse_no_scale(V_gen_struct, V_target_struct, mask)

    G_gen = V_gen_struct @ V_gen_struct.transpose(1, 2)
    G_tgt = V_target_struct @ V_target_struct.transpose(1, 2)
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    diff_gram = (G_gen - G_tgt).pow(2) * mask_2d
    L_gen_gram = diff_gram.sum() / mask_2d.sum().clamp(min=1)

    V_target_batch = V_target.float()
    L_gen_scale = rms_log_loss(V_gen_centered, V_target_batch, mask)
    if knn_indices is not None:
        L_gen_scale_local = knn_scale_loss(
            V_gen_centered, V_target_batch, struct_mask,
            knn_indices=knn_indices, k=15, return_per_sample=False,
        )
        L_gen_scale = L_gen_scale + gen_scale_local_weight * L_gen_scale_local

    return L_gen_align, L_gen_gram, L_gen_scale


# ============================================================================
# Loss weights + total assembly
# ============================================================================
# Active Stage-C loss weights.
STAGE_C_WEIGHTS: Dict[str, float] = {
    "score": 16.0,        # EDM residual denoising
    "gram": 1.0,          # scale-normalized Gram Frobenius
    "gram_scale": 1.0,    # log-trace scale term
    "out_scale": 2.0,     # learned-branch F_x scale
    "gram_learn": 1.0,    # high-sigma learned-branch Gram
    "knn_nca": 1.0,       # kNN neighborhood (NCA)
    "knn_scale": 0.1,     # edge-wise local scale penalty
    "edge": 2.0,          # multiplicative edge-length
    "subspace": 0.25,     # low-rank subspace penalty
    "gen_align": 10.0,    # generator Procrustes MSE
    "gen_gram": 10.0,     # generator Gram
    "gen_scale": 10.0,    # generator scale (global + local)
}

# Convenience alias.
WEIGHTS = STAGE_C_WEIGHTS


def assemble_total_loss(losses: Dict[str, torch.Tensor],
                        weights: Dict[str, float] = STAGE_C_WEIGHTS,
                        score_multiplier: float = 1.0) -> torch.Tensor:
    """
    Weighted sum of the active Stage-C terms. `losses` maps the STAGE_C_WEIGHTS
    keys to their scalar values; missing keys contribute zero.
    """
    device = None
    for v in losses.values():
        if torch.is_tensor(v):
            device = v.device
            break
    total = torch.zeros((), device=device)
    for key, w in weights.items():
        if key not in losses:
            continue
        mult = score_multiplier if key == "score" else 1.0
        total = total + w * mult * losses[key]
    return total


__all__ = [
    "edm_loss_weight",
    "edm_precond",
    "center_only",
    "rigid_align_apply_no_scale",
    "rigid_align_mse_no_scale",
    "rms_log_loss",
    "variance_outside_topk",
    "knn_nca_loss",
    "knn_scale_loss",
    "edge_log_ratio_loss",
    "AdaptiveQuantileGate",
    "make_geometry_gates",
    "noise_score_from_sigma",
    "ensure_minimum_coverage",
    "build_geometry_gates",
    "build_structure_tensors",
    "within_miniset_knn",
    "edm_residual_score_loss",
    "gram_losses",
    "gram_learn_loss",
    "out_scale_loss",
    "knn_nca",
    "knn_scale",
    "edge_loss",
    "subspace_loss",
    "generator_supervision",
    "assemble_total_loss",
    "STAGE_C_WEIGHTS",
    "WEIGHTS",
]
