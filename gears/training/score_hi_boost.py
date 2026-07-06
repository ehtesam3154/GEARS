"""High-noise score-loss terms from the original full_v1 Stage-C recipe.

The original training (model/core_models_et_p2.py) had two *experimental* terms
ACTIVE that shaped the score loss at high noise, and which the clean gears/ port
initially omitted:

  * HI_BOOST  (EXP_SCORE_HI_BOOST): up to ``boost_factor``x extra weight on the
    noisiest ``target_rate`` fraction of samples in L_score, gated on a
    data-driven readiness signal and ramped in over ``ramp_steps`` steps, with a
    tail-safety cap so the extreme-sigma tail is not boosted as hard.
  * FX_HI     (EXP_SCORE_FX_HI): direct F_x-space MSE supervision of the learned
    EDM branch at high noise (weight ``fx_hi_weight``). At high sigma
    c_skip -> 0 so x0 ~= c_out * F_x; supervising F_x forces the learned branch
    to carry correct geometry where the skip connection cannot.

The two are coupled: FX_HI produces the F_x scale ratio that drives HI_BOOST
readiness. The original scattered that ratio computation ~2000 lines away from
the loss; here it is consolidated into :func:`fx_hi_loss`. The math matches the
original (edm_precond c_skip/c_out, per-set masked means, c_out^2 unit match,
readiness EMA of |log(fx_ratio)|). Both terms are opt-in and default off, so
existing runs are byte-unchanged.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch

from ..models.denoiser import center_only, edm_precond


@dataclass
class HiBoostState:
    """Data-driven readiness state for HI_BOOST (was a dict in the original)."""
    ready: bool = False
    start_step: Optional[int] = None
    ema_fx_ratio_hi: float = 1.0
    last_fx_ratio_hi: Optional[float] = None   # set by fx_hi_loss each step
    stable_count: int = 0
    stability_tol: float = 0.10
    min_stable_checks: int = 3
    ema_decay: float = 0.98


def high_noise_gate(sigma_flat: torch.Tensor, sigma_data_resid: float, gate) -> torch.Tensor:
    """Update the adaptive quantile gate and return the per-sample high-noise
    gate (B,), 1.0 for the noisiest ``target_rate`` fraction (0/1 after warmup,
    all-ones during warmup). ``gate`` is an AdaptiveQuantileGate(mode="high")."""
    with torch.no_grad():
        c_skip = (sigma_data_resid ** 2) / (sigma_flat ** 2 + sigma_data_resid ** 2 + 1e-12)
        noise = -torch.log(c_skip + 1e-12)           # (B,) higher = noisier
        gate.update(noise)
        return gate.gate(noise, torch.ones_like(noise))


def hi_boost_multiplier(gate_hi: torch.Tensor, sigma_flat: torch.Tensor,
                        state: HiBoostState, global_step: int,
                        boost_factor: float = 4.0, ramp_steps: int = 200,
                        tail_quantile: float = 0.95, tail_cap: float = 2.0) -> torch.Tensor:
    """Per-sample score-loss weight multiplier (B,). Updates ``state`` readiness
    from ``state.last_fx_ratio_hi`` (the previous step's F_x ratio) and ramps the
    boost in once ready. Faithful to EXP_SCORE_HI_BOOST + tail-safety cap."""
    fx_ratio = state.last_fx_ratio_hi
    if fx_ratio is not None and fx_ratio > 0:
        state.ema_fx_ratio_hi = (state.ema_decay * state.ema_fx_ratio_hi
                                 + (1.0 - state.ema_decay) * fx_ratio)
        stable = abs(math.log(state.ema_fx_ratio_hi)) < state.stability_tol
        state.stable_count = state.stable_count + 1 if stable else 0
        if (not state.ready) and state.stable_count >= state.min_stable_checks:
            state.ready = True
            state.start_step = global_step

    if not state.ready:
        return torch.ones_like(sigma_flat)
    ramp = min(1.0, (global_step - state.start_step) / max(1, ramp_steps))

    base_boost = 1.0 + ramp * (boost_factor - 1.0) * gate_hi
    tail_thr = sigma_flat.quantile(tail_quantile)
    is_tail = sigma_flat >= tail_thr
    tail_boost = 1.0 + ramp * (tail_cap - 1.0) * gate_hi
    return torch.where(is_tail, torch.minimum(base_boost, tail_boost), base_boost)


def fx_hi_loss(R0_hat: torch.Tensor, R_target: torch.Tensor, R_t: torch.Tensor,
               sigma_t: torch.Tensor, sigma_data_resid: float, mask: torch.Tensor,
               gate_hi: torch.Tensor, w: torch.Tensor,
               cout2_match: bool = True) -> Tuple[torch.Tensor, Optional[float]]:
    """Gated F_x-space MSE at high noise, plus the high-sigma F_x scale ratio.

    ``R0_hat`` = predicted residual (x0), ``R_target`` = target residual,
    ``R_t`` = noisy residual input (diff_target + sigma*eps), ``sigma_t`` the
    per-set sigma, ``w`` the base EDM weights, ``gate_hi`` the high-noise gate.
    Returns (L_fx_hi, fx_ratio_hi) where fx_ratio_hi feeds HI_BOOST readiness
    (None if no high-sigma samples this batch). Faithful to EXP_SCORE_FX_HI."""
    mask_f = mask.float()
    m3 = mask_f.unsqueeze(-1)
    den = mask_f.sum(dim=1).clamp_min(1.0)                       # (B,)
    sigma_flat = sigma_t.view(-1).float()
    c_skip, c_out, _, _ = edm_precond(sigma_flat, sigma_data_resid)   # (B,1,1)

    V_c, _ = center_only(R_t.float(), mask);      V_c = V_c * m3
    x0_c, _ = center_only(R0_hat.float(), mask);  x0_c = x0_c * m3
    tgt_c, _ = center_only(R_target.float(), mask); tgt_c = tgt_c * m3

    eps = 1e-8
    F_pred = (x0_c - c_skip * V_c) / (c_out + eps)               # (B,N,D)
    F_tgt = (tgt_c - c_skip * V_c) / (c_out + eps)

    err2_point = (F_pred - F_tgt).pow(2).mean(dim=-1)            # (B,N)
    err2_sample = (err2_point * mask_f).sum(dim=1) / den         # (B,)
    if cout2_match:
        err2_sample = err2_sample * (c_out.view(-1) ** 2).detach()
    err2_sample = torch.nan_to_num(err2_sample, nan=0.0, posinf=0.0, neginf=0.0)

    num = (w * err2_sample * gate_hi).sum()
    den_g = (w * gate_hi).sum().clamp(min=1e-8)
    L_fx_hi = num / den_g

    with torch.no_grad():
        d3 = m3.sum(dim=(1, 2)).clamp(min=1.0)                   # (B,)
        rms_pred = ((F_pred.pow(2) * m3).sum(dim=(1, 2)) / d3).sqrt()
        rms_tgt = ((F_tgt.pow(2) * m3).sum(dim=(1, 2)) / d3).sqrt()
        ratio = rms_pred / rms_tgt.clamp(min=1e-8)               # (B,)
        hi_mask = c_skip.view(-1) < 0.25
        fx_ratio_hi = ratio[hi_mask].median().item() if bool(hi_mask.any()) else None

    return L_fx_hi, fx_ratio_hi
