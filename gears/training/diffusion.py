"""
Stage-C training: conditional geometry generator + EDM residual-diffusion refiner.

Trains three networks jointly on pose-free ST mini-sets:
    context_encoder : SetEncoderContext  -- set -> per-point conditioning H.
    generator       : MetricSetGenerator -- H -> base coordinates V_base.
    score_net       : DiffusionScoreNet  -- EDM-preconditioned residual denoiser.

The generator produces a base embedding V_base; the target coordinates are
Procrustes-aligned (rotation/reflection only) into the V_base frame and the
residual R = V_target_aligned - V_base is what the score network denoises. The
denoiser is EDM-preconditioned; the residual data scale sigma_data_resid is
estimated empirically and re-estimated once the generator has warmed up.

A sigma-cap curriculum controls the noise range: training starts at a low cap
(1x sigma_data) and is promoted stage-by-stage up to 4x. Stages advance on a
step budget -- reaching `curriculum_target_stage` with ~`target_dwell_frac` of
the run left to dwell at the top cap. If a fixed-batch evaluator is supplied via
`eval_fixed_batch_fn`, it can additionally promote earlier once the model shows
competence.

Set `use_residual_diffusion=False` to train the denoiser directly on the aligned
absolute coordinates instead of the residual.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from gears.data import collate_minisets, factor_from_gram
from gears.training.score_hi_boost import HiBoostState
from gears.training.losses_geom import (
    STAGE_C_WEIGHTS,
    AdaptiveQuantileGate,
    rigid_align_apply_no_scale,
    make_geometry_gates,
    build_geometry_gates,
    build_structure_tensors,
    within_miniset_knn,
    edm_residual_score_loss,
    gram_losses,
    gram_learn_loss,
    out_scale_loss,
    knn_nca,
    knn_scale,
    edge_loss,
    subspace_loss,
    generator_supervision,
    assemble_total_loss,
)


# =============================================================================
# Vendored helpers
# =============================================================================
def apply_context_augmentation(Z_set, mask, noise_std=0.02, dropout_rate=0.1):
    """
    Stochastic augmentation of the conditioning embeddings Z_set only (geometry
    targets are untouched): feature-RMS-scaled Gaussian noise plus per-batch
    feature dropout with inverted-dropout rescaling.
    """
    B, N, h_dim = Z_set.shape
    Z_aug = Z_set.clone()

    mask_expanded = mask.unsqueeze(-1).float()
    valid_count = mask_expanded.sum(dim=(0, 1)).clamp(min=1)
    feature_rms = ((Z_set ** 2) * mask_expanded).sum(dim=(0, 1)) / valid_count
    feature_rms = torch.sqrt(feature_rms + 1e-8)

    if noise_std > 0:
        noise = torch.randn_like(Z_aug) * (noise_std * feature_rms)
        Z_aug = Z_aug + noise * mask_expanded

    if dropout_rate > 0:
        dropout_mask = (torch.rand(B, 1, h_dim, device=Z_aug.device) > dropout_rate).float()
        Z_aug = Z_aug * dropout_mask / (1.0 - dropout_rate + 1e-8)

    return Z_aug


def normalize_and_scale_conditioning(H, mask, alpha, eps=1e-8):
    """
    Normalize H to unit RMS per sample (over valid points and channels), then
    scale by alpha. RMS is detached so the context encoder cannot game it.
    Returns (H_scaled, rms_before).
    """
    B, N, c_dim = H.shape
    mask_f = mask.unsqueeze(-1).float()
    H_masked = H * mask_f

    n_valid = mask.sum(dim=1).float()
    denominator = (n_valid * c_dim).clamp(min=1.0)
    sq_sum = (H_masked ** 2).sum(dim=(1, 2))
    rms_H = (sq_sum / denominator + eps).sqrt().detach()

    H_norm = H / rms_H.view(B, 1, 1).clamp(min=eps)
    H_scaled = H_norm * alpha
    H_scaled = H_scaled * mask_f
    return H_scaled, rms_H


def apply_z_ln(Z_set, context_encoder):
    """LayerNorm over the feature dimension of Z_set."""
    return F.layer_norm(Z_set, (Z_set.shape[-1],))


def sample_sigma_capband(batch_size, sigma_cap, cap_band_frac,
                         cap_band_lo_mult=0.6, sigma_min=0.02,
                         P_mean=-1.2, P_std=1.2, device='cuda'):
    """
    Cap-band emphasis noise-level sampling. A fraction `cap_band_frac` of the
    batch is drawn log-uniform from [cap_band_lo_mult * sigma_cap, sigma_cap];
    the remainder is drawn from a clamped log-normal over [sigma_min, sigma_cap].
    Samples are shuffled so the two groups interleave.
    """
    cap_band_lo = max(cap_band_lo_mult * sigma_cap, sigma_min)
    n_cap_band = int(batch_size * cap_band_frac)
    n_full_range = batch_size - n_cap_band

    sigma_parts = []
    if n_cap_band > 0:
        log_lo = math.log(cap_band_lo)
        log_hi = math.log(sigma_cap)
        if log_hi > log_lo + 1e-8:
            log_sigma_cb = torch.rand(n_cap_band, device=device) * (log_hi - log_lo) + log_lo
            sigma_parts.append(log_sigma_cb.exp())
        else:
            sigma_parts.append(torch.full((n_cap_band,), sigma_cap, device=device))
    if n_full_range > 0:
        rnd_normal = torch.randn(n_full_range, device=device)
        sigma_full = (rnd_normal * P_std + P_mean).exp().clamp(sigma_min, sigma_cap)
        sigma_parts.append(sigma_full)

    sigma = torch.cat(sigma_parts, dim=0)
    sigma = sigma[torch.randperm(sigma.shape[0], device=device)]
    return sigma


def init_st_dist_bins_from_data(coords, n_bins=24, mode="log", max_quantile=0.99):
    """
    Build distance bin edges from ST coordinates. Returns a (n_bins+1,) tensor of
    increasing edges used by the score network's distance-bias head.
    """
    from scipy.spatial.distance import pdist

    if coords.shape[0] > 2000:
        idx = np.random.choice(coords.shape[0], 2000, replace=False)
        coords = coords[idx]

    D = torch.from_numpy(pdist(coords, metric="euclidean").astype(np.float32))
    D_pos = D[D > 0]

    if mode == "quantile":
        qs = torch.linspace(0., max_quantile, n_bins + 1)
        edges = torch.quantile(D_pos, qs)
        edges[0] = 0.0
    elif mode == "log":
        eps = 1e-3
        d_min = torch.quantile(D_pos, 0.01).item()
        d_max = torch.quantile(D_pos, max_quantile).item()
        edges = torch.exp(torch.linspace(np.log(d_min + eps), np.log(d_max + eps), n_bins + 1))
        edges[0] = 0.0
    else:  # linear
        d_max = torch.quantile(D_pos, max_quantile).item()
        edges = torch.linspace(0.0, d_max, n_bins + 1)

    return edges


def _compute_sigma_data_resid_aligned(st_loader, context_encoder, generator, device, n_batches=10):
    """Median per-sample RMS of Procrustes-aligned residuals R = align(V) - V_base."""
    resid_rms_list = []
    it = iter(st_loader)
    for _ in range(min(n_batches, len(st_loader))):
        sample_batch = next(it, None)
        if sample_batch is None:
            break
        V_batch = sample_batch['V_target'].to(device, non_blocking=True)
        Z_batch = sample_batch['Z_set'].to(device, non_blocking=True)
        mask_batch = sample_batch['mask'].to(device, non_blocking=True)
        with torch.no_grad():
            H_batch = context_encoder(Z_batch, mask_batch)
            V_base_batch = generator(H_batch, mask_batch)
            V_target_aligned = rigid_align_apply_no_scale(V_batch, V_base_batch, mask_batch)
            R_batch = V_target_aligned - V_base_batch
        for i in range(min(4, V_batch.shape[0])):
            m = mask_batch[i].bool()
            if m.sum() > 0:
                resid_rms_list.append(R_batch[i, m].pow(2).mean().sqrt().item())
    return float(np.median(resid_rms_list)) if resid_rms_list else None


# =============================================================================
# Config
# =============================================================================
@dataclass
class StageCConfig:
    """Stage-C hyperparameters (defaults for the residual-diffusion run)."""

    # optimization
    n_epochs: int = 1000
    batch_size: int = 8
    lr: float = 3e-5
    ema_decay: float = 0.999

    # diffusion / EDM
    sigma_min: float = 0.02
    sigma_max: float = 3.0
    use_edm: bool = True
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_refine_max: Optional[float] = None      # None -> 20.0 * sigma_data
    use_residual_diffusion: bool = True
    sigma_resid_recompute_step: int = 3000

    # conditioning augmentation
    z_noise_std: float = 0.02
    z_dropout_rate: float = 0.1
    aug_prob: float = 0.5
    use_z_ln: bool = False
    cond_alpha: float = 0.5

    # high-noise score-loss shaping (full_v1 EXP_SCORE_HI_BOOST / FX_HI; off by default)
    score_hi_boost: bool = False
    score_fx_hi: bool = False
    score_hi_boost_factor: float = 4.0
    score_hi_boost_ramp: int = 200
    score_hi_target_rate: float = 0.25
    score_fx_hi_weight: float = 2.0

    # classifier-free-guidance context dropout
    cfg_context_dropout: bool = True
    cfg_warmup_start: int = 20
    cfg_warmup_len: int = 20
    p_uncond_max: float = 0.10

    # curriculum sigma-cap schedule
    curriculum_target_stage: int = 6
    curriculum_min_epochs: int = 100
    curriculum_early_stop: bool = True
    sigma_cap_safe_mult: Optional[float] = None
    sigma_cap_abs_max: Optional[float] = None
    sigma_cap_abs_min: Optional[float] = None

    # legacy loss-plateau early stop (used only when curriculum disabled)
    enable_early_stop: bool = True
    early_stop_min_epochs: int = 12
    early_stop_patience: int = 6
    early_stop_threshold: float = 0.01

    # reproducibility / io
    seed: Optional[int] = None
    precision: str = '16-mixed'   # "32-true" | "16-mixed" | "bf16-mixed"
    num_workers: int = 0          # DataLoader workers for miniset sampling (0 = main process)

    weights: Dict[str, float] = field(default_factory=lambda: dict(STAGE_C_WEIGHTS))


# Curriculum stage multipliers (x sigma_data / sigma_data_resid).
_CURRICULUM_MULTS = [1.0, 2.0, 3.0, 4.0]


# =============================================================================
# Curriculum state + helpers
# =============================================================================
def _build_curriculum_state(cfg: StageCConfig, sigma_cap_safe_mult: float) -> dict:
    return {
        'sigma_cap_mults': list(_CURRICULUM_MULTS),
        'current_stage': 0,
        'sigma_cap_safe_mult': sigma_cap_safe_mult,
        'sigma_cap_abs_max': cfg.sigma_cap_abs_max,
        'sigma_cap_abs_min': cfg.sigma_cap_abs_min,
        'consecutive_passes': 0,
        'promotion_threshold': 5,
        'stall_count': 0,
        'stall_limit': 6,
        'generator_stable': False,
        'gen_consecutive_passes': 0,
        'jacc_min_by_mult': {1.0: 0.10, 2.0: 0.10, 3.0: 0.10, 4.0: 0.10},
        'scale_r_min_by_mult': {1.0: 0.85, 2.0: 0.80, 3.0: 0.75, 4.0: 0.70},
        'scale_r_min_default': 0.70,
        'trace_r_min_by_mult': {1.0: 0.69, 2.0: 0.61, 3.0: 0.53, 4.0: 0.45},
        'trace_r_min_default': 0.45,
        'scale_r_min_final': 0.80,
        'trace_r_min_final': 0.60,
        'cap_band_frac_by_stage': {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5},
        'cap_band_frac_default': 0.3,
        'cap_band_lo_mult': 0.6,
        'eval_history': [],
        'target_stage': cfg.curriculum_target_stage,
        'curriculum_early_stop': cfg.curriculum_early_stop,
        'steps_in_stage': 0,
        'min_steps_per_stage': 1000,
        'min_steps_at_target': 2000,
        'max_steps_per_stage': None,      # set once steps_per_epoch is known
        'target_dwell_frac': 0.20,
        'ramp_steps': 300,
        'ramp_start_step': None,
        'ramp_prev_cap': None,
        'ramp_target_cap': None,
        'min_epochs': cfg.curriculum_min_epochs,
        'metrics_history_K': 5,
        'metrics_pass_history': [],
        'cap_band_loss_sum': 0.0,
        'cap_band_loss_count': 0,
        'cap_band_loss_history': [],
        'loss_plateau_patience': 6,
        'loss_plateau_threshold': 0.02,
        'loss_no_improve_count': 0,
        'best_cap_band_loss': float('inf'),
        'cond_alpha': cfg.cond_alpha,
        'use_residual_diffusion': cfg.use_residual_diffusion,
        'sigma_resid_valid': False,
        'sigma_data_resid_locked': None,
        'sigma_resid_recompute_step': cfg.sigma_resid_recompute_step,
        'sigma_resid_recomputed': False,
        'sigma_cap_eff_last': None,
        'max_sigma_cap_eff_seen': 0.0,
    }


def _sigma0_for_curriculum(cs: dict, sigma_data: float) -> float:
    """sigma0 driving the curriculum: residual scale if valid, else sigma_data."""
    if cs.get('use_residual_diffusion', False) and cs.get('sigma_resid_valid', False):
        return cs.get('sigma_data_resid_locked', cs.get('sigma_data_resid', sigma_data))
    return sigma_data


def get_sigma_cap_eff(cs: dict, global_step: int, sigma_data: float, do_log: bool = False):
    """
    Effective sigma_cap for the current stage, accounting for the promotion ramp
    and the data-dependent safety cap. Pure: does not mutate state.

    Returns (sigma_cap_eff, sigma_cap_target, ramp_active, ramp_progress, debug_info).
    """
    curr_stage = cs['current_stage']
    curr_mult = cs['sigma_cap_mults'][curr_stage]

    sigma0 = _sigma0_for_curriculum(cs, sigma_data)
    sigma_cap_stage = curr_mult * sigma0

    if cs.get('sigma_cap_safe_mult') is not None:
        sigma_cap_safe_abs = cs['sigma_cap_safe_mult'] * sigma0
    else:
        target_stage = cs.get('target_stage', len(cs['sigma_cap_mults']) - 1)
        mults = cs['sigma_cap_mults']
        default_mult = mults[min(target_stage, len(mults) - 1)]
        sigma_cap_safe_abs = default_mult * sigma0

    sigma_cap_target = min(sigma_cap_stage, sigma_cap_safe_abs)

    sigma_cap_abs_max = cs.get('sigma_cap_abs_max')
    sigma_cap_abs_min = cs.get('sigma_cap_abs_min')
    if sigma_cap_abs_max is not None:
        sigma_cap_target = min(sigma_cap_target, sigma_cap_abs_max)
    if sigma_cap_abs_min is not None:
        sigma_cap_target = max(sigma_cap_target, sigma_cap_abs_min)

    ramp_start = cs.get('ramp_start_step')
    ramp_steps = cs.get('ramp_steps', 300)
    debug_info = {
        'sigma0': sigma0, 'curr_mult': curr_mult, 'cap_stage': sigma_cap_stage,
        'cap_safe_abs': sigma_cap_safe_abs, 'cap_target': sigma_cap_target,
    }

    if ramp_start is not None:
        prev_cap = cs.get('ramp_prev_cap')
        target_cap = cs.get('ramp_target_cap')
        if prev_cap is None or target_cap is None:
            debug_info['cap_final'] = sigma_cap_target
            return sigma_cap_target, sigma_cap_target, False, 1.0, debug_info

        steps_since_ramp = global_step - ramp_start
        if steps_since_ramp < ramp_steps:
            t = steps_since_ramp / ramp_steps
            sigma_cap_eff = prev_cap + t * (target_cap - prev_cap)
            sigma_cap_eff = min(sigma_cap_eff, sigma_cap_safe_abs)
            if sigma_cap_abs_max is not None:
                sigma_cap_eff = min(sigma_cap_eff, sigma_cap_abs_max)
            if sigma_cap_abs_min is not None:
                sigma_cap_eff = max(sigma_cap_eff, sigma_cap_abs_min)
            debug_info['cap_final'] = sigma_cap_eff
            return sigma_cap_eff, sigma_cap_target, True, t, debug_info

    debug_info['cap_final'] = sigma_cap_target
    return sigma_cap_target, sigma_cap_target, False, 1.0, debug_info


def _reset_stage_local(cs: dict):
    """Reset the stage-local three-gate accumulators (on promotion/demotion)."""
    cs['steps_in_stage'] = 0
    cs['cap_band_loss_sum'] = 0.0
    cs['cap_band_loss_count'] = 0
    cs['cap_band_loss_history'] = []
    cs['best_cap_band_loss'] = float('inf')
    cs['loss_no_improve_count'] = 0
    cs['metrics_pass_history'] = []


def _start_ramp(cs: dict, old_mult: float, new_mult: float, global_step: int, sigma_data: float):
    """Arm the smooth sigma_cap ramp between two stage multipliers."""
    sigma0 = _sigma0_for_curriculum(cs, sigma_data)
    old_cap = old_mult * sigma0
    new_cap = new_mult * sigma0
    cs['ramp_start_step'] = global_step
    cs['ramp_prev_cap'] = old_cap
    cs['ramp_target_cap'] = new_cap
    return old_cap, new_cap


def _promote_stage(cs: dict, global_step: int, sigma_data: float):
    """Advance one stage, reset windows/stall, arm ramp, reset stage-local state."""
    old_stage = cs['current_stage']
    old_mult = cs['sigma_cap_mults'][old_stage]
    cs['current_stage'] += 1
    cs['stall_count'] = 0
    cs['promo_pass_window'] = []
    cs['structure_pass_window'] = []
    new_mult = cs['sigma_cap_mults'][cs['current_stage']]
    old_cap, new_cap = _start_ramp(cs, old_mult, new_mult, global_step, sigma_data)
    _reset_stage_local(cs)
    return old_stage, old_cap, new_cap


def _demote_stage(cs: dict, global_step: int, sigma_data: float):
    """Step down one stage (stall safety valve)."""
    old_stage = cs['current_stage']
    old_mult = cs['sigma_cap_mults'][old_stage]
    cs['current_stage'] -= 1
    cs['promo_pass_window'] = []
    cs['structure_pass_window'] = []
    cs['stall_count'] = 0
    new_mult = cs['sigma_cap_mults'][cs['current_stage']]
    old_cap, new_cap = _start_ramp(cs, old_mult, new_mult, global_step, sigma_data)
    _reset_stage_local(cs)
    return old_stage, old_cap, new_cap


def _curriculum_promotion_step(cs, epoch, global_step, sigma_data,
                               sigma_cap_eff, sigma_cap_target, ramp_active, metrics):
    """
    Consume the fixed-batch evaluation metrics and update the curriculum:
    threshold checks -> windowed pass tracking -> promotion (structure-based
    pre-target / full at target) -> stall/demotion -> gate bookkeeping.
    `metrics` = dict(scale_r, trace_r, jacc, gen_scale_r, gen_trace_r).
    """
    scale_r_promo = metrics['scale_r']
    trace_r_promo = metrics['trace_r']
    jacc_promo = metrics['jacc']

    curr_stage = cs['current_stage']
    curr_mult_eff = sigma_cap_eff / sigma_data
    available_mults = list(cs['jacc_min_by_mult'].keys())
    closest_mult = min(available_mults, key=lambda m: abs(m - curr_mult_eff))
    jacc_min = cs['jacc_min_by_mult'].get(closest_mult, 0.07)
    target_stage = cs.get('target_stage', len(cs['sigma_cap_mults']) - 1)

    if curr_stage >= target_stage:
        scale_r_min = cs.get('scale_r_min_final', 0.80)
        trace_r_min = cs.get('trace_r_min_final', 0.60)
    else:
        scale_r_min = cs['scale_r_min_by_mult'].get(closest_mult, cs['scale_r_min_default'])
        trace_r_min = cs['trace_r_min_by_mult'].get(closest_mult, cs['trace_r_min_default'])

    scale_ok = scale_r_promo >= scale_r_min
    trace_ok = trace_r_promo >= trace_r_min
    jacc_ok = jacc_promo >= jacc_min

    prev_jacc = cs['eval_history'][-1].get('jacc', 0) if cs['eval_history'] else None
    jacc_not_decreasing = (prev_jacc is None) or (jacc_promo >= prev_jacc - 0.01)

    structure_pass = jacc_ok and jacc_not_decreasing
    all_pass = structure_pass and scale_ok and trace_ok

    cs['eval_history'].append({
        'epoch': epoch, 'sigma_cap': sigma_cap_eff, 'sigma_cap_target': sigma_cap_target,
        'scale_r': scale_r_promo, 'trace_r': trace_r_promo, 'jacc': jacc_promo,
        'passed': all_pass, 'structure_pass': structure_pass,
    })

    # Generator warm-start gate.
    if not cs['generator_stable']:
        gen_scale_ok = 0.85 <= metrics['gen_scale_r'] <= 1.15
        gen_trace_ok = metrics['gen_trace_r'] >= 0.70
        if gen_scale_ok and gen_trace_ok:
            cs['gen_consecutive_passes'] += 1
        else:
            cs['gen_consecutive_passes'] = 0
        if cs['gen_consecutive_passes'] >= 3:
            cs['generator_stable'] = True

    # Windowed pass tracking (5-of-last-6).
    window_size = 6
    cs.setdefault('promo_pass_window', []).append(all_pass)
    cs['promo_pass_window'] = cs['promo_pass_window'][-window_size:]
    cs.setdefault('structure_pass_window', []).append(structure_pass)
    cs['structure_pass_window'] = cs['structure_pass_window'][-window_size:]

    window = cs['promo_pass_window']
    structure_window = cs['structure_pass_window']
    passes_in_window = sum(window)
    structure_passes_in_window = sum(structure_window)
    required_passes = cs['promotion_threshold']

    steps_in_stage = cs.get('steps_in_stage', 0)
    warmup_steps = 200
    if len(window) >= window_size and steps_in_stage >= warmup_steps and not ramp_active:
        if passes_in_window >= 2:
            cs['stall_count'] = 0
        else:
            cs['stall_count'] += 1

    cs['structure_passes_in_window'] = structure_passes_in_window
    scale_collapsed = scale_r_promo < 0.60
    cs['scale_collapsed'] = scale_collapsed

    at_target = (curr_stage >= target_stage)
    if at_target:
        should_promote = (passes_in_window >= required_passes and
                          len(window) >= required_passes and not scale_collapsed)
    else:
        should_promote = (structure_passes_in_window >= required_passes and
                          len(structure_window) >= required_passes and not scale_collapsed)

    if should_promote and curr_stage < len(cs['sigma_cap_mults']) - 1:
        _promote_stage(cs, global_step, sigma_data)

    # Stall safety valve: demote if genuinely failing below target.
    if cs['stall_count'] >= cs['stall_limit']:
        struct_passes = sum(cs['structure_pass_window'])
        if cs['current_stage'] > 0 and struct_passes < 4:
            _demote_stage(cs, global_step, sigma_data)

    # Metric pass history.
    cs['metrics_pass_history'].append(all_pass)
    K = cs['metrics_history_K']
    cs['metrics_pass_history'] = cs['metrics_pass_history'][-K:]

    # Cap-band loss plateau (cap_band_loss_* accumulated in the batch step).
    if cs['cap_band_loss_count'] > 0:
        avg_cap_band_loss = cs['cap_band_loss_sum'] / cs['cap_band_loss_count']
        cs['cap_band_loss_history'].append(avg_cap_band_loss)
        loss_threshold = cs['loss_plateau_threshold']
        best_loss = cs['best_cap_band_loss']
        if best_loss == float('inf'):
            cs['best_cap_band_loss'] = avg_cap_band_loss
            cs['loss_no_improve_count'] = 0
        elif avg_cap_band_loss < best_loss:
            rel_improv = (best_loss - avg_cap_band_loss) / max(best_loss, 1e-8)
            if rel_improv > loss_threshold:
                cs['best_cap_band_loss'] = avg_cap_band_loss
                cs['loss_no_improve_count'] = 0
            else:
                cs['loss_no_improve_count'] += 1
        else:
            cs['loss_no_improve_count'] += 1
    cs['cap_band_loss_sum'] = 0.0
    cs['cap_band_loss_count'] = 0

    return all_pass, structure_pass, scale_collapsed


def _curriculum_should_stop(cs, cfg, epoch, avg_total, global_step, sigma_data):
    """
    Three-gate curriculum-aware stopping. Also performs the pre-target
    force-promote / max-steps-escape safety promotions. Returns should_stop.
    """
    curriculum_enabled = len(cs.get('sigma_cap_mults', [])) > 0
    curriculum_stop_enabled = curriculum_enabled and cs.get('curriculum_early_stop', True)

    if curriculum_enabled and curriculum_stop_enabled:
        curr_stage = cs['current_stage']
        target_stage = cs.get('target_stage', len(cs['sigma_cap_mults']) - 1)
        min_epochs_curriculum = cs.get('min_epochs', 15)
        min_steps_per_stage = cs.get('min_steps_per_stage', 500)
        steps_in_stage = cs.get('steps_in_stage', 0)

        gate_a = (curr_stage >= target_stage)
        promo_window = cs.get('promo_pass_window', [])
        passes_in_window = sum(promo_window) if promo_window else 0
        gate_b = (len(promo_window) >= 6 and passes_in_window >= 5)
        loss_plateau_patience = cs.get('loss_plateau_patience', 6)
        loss_no_improve = cs.get('loss_no_improve_count', 0)
        gate_c = (loss_no_improve >= loss_plateau_patience)

        min_epochs_met = (epoch + 1) >= min_epochs_curriculum
        min_steps_met = steps_in_stage >= min_steps_per_stage

        stall_count = cs.get('stall_count', 0)
        stall_limit = cs.get('stall_limit', 6)
        structure_passes = cs.get('structure_passes_in_window', 0)
        structure_competent = structure_passes >= 4
        structure_failing = structure_passes < 2

        # STOP 1: three-gate convergence success (checked first).
        if gate_a and gate_b and gate_c and min_epochs_met and min_steps_met:
            return True

        # STOP 2: stall handling.
        if stall_count >= stall_limit and min_steps_met:
            scale_collapsed = cs.get('scale_collapsed', False)
            if curr_stage < target_stage:
                if structure_competent and not scale_collapsed:
                    if curr_stage < len(cs['sigma_cap_mults']) - 1:
                        _promote_stage(cs, global_step, sigma_data)  # scale-stall force promote
                elif structure_failing:
                    max_steps = cs.get('max_steps_per_stage', 1500)
                    if steps_in_stage >= max_steps and curr_stage < len(cs['sigma_cap_mults']) - 1:
                        _promote_stage(cs, global_step, sigma_data)  # max-steps escape
                # borderline (2-3) or collapsed-scale: continue training, no stop.
            else:
                min_steps_at_target = cs.get('min_steps_at_target', 2000)
                if min_epochs_met and steps_in_stage >= min_steps_at_target:
                    return True   # stall at target stage
        return False

    # No curriculum: legacy loss-plateau early stop.
    if cfg.enable_early_stop and (epoch + 1) >= cfg.early_stop_min_epochs:
        best = cs.setdefault('_legacy_best', float('inf'))
        no_improve = cs.setdefault('_legacy_no_improve', 0)
        if avg_total < best:
            rel_improv = (best - avg_total) / max(best, 1e-8)
            if rel_improv > cfg.early_stop_threshold:
                cs['_legacy_best'] = avg_total
                cs['_legacy_no_improve'] = 0
            else:
                cs['_legacy_no_improve'] = no_improve + 1
        else:
            cs['_legacy_no_improve'] = no_improve + 1
        return cs['_legacy_no_improve'] >= cfg.early_stop_patience
    elif cfg.enable_early_stop:
        cs['_legacy_best'] = min(cs.setdefault('_legacy_best', float('inf')), avg_total)
    return False


# =============================================================================
# Per-batch step
# =============================================================================
def _stage_c_batch_step(batch, Z_set, mask, struct_mask,
                        context_encoder, generator, score_net,
                        sigma, sigma_t, sigma_edm, sigma_cap,
                        weights, tau_reference, cond_alpha, p_uncond,
                        gates, cap_state, curriculum_state,
                        use_residual, device, amp_dtype,
                        score_hi_gate=None, boost_state=None, global_step=0,
                        hi_boost=False, fx_hi=False, boost_factor=4.0,
                        boost_ramp=200, fx_hi_weight=2.0):
    """
    One residual-diffusion training forward pass + Stage-C losses for a mini-set
    batch. Returns (L_total, loss_terms_dict, V_hat).
    """
    B = Z_set.shape[0]
    D_latent = score_net.D_latent
    use_amp = str(device).startswith('cuda')

    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        # ---- context ----
        H = context_encoder(Z_set, mask)

        # ---- target coordinates from the ground-truth Gram matrix ----
        V_target_raw = batch['V_target'].to(device)
        G_target = batch['G_target'].to(device)
        V_target = torch.zeros_like(V_target_raw)
        for i in range(B):
            nv = int(mask[i].sum().item())
            if nv <= 1:
                continue
            G_i = G_target[i, :nv, :nv].float()
            V_target[i, :nv] = factor_from_gram(G_i, D_latent).to(V_target_raw.dtype)
        V_gen = generator(H, mask) if weights.get('gen_align', 0.0) > 0 else None

        # ---- detached base + Procrustes-align target into base frame ----
        with torch.no_grad():
            V_base = generator(H, mask).detach()
        with torch.no_grad():
            V_target_aligned = rigid_align_apply_no_scale(V_target, V_base, mask)
        R_target = V_target_aligned - V_base

        # ---- CFG context dropout + conditioning strength ----
        drop_bool = (torch.rand(B, device=device) < p_uncond).float()
        drop_mask = drop_bool.view([B] + [1] * (H.ndim - 1))
        H_train = H * (1.0 - drop_mask)
        H_train, _ = normalize_and_scale_conditioning(H_train, mask, cond_alpha)
        cond_only = (1.0 - drop_bool)

        # ---- diffusion target: residual (default) or aligned absolute coords ----
        diff_target = R_target if use_residual else V_target_aligned

        # ---- noise around the diffusion target ----
        eps = torch.randn_like(V_target)
        V_t = diff_target + sigma_t * eps
        V_t = V_t * mask.unsqueeze(-1).float()

        # ---- two-pass self-conditioning EDM denoise ----
        sigma_flat = sigma_t.view(-1)
        with torch.no_grad():
            x0_0 = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_edm, self_cond=None)
            if isinstance(x0_0, tuple):
                x0_0 = x0_0[0]
        x0_pred = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_edm,
                                        self_cond=x0_0.detach())
        if isinstance(x0_pred, tuple):
            x0_pred = x0_pred[0]

        # ---- compose ----
        V_hat = (V_base + x0_pred) if use_residual else x0_pred

    # ===== losses in fp32 =====
    with torch.autocast(device_type='cuda', enabled=False):
        losses = {}
        losses['score'] = edm_residual_score_loss(
            x0_pred, diff_target, sigma, sigma_edm, mask,
            cap_state=cap_state, sigma_cap=sigma_cap, curriculum_state=curriculum_state,
            R_t=V_t, score_hi_gate=score_hi_gate, boost_state=boost_state,
            hi_boost=hi_boost, fx_hi=fx_hi, boost_factor=boost_factor,
            boost_ramp=boost_ramp, fx_hi_weight=fx_hi_weight, global_step=global_step)

        st = build_structure_tensors(V_hat, V_target, mask)
        knn = within_miniset_knn(V_target, mask, k=15)
        gg = build_geometry_gates(sigma, sigma_edm, mask, gates, cond_only=cond_only)

        L_gram, L_gram_scale = gram_losses(
            st['V_struct'], st['V_target_struct'], gg['gram'], gg['gram_scale'],
            st['m_bool'], st['m_float'])
        losses['gram'] = L_gram
        losses['gram_scale'] = L_gram_scale
        losses['gram_learn'] = gram_learn_loss(
            st['V_hat_centered'], V_target, V_t, sigma, sigma_edm, mask,
            st['m_bool'], st['m_float'])
        losses['out_scale'] = out_scale_loss(
            st['V_hat_centered'], V_target, V_t, sigma, sigma_edm, mask, st['m_float'])
        losses['knn_nca'] = knn_nca(
            st['V_struct'], st['V_target_struct'], struct_mask, gg['nca'], tau_reference)
        losses['knn_scale'] = knn_scale(
            st['V_hat_centered'], V_target, struct_mask, knn, gg['noise_score'], mask)
        losses['edge'] = edge_loss(
            st['V_struct'], st['V_target_struct'], struct_mask, knn, gg['edge'])
        losses['subspace'] = subspace_loss(st['V_struct'], mask)
        if V_gen is not None:
            L_ga, L_gg, L_gs = generator_supervision(
                V_gen, V_target, st['V_target_struct'], struct_mask, knn, mask)
            losses['gen_align'] = L_ga
            losses['gen_gram'] = L_gg
            losses['gen_scale'] = L_gs

        L_total = assemble_total_loss(losses, weights)

    loss_terms = {k: float(v.detach().item()) for k, v in losses.items()}
    return L_total, loss_terms, V_hat


# =============================================================================
# Training entry point
# =============================================================================
def train_stageC(
    context_encoder,        # SetEncoderContext
    generator,              # MetricSetGenerator
    score_net,              # DiffusionScoreNet
    st_dataset,             # STSetDataset
    encoder=None,           # SharedEncoder (frozen; saved into checkpoints)
    config: Optional[StageCConfig] = None,
    device: str = 'cuda',
    out_dir: str = 'stageC_out',
    resume_ckpt: Optional[str] = None,   # path to a ckpt_*.pt to resume training from
    fabric=None,            # optional lightning.fabric.Fabric for DDP training
    eval_fixed_batch_fn: Optional[Callable] = None,
) -> Dict:
    """
    Train the Stage-C residual-diffusion generator jointly with the context
    encoder and generator prior on ST mini-sets. Returns the training history
    (which also carries the EMA state dicts for downstream use).

    `eval_fixed_batch_fn`, when provided, is called once per epoch to produce the
    fixed-batch metrics that drive the sigma-cap curriculum; without it the
    curriculum remains at stage 0.
    """
    from torch.utils.data import DataLoader

    cfg = config or StageCConfig()
    WEIGHTS = dict(cfg.weights)

    # Under DDP only rank 0 writes files / logs; single-GPU (fabric=None) is rank0.
    is_rank0 = (fabric is None) or fabric.is_global_zero

    if cfg.seed is not None:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    if is_rank0:
        os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_fp16 = (cfg.precision == '16-mixed')
    # Fabric owns AMP/grad-scaling; only build the manual scaler on the single-GPU path.
    scaler = None if fabric is not None else torch.cuda.amp.GradScaler(enabled=use_fp16)

    context_encoder = context_encoder.to(device).train()
    score_net = score_net.to(device).train()
    generator = generator.to(device).train()

    # Raw (unwrapped) module handles. EMA reads these and buffers/sigma_data are
    # set on these so forward() sees them; fabric.setup reassigns the names below.
    raw_context_encoder = context_encoder
    raw_score_net = score_net
    raw_generator = generator

    # ---- Resume: load checkpoint + restore RAW module weights BEFORE fabric.setup
    # so DDP wraps the loaded parameters. Every rank loads the same shared-on-disk
    # file (map_location=device) -> all ranks restore byte-identical state (no
    # broadcast needed). Optimizer / scheduler / EMA / curriculum / sigma_data are
    # restored further below, each once its target object exists (and, for the
    # optimizer, after fabric.setup has returned the wrapped instance).
    resume_state = None
    start_epoch = 0
    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        resume_state = torch.load(resume_ckpt, map_location=device, weights_only=False)
        raw_context_encoder.load_state_dict(resume_state['context_encoder'])
        raw_generator.load_state_dict(resume_state['generator'])
        # score_net registers 'st_dist_bin_edges' as a None-valued buffer, which a
        # freshly built net omits from its state_dict; populate it from the ckpt
        # first so the strict load treats the key as expected (it is recomputed
        # from the ST coordinates a few lines below regardless).
        _sn_sd = resume_state['score_net']
        if _sn_sd.get('st_dist_bin_edges', None) is not None:
            raw_score_net.st_dist_bin_edges = _sn_sd['st_dist_bin_edges'].to(device)
        raw_score_net.load_state_dict(_sn_sd)
        # Saved 'epoch' is the 0-based index of the last COMPLETED epoch; resume at
        # the next one so a run that finished epoch N-1 (N epochs done) starts at N.
        start_epoch = int(resume_state['epoch']) + 1
        if is_rank0:
            print(f"[StageC] RESUME from {resume_ckpt}: restored nets @ saved "
                  f"epoch {resume_state['epoch']} -> starting at epoch {start_epoch}")

    # Optimizer: generator gets 5x LR (small gradients); cosine schedule.
    wd_default = 1e-4
    optimizer = torch.optim.AdamW([
        {"params": context_encoder.parameters(), "lr": cfg.lr,       "weight_decay": wd_default},
        {"params": score_net.parameters(),       "lr": cfg.lr,       "weight_decay": wd_default},
        {"params": generator.parameters(),       "lr": cfg.lr * 5.0, "weight_decay": wd_default},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    # EMA copies of score_net + context_encoder (deep-copied from raw modules).
    ema_decay = cfg.ema_decay
    score_net_ema = copy.deepcopy(raw_score_net).eval()
    for p in score_net_ema.parameters():
        p.requires_grad_(False)
    context_encoder_ema = copy.deepcopy(raw_context_encoder).eval()
    for p in context_encoder_ema.parameters():
        p.requires_grad_(False)

    # Resume: the EMA copies above were deep-copied from the (just-loaded) raw
    # modules, not from the saved EMA weights -- overwrite with the saved EMA.
    if resume_state is not None:
        context_encoder_ema.load_state_dict(resume_state['context_encoder_ema'])
        # Align the EMA's 'st_dist_bin_edges' buffer with whatever the saved EMA
        # carries (typically None -- the EMA is copied before the buffer is
        # populated and ema_update skips None buffers) so the strict load matches.
        _ema_sd = resume_state['score_net_ema']
        _ema_edges = _ema_sd.get('st_dist_bin_edges', None)
        score_net_ema.st_dist_bin_edges = _ema_edges.to(device) if _ema_edges is not None else None
        score_net_ema.load_state_dict(_ema_sd)

    @torch.no_grad()
    def ema_update(ema_model, model, decay):
        msd = model.state_dict()
        for k, v_ema in ema_model.state_dict().items():
            v = msd[k]
            if not torch.is_floating_point(v):
                v_ema.copy_(v)
            else:
                v_ema.mul_(decay).add_(v, alpha=1.0 - decay)

    # Wrap nets + optimizer for DDP when a Fabric is supplied. fabric.setup
    # re-points the optimizer's param groups onto the (device-resident) params.
    if fabric is not None:
        context_encoder, generator, score_net, optimizer = fabric.setup(
            context_encoder, generator, score_net, optimizer)

    # Resume: restore optimizer + scheduler AFTER they are built and (under fabric)
    # AFTER fabric.setup has returned the wrapped optimizer (whose load_state_dict
    # delegates to the underlying optimizer).
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])

    # Clip / non-finite param list references the raw params the optimizer steps
    # (the wrapped modules expose duplicated params, so read from the raw handles).
    params = (list(raw_context_encoder.parameters()) +
              list(raw_generator.parameters()) +
              list(raw_score_net.parameters()))

    # Curriculum.
    sigma_cap_safe_mult = cfg.sigma_cap_safe_mult
    if sigma_cap_safe_mult is None:
        _target_mult = _CURRICULUM_MULTS[min(cfg.curriculum_target_stage, len(_CURRICULUM_MULTS) - 1)]
        sigma_cap_safe_mult = _target_mult
    curriculum_state = _build_curriculum_state(cfg, sigma_cap_safe_mult)

    # Dataloader (sharded across ranks under DDP via fabric.setup_dataloaders).
    # __getitem__ is CPU-only (Z_dict is precomputed on CPU), so miniset sampling
    # can be parallelized across worker processes. Each worker re-seeds numpy from
    # its torch seed so the np.random draws in __getitem__ are decorrelated.
    def _seed_worker(worker_id):
        s = torch.initial_seed() % (2 ** 31)
        np.random.seed(s)
    nw = int(getattr(cfg, "num_workers", 0))
    st_loader = DataLoader(
        st_dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_minisets, num_workers=nw, pin_memory=False,
        persistent_workers=(nw > 0), worker_init_fn=(_seed_worker if nw > 0 else None),
        prefetch_factor=(4 if nw > 0 else None))
    if fabric is not None:
        st_loader = fabric.setup_dataloaders(st_loader)
    steps_per_epoch = len(st_loader)

    # Dynamic per-stage step budget (guarantees reaching the target stage).
    total_steps = cfg.n_epochs * steps_per_epoch
    target_stage = curriculum_state['target_stage']
    target_dwell_steps = int(total_steps * curriculum_state.get('target_dwell_frac', 0.20))
    if target_stage > 0:
        max_steps_per_stage = max(500, (total_steps - target_dwell_steps) // target_stage)
    else:
        max_steps_per_stage = total_steps
    curriculum_state['max_steps_per_stage'] = max_steps_per_stage
    curriculum_state['min_steps_at_target'] = max(500, target_dwell_steps // 2)

    # Resume: replace the freshly built curriculum with the saved one, re-deriving
    # the step budgets from THIS run's steps_per_epoch only if the checkpoint lacks
    # them (older / interrupted checkpoints may not carry them).
    if resume_state is not None:
        _fresh_max_steps = curriculum_state['max_steps_per_stage']
        _fresh_min_at_target = curriculum_state['min_steps_at_target']
        curriculum_state = resume_state['curriculum_state']
        if curriculum_state.get('max_steps_per_stage') is None:
            curriculum_state['max_steps_per_stage'] = _fresh_max_steps
        if curriculum_state.get('min_steps_at_target') is None:
            curriculum_state['min_steps_at_target'] = _fresh_min_at_target

    # score_net distance-bias bin edges from ST coordinates.
    all_st_coords = []
    for slide_id in st_dataset.targets_dict:
        y_hat = st_dataset.targets_dict[slide_id].y_hat
        all_st_coords.append(y_hat.cpu().numpy() if torch.is_tensor(y_hat) else y_hat)
    st_coords_np = np.concatenate(all_st_coords, axis=0)
    raw_score_net.st_dist_bin_edges = init_st_dist_bins_from_data(
        st_coords_np, n_bins=raw_score_net.dist_bins, mode='log').to(device)

    # kNN-NCA temperature (squared 15th-NN distance in raw target space).
    slide_d15_medians = []
    for slide_id in st_dataset.targets_dict:
        y_hat = st_dataset.targets_dict[slide_id].y_hat.to(device)
        n_slide = y_hat.shape[0]
        if n_slide < 20:
            continue
        with torch.no_grad():
            D_slide = torch.cdist(y_hat, y_hat)
            D_slide[torch.arange(n_slide), torch.arange(n_slide)] = float('inf')
            knn_dists, _ = torch.topk(D_slide, k=min(15, n_slide - 1), dim=1, largest=False)
            slide_d15_medians.append(knn_dists[:, -1].median().item())
    r_15_median = float(np.median(slide_d15_medians)) if slide_d15_medians else 1.0
    tau_reference = float(max(1e-4, min(r_15_median ** 2, 1e2)))

    # Estimate sigma_data over a few ST batches (skipped on resume -- the
    # checkpoint's value is authoritative and must not be re-estimated).
    if resume_state is not None:
        sigma_data = float(resume_state['sigma_data'])
    else:
        sample_stds = []
        it = iter(st_loader)
        with torch.no_grad():
            for _ in range(min(10, len(st_loader))):
                sb = next(it, None)
                if sb is None:
                    break
                V_b = sb['V_target'].to(device, non_blocking=True)
                m_b = sb['mask'].to(device, non_blocking=True)
                for i in range(min(4, V_b.shape[0])):
                    m = m_b[i]
                    if m.sum() > 0:
                        sample_stds.append(V_b[i, m].std().item())
        sigma_data = float(np.median(sample_stds)) if sample_stds else 1.0
    # sigma_data feeds the loss and must be identical on every rank (each rank
    # estimated it from its own data shard); broadcast rank-0's value.
    if fabric is not None:
        sigma_data = float(fabric.broadcast(
            torch.tensor(sigma_data, device=device), src=0).item())

    # EDM sigma clamps + refinement ceiling.
    sigma_min = cfg.sigma_min
    sigma_max = cfg.sigma_max
    sigma_refine_max = cfg.sigma_refine_max
    if cfg.use_edm:
        sigma_max = min(sigma_max, sigma_data * 100)
        sigma_min = max(sigma_min, sigma_data * 0.001)
        if sigma_refine_max is None:
            sigma_refine_max = 20.0 * sigma_data
        sigma_refine_max = min(sigma_refine_max, sigma_max)

    raw_score_net.sigma_data = sigma_data

    # Residual data scale: only valid once the generator has trained. Fresh start
    # keeps sigma_data_resid == sigma_data until the recompute trigger; on resume
    # the restored curriculum_state already carries the (possibly recomputed and
    # locked) residual scale + validity flag, so leave those untouched.
    if resume_state is not None:
        sigma_data_resid = float(curriculum_state.get('sigma_data_resid', sigma_data))
    else:
        sigma_data_resid = float(sigma_data)
        if fabric is not None:
            sigma_data_resid = float(fabric.broadcast(
                torch.tensor(sigma_data_resid, device=device), src=0).item())
        sigma_resid_valid = False
        curriculum_state['sigma_data_resid'] = sigma_data_resid
        curriculum_state['sigma_resid_valid'] = sigma_resid_valid

    if is_rank0:
        print(f"[StageC] sigma_data={sigma_data:.4f} sigma_min={sigma_min:.6f} "
              f"sigma_max={sigma_max:.2f} sigma_refine_max={sigma_refine_max:.4f}")
        print(f"[StageC] curriculum stages={curriculum_state['sigma_cap_mults']} "
              f"target_stage={target_stage} residual_diffusion={cfg.use_residual_diffusion}")

    # Persistent loss state across the whole run.
    gates = make_geometry_gates()
    cap_state = {}
    # High-noise score-loss state (only when HI_BOOST / FX_HI are enabled).
    score_hi_gate = None
    boost_state = None
    if cfg.score_hi_boost or cfg.score_fx_hi:
        score_hi_gate = AdaptiveQuantileGate(
            target_rate=cfg.score_hi_target_rate, mode="high",
            warmup_steps=200, reservoir_size=8192, update_every=50, ema=0.9)
        boost_state = HiBoostState()

    history = {'epoch': [], 'epoch_avg': {k: [] for k in list(WEIGHTS.keys()) + ['total']}}
    global_step = int(resume_state['global_step']) if resume_state is not None else 0
    should_stop = False
    ckpt = None

    for epoch in range(start_epoch, cfg.n_epochs):
        context_encoder.train()
        score_net.train()
        generator.train()

        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0
        n_batches = 0

        # CFG context-dropout probability for this epoch.
        if cfg.cfg_context_dropout and epoch >= cfg.cfg_warmup_start:
            ramp = min(1.0, (epoch - cfg.cfg_warmup_start) / max(1, cfg.cfg_warmup_len))
            p_uncond = cfg.p_uncond_max * ramp
        else:
            p_uncond = 0.0

        for batch in st_loader:
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            if cfg.use_z_ln:
                Z_set = apply_z_ln(Z_set, context_encoder)

            is_landmark = batch.get('is_landmark', None)
            if is_landmark is not None:
                is_landmark = is_landmark.to(device).bool() & mask
            else:
                is_landmark = torch.zeros_like(mask)
            # Points where kNN-based structure losses are valid (non-landmark).
            struct_mask = mask & (~is_landmark)
            batch_size_real = Z_set.shape[0]

            # Stochastic conditioning augmentation.
            if torch.rand(1).item() < cfg.aug_prob:
                Z_set = apply_context_augmentation(
                    Z_set, mask, noise_std=cfg.z_noise_std, dropout_rate=cfg.z_dropout_rate)

            # Curriculum sigma cap for this step.
            curr_stage = curriculum_state['current_stage']
            sigma_cap_eff, sigma_cap_target, ramp_active, _, _ = get_sigma_cap_eff(
                curriculum_state, global_step, sigma_data)
            curriculum_state['sigma_cap_eff_last'] = sigma_cap_eff
            curriculum_state['max_sigma_cap_eff_seen'] = max(
                curriculum_state.get('max_sigma_cap_eff_seen', 0.0), sigma_cap_eff)
            if not ramp_active and curriculum_state.get('ramp_start_step') is not None:
                curriculum_state['ramp_start_step'] = None
                curriculum_state['ramp_prev_cap'] = None
                curriculum_state['ramp_target_cap'] = None
            sigma_cap = min(sigma_cap_eff, sigma_refine_max)
            cap_band_frac = curriculum_state['cap_band_frac_by_stage'].get(
                curr_stage, curriculum_state['cap_band_frac_default'])

            # Sample noise levels.
            sigma = sample_sigma_capband(
                batch_size_real, sigma_cap, cap_band_frac,
                cap_band_lo_mult=curriculum_state['cap_band_lo_mult'],
                sigma_min=sigma_min, P_mean=cfg.P_mean, P_std=cfg.P_std, device=device)
            sigma_t = sigma.view(-1, 1, 1)

            # sigma_edm: residual data scale if valid, else sigma_data.
            if curriculum_state.get('use_residual_diffusion', False) and \
                    curriculum_state.get('sigma_resid_valid', False):
                sigma_edm = curriculum_state.get(
                    'sigma_data_resid_locked',
                    curriculum_state.get('sigma_data_resid', sigma_data))
            else:
                sigma_edm = sigma_data

            L_total, loss_terms, V_hat = _stage_c_batch_step(
                batch, Z_set, mask, struct_mask,
                context_encoder, generator, score_net,
                sigma, sigma_t, sigma_edm, sigma_cap,
                WEIGHTS, tau_reference, cfg.cond_alpha, p_uncond,
                gates, cap_state, curriculum_state,
                cfg.use_residual_diffusion, device, amp_dtype,
                score_hi_gate=score_hi_gate, boost_state=boost_state,
                global_step=global_step, hi_boost=cfg.score_hi_boost,
                fx_hi=cfg.score_fx_hi, boost_factor=cfg.score_hi_boost_factor,
                boost_ramp=cfg.score_hi_boost_ramp, fx_hi_weight=cfg.score_fx_hi_weight)

            # Backward / clip / step.
            optimizer.zero_grad(set_to_none=True)
            if fabric is not None:
                # DDP path: Fabric owns AMP (its GradScaler) + grad all-reduce.
                fabric.backward(L_total)
                nonfinite = any(p.grad is not None and not torch.isfinite(p.grad).all()
                                for p in params)
                # All ranks must handle non-finite grads identically or the next
                # collective desyncs and the run hangs. score_net grads are not
                # DDP-synced (forward_edm bypasses the DDP wrapper), so trust the
                # all-reduced flag rather than each rank's local view.
                flag = fabric.all_reduce(
                    torch.tensor(float(nonfinite), device=device), reduce_op="sum")
                nonfinite = flag.item() > 0
                if nonfinite:
                    # Zero the offending grads and still step (as the original
                    # does). Skipping optimizer.step() would strand Fabric's fp16
                    # GradScaler: with no step it never lowers its loss scale, so
                    # every subsequent batch overflows and is skipped forever.
                    for p in params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(params, 1000.0)
                optimizer.step()
            else:
                # Single-GPU path: manual GradScaler (unchanged).
                scaler.scale(L_total).backward()
                scaler.unscale_(optimizer)
                nonfinite = any(p.grad is not None and not torch.isfinite(p.grad).all()
                                for p in params)
                if nonfinite:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1000.0)
                scaler.step(optimizer)
                scaler.update()

            # EMA update (after the optimizer step); read the raw (unwrapped) nets.
            ema_update(score_net_ema, raw_score_net, ema_decay)
            ema_update(context_encoder_ema, raw_context_encoder, ema_decay)

            epoch_losses['total'] += float(L_total.item())
            for k in WEIGHTS.keys():
                if k in loss_terms:
                    epoch_losses[k] += float(loss_terms[k])
            n_batches += 1
            global_step += 1
            curriculum_state['steps_in_stage'] += 1

            # Residual data-scale recompute once the generator has warmed up.
            if (cfg.use_residual_diffusion and
                    not curriculum_state.get('sigma_resid_recomputed', False) and
                    global_step == curriculum_state.get('sigma_resid_recompute_step', 3000)):
                computed = _compute_sigma_data_resid_aligned(
                    st_loader, context_encoder, generator, device)
                if computed is not None:
                    curriculum_state['sigma_data_resid'] = computed
                    curriculum_state['sigma_data_resid_locked'] = computed
                    curriculum_state['sigma_resid_valid'] = True
                # Each rank recomputed from its own shard; sync to rank-0's value.
                if fabric is not None:
                    _r = float(fabric.broadcast(torch.tensor(
                        float(curriculum_state['sigma_data_resid']), device=device),
                        src=0).item())
                    _v = fabric.broadcast(torch.tensor(
                        1.0 if curriculum_state.get('sigma_resid_valid', False) else 0.0,
                        device=device), src=0).item() > 0.5
                    curriculum_state['sigma_data_resid'] = _r
                    curriculum_state['sigma_data_resid_locked'] = _r
                    curriculum_state['sigma_resid_valid'] = _v
                curriculum_state['sigma_resid_recomputed'] = True

        # ---- end of epoch: fixed-batch eval drives curriculum promotion ----
        sigma_cap_eff, sigma_cap_target, ramp_active, _, _ = get_sigma_cap_eff(
            curriculum_state, global_step, sigma_data)
        if eval_fixed_batch_fn is not None:
            eval_metrics = eval_fixed_batch_fn(
                score_net, context_encoder, generator, curriculum_state,
                sigma_cap_eff, sigma_data, device)
            _curriculum_promotion_step(
                curriculum_state, epoch, global_step, sigma_data,
                sigma_cap_eff, sigma_cap_target, ramp_active, eval_metrics)

        epoch_means = {k: (epoch_losses[k] / max(n_batches, 1)) for k in WEIGHTS.keys()}
        epoch_means['total'] = epoch_losses['total'] / max(n_batches, 1)
        for k, v in epoch_means.items():
            history['epoch_avg'].setdefault(k, []).append(v)
        history['epoch'].append(epoch + 1)

        avg_total = epoch_means['total']
        if is_rank0:
            print(f"[Epoch {epoch+1}/{cfg.n_epochs}] total={avg_total:.4f} "
                  f"score={epoch_means.get('score', 0.0):.4f} "
                  f"stage={curriculum_state['current_stage']} "
                  f"sigma_cap_eff={sigma_cap_eff:.4f} (batches={n_batches})")

        # Time-based curriculum advance: once this stage's step budget is spent,
        # force-promote so the sigma-cap ramps up on schedule (no evaluator
        # needed). steps_in_stage is incremented identically on every rank, so
        # all ranks promote in lockstep under DDP. When a fixed-batch evaluator
        # IS supplied, it can additionally promote earlier via _curriculum_should_stop.
        _max_stage = min(curriculum_state['target_stage'],
                         len(curriculum_state['sigma_cap_mults']) - 1)
        if (curriculum_state['current_stage'] < _max_stage
                and curriculum_state['steps_in_stage'] >= curriculum_state['max_steps_per_stage']):
            old_stage, old_cap, new_cap = _promote_stage(
                curriculum_state, global_step, sigma_data)
            if is_rank0:
                print(f"[Curriculum] epoch {epoch+1}: stage {old_stage} -> "
                      f"{curriculum_state['current_stage']} "
                      f"(sigma_cap {old_cap:.3f} -> {new_cap:.3f})")

        should_stop = _curriculum_should_stop(
            curriculum_state, cfg, epoch, avg_total, global_step, sigma_data)
        # Every rank must agree on the break, else the run desyncs on the next
        # collective; broadcast rank-0's decision.
        if fabric is not None:
            should_stop = fabric.broadcast(
                torch.tensor(1.0 if should_stop else 0.0, device=device),
                src=0).item() > 0.5
        scheduler.step()

        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'context_encoder': raw_context_encoder.state_dict(),
            'score_net': raw_score_net.state_dict(),
            'generator': raw_generator.state_dict(),
            'context_encoder_ema': context_encoder_ema.state_dict(),
            'score_net_ema': score_net_ema.state_dict(),
            'ema_decay': ema_decay,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history,
            'sigma_data': sigma_data,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'curriculum_state': curriculum_state,
        }
        if encoder is not None:
            ckpt['encoder'] = encoder.state_dict()
        # End-of-epoch barrier keeps ranks aligned; only rank-0 writes files.
        if fabric is not None:
            fabric.barrier()
        if is_rank0:
            torch.save(ckpt, os.path.join(out_dir, 'ckpt_latest.pt'))
            if (epoch + 1) % 50 == 0:
                torch.save(ckpt, os.path.join(out_dir, f'ckpt_epoch_{epoch+1}.pt'))

        if should_stop:
            if is_rank0:
                print(f"[StageC] Early stop at epoch {epoch+1}")
            break

    # Barrier before the final checkpoint save; only rank-0 touches the disk.
    if fabric is not None:
        fabric.barrier()
    if is_rank0:
        if ckpt is not None:
            torch.save(ckpt, os.path.join(out_dir, 'ckpt_final.pt'))
        with open(os.path.join(out_dir, 'stageC_history.json'), 'w') as f:
            json.dump({'epoch': history['epoch']}, f, indent=2)
    history.update({
        'sigma_data': sigma_data,
        'sigma_data_resid': curriculum_state.get('sigma_data_resid', sigma_data),
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'score_net_ema_state': score_net_ema.state_dict(),
        'context_encoder_ema_state': context_encoder_ema.state_dict(),
        'ema_decay': ema_decay,
    })
    return history


# Compatibility alias.
train_diffusion = train_stageC
