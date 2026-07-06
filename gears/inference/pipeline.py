"""Distance-first patchwise single-cell reconstruction.

Entry point for the inference pipeline: it turns a bag of dissociated single
cells (raw gene expression) into 2D coordinates plus a dense pairwise distance
matrix, using the frozen Stage-A encoder together with the trained Stage-C
context encoder / generator / score network.

The reconstruction never solves the full cell population at once. Instead it:

    0. Encodes every cell with the frozen encoder -> embeddings Z.
    1. Builds a locality graph on Z (mutual-kNN + Jaccard overlap filter).
    2. Samples many small, mutually-overlapping patches from that graph.
    3. Recovers each patch geometry independently by reverse residual-diffusion
       sampling (generator proposal + denoised residual).
    4. Reads a sparse set of pairwise distance measurements off each patch.
    5. Scores per-patch reliability from overlap agreement.
    6. Fuses all measurements into one stitched global distance graph
       (reliability-weighted median per edge).
    7. Solves global 2D coordinates from that graph (Landmark Isomap init +
       weighted-Huber distance-geometry refinement).

All stitching happens on distances (gauge invariant), never on raw coordinates.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_z_ln(Z_set: torch.Tensor, context_encoder: nn.Module) -> torch.Tensor:
    """Per-point LayerNorm over Z feature channels (anchor channel untouched).

    Matches the training-time ``apply_z_ln``: the model is trained on
    feature-standardized embeddings, so inference must standardize too. No
    learned affine (F.layer_norm with weight=bias=None).
    """
    input_dim = Z_set.shape[-1]
    anchor_train = getattr(context_encoder, "anchor_train", False)
    expected_in = getattr(context_encoder, "input_dim", None)
    if expected_in is None and hasattr(context_encoder, "input_proj"):
        expected_in = context_encoder.input_proj.in_features

    if anchor_train and expected_in is not None and input_dim == expected_in:
        z_feat = F.layer_norm(Z_set[..., :-1], (Z_set.shape[-1] - 1,))
        return torch.cat([z_feat, Z_set[..., -1:]], dim=-1)
    return F.layer_norm(Z_set, (input_dim,))

from .locality import build_locality_graph_v2, sample_patches_random_walk_v2
from .patch_geometry import (
    sample_patch_residual_diffusion_v2,
    extract_patch_distances_v2,
)
from .stitch import (
    compute_overlap_consistency_v2,
    aggregate_distance_measurements_v2,
)
from .solve import landmark_isomap_init_v2, global_distance_geometry_solve_v2


@dataclass
class InferConfig:
    """Resolved defaults for the residual distance-first SC inference pipeline."""

    # Locality graph (Step 1)
    k_Z: int = 40
    k_sigma: int = 10
    tau_jaccard: float = 0.10
    min_shared: int = 5

    # Patch sampling (Step 2)
    patch_size: int = 192
    overlap_frac: float = 0.5
    coverage_per_cell: float = 6.0
    min_overlap: int = 30

    # Per-patch residual diffusion (Step 3)
    use_residual_diffusion: bool = True
    generator_only: bool = False
    use_z_ln: bool = False
    n_diffusion_steps: int = 200
    guidance_scale: float = 2.0
    sigma_data: float = 0.5
    sigma_data_resid: Optional[float] = None
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    # Patch -> sparse distances (Step 4)
    k_edge: int = 15

    # Distance aggregation (Step 6)
    M_min: int = 2
    tau_spread: float = 0.30
    spread_alpha: float = 10.0

    # Global solve (Step 7)
    clip_init_runaway: bool = True
    n_landmarks: int = 256
    global_iters: int = 1000
    global_lr: float = 0.01
    huber_delta: float = 0.1
    anchor_lambda: float = 0.1

    verbose: bool = False


def reconstruct_sc(
    encoder: nn.Module,
    context_encoder: nn.Module,
    generator: nn.Module,
    score_net: nn.Module,
    sc_gene_expr: torch.Tensor,
    config: InferConfig,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Reconstruct 2D coordinates + a dense distance matrix from SC expression.

    Args:
        encoder:         frozen Stage-A shared encoder (expression -> embedding).
        context_encoder: trained per-point conditioning encoder.
        generator:       trained base-coordinate proposal network.
        score_net:       trained residual-diffusion score network.
        sc_gene_expr:    (N, G) single-cell expression tensor.
        config:          InferConfig with the resolved v2/residual settings.
        device:          'cpu' or 'cuda'.

    Returns:
        dict with:
            'coords':      (N, 2) canonicalized 2D coordinates (centered, RMS=1).
            'distances':   (N, N) dense pairwise distance matrix, cdist(coords).
            'diagnostics': per-stage diagnostics from every pipeline stage.
    """
    N = sc_gene_expr.shape[0]

    if generator is None:
        raise ValueError(
            "A trained generator is required: the pipeline uses the generator "
            "for the base-coordinate proposal in every patch."
        )

    # Diffusion start / preconditioning scales.
    if config.generator_only:
        sigma_start = 0.0
        sigma_data_effective = (
            config.sigma_data_resid
            if (config.use_residual_diffusion and config.sigma_data_resid is not None)
            else config.sigma_data
        )
    elif config.use_residual_diffusion and config.sigma_data_resid is not None:
        sigma_start = min(3.0 * config.sigma_data_resid, config.sigma_max)
        # Match training: residual mode preconditions on sigma_data_resid so the
        # inference c_skip / c_out coefficients line up with the trained net.
        sigma_data_effective = config.sigma_data_resid
    else:
        raise ValueError(
            "This pipeline extracts the residual diffusion path only. Set "
            "use_residual_diffusion=True and provide sigma_data_resid (or set "
            "generator_only=True)."
        )

    # Step 0: encode all cells.
    with torch.no_grad():
        encoder.eval()
        Z_all = encoder(sc_gene_expr.to(device))  # (N, h)

    # Step 1: locality graph.
    locality_graph = build_locality_graph_v2(
        Z=Z_all,
        k_Z=config.k_Z,
        k_sigma=config.k_sigma,
        tau_jaccard=config.tau_jaccard,
        min_shared=config.min_shared,
        device=device,
        verbose=config.verbose,
    )

    diag = locality_graph['diagnostics']
    if config.verbose and diag['isolated_nodes'] > N * 0.1:
        print(f"[WARNING] {diag['isolated_nodes']} isolated nodes "
              f"({100 * diag['isolated_nodes'] / N:.1f}%)")

    # Step 2: overlapping patches.
    patch_result = sample_patches_random_walk_v2(
        graph=locality_graph,
        N=N,
        patch_size=config.patch_size,
        overlap_frac=config.overlap_frac,
        coverage_per_cell=config.coverage_per_cell,
        min_overlap=config.min_overlap,
        device=device,
        verbose=config.verbose,
    )

    patch_indices = patch_result['patch_indices']
    patch_overlaps = patch_result['patch_overlaps']
    K = len(patch_indices)

    if config.verbose and not patch_result['diagnostics']['is_connected']:
        print(f"[WARNING] Patch graph disconnected "
              f"({patch_result['diagnostics']['n_components']} components)")

    # Step 3: per-patch geometry via residual diffusion.
    expected_in_ctx = getattr(context_encoder, "input_dim", None)
    if expected_in_ctx is None and hasattr(context_encoder, 'input_proj'):
        expected_in_ctx = context_encoder.input_proj.in_features

    patch_V = []
    residual_ratios = []
    rank2_energies = []

    context_encoder.eval()
    score_net.eval()
    generator.eval()

    for k in range(K):
        S_k = patch_indices[k]
        m_k = S_k.numel()

        Z_k = Z_all[S_k].unsqueeze(0)  # (1, m, h)
        mask_k = torch.ones(1, m_k, dtype=torch.bool, device=device)

        # Append an anchor channel if the context encoder expects one.
        if expected_in_ctx is not None and expected_in_ctx == Z_k.shape[-1] + 1:
            zeros_anchor = torch.zeros(1, m_k, 1, device=device, dtype=Z_k.dtype)
            Z_k = torch.cat([Z_k, zeros_anchor], dim=-1)

        # Feature-standardize Z (training used z-layernorm; must match at inference).
        if config.use_z_ln:
            Z_k = _apply_z_ln(Z_k, context_encoder)

        # Deterministic seed based on patch cell IDs.
        seed = hash(tuple(sorted(S_k.cpu().tolist()))) % (2 ** 31)

        if config.generator_only:
            with torch.no_grad():
                H_k = context_encoder(Z_k, mask_k)
                V_k = generator(H_k, mask_k).squeeze(0)
                residual_ratios.append(0.0)
        else:
            result = sample_patch_residual_diffusion_v2(
                Z_patch=Z_k,
                mask_patch=mask_k,
                context_encoder=context_encoder,
                generator=generator,
                score_net=score_net,
                sigma_data=sigma_data_effective,
                sigma_start=sigma_start,
                sigma_min=config.sigma_min,
                n_steps=config.n_diffusion_steps,
                guidance_scale=config.guidance_scale,
                seed=seed,
                device=device,
            )
            V_k = result['V_final']
            residual_ratios.append(result['residual_ratio'])

        patch_V.append(V_k)

    # Step 4: read sparse distance measurements off each patch.
    patch_measurements = []
    for k in range(K):
        meas = extract_patch_distances_v2(
            V_patch=patch_V[k],
            mask_patch=torch.ones(patch_V[k].shape[0], dtype=torch.bool, device=device),
            k_edge=config.k_edge,
            device=device,
        )
        patch_measurements.append(meas)
        rank2_energies.append(meas['diagnostics'].get('rank2_energy', 0))

    # Step 5: per-patch reliability from overlap agreement.
    overlap_result = compute_overlap_consistency_v2(
        patch_V=patch_V,
        patch_indices=patch_indices,
        patch_overlaps=patch_overlaps,
        verbose=config.verbose,
    )

    # Step 6: fuse measurements into one global distance graph.
    patch_consistency = overlap_result.get('patch_consistency', None)

    agg_result = aggregate_distance_measurements_v2(
        patch_measurements=patch_measurements,
        patch_indices=patch_indices,
        N=N,
        M_min=config.M_min,
        tau_spread=config.tau_spread,
        spread_alpha=config.spread_alpha,
        patch_consistency=patch_consistency,
        verbose=config.verbose,
    )

    global_edges = agg_result['edges']
    global_distances = agg_result['distances']
    global_weights = agg_result['weights']

    if config.verbose and len(global_edges) < N:
        print(f"[WARNING] Only {len(global_edges)} global edges for {N} nodes - "
              f"may have connectivity issues")

    # Step 7a: Landmark Isomap initialization.
    X_init = landmark_isomap_init_v2(
        edges=global_edges,
        distances=global_distances,
        N=N,
        n_landmarks=config.n_landmarks,
        device=device,
        DEBUG_FLAG=config.verbose,
        clip_runaway=config.clip_init_runaway,
    )

    # Step 7b: weighted-Huber distance-geometry refinement.
    solve_result = global_distance_geometry_solve_v2(
        edges=global_edges,
        distances=global_distances,
        weights=global_weights,
        N=N,
        X_init=X_init,
        n_iters=config.global_iters,
        lr=config.global_lr,
        huber_delta=config.huber_delta,
        anchor_lambda=config.anchor_lambda,
        log_every=max(1, config.global_iters // 10),
        device=device,
        DEBUG_FLAG=config.verbose,
    )

    X_final = solve_result['X_final']  # (N, 2)

    # Canonicalize: center and isotropic-scale to unit RMS.
    X_canon = X_final - X_final.mean(dim=0)
    rms = X_canon.pow(2).mean().sqrt()
    if rms > 1e-8:
        X_canon = X_canon / rms

    D = torch.cdist(X_canon, X_canon)  # (N, N)

    diagnostics = {
        'locality_graph': locality_graph['diagnostics'],
        'patch_sampling': patch_result['diagnostics'],
        'overlap_consistency': overlap_result['diagnostics'],
        'aggregation': agg_result['diagnostics'],
        'global_solve': solve_result['diagnostics'],
        'residual_ratio_median': float(np.median(residual_ratios)) if residual_ratios else 0.0,
        'rank2_energy_median': float(np.median(rank2_energies)) if rank2_energies else 0.0,
    }

    return {
        'coords': X_canon,
        'distances': D,
        'diagnostics': diagnostics,
    }
