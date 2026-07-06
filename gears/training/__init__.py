"""
Stage-C training: conditional geometry generator + EDM residual-diffusion refiner.

Public API:
    StageCConfig   -- Stage-C hyperparameters (residual-diffusion defaults).
    train_stageC   -- joint training loop for context encoder + generator + score net.
    train_diffusion -- alias for train_stageC.

    plus the Stage-C loss module public API (STAGE_C_WEIGHTS + loss terms).
"""

from .diffusion import StageCConfig, train_stageC, train_diffusion
from .losses_geom import (
    STAGE_C_WEIGHTS,
    WEIGHTS,
    assemble_total_loss,
    edm_loss_weight,
    edm_residual_score_loss,
    gram_losses,
    gram_learn_loss,
    out_scale_loss,
    knn_nca,
    knn_scale,
    edge_loss,
    subspace_loss,
    generator_supervision,
    build_structure_tensors,
    within_miniset_knn,
    build_geometry_gates,
    make_geometry_gates,
    AdaptiveQuantileGate,
    rigid_align_apply_no_scale,
    rigid_align_mse_no_scale,
)

__all__ = [
    "StageCConfig",
    "train_stageC",
    "train_diffusion",
    "STAGE_C_WEIGHTS",
    "WEIGHTS",
    "assemble_total_loss",
    "edm_loss_weight",
    "edm_residual_score_loss",
    "gram_losses",
    "gram_learn_loss",
    "out_scale_loss",
    "knn_nca",
    "knn_scale",
    "edge_loss",
    "subspace_loss",
    "generator_supervision",
    "build_structure_tensors",
    "within_miniset_knn",
    "build_geometry_gates",
    "make_geometry_gates",
    "AdaptiveQuantileGate",
    "rigid_align_apply_no_scale",
    "rigid_align_mse_no_scale",
]
