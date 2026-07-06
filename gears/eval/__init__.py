"""
Reconstruction evaluation metrics.

`evaluate_reconstruction` runs the full metric suite; the individual metric
functions are re-exported for standalone use.
"""

from .metrics import (
    evaluate_reconstruction,
    compute_scale_alignment,
    knn_from_distance_matrix,
    get_ranks,
    trustworthiness,
    continuity,
    local_rank_correlation,
    local_kendall_tau,
    shell_preservation_f1,
    edge_roc_auc,
    balanced_average_precision,
    edge_recall_multiscale,
    local_distance_rmse,
    local_stress,
    nearmiss_radius,
    normalized_stress,
    pairwise_distance_pearson,
    pairwise_distance_spearman,
    relative_frobenius_error,
    wasserstein_knn_distances,
    sliced_wasserstein_distance,
    density_earth_movers_distance,
    radial_profile_comparison,
    optimal_matching_evaluation,
)
from .plotting import plot_reconstruction, score_reconstruction

__all__ = [
    "evaluate_reconstruction",
    "plot_reconstruction",
    "score_reconstruction",
    "compute_scale_alignment",
    "knn_from_distance_matrix",
    "get_ranks",
    "trustworthiness",
    "continuity",
    "local_rank_correlation",
    "local_kendall_tau",
    "shell_preservation_f1",
    "edge_roc_auc",
    "balanced_average_precision",
    "edge_recall_multiscale",
    "local_distance_rmse",
    "local_stress",
    "nearmiss_radius",
    "normalized_stress",
    "pairwise_distance_pearson",
    "pairwise_distance_spearman",
    "relative_frobenius_error",
    "wasserstein_knn_distances",
    "sliced_wasserstein_distance",
    "density_earth_movers_distance",
    "radial_profile_comparison",
    "optimal_matching_evaluation",
]
