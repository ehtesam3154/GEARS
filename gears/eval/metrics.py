"""
Reconstruction evaluation metrics for 2D spatial coordinate recovery.

Compares predicted 2D coordinates / pairwise distances against ground-truth
coordinates / distances. Metrics are grouped as:

- Global geometry: Spearman / Pearson correlation of upper-triangular pairwise
  distances, Kruskal Stress-1, relative Frobenius error.
- Local geometry: edge ROC-AUC and balanced Average Precision (near vs. far
  pairs relative to a ground-truth radius), macro Shell-F1 over distance shells,
  local rank correlations, local stress and local RMSE with a locally refit scale.
- Neighborhood quality: Trustworthiness@k and Continuity@k.
- Distribution: Sliced Wasserstein Distance on point clouds, W1 on kNN-distance
  distributions, density EMD, radial-profile comparison.
- Diagnostics: optimal (Hungarian) point matching, near-miss fraction.

All distance-based metrics operate on Euclidean distance matrices. The predicted
distances are first rescaled by an optimal global scale factor alpha* (least
squares on upper-triangular distances); a few local metrics refit that scale on
the local pairs only.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau, wasserstein_distance, ks_2samp
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def knn_from_distance_matrix(D, k):
    """Return k nearest-neighbor indices per row (excluding self)."""
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    return idx


def get_ranks(D):
    """Return a rank matrix: ranks[i, j] is the rank of column j within row i."""
    n = D.shape[0]
    ranks = np.zeros_like(D, dtype=int)
    for i in range(n):
        ranks[i] = np.argsort(np.argsort(D[i]))
    return ranks


def compute_scale_alignment(D_gt, D_pred):
    """
    Optimal scale factor aligning predicted distances to ground truth.

    alpha* = argmin_alpha sum (D_gt - alpha * D_pred)^2 on upper-triangular pairs.

    Returns:
        alpha_star, D_pred_scaled (= alpha_star * D_pred)
    """
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred[triu_idx]

    alpha_star = np.sum(d_gt * d_pred) / np.sum(d_pred ** 2)
    D_pred_scaled = alpha_star * D_pred

    return alpha_star, D_pred_scaled


# ---------------------------------------------------------------------------
# Neighborhood quality
# ---------------------------------------------------------------------------

def trustworthiness(D_gt, D_pred, k=20):
    """
    Trustworthiness@k: penalizes predicted neighbors that are far in GT.

    Rank-based and scale-invariant. Returns a value in [0, 1] (1 = perfect).
    """
    n = D_gt.shape[0]

    knn_gt = knn_from_distance_matrix(D_gt, k)
    knn_pred = knn_from_distance_matrix(D_pred, k)

    ranks_gt = get_ranks(D_gt)

    trust_sum = 0.0
    for i in range(n):
        gt_set = set(knn_gt[i])
        pred_set = set(knn_pred[i])

        # Intruders: predicted neighbors not in GT top-k.
        intruders = pred_set - gt_set

        for j in intruders:
            rank_j_in_gt = ranks_gt[i, j]
            trust_sum += max(0, rank_j_in_gt - k)

    T = 1 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * trust_sum
    return T


def continuity(D_gt, D_pred, k=20):
    """
    Continuity@k: penalizes GT neighbors that are far in the prediction.

    Rank-based and scale-invariant. Returns a value in [0, 1] (1 = perfect).
    """
    n = D_gt.shape[0]

    knn_gt = knn_from_distance_matrix(D_gt, k)
    knn_pred = knn_from_distance_matrix(D_pred, k)

    ranks_pred = get_ranks(D_pred)

    cont_sum = 0.0
    for i in range(n):
        gt_set = set(knn_gt[i])
        pred_set = set(knn_pred[i])

        # Missing: GT neighbors not in predicted top-k.
        missing = gt_set - pred_set

        for j in missing:
            rank_j_in_pred = ranks_pred[i, j]
            cont_sum += max(0, rank_j_in_pred - k)

    C = 1 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * cont_sum
    return C


# ---------------------------------------------------------------------------
# Local geometry
# ---------------------------------------------------------------------------

def local_rank_correlation(D_gt, D_pred, k_local=50, method='spearman'):
    """
    Per-point rank correlation of distances to each cell's GT neighborhood.

    Returns (mean, median, per_point) correlations.
    """
    n = D_gt.shape[0]

    knn_gt = knn_from_distance_matrix(D_gt, k_local)

    local_corrs = np.empty(n, dtype=np.float32)

    for i in range(n):
        neighbors = knn_gt[i]

        d_gt_local = D_gt[i, neighbors]
        d_pred_local = D_pred[i, neighbors]

        if method == 'spearman':
            corr, _ = spearmanr(d_gt_local, d_pred_local)
        elif method == 'kendall':
            corr, _ = kendalltau(d_gt_local, d_pred_local)
        else:
            raise ValueError(f"Unknown method: {method}")

        local_corrs[i] = corr if not np.isnan(corr) else 0.0

    return local_corrs.mean(), np.median(local_corrs), local_corrs


def local_kendall_tau(D_gt, D_pred, k_local=50):
    """
    Per-point tie-aware Kendall tau_b over each cell's GT neighborhood.

    Returns (mean, median, per_point) tau values.
    """
    n = D_gt.shape[0]

    knn_gt = knn_from_distance_matrix(D_gt, k_local)

    local_taus = np.empty(n, dtype=np.float32)

    for i in range(n):
        neighbors = knn_gt[i]

        d_gt_local = D_gt[i, neighbors]
        d_pred_local = D_pred[i, neighbors]

        tau, _ = kendalltau(d_gt_local, d_pred_local)

        local_taus[i] = tau if not np.isnan(tau) else 0.0

    return local_taus.mean(), np.median(local_taus), local_taus


def shell_preservation_f1(D_gt, D_pred_scaled, n_bins=10, local_only=True, local_bins=5):
    """
    Macro-F1 over distance shells (quantile bins of GT distances).

    A pair is "correct" for shell t when both its GT and predicted distances fall
    in shell t. Returns (macro_f1, bin_edges, per_bin_f1).
    """
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred_scaled[triu_idx]

    # Quantile bin edges; open first / last bin.
    bin_edges = np.quantile(d_gt, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = 0
    bin_edges[-1] = np.inf

    s_gt = np.digitize(d_gt, bin_edges[1:])
    s_pred = np.digitize(d_pred, bin_edges[1:])

    per_bin_f1 = []
    bins_to_eval = range(local_bins) if local_only else range(n_bins)

    for t in bins_to_eval:
        tp = np.sum((s_gt == t) & (s_pred == t))
        fp = np.sum((s_gt != t) & (s_pred == t))
        fn = np.sum((s_gt == t) & (s_pred != t))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_bin_f1.append(f1)

    macro_f1 = np.mean(per_bin_f1)

    return macro_f1, bin_edges, np.array(per_bin_f1)


def edge_roc_auc(D_gt, D_pred_scaled, k_for_radius=50):
    """
    Binary near/far edge classification.

    Radius R = median over cells of the distance to the k_for_radius-th GT
    neighbor. Pairs with GT distance <= R are "near" (positive). Score is the
    negative predicted distance. Returns (roc_auc, pr_auc, R).
    """
    n = D_gt.shape[0]

    knn_distances = np.sort(D_gt, axis=1)[:, 1:k_for_radius + 1]
    k_th_distances = knn_distances[:, -1]
    R = np.median(k_th_distances)

    triu_idx = np.triu_indices(n, k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred_scaled[triu_idx]

    y_true = (d_gt <= R).astype(int)
    y_score = -d_pred

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return roc_auc, pr_auc, R


def balanced_average_precision(D_gt, D_pred_scaled, k_for_radius=50):
    """
    Per-anchor Average Precision with balanced positive/negative sampling.

    For each anchor, positives are GT neighbors within radius R and negatives are
    sampled (without replacement) to match the positive count. Score is the
    negative predicted distance. Returns (mean_bAP, per_anchor_AP).
    """
    n = D_gt.shape[0]

    knn_distances = np.sort(D_gt, axis=1)[:, 1:k_for_radius + 1]
    k_th_distances = knn_distances[:, -1]
    R = np.median(k_th_distances)

    per_anchor_ap = []

    for i in range(n):
        pos_mask = (D_gt[i] <= R) & (np.arange(n) != i)
        pos_indices = np.where(pos_mask)[0]

        neg_mask = (D_gt[i] > R)
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        n_pos = len(pos_indices)
        if len(neg_indices) > n_pos:
            neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)

        indices = np.concatenate([pos_indices, neg_indices])
        y_true = np.concatenate([np.ones(len(pos_indices)), np.zeros(len(neg_indices))])
        y_score = -D_pred_scaled[i, indices]

        ap = average_precision_score(y_true, y_score)
        per_anchor_ap.append(ap)

    mean_bAP = np.mean(per_anchor_ap)
    return mean_bAP, np.array(per_anchor_ap)


def edge_recall_multiscale(D_gt, D_pred_scaled, k_values=[6, 18, 50]):
    """
    Fraction of GT near-pairs also predicted near, at several radius scales.

    For each k, R = median distance to the k-th GT neighbor. Returns a dict
    mapping k -> recall.
    """
    n = D_gt.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred_scaled[triu_idx]

    recalls = {}

    for k in k_values:
        knn_distances = np.sort(D_gt, axis=1)[:, 1:k + 1]
        k_th_distances = knn_distances[:, -1]
        R = np.median(k_th_distances)

        gt_near = d_gt <= R
        pred_near = d_pred <= R

        if gt_near.sum() == 0:
            recalls[k] = 0.0
        else:
            recall = (gt_near & pred_near).sum() / gt_near.sum()
            recalls[k] = recall

    return recalls


def local_distance_rmse(D_gt, D_pred, k_for_radius=50):
    """
    RMSE of distances on local pairs, with scale refit on those local pairs.

    D_pred is the UNSCALED predicted distance matrix. Local pairs are those with
    GT distance <= R (median distance to the k_for_radius-th GT neighbor).
    Returns (rmse, R, alpha_R).
    """
    n = D_gt.shape[0]

    knn_distances = np.sort(D_gt, axis=1)[:, 1:k_for_radius + 1]
    k_th_distances = knn_distances[:, -1]
    R = np.median(k_th_distances)

    triu_idx = np.triu_indices(n, k=1)
    d_gt_all = D_gt[triu_idx]
    d_pred_all = D_pred[triu_idx]

    local_mask = d_gt_all <= R
    d_gt_local = d_gt_all[local_mask]
    d_pred_local = d_pred_all[local_mask]

    alpha_R = np.sum(d_gt_local * d_pred_local) / np.sum(d_pred_local ** 2)
    d_pred_local_scaled = alpha_R * d_pred_local

    rmse = np.sqrt(np.mean((d_gt_local - d_pred_local_scaled) ** 2))

    return rmse, R, alpha_R


def local_stress(D_gt, D_pred, k_for_radius=50):
    """
    Kruskal Stress-1 restricted to local pairs, with scale refit on those pairs.

    D_pred is the UNSCALED predicted distance matrix. Returns
    (local_stress_value, R, alpha_R).
    """
    n = D_gt.shape[0]

    knn_distances = np.sort(D_gt, axis=1)[:, 1:k_for_radius + 1]
    k_th_distances = knn_distances[:, -1]
    R = np.median(k_th_distances)

    triu_idx = np.triu_indices(n, k=1)
    d_gt_all = D_gt[triu_idx]
    d_pred_all = D_pred[triu_idx]

    local_mask = d_gt_all <= R
    d_gt_local = d_gt_all[local_mask]
    d_pred_local = d_pred_all[local_mask]

    alpha_R = np.sum(d_gt_local * d_pred_local) / np.sum(d_pred_local ** 2)
    d_pred_local_scaled = alpha_R * d_pred_local

    numerator = np.sum((d_gt_local - d_pred_local_scaled) ** 2)
    denominator = np.sum(d_gt_local ** 2)

    local_stress_val = np.sqrt(numerator / denominator)

    return local_stress_val, R, alpha_R


def nearmiss_radius(coords_gt, coords_pred, radius=None, k_for_radius=50):
    """
    Fraction of cells whose predicted position is within radius R of GT.

    If radius is None, R = median distance to the k_for_radius-th GT neighbor,
    making the metric scale-adaptive. Returns
    (nearmiss_frac, radius_used, per_point_errors).
    """
    errors = np.linalg.norm(coords_gt - coords_pred, axis=1)

    if radius is None:
        D_gt = squareform(pdist(coords_gt, 'euclidean'))
        knn_distances = np.sort(D_gt, axis=1)[:, 1:k_for_radius + 1]
        k_th_distances = knn_distances[:, -1]
        radius = np.median(k_th_distances)

    within_radius = errors <= radius
    nearmiss_frac = within_radius.mean()

    return nearmiss_frac, radius, errors


# ---------------------------------------------------------------------------
# Global geometry
# ---------------------------------------------------------------------------

def normalized_stress(D_gt, D_pred_scaled):
    """
    Kruskal Stress-1: sqrt(sum (D_gt - D_pred)^2 / sum D_gt^2) on upper triangle.

    Uses the scale-aligned predicted distances. Lower is better (0 = perfect).
    """
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred_scaled[triu_idx]

    numerator = np.sum((d_gt - d_pred) ** 2)
    denominator = np.sum(d_gt ** 2)

    stress = np.sqrt(numerator / denominator)
    return stress


def pairwise_distance_pearson(D_gt, D_pred):
    """Pearson correlation of upper-triangular pairwise distances."""
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred[triu_idx]

    corr = np.corrcoef(d_gt, d_pred)[0, 1]
    return corr


def pairwise_distance_spearman(D_gt, D_pred):
    """Spearman (rank) correlation of upper-triangular pairwise distances."""
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred[triu_idx]

    corr, _ = spearmanr(d_gt, d_pred)
    return corr


def relative_frobenius_error(D_gt, D_pred_scaled):
    """
    Relative Frobenius error ||D_gt - D_pred|| / ||D_gt|| on upper-triangular pairs.

    Uses the scale-aligned predicted distances. Lower is better (0 = perfect).
    """
    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    d_gt = D_gt[triu_idx]
    d_pred = D_pred_scaled[triu_idx]

    numerator = np.linalg.norm(d_gt - d_pred)
    denominator = np.linalg.norm(d_gt)

    rel_err = numerator / denominator
    return rel_err


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def wasserstein_knn_distances(D_gt, D_pred, k=20):
    """
    Wasserstein-1 distance between kNN-distance distributions (identity-free).

    Compares the distribution of distances to the k nearest neighbors, globally
    and per point. Returns (W1_global, W1_per_point).
    """
    n = D_gt.shape[0]

    knn_dists_gt = np.sort(D_gt, axis=1)[:, 1:k + 1]
    knn_dists_pred = np.sort(D_pred, axis=1)[:, 1:k + 1]

    all_gt = knn_dists_gt.flatten()
    all_pred = knn_dists_pred.flatten()
    W1_global = wasserstein_distance(all_gt, all_pred)

    W1_per_point = np.array([
        wasserstein_distance(knn_dists_gt[i], knn_dists_pred[i])
        for i in range(n)
    ])

    return W1_global, W1_per_point


def _canonicalize(X, eps=1e-12):
    """Center a point cloud and rescale it to unit RMS radius."""
    X = X - X.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(X ** 2, axis=1))) + eps
    return X / rms


def _pca_align(A, B):
    """Rotate A onto B's principal axes (no correspondences)."""
    Ua = np.linalg.svd(np.cov(A.T), full_matrices=False)[0]
    Ub = np.linalg.svd(np.cov(B.T), full_matrices=False)[0]
    R = Ua @ Ub.T
    return A @ R


def sliced_wasserstein_distance(coords_gt, coords_pred, n_projections=1000,
                                seed=42, canonicalize=True, pca_align=True):
    """
    Sliced Wasserstein Distance between two point clouds.

    Optionally canonicalizes (center + unit RMS radius) and PCA-aligns the
    prediction to the GT before averaging 1D Wasserstein distances over
    n_projections random directions.
    """
    coords_gt = np.asarray(coords_gt, dtype=np.float64)
    coords_pred = np.asarray(coords_pred, dtype=np.float64)

    if coords_gt.shape != coords_pred.shape:
        raise ValueError(f"Shape mismatch: gt {coords_gt.shape} vs pred {coords_pred.shape}")

    if canonicalize:
        coords_gt = _canonicalize(coords_gt)
        coords_pred = _canonicalize(coords_pred)

    if pca_align:
        coords_pred = _pca_align(coords_pred, coords_gt)

    rng = np.random.default_rng(seed)
    n, d = coords_gt.shape

    directions = rng.normal(size=(n_projections, d))
    directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)

    proj_gt = coords_gt @ directions.T
    proj_pr = coords_pred @ directions.T

    distances = [wasserstein_distance(proj_gt[:, p], proj_pr[:, p]) for p in range(n_projections)]
    return float(np.mean(distances))


def density_earth_movers_distance(coords_gt, coords_pred, grid_size=64):
    """
    Wasserstein-1 distance between 2D occupancy histograms.

    Both clouds are binned on a shared padded grid, normalized to probability
    distributions, and compared with a 1D Wasserstein distance over the flattened
    bins. Returns the EMD.
    """
    all_coords = np.vstack([coords_gt, coords_pred])
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)

    padding = 0.1 * (maxs - mins)
    mins -= padding
    maxs += padding

    hist_gt, xedges, yedges = np.histogram2d(
        coords_gt[:, 0], coords_gt[:, 1],
        bins=grid_size, range=[[mins[0], maxs[0]], [mins[1], maxs[1]]]
    )

    hist_pred, _, _ = np.histogram2d(
        coords_pred[:, 0], coords_pred[:, 1],
        bins=grid_size, range=[[mins[0], maxs[0]], [mins[1], maxs[1]]]
    )

    hist_gt = hist_gt.flatten() / hist_gt.sum()
    hist_pred = hist_pred.flatten() / hist_pred.sum()

    emd = wasserstein_distance(hist_gt, hist_pred)

    return emd


def radial_profile_comparison(coords_gt, coords_pred):
    """
    Compare radial distances from each cloud's centroid (rotation invariant).

    Returns (wasserstein_dist, ks_statistic, ks_pvalue).
    """
    centroid_gt = coords_gt.mean(axis=0)
    centroid_pred = coords_pred.mean(axis=0)

    radii_gt = np.linalg.norm(coords_gt - centroid_gt, axis=1)
    radii_pred = np.linalg.norm(coords_pred - centroid_pred, axis=1)

    wd = wasserstein_distance(radii_gt, radii_pred)

    ks_stat, ks_pval = ks_2samp(radii_gt, radii_pred)

    return wd, ks_stat, ks_pval


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def optimal_matching_evaluation(coords_gt, coords_pred, D_gt, return_matching=False):
    """
    Optimal point-to-point matching (Hungarian) between the two clouds.

    Distinguishes correct geometry from correct correspondence. Returns a dict
    with the assignment, total matching cost, and mean/median matched-point
    coordinate error. When return_matching is True, also returns the reordered
    predicted coordinates and their distance matrix.
    """
    n = coords_gt.shape[0]

    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(coords_gt[i] - coords_pred[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    coords_pred_matched = coords_pred[col_ind]

    D_pred_matched = squareform(pdist(coords_pred_matched, 'euclidean'))

    coord_errors = np.linalg.norm(coords_gt - coords_pred_matched, axis=1)

    results = {
        'optimal_assignment': col_ind,
        'matching_cost': cost_matrix[row_ind, col_ind].sum(),
        'mean_matched_distance': coord_errors.mean(),
        'median_matched_distance': np.median(coord_errors),
        'coord_errors': coord_errors,
    }

    if return_matching:
        return results, coords_pred_matched, D_pred_matched
    else:
        return results


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate_reconstruction(D_gt, D_pred, coords_gt=None, coords_pred=None,
                            k_values=[10, 20, 50], k_local=50, verbose=True,
                            compute_matching=True):
    """
    Full reconstruction evaluation.

    Args:
        D_gt: ground-truth pairwise distance matrix (N x N).
        D_pred: predicted pairwise distance matrix (N x N).
        coords_gt: ground-truth coordinates (N x d); enables coordinate-space
            and distribution metrics when provided.
        coords_pred: predicted coordinates (N x d); enables coordinate-space and
            distribution metrics when provided.
        k_values: neighborhood sizes for Trustworthiness / Continuity.
        k_local: neighborhood size for local rank correlations.
        verbose: print a running report.

    Returns:
        results: dict of all metric values.
    """
    results = {}

    # Global scale alignment.
    alpha_star, D_pred_scaled = compute_scale_alignment(D_gt, D_pred)
    results['alpha_star'] = alpha_star

    triu_idx = np.triu_indices(D_gt.shape[0], k=1)
    rms_gt = np.sqrt(np.mean(D_gt[triu_idx] ** 2))
    rms_pred = np.sqrt(np.mean(D_pred[triu_idx] ** 2))
    scale_bias = rms_pred / rms_gt
    results['scale_bias_rms'] = scale_bias

    if verbose:
        print("=" * 70)
        print("SPATIAL RECONSTRUCTION EVALUATION")
        print("=" * 70)
        print(f"\n[SCALE ALIGNMENT]")
        print(f"  alpha* (optimal scale factor): {alpha_star:.4f}")
        print(f"  Scale bias (RMS ratio):        {scale_bias:.4f}")

    if verbose:
        print(f"\n{'=' * 70}")
        print("LOCAL GEOMETRY")
        print(f"{'=' * 70}")

    # Trustworthiness & Continuity.
    for k in k_values:
        T = trustworthiness(D_gt, D_pred_scaled, k=k)
        C = continuity(D_gt, D_pred_scaled, k=k)
        results[f'trustworthiness@{k}'] = T
        results[f'continuity@{k}'] = C

        if verbose:
            print(f"\n[k={k}]")
            print(f"  Trustworthiness: {T:.4f}")
            print(f"  Continuity:      {C:.4f}")

    # Local rank correlation (Spearman).
    mean_corr, median_corr, per_point_corr = local_rank_correlation(
        D_gt, D_pred_scaled, k_local=k_local, method='spearman'
    )
    results['local_spearman_mean'] = mean_corr
    results['local_spearman_median'] = median_corr
    results['local_spearman_per_point'] = per_point_corr

    if verbose:
        print(f"\n[Local Rank Correlation (Spearman, k={k_local})]")
        print(f"  Mean:   {mean_corr:.4f}")
        print(f"  Median: {median_corr:.4f}")

    # Local Kendall tau_b (tie-aware).
    mean_tau, median_tau, per_point_tau = local_kendall_tau(D_gt, D_pred_scaled, k_local=k_local)
    results['local_kendall_mean'] = mean_tau
    results['local_kendall_median'] = median_tau
    results['local_kendall_per_point'] = per_point_tau

    if verbose:
        print(f"\n[Local Kendall tau_b (k={k_local})]")
        print(f"  Mean:   {mean_tau:.4f}")
        print(f"  Median: {median_tau:.4f}")

    # Shell preservation F1.
    shell_f1, bin_edges, per_bin_f1 = shell_preservation_f1(
        D_gt, D_pred_scaled, n_bins=10, local_only=True, local_bins=5
    )
    results['shell_f1'] = shell_f1
    results['shell_bin_edges'] = bin_edges
    results['shell_per_bin_f1'] = per_bin_f1

    if verbose:
        print(f"\n[Shell Preservation F1 (local bins)]")
        print(f"  Macro-F1: {shell_f1:.4f}")

    # Edge ROC-AUC.
    roc_auc, pr_auc, edge_R = edge_roc_auc(D_gt, D_pred_scaled, k_for_radius=50)
    results['edge_roc_auc'] = roc_auc
    results['edge_pr_auc'] = pr_auc
    results['edge_radius'] = edge_R

    if verbose:
        print(f"\n[Edge Classification (near vs far)]")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  Radius:  {edge_R:.4f}")

    # Balanced Average Precision.
    mean_bAP, per_anchor_bAP = balanced_average_precision(D_gt, D_pred_scaled, k_for_radius=50)
    results['balanced_AP'] = mean_bAP
    results['balanced_AP_per_anchor'] = per_anchor_bAP

    if verbose:
        print(f"\n[Balanced Average Precision]")
        print(f"  bAP (balanced): {mean_bAP:.4f}")

    # Multi-scale edge recall.
    edge_recalls = edge_recall_multiscale(D_gt, D_pred_scaled, k_values=[6, 18, 50])
    results['edge_recalls'] = edge_recalls

    if verbose:
        print(f"\n[Multi-scale Edge Recall]")
        for k, recall in edge_recalls.items():
            print(f"  Recall@k={k}: {recall:.4f}")

    # Local distance RMSE (uses unscaled D_pred).
    local_rmse, local_rmse_R, local_rmse_alpha = local_distance_rmse(D_gt, D_pred, k_for_radius=50)
    results['local_rmse'] = local_rmse
    results['local_rmse_radius'] = local_rmse_R

    if verbose:
        print(f"\n[Local Distance RMSE (R={local_rmse_R:.4f})]")
        print(f"  RMSE: {local_rmse:.4f}")

    # Near-miss fraction (coordinate space).
    if coords_gt is not None and coords_pred is not None:
        nearmiss_frac, radius, errors = nearmiss_radius(coords_gt, coords_pred)
        results['nearmiss_frac'] = nearmiss_frac
        results['nearmiss_radius'] = radius
        results['nearmiss_errors'] = errors

        if verbose:
            print(f"\n[NearMiss@R]")
            print(f"  Radius (auto-computed): {radius:.4f}")
            print(f"  Fraction within R: {nearmiss_frac:.4f}")
            print(f"  Median error: {np.median(errors):.4f}")

    # Distribution & diagnostic metrics (coordinate space).
    if coords_gt is not None and coords_pred is not None:
        swd = sliced_wasserstein_distance(coords_gt, coords_pred)
        results['sliced_wasserstein'] = swd

        density_emd = density_earth_movers_distance(coords_gt, coords_pred)
        results['density_emd'] = density_emd

        radial_wd, radial_ks_stat, radial_ks_pval = radial_profile_comparison(coords_gt, coords_pred)
        results['radial_wasserstein'] = radial_wd
        results['radial_ks_statistic'] = radial_ks_stat
        results['radial_ks_pvalue'] = radial_ks_pval

        # Hungarian matching is O(N^3); skip on large clouds (compute_matching=False).
        if compute_matching:
            matching_results = optimal_matching_evaluation(coords_gt, coords_pred, D_gt)
            results['matching_cost'] = matching_results['matching_cost']
            results['mean_matched_distance'] = matching_results['mean_matched_distance']
            results['median_matched_distance'] = matching_results['median_matched_distance']

        if verbose:
            print(f"\n[Distribution Metrics (unlabeled)]")
            print(f"  Sliced Wasserstein Distance: {swd:.4f}")
            print(f"  Density EMD: {density_emd:.4f}")
            print(f"  Radial Wasserstein: {radial_wd:.4f}")
            print(f"  Radial KS statistic: {radial_ks_stat:.4f}")

            if compute_matching:
                print(f"\n[Optimal Matching Diagnostic (Hungarian)]")
                print(f"  Mean matched distance: {matching_results['mean_matched_distance']:.4f}")
                print(f"  Median matched distance: {matching_results['median_matched_distance']:.4f}")

    # Wasserstein between kNN-distance distributions.
    W1_global, W1_per_point = wasserstein_knn_distances(D_gt, D_pred_scaled, k=20)
    results['wasserstein_knn_global'] = W1_global
    results['wasserstein_knn_per_point'] = W1_per_point

    if verbose:
        print(f"\n[Wasserstein k-NN Distance (k=20)]")
        print(f"  Global W1: {W1_global:.4f}")

    if verbose:
        print(f"\n{'=' * 70}")
        print("GLOBAL GEOMETRY")
        print(f"{'=' * 70}")

    # Global stress.
    stress = normalized_stress(D_gt, D_pred_scaled)
    results['stress'] = stress
    if verbose:
        print(f"\n[Normalized Stress]")
        print(f"  Stress: {stress:.4f}")

    # Local stress (uses unscaled D_pred).
    local_stress_val, local_R, alpha_R = local_stress(D_gt, D_pred, k_for_radius=50)
    results['local_stress'] = local_stress_val
    results['local_stress_radius'] = local_R
    results['local_stress_alpha_R'] = alpha_R

    if verbose:
        print(f"\n[Local Stress (restricted to R={local_R:.4f})]")
        print(f"  Local Stress: {local_stress_val:.4f}")
        print(f"  alpha_R (local scale): {alpha_R:.4f}")

    # Distance correlations.
    pearson = pairwise_distance_pearson(D_gt, D_pred_scaled)
    spearman = pairwise_distance_spearman(D_gt, D_pred_scaled)
    results['distance_pearson'] = pearson
    results['distance_spearman'] = spearman

    if verbose:
        print(f"\n[Distance Correlations]")
        print(f"  Pearson:  {pearson:.4f}")
        print(f"  Spearman: {spearman:.4f}")

    # Relative Frobenius error.
    rel_err = relative_frobenius_error(D_gt, D_pred_scaled)
    results['relative_frobenius_error'] = rel_err

    if verbose:
        print(f"\n[Relative Frobenius Error]")
        print(f"  RelErr_F: {rel_err:.4f}")

    if verbose:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")

        print(f"\n{'METRIC':<40} {'VALUE':>10}")
        print(f"{'-' * 70}")

        print(f"\nScale Alignment:")
        print(f"  {'alpha* (optimal scale)':<38} {results['alpha_star']:>10.4f}")
        print(f"  {'Scale bias (RMS ratio)':<38} {results['scale_bias_rms']:>10.4f}")

        print(f"\nLocal Geometry:")
        print(f"  {'Trustworthiness@20':<38} {results['trustworthiness@20']:>10.4f}")
        print(f"  {'Continuity@20':<38} {results['continuity@20']:>10.4f}")
        print(f"  {'Local Kendall tau_b (k=50)':<38} {results['local_kendall_mean']:>10.4f}")
        print(f"  {'Shell F1 (local)':<38} {results['shell_f1']:>10.4f}")
        print(f"  {'Edge ROC-AUC':<38} {results['edge_roc_auc']:>10.4f}")
        if 'nearmiss_frac' in results:
            print(f"  {'NearMiss@R':<38} {results['nearmiss_frac']:>10.4f}")

        print(f"\nGlobal Geometry:")
        print(f"  {'Stress (global)':<38} {results['stress']:>10.4f}")
        print(f"  {'Local Stress':<38} {results['local_stress']:>10.4f}")
        print(f"  {'Distance Pearson':<38} {results['distance_pearson']:>10.4f}")
        print(f"  {'Distance Spearman':<38} {results['distance_spearman']:>10.4f}")
        print(f"  {'Relative Frobenius Error':<38} {results['relative_frobenius_error']:>10.4f}")

        if 'sliced_wasserstein' in results:
            print(f"\nDistribution Metrics:")
            print(f"  {'Sliced Wasserstein Distance':<38} {results['sliced_wasserstein']:>10.4f}")
            print(f"  {'Density EMD':<38} {results['density_emd']:>10.4f}")
            print(f"  {'Wasserstein k-NN (global)':<38} {results['wasserstein_knn_global']:>10.4f}")

        print(f"\n{'=' * 70}")

    return results
