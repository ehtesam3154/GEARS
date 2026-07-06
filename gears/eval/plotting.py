"""Score and plot a reconstruction against ground-truth spatial coordinates."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..inference.outliers import inlier_mask
from .metrics import evaluate_reconstruction


def _canon(X):
    X = X - X.mean(0)
    return X / (np.sqrt((X ** 2).sum(1)).mean() + 1e-8)


def _align(P, G):  # orthogonal Procrustes P -> G
    U, _, Vt = np.linalg.svd(P.T @ G)
    return P @ (U @ Vt)


def score_reconstruction(coords, gt_coords, is_outlier=None, full=False,
                         subsample=5000, seed=1):
    """Distance-geometry metrics (spearman/pearson/edge-ROC/trust/cont/SWD/W1) of
    a reconstruction vs GT. Evaluated on the inlier set; large clouds are
    subsampled for the O(N^2) metrics unless ``full=True``."""
    coords = np.asarray(coords)
    gt = np.asarray(gt_coords)
    keep = (~np.asarray(is_outlier)) if is_outlier is not None else inlier_mask(coords)
    cp, cg = coords[keep], gt[keep]
    if full or len(cg) <= subsample:
        idx = np.arange(len(cg))
    else:
        idx = np.random.RandomState(seed).choice(len(cg), subsample, replace=False)
    return evaluate_reconstruction(
        squareform(pdist(cg[idx])), squareform(pdist(cp[idx])),
        cg[idx], cp[idx], verbose=False, compute_matching=False)


def plot_reconstruction(coords, gt_coords, color=None, is_outlier=None,
                        title="reconstruction", save_path=None, point_size=12):
    """Side-by-side GT vs reconstruction, Procrustes-aligned and colored by
    ``color`` (default: GT second axis). Returns (fig, ax)."""
    import matplotlib.pyplot as plt
    coords = np.asarray(coords)
    gt = np.asarray(gt_coords)
    keep = (~np.asarray(is_outlier)) if is_outlier is not None else inlier_mask(coords)
    G = _canon(gt[keep])
    P = _align(_canon(coords[keep]), G)
    c = np.asarray(color)[keep] if color is not None else G[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, X, t in [(ax[0], G, "ground truth"), (ax[1], P, title)]:
        a.scatter(X[:, 0], X[:, 1], c=c, cmap="viridis", s=point_size, alpha=0.85, linewidths=0)
        a.set_title(t)
        a.set_aspect("equal")
        a.axis("off")
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, ax
