"""Geometry helpers for evaluation / plotting."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def inlier_mask(coords: np.ndarray, k: int = 8, z_thresh: float = 5.0) -> np.ndarray:
    """Keep-mask over a point cloud (True = keep). Drops points whose kNN
    isolation is an extreme robust (median/MAD) z-score outlier."""
    coords = np.asarray(coords)
    n = coords.shape[0]
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    iso = d[:, 1:].mean(axis=1)
    med = np.median(iso)
    mad = np.median(np.abs(iso - med)) + 1e-12
    return (0.6745 * (iso - med) / mad) <= z_thresh
