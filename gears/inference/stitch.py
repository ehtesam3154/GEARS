"""
Patchwise distance stitching for single-cell reconstruction.

After each local patch has been embedded independently (distance-first, in its
own gauge), this module fuses the overlapping patches into a single global
sparse distance graph:

    compute_overlap_consistency_v2
        For every pair of overlapping patches, compare the pairwise distance
        matrices restricted to their shared cells. Disagreement is measured in
        log-distance space (gauge-invariant), reduced to a per-patch mean and
        turned into a per-patch reliability score exp(-disagreement / scale).

    aggregate_distance_measurements_v2
        Collect every per-patch measurement of each global edge, weight each
        measurement by its source patch's reliability, take a reliability-
        weighted median as the consensus distance, and keep only edges seen at
        least M_min times with a small relative spread. Surviving edges get a
        trust weight count * exp(-alpha * spread^2). Isolated nodes are rescued
        with single-measurement fallback edges so the global solve stays
        connected.

The output of `aggregate_distance_measurements_v2` is the stitched sparse
distance graph (edges, consensus distances, confidence weights) consumed by the
global distance-geometry solve.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_overlap_consistency_v2(
    patch_V: List[torch.Tensor],
    patch_indices: List[torch.Tensor],
    patch_overlaps: Dict[Tuple[int, int], set],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Score how well overlapping patches agree on their shared geometry.

    For each overlapping patch pair, compare the pairwise distance matrices on
    the overlap in log space (scale/gauge invariant). High disagreement flags a
    patch whose local embedding is unreliable.

    Args:
        patch_V:         per-patch coordinates, tensor (n_k, D) for patch k.
        patch_indices:   per-patch global cell indices, tensor of length n_k.
        patch_overlaps:  {(k1, k2): set(global indices shared by k1 and k2)}.

    Returns:
        dict with per-pair disagreements, per-patch consistency scores, and
        aggregate statistics.
    """
    pair_disagreements = {}
    disagreement_values = []

    for (k1, k2), overlap_global in patch_overlaps.items():
        if len(overlap_global) < 5:
            continue

        # Map global indices to local indices in each patch
        idx1_list = patch_indices[k1].cpu().tolist()
        idx2_list = patch_indices[k2].cpu().tolist()

        local1 = [idx1_list.index(g) for g in overlap_global if g in idx1_list]
        local2 = [idx2_list.index(g) for g in overlap_global if g in idx2_list]

        if len(local1) < 5 or len(local2) < 5:
            continue

        # Coordinates for the overlap in each patch's own frame
        V1_overlap = patch_V[k1][local1]  # (n_overlap, D)
        V2_overlap = patch_V[k2][local2]  # (n_overlap, D)

        # Pairwise distance matrices on the overlap
        D1 = torch.cdist(V1_overlap, V1_overlap)
        D2 = torch.cdist(V2_overlap, V2_overlap)

        # Compare in log space for scale stability
        D1_log = (D1 + 1e-8).log()
        D2_log = (D2 + 1e-8).log()

        # Mean absolute difference of log pairwise distances
        disagreement = (D1_log - D2_log).abs().mean().item()

        pair_disagreements[(k1, k2)] = disagreement
        disagreement_values.append(disagreement)

    if verbose and disagreement_values:
        print(f"\n[OVERLAP-CONSIST] Checked {len(disagreement_values)} patch pairs")
        print(f"[OVERLAP-CONSIST] Disagreement: min={min(disagreement_values):.4f}, "
              f"median={np.median(disagreement_values):.4f}, max={max(disagreement_values):.4f}")

    diagnostics = {
        'n_pairs_checked': len(disagreement_values),
        'disagreement_min': min(disagreement_values) if disagreement_values else 0,
        'disagreement_median': float(np.median(disagreement_values)) if disagreement_values else 0,
        'disagreement_max': max(disagreement_values) if disagreement_values else 0,
    }

    # Per-patch consistency scores: patches that agree with their neighbors
    # score higher. A patch with no overlaps inherits the median disagreement.
    K = len(patch_V)
    patch_disagreements = {k: [] for k in range(K)}

    for (k1, k2), disagreement in pair_disagreements.items():
        patch_disagreements[k1].append(disagreement)
        patch_disagreements[k2].append(disagreement)

    patch_mean_disagreement = {}
    for k in range(K):
        if patch_disagreements[k]:
            patch_mean_disagreement[k] = np.mean(patch_disagreements[k])
        else:
            patch_mean_disagreement[k] = diagnostics['disagreement_median']

    # Lower disagreement -> higher consistency. Scale by the median disagreement
    # so the median patch lands near exp(-1) ~ 0.37.
    scale = diagnostics['disagreement_median'] + 1e-8
    patch_consistency = {}
    for k in range(K):
        patch_consistency[k] = np.exp(-patch_mean_disagreement[k] / scale)

    if verbose:
        consistency_values = list(patch_consistency.values())
        print(f"[OVERLAP-CONSIST] Per-patch consistency scores:")
        print(f"[OVERLAP-CONSIST]   min={min(consistency_values):.3f}, "
              f"median={np.median(consistency_values):.3f}, max={max(consistency_values):.3f}")

    return {
        'pair_disagreements': pair_disagreements,
        'patch_consistency': patch_consistency,
        'diagnostics': diagnostics,
    }


def aggregate_distance_measurements_v2(
    patch_measurements: List[Dict],
    patch_indices: List[torch.Tensor],
    N: int,
    M_min: int = 2,
    tau_spread: float = 0.30,
    spread_alpha: float = 10.0,
    patch_consistency: Optional[Dict[int, float]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fuse per-patch distance measurements into a global sparse distance graph.

    For each global edge (i, j), collect every patch measurement, take a
    reliability-weighted median as the consensus distance, measure the spread,
    and keep the edge only if it was measured at least M_min times with relative
    spread <= tau_spread. Surviving edges carry a trust weight.

    Args:
        patch_measurements: per-patch dicts with 'edges' (local (u, v) pairs),
                            'distances', and 'weights'.
        patch_indices:      per-patch global cell indices, tensor of length n_k.
        N:                  total number of cells.
        M_min:              minimum measurements to keep an edge.
        tau_spread:         maximum relative spread to keep an edge.
        spread_alpha:       decay of the trust weight with spread.
        patch_consistency:  optional {patch_id: score}; measurements from higher-
                            consistency patches are weighted up.

    Returns:
        dict with global edges, consensus distances, trust weights, per-edge
        spreads/counts, and diagnostics.
    """
    if verbose:
        print(f"\n[AGGREGATE] Aggregating measurements from {len(patch_measurements)} patches")
        if patch_consistency:
            print(f"[AGGREGATE] Using patch consistency weighting")

    # Collect measurements per global edge
    edge_measurements = defaultdict(list)  # (i, j) -> [(distance, weight, patch_id), ...]

    for k, (patch_meas, patch_idx) in enumerate(zip(patch_measurements, patch_indices)):
        patch_idx_list = patch_idx.cpu().tolist()

        # Patch consistency weight (default 1.0 if not provided)
        patch_weight = patch_consistency.get(k, 1.0) if patch_consistency else 1.0

        for (u, v), d, w in zip(patch_meas['edges'], patch_meas['distances'], patch_meas['weights']):
            # Map local indices to global
            i_global = patch_idx_list[u]
            j_global = patch_idx_list[v]

            # Canonical ordering
            edge = (min(i_global, j_global), max(i_global, j_global))
            w_adjusted = w * patch_weight
            edge_measurements[edge].append((d, w_adjusted, k))

    if verbose:
        print(f"[AGGREGATE] Total unique edges: {len(edge_measurements)}")

    # Consensus per edge
    global_edges = []
    consensus_distances = []
    consensus_weights = []
    edge_spreads = []
    edge_counts = []

    for (i, j), measurements in edge_measurements.items():
        count = len(measurements)

        # Filter by minimum count
        if count < M_min:
            continue

        distances = [m[0] for m in measurements]
        weights = [m[1] for m in measurements]

        # Weighted median
        sorted_pairs = sorted(zip(distances, weights))
        cumsum = 0
        total_weight = sum(weights)
        d_median = sorted_pairs[-1][0]  # Default to last
        for d, w in sorted_pairs:
            cumsum += w
            if cumsum >= total_weight / 2:
                d_median = d
                break

        # Spread (relative IQR, or relative std for small samples)
        if len(distances) >= 4:
            q10 = np.percentile(distances, 10)
            q90 = np.percentile(distances, 90)
            spread = (q90 - q10) / (d_median + 1e-8)
        else:
            spread = np.std(distances) / (d_median + 1e-8) if d_median > 0 else 0

        # Filter by spread
        if spread > tau_spread:
            continue

        # Trust weight: count * exp(-alpha * spread^2)
        trust_weight = count * np.exp(-spread_alpha * spread**2)

        global_edges.append((i, j))
        consensus_distances.append(d_median)
        consensus_weights.append(trust_weight)
        edge_spreads.append(spread)
        edge_counts.append(count)

    if verbose:
        print(f"[AGGREGATE] Edges after filtering (M_min={M_min}, tau_spread={tau_spread}): {len(global_edges)}")
        if consensus_distances:
            print(f"[AGGREGATE] Consensus distances: min={min(consensus_distances):.4f}, median={np.median(consensus_distances):.4f}, max={max(consensus_distances):.4f}")
            print(f"[AGGREGATE] Spreads: min={min(edge_spreads):.3f}, median={np.median(edge_spreads):.3f}, max={max(edge_spreads):.3f}")
            print(f"[AGGREGATE] Counts: min={min(edge_counts)}, median={np.median(edge_counts):.1f}, max={max(edge_counts)}")

    # Rescue isolated nodes using single-measurement fallback edges so the
    # global solve stays connected.
    connected_nodes = set()
    for (i, j) in global_edges:
        connected_nodes.add(i)
        connected_nodes.add(j)

    isolated_nodes = [i for i in range(N) if i not in connected_nodes]

    if len(isolated_nodes) > 0:
        if verbose:
            print(f"[AGGREGATE] {len(isolated_nodes)} isolated nodes, adding rescue edges...")

        rescue_edges_added = 0
        for node in isolated_nodes:
            # All edges touching this node, from the unfiltered measurements
            node_edges = []
            for (i, j), measurements in edge_measurements.items():
                if i == node or j == node:
                    distances_ij = [m[0] for m in measurements]
                    d_median = np.median(distances_ij)
                    # Reduced weight for rescue edges
                    w = len(measurements) * 0.5
                    node_edges.append(((i, j), d_median, w))

            # Prefer edges with more measurements
            node_edges.sort(key=lambda x: -x[2])

            k_rescue = min(5, len(node_edges))
            for idx in range(k_rescue):
                edge, d, w = node_edges[idx]
                if edge not in global_edges:
                    global_edges.append(edge)
                    consensus_distances.append(d)
                    consensus_weights.append(w)
                    edge_spreads.append(0.5)  # Mark as uncertain
                    edge_counts.append(1)
                    rescue_edges_added += 1

        if verbose:
            print(f"[AGGREGATE] Added {rescue_edges_added} rescue edges")
            connected_nodes = set()
            for (i, j) in global_edges:
                connected_nodes.add(i)
                connected_nodes.add(j)
            still_isolated = [i for i in range(N) if i not in connected_nodes]
            print(f"[AGGREGATE] {len(still_isolated)} nodes still isolated after rescue")

    diagnostics = {
        'n_global_edges': len(global_edges),
        'n_total_measurements': sum(len(m) for m in edge_measurements.values()),
        'edges_filtered_by_count': sum(1 for m in edge_measurements.values() if len(m) < M_min),
        'edges_filtered_by_spread': sum(1 for (i, j), m in edge_measurements.items()
                                        if len(m) >= M_min and
                                        (np.percentile([x[0] for x in m], 90) - np.percentile([x[0] for x in m], 10)) /
                                        (np.median([x[0] for x in m]) + 1e-8) > tau_spread),
        'spread_median': float(np.median(edge_spreads)) if edge_spreads else 0,
        'count_median': float(np.median(edge_counts)) if edge_counts else 0,
    }

    return {
        'edges': global_edges,
        'distances': consensus_distances,
        'weights': consensus_weights,
        'spreads': edge_spreads,
        'counts': edge_counts,
        'diagnostics': diagnostics,
    }
