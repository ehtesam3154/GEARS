"""Single-cell locality graph + overlapping-patch sampler.

The distance-first inference pipeline reconstructs 2D coordinates for a bag of
dissociated single cells from their frozen Stage-A embeddings ``Z`` (B, h). It
never sees the whole cell population at once; instead it decomposes the cells
into many small, spatially-coherent, mutually-overlapping patches, solves each
patch geometry independently, and stitches the patches back together.

This module holds the first two steps of that pipeline:

    build_locality_graph_v2       -- Z-space embeddings -> a filtered locality
                                     graph (mutual-kNN + Jaccard overlap
                                     filtering + self-tuned edge weights).
    sample_patches_random_walk_v2 -- locality graph -> a set of overlapping
                                     patches (weighted random walks with a
                                     sliding window, then a connected cover
                                     that guarantees every cell is placed in at
                                     least one patch).

Both operate purely on the embedding geometry; no spatial coordinates are used.
"""

from typing import Any, Dict, List

import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_locality_graph_v2(
    Z: torch.Tensor,
    k_Z: int = 40,
    k_sigma: int = 10,
    tau_jaccard: float = 0.10,
    min_shared: int = 5,
    device: str = 'cuda',
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build a locality graph on the single cells from Z-space embeddings.

    Unlike raw kNN, this filters for mutual neighbors and shared-neighborhood
    to suppress "teleport" edges (same cell type but spatially far).

    Returns:
        dict with:
            'adj_list': Dict[int, List[int]] - adjacency list
            'edge_weights': Dict[Tuple[int,int], float] - edge weights
            'local_scales': Tensor (N,) - sigma_i for each node
            'diagnostics': dict - mutuality rate, etc.
    """
    N, h = Z.shape
    Z_np = Z.cpu().numpy() if Z.is_cuda else Z.numpy()

    if verbose:
        print(f"\n[LOCALITY-GRAPH] Building locality graph: N={N}, k_Z={k_Z}")

    # Base directed kNN graph in Z-space
    nbrs = NearestNeighbors(n_neighbors=min(k_Z + 1, N), algorithm='auto').fit(Z_np)
    distances, indices = nbrs.kneighbors(Z_np)

    # indices[:, 0] is self, so neighbors are indices[:, 1:]
    knn_sets = [set(indices[i, 1:k_Z+1].tolist()) for i in range(N)]

    # Symmetrize to mutual-kNN
    mutual_edges = set()
    for i in range(N):
        for j in knn_sets[i]:
            if i in knn_sets[j]:  # Mutual
                edge = (min(i, j), max(i, j))
                mutual_edges.add(edge)

    mutuality_rate = len(mutual_edges) / (N * k_Z / 2 + 1e-8)
    if verbose:
        print(f"[LOCALITY-GRAPH] Mutual-kNN edges: {len(mutual_edges)} (mutuality rate: {mutuality_rate:.3f})")

    # Shared-neighborhood / Jaccard filter
    filtered_edges = []
    jaccard_scores = []
    shared_counts = []

    for (i, j) in mutual_edges:
        intersection = len(knn_sets[i] & knn_sets[j])
        union = len(knn_sets[i] | knn_sets[j])
        jaccard = intersection / (union + 1e-8)

        # Keep edge if Jaccard >= threshold OR shared neighbors >= min
        if jaccard >= tau_jaccard or intersection >= min_shared:
            filtered_edges.append((i, j))
            jaccard_scores.append(jaccard)
            shared_counts.append(intersection)

    if verbose:
        print(f"[LOCALITY-GRAPH] After Jaccard filter (tau={tau_jaccard}, min_shared={min_shared}): {len(filtered_edges)} edges")
        if jaccard_scores:
            print(f"[LOCALITY-GRAPH] Jaccard stats: min={min(jaccard_scores):.3f}, median={np.median(jaccard_scores):.3f}, max={max(jaccard_scores):.3f}")

    # Self-tuned edge weights.
    # Local scale: distance to k_sigma-th neighbor
    local_scales = torch.tensor(distances[:, min(k_sigma, distances.shape[1]-1)],
                                 dtype=torch.float32, device=device)
    local_scales = local_scales.clamp(min=1e-8)

    # Build adjacency list and edge weights
    adj_list = {i: [] for i in range(N)}
    edge_weights = {}

    Z_t = Z.to(device)
    for (i, j) in filtered_edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

        # Weight: exp(-d^2 / (sigma_i * sigma_j))
        d_ij = (Z_t[i] - Z_t[j]).norm().item()
        sigma_i = local_scales[i].item()
        sigma_j = local_scales[j].item()
        w_ij = np.exp(-d_ij**2 / (sigma_i * sigma_j + 1e-8))

        edge_weights[(i, j)] = w_ij
        edge_weights[(j, i)] = w_ij

    # Compute degree distribution for diagnostics
    degrees = [len(adj_list[i]) for i in range(N)]
    isolated_nodes = sum(1 for d in degrees if d == 0)

    if verbose:
        print(f"[LOCALITY-GRAPH] Degree stats: min={min(degrees)}, median={np.median(degrees):.1f}, max={max(degrees)}")
        print(f"[LOCALITY-GRAPH] Isolated nodes: {isolated_nodes} ({100*isolated_nodes/N:.1f}%)")

    diagnostics = {
        'mutuality_rate': mutuality_rate,
        'n_edges_mutual': len(mutual_edges),
        'n_edges_filtered': len(filtered_edges),
        'jaccard_median': float(np.median(jaccard_scores)) if jaccard_scores else 0.0,
        'degree_median': float(np.median(degrees)),
        'isolated_nodes': isolated_nodes,
    }

    return {
        'adj_list': adj_list,
        'edge_weights': edge_weights,
        'local_scales': local_scales,
        'knn_sets': knn_sets,
        'filtered_edges': filtered_edges,
        'diagnostics': diagnostics,
    }


def sample_patches_random_walk_v2(
    graph: Dict[str, Any],
    N: int,
    patch_size: int = 256,
    overlap_frac: float = 0.5,
    coverage_per_cell: float = 4.0,
    min_overlap: int = 30,
    max_patches: int = 500,
    device: str = 'cuda',
    verbose: bool = False,
) -> Dict[str, Any]:
    """Sample overlapping patches as connected subgraphs via random walk.

    Uses a sliding window over a weighted random walk to guarantee structural
    overlap between consecutive patches.

    Returns:
        dict with:
            'patch_indices': List[Tensor] - cell indices per patch
            'patch_overlaps': Dict[(k1,k2), Set[int]] - overlap sets
            'coverage_counts': Tensor (N,) - how many patches each cell appears in
            'diagnostics': dict
    """
    adj_list = graph['adj_list']
    edge_weights = graph['edge_weights']

    stride = int(patch_size * (1 - overlap_frac))
    target_patches = int(np.ceil(N * coverage_per_cell / patch_size))
    target_patches = min(target_patches, max_patches)

    if verbose:
        print(f"\n[PATCH-SAMPLE] Sampling patches: size={patch_size}, overlap={overlap_frac:.0%}, stride={stride}")
        print(f"[PATCH-SAMPLE] Target patches: {target_patches} (coverage={coverage_per_cell})")

    # Find nodes with edges (non-isolated)
    active_nodes = [i for i in range(N) if len(adj_list[i]) > 0]
    if len(active_nodes) < patch_size:
        print(f"[PATCH-SAMPLE] WARNING: Only {len(active_nodes)} active nodes, need {patch_size}")
        patch_size = max(32, len(active_nodes) // 2)

    # Random walk with weighted transitions
    def weighted_random_walk(start_node: int, length: int) -> List[int]:
        walk = [start_node]
        current = start_node
        for _ in range(length - 1):
            neighbors = adj_list[current]
            if not neighbors:
                # Stuck at isolated node - jump to random active node
                current = random.choice(active_nodes)
            else:
                # Weighted choice
                weights = [edge_weights.get((current, n), 0.01) for n in neighbors]
                total = sum(weights)
                weights = [w/total for w in weights]
                current = random.choices(neighbors, weights=weights, k=1)[0]
            walk.append(current)
        return walk

    # Generate long random walk
    walk_length = stride * (target_patches + 5) + patch_size
    start_node = random.choice(active_nodes)
    walk = weighted_random_walk(start_node, walk_length)

    if verbose:
        unique_in_walk = len(set(walk))
        print(f"[PATCH-SAMPLE] Random walk length: {len(walk)}, unique nodes visited: {unique_in_walk}")

    # Extract patches via sliding window (with deduplication)
    patch_indices_list = []
    coverage_counts = torch.zeros(N, dtype=torch.long, device=device)

    pos = 0
    while pos + patch_size <= len(walk) and len(patch_indices_list) < target_patches:
        window = walk[pos:pos + patch_size]
        unique_nodes = list(dict.fromkeys(window))  # Preserve order, remove dups

        # If too few unique nodes, extend by BFS
        if len(unique_nodes) < patch_size * 0.8:
            unique_nodes = _extend_patch_bfs(unique_nodes, adj_list, patch_size, N)

        # Truncate to patch_size
        unique_nodes = unique_nodes[:patch_size]

        patch_tensor = torch.tensor(unique_nodes, dtype=torch.long, device=device)
        patch_indices_list.append(patch_tensor)

        for idx in unique_nodes:
            coverage_counts[idx] += 1

        pos += stride

    # If not enough patches or coverage, sample more starting from under-covered nodes
    under_covered = (coverage_counts < coverage_per_cell * 0.5).nonzero(as_tuple=True)[0]
    attempts = 0
    while len(patch_indices_list) < target_patches and len(under_covered) > 0 and attempts < 100:
        start_idx = under_covered[random.randint(0, len(under_covered)-1)].item()
        if start_idx in active_nodes or len(adj_list[start_idx]) > 0:
            walk = weighted_random_walk(start_idx, patch_size * 2)
            unique_nodes = list(dict.fromkeys(walk))[:patch_size]

            if len(unique_nodes) >= patch_size * 0.7:
                unique_nodes = _extend_patch_bfs(unique_nodes, adj_list, patch_size, N)
                unique_nodes = unique_nodes[:patch_size]

                patch_tensor = torch.tensor(unique_nodes, dtype=torch.long, device=device)
                patch_indices_list.append(patch_tensor)

                for idx in unique_nodes:
                    coverage_counts[idx] += 1

        under_covered = (coverage_counts < coverage_per_cell * 0.5).nonzero(as_tuple=True)[0]
        attempts += 1

    K = len(patch_indices_list)

    # Compute patch overlaps
    patch_sets = [set(p.cpu().tolist()) for p in patch_indices_list]
    patch_overlaps = {}
    overlap_sizes = []

    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            overlap = patch_sets[k1] & patch_sets[k2]
            if len(overlap) >= min_overlap:
                patch_overlaps[(k1, k2)] = overlap
                overlap_sizes.append(len(overlap))

    # Check patch graph connectivity
    patch_adj = {k: set() for k in range(K)}
    for (k1, k2) in patch_overlaps:
        patch_adj[k1].add(k2)
        patch_adj[k2].add(k1)

    # BFS to find connected components
    visited = set()
    components = []
    for start in range(K):
        if start not in visited:
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    queue.extend(patch_adj[node] - visited)
            components.append(component)

    is_connected = len(components) == 1

    # Ensure full coverage: attach any uncovered cell to the patch that shares
    # the most of its neighbors (isolated cells go to the largest patch).
    uncovered_cells = (coverage_counts == 0).nonzero(as_tuple=True)[0].cpu().tolist()
    if len(uncovered_cells) > 0:
        if verbose:
            print(f"[PATCH-SAMPLE] {len(uncovered_cells)} uncovered cells, adding to nearest patches...")

        for cell_idx in uncovered_cells:
            cell_neighbors = set(adj_list[cell_idx])

            if len(cell_neighbors) > 0:
                # Find patch with most overlap with cell's neighbors
                best_patch_idx = 0
                best_overlap = 0
                for k, patch_set in enumerate(patch_sets):
                    overlap = len(cell_neighbors & patch_set)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_patch_idx = k

                # Add cell to the best patch
                patch_sets[best_patch_idx].add(cell_idx)
                patch_indices_list[best_patch_idx] = torch.tensor(
                    list(patch_sets[best_patch_idx]), dtype=torch.long, device=device
                )
                coverage_counts[cell_idx] = 1
            else:
                # Isolated cell (no neighbors) - add to largest patch
                largest_patch_idx = max(range(K), key=lambda k: len(patch_sets[k]))
                patch_sets[largest_patch_idx].add(cell_idx)
                patch_indices_list[largest_patch_idx] = torch.tensor(
                    list(patch_sets[largest_patch_idx]), dtype=torch.long, device=device
                )
                coverage_counts[cell_idx] = 1

        if verbose:
            print(f"[PATCH-SAMPLE] All cells now covered")

        # Recompute overlaps after adding cells
        patch_overlaps = {}
        overlap_sizes = []
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                overlap = patch_sets[k1] & patch_sets[k2]
                if len(overlap) >= min_overlap:
                    patch_overlaps[(k1, k2)] = overlap
                    overlap_sizes.append(len(overlap))

    if verbose:
        print(f"[PATCH-SAMPLE] Generated {K} patches")
        print(f"[PATCH-SAMPLE] Coverage: min={coverage_counts.min().item()}, median={coverage_counts.float().median().item():.1f}, max={coverage_counts.max().item()}")
        uncovered = (coverage_counts == 0).sum().item()
        print(f"[PATCH-SAMPLE] Uncovered cells: {uncovered} ({100*uncovered/N:.1f}%)")
        print(f"[PATCH-SAMPLE] Overlapping patch pairs: {len(patch_overlaps)}")
        if overlap_sizes:
            print(f"[PATCH-SAMPLE] Overlap sizes: min={min(overlap_sizes)}, median={np.median(overlap_sizes):.0f}, max={max(overlap_sizes)}")
        print(f"[PATCH-SAMPLE] Patch graph connected: {is_connected} ({len(components)} components)")

    diagnostics = {
        'n_patches': K,
        'coverage_min': coverage_counts.min().item(),
        'coverage_median': coverage_counts.float().median().item(),
        'uncovered_cells': (coverage_counts == 0).sum().item(),
        'n_overlap_pairs': len(patch_overlaps),
        'overlap_median': float(np.median(overlap_sizes)) if overlap_sizes else 0,
        'is_connected': is_connected,
        'n_components': len(components),
    }

    return {
        'patch_indices': patch_indices_list,
        'patch_sets': patch_sets,
        'patch_overlaps': patch_overlaps,
        'coverage_counts': coverage_counts,
        'components': components,
        'diagnostics': diagnostics,
    }


def _extend_patch_bfs(nodes: List[int], adj_list: Dict[int, List[int]], target_size: int, N: int) -> List[int]:
    """Extend a patch by BFS until reaching target size."""
    node_set = set(nodes)
    frontier = list(nodes)
    random.shuffle(frontier)

    while len(node_set) < target_size and frontier:
        current = frontier.pop(0)
        for neighbor in adj_list[current]:
            if neighbor not in node_set:
                node_set.add(neighbor)
                frontier.append(neighbor)
                if len(node_set) >= target_size:
                    break

    return list(node_set)
