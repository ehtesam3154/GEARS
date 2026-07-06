"""Simulate a mouse-brain ST slide from the spatially-resolved SC cells.

The mouse-brain "ST" is not an independent assay — it is a pseudo-Visium slide
built from the dissociated SC cohort, which carries per-cell spatial coordinates
(`metadata.csv` x_global / y_global). We lay a regular grid over the tissue and
sum the SC cells that fall inside each grid bin into one spot. Different grid
resolutions / offsets give different slides; the dataset ships the exact ones we
trained on (`simu_st3_*`), and this script (cleaned from `GenerateData.ipynb`)
reproduces the recipe so you can regenerate them or make your own.

    # ST1- and ST2-style slides (grid resolution + offset decorrelate them)
    python scripts/simulate_mousebrain_st.py --grid 30            --out_prefix simu_st1  # ~570 spots
    python scripts/simulate_mousebrain_st.py --grid 40 --offset 0.3 --out_prefix simu_st2  # ~950 spots

    # a finer slide
    python scripts/simulate_mousebrain_st.py --grid 60 --out_prefix simu_st_fine
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def simulate_st(sc_counts, coords, celltype_onehot, grid, offset=0.0, bin_frac=0.70, min_cells=2):
    """Aggregate SC cells into grid spots.

    Args:
        sc_counts:       (n_cells, n_genes) SC expression.
        coords:          (n_cells, 2) SC spatial coordinates.
        celltype_onehot: (n_cells, n_types) one-hot cell types (for spot proportions).
        grid:            grid resolution (grid x grid points).
        offset:          shift the grid origin by `offset` bins (to decorrelate slides).
        bin_frac:        fraction of the grid spacing each spot integrates over.
        min_cells:       keep only spots covering at least this many cells.

    Returns:
        st_counts (n_spots, n_genes), st_coords (n_spots, 2), st_types (n_spots, n_types).
    """
    (min_x, min_y), (max_x, max_y) = coords.min(0), coords.max(0)
    span_x = (max_x - min_x) / (grid - 1)
    span_y = (max_y - min_y) / (grid - 1)
    xs = np.linspace(min_x + offset * span_x, max_x, grid)
    ys = np.linspace(min_y + offset * span_y, max_y, grid)
    half_x, half_y = bin_frac / 2 * span_x, bin_frac / 2 * span_y

    counts, cs, types = [], [], []
    for gx in tqdm(xs, desc=f"grid {grid}x{grid}"):
        in_x = (coords[:, 0] > gx - half_x) & (coords[:, 0] <= gx + half_x)
        for gy in ys:
            m = in_x & (coords[:, 1] > gy - half_y) & (coords[:, 1] <= gy + half_y)
            if m.sum() >= min_cells:
                counts.append(sc_counts[m].sum(0))
                types.append(celltype_onehot[m].sum(0))
                cs.append((gx, gy))
    return np.asarray(counts), np.asarray(cs), np.asarray(types)


def main():
    ap = argparse.ArgumentParser(description="Simulate a mouse-brain ST slide from SC")
    ap.add_argument("--data_dir", default="data/mousedata_2020/E1z2")
    ap.add_argument("--grid", type=int, default=60, help="grid resolution (grid x grid)")
    ap.add_argument("--offset", type=float, default=0.0, help="grid-origin offset in bins")
    ap.add_argument("--bin_frac", type=float, default=0.70)
    ap.add_argument("--min_cells", type=int, default=2)
    ap.add_argument("--out_prefix", default="simu_st3", help="output file prefix, e.g. simu_st3")
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"), index_col=0)
    sc = pd.read_csv(os.path.join(args.data_dir, "simu_sc_counts.csv"), index_col=0).T  # cells x genes
    coords = meta[["x_global", "y_global"]].values.astype(np.float64)
    onehot = pd.get_dummies(meta["celltype_mapped_refined"])
    print(f"SC: {sc.shape[0]} cells x {sc.shape[1]} genes")

    counts, cs, types = simulate_st(
        sc.values.astype(np.float64), coords, onehot.values.astype(np.float64),
        grid=args.grid, offset=args.offset, bin_frac=args.bin_frac, min_cells=args.min_cells)
    print(f"-> {counts.shape[0]} spots (grid {args.grid}, min_cells {args.min_cells})")

    spot_ids = [f"spot_{i}" for i in range(counts.shape[0])]
    # counts saved gene x spot (matches simu_st*_counts_et.csv layout)
    pd.DataFrame(counts.T, index=sc.columns, columns=spot_ids).to_csv(
        os.path.join(args.data_dir, f"{args.out_prefix}_counts_et.csv"))
    pd.DataFrame({"coord_x": cs[:, 0], "coord_y": cs[:, 1]}, index=spot_ids).to_csv(
        os.path.join(args.data_dir, f"{args.out_prefix}_metadata_et.csv"))
    pd.DataFrame(types, index=spot_ids, columns=onehot.columns).to_csv(
        os.path.join(args.data_dir, f"{args.out_prefix}_celltype_et.csv"))
    print(f"saved {args.out_prefix}_{{counts,metadata,celltype}}_et.csv -> {args.data_dir}")


if __name__ == "__main__":
    main()
