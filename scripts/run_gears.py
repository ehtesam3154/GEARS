"""End-to-end GEARS pipeline: encoder -> targets -> geometry -> reconstruction.

This script shows the full API. Plug your own data loading into `load_data()`:
it must return spatial-transcriptomics (ST) coordinates + expression per slide,
and dissociated single-cell (SC) expression, all over a shared gene set.

    python scripts/run_gears.py
"""

import argparse

import numpy as np
import torch

from gears import GEARS, EncoderConfig
from gears.training import StageCConfig
from gears.inference import InferConfig
from gears.eval import evaluate_reconstruction


def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    """Center and isotropically rescale coordinates (unit median nearest-neighbor
    distance). GEARS consumes pose-normalized coordinates."""
    coords = coords - coords.mean(0, keepdim=True)
    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values.median().clamp(min=1e-8)
    return coords / nn


def load_data():
    """Return (st_slides, sc_expr[, sc_gt_coords]).

    st_slides: dict slide_id -> (coords (n,2), expr (n,G) log1p) with a SHARED gene set.
    sc_expr:   (N, G) log1p dissociated single-cell expression on the same genes.
    Optionally return ground-truth SC coords (N,2) for evaluation.

    Replace this stub with your loader (e.g. read AnnData, intersect var_names,
    log1p-normalize, and build the dicts).
    """
    raise NotImplementedError("Wire your ST/SC data here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="gears_out")
    ap.add_argument("--stageA_epochs", type=int, default=1000)
    ap.add_argument("--stageC_epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    st_slides, sc_expr, *rest = load_data()
    sc_gt = rest[0] if rest else None
    n_genes = sc_expr.shape[1]

    st_expr_dict = {sid: expr for sid, (coords, expr) in st_slides.items()}
    st_coords_dict = {sid: normalize_coords(coords) for sid, (coords, expr) in st_slides.items()}

    # Stage A trains on the union of all ST spots + SC cells.
    st_expr_all = torch.cat([e for e in st_expr_dict.values()], 0)
    st_slide_ids = torch.cat([torch.full((e.shape[0],), sid) for sid, e in st_expr_dict.items()])

    model = GEARS(n_genes=n_genes, device=args.device)

    model.train_encoder(
        st_expr_all, sc_expr, st_slide_ids=st_slide_ids,
        config=EncoderConfig(n_epochs=args.stageA_epochs, seed=args.seed),
        out_dir=args.out_dir)

    model.precompute_targets(st_coords_dict)

    model.train_geometry(
        st_expr_dict,
        config=StageCConfig(n_epochs=args.stageC_epochs, use_residual_diffusion=True,
                            seed=args.seed),
        out_dir=args.out_dir)

    model.save(f"{args.out_dir}/gears_model.pt")

    result = model.reconstruct(sc_expr, config=InferConfig(use_residual_diffusion=True))
    coords, distances = result["coords"], result["distances"]
    torch.save({"coords": coords, "distances": distances}, f"{args.out_dir}/reconstruction.pt")
    print(f"Reconstructed {coords.shape[0]} cells -> 2D coordinates.")

    if sc_gt is not None:
        from scipy.spatial.distance import pdist, squareform
        D_gt = squareform(pdist(sc_gt.cpu().numpy()))
        metrics = evaluate_reconstruction(
            D_gt, distances.cpu().numpy(), sc_gt.cpu().numpy(), coords.cpu().numpy(),
            verbose=False)
        print("Global: spearman={distance_spearman:.4f} pearson={distance_pearson:.4f} "
              "stress={stress:.4f}".format(**metrics))
        print("Local:  edge_auc={edge_roc_auc:.4f} bAP={balanced_AP:.4f} "
              "trust@20={trustworthiness@20:.4f}".format(**metrics))


if __name__ == "__main__":
    main()
