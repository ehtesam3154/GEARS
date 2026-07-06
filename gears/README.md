# GEARS

Geometry-first reconstruction of single-cell spatial organization. GEARS learns
an intrinsic 2D spatial geometry for dissociated scRNA-seq cells using spatial
transcriptomics (ST) as pose-invariant geometric supervision — without cell-type
labels, histology, or cell-to-spot assignment.

## Pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| A. Encoder | `gears/encoder.py`, `gears/train_encoder.py` | Domain-invariant expression encoder aligning ST spots and dissociated cells into a shared embedding (VICReg + gradient-reversal domain adversary + CORAL/MMD/local-alignment). Frozen afterwards. |
| B. Targets | `gears/data/` | Pose-free per-slide geometric targets and spatially-coherent training mini-sets (centered-Gram factor `V_target`, spatial kNN). |
| C. Geometry | `gears/models/`, `gears/training/` | A permutation-equivariant Set-Transformer generator produces a coarse geometry `V_base`; an EDM-preconditioned diffusion refiner denoises residuals under Gram / neighborhood supervision. |
| Inference | `gears/inference/` | Encode all cells → mutual-kNN locality graph → overlapping random-walk patches → per-patch geometry via residual diffusion → reliability-weighted distance stitching → Landmark-Isomap + weighted-Huber distance-geometry solve → 2D coordinates + dense distance matrix. |
| Eval | `gears/eval/` | Global (Spearman/Pearson/Stress-1), local (Edge ROC-AUC, bAP, Shell-F1), neighborhood (Trust/Cont@k), and distribution (SWD, W1) metrics. |

## Usage

```python
from gears import GEARS, EncoderConfig
from gears.training import StageCConfig
from gears.inference import InferConfig

model = GEARS(n_genes=G, D_latent=32, device="cuda")

# Stage A — encoder (ST spots + dissociated cells, shared genes; log1p)
model.train_encoder(st_expr, sc_expr, st_slide_ids=slide_ids,
                    config=EncoderConfig(n_epochs=1000))

# Stage B — pose-free targets (coords must be pose-normalized; see scripts/run_gears.py)
model.precompute_targets({0: st_coords})           # slide_id -> (n, 2)

# Stage C — geometry generator + diffusion refiner
model.train_geometry({0: st_expr},                 # slide_id -> (n, G)
                    config=StageCConfig(n_epochs=200, use_residual_diffusion=True))

# Inference — expression -> 2D. mode="auto" runs one-shot when the set fits,
# otherwise patchwise; the result carries an is_outlier mask.
out = model.reconstruct(sc_expr)                    # or mode="single" / "patchwise"
coords, distances = out["coords"], out["distances"]

from gears.eval import plot_reconstruction, score_reconstruction
score_reconstruction(coords.cpu().numpy(), gt_coords, out["is_outlier"])
plot_reconstruction(coords.cpu().numpy(), gt_coords, is_outlier=out["is_outlier"],
                    save_path="reconstruction.png")
```

A full runnable template (data loading, coordinate normalization, evaluation) is in
[`scripts/run_gears.py`](../scripts/run_gears.py).

## Environment

Python 3.10, PyTorch (tested on 2.0–2.2, CUDA 11.8), plus numpy / scipy / scikit-learn.
Stage A/C train on a single GPU; inference runs patchwise and scales to large cohorts.
