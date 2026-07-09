# GEARS

**Official code for ["Geometry-First Generative Spatial Single-Cell Reconstruction"](https://arxiv.org/abs/2605.28200) (KDD 2026).**
&nbsp; 📄 [Paper (arXiv)](https://arxiv.org/abs/2605.28200)

GEARS learns an intrinsic 2D spatial geometry for dissociated scRNA-seq cells,
using spatial transcriptomics (ST) as pose-invariant geometric supervision — no
cell-type labels, no histology, no cell-to-spot assignment. A domain-invariant
encoder aligns ST spots and dissociated cells into a shared embedding; a
Set-Transformer generator plus an EDM-preconditioned diffusion refiner produce
per-cell geometry; and a patchwise inference pipeline stitches local geometries
into a global 2D reconstruction.

The reusable package lives in [`gears/`](gears/) — see [`gears/README.md`](gears/README.md)
for the stage-by-stage API. This top-level README covers **installation, data,
and reproducing the two reference runs**.

## Repository layout

```
gears/                     the GEARS package (encoder, targets, models, training, inference, eval)
scripts/
  train_mousebrain_full.py  reproduce the mouse-brain checkpoint (train ST3 → reconstruct ~10k SC cells)
  train_hscc.py             reproduce the hSCC (cSCC patient P2) checkpoint (train ST1+ST2 → reconstruct ST3)
```

## Installation

Python 3.10, PyTorch 2.1 (CUDA 11.8). The multi-GPU scripts use Lightning Fabric.

```bash
conda create -n gears python=3.10 -y
conda activate gears
pip install torch==2.1.0                     # match your CUDA build
pip install scanpy anndata numpy scipy scikit-learn pandas
pip install lightning-fabric==2.1.4          # only needed for the DDP scripts
```

Stage A / Stage C train on GPU; inference runs patchwise on a single GPU and
scales to large cohorts. The DDP scripts were validated on 4× RTX A4500 (20 GB).

## Data

The two reference **datasets** are hosted externally — **[download (MEGA)](https://mega.nz/folder/DjhVVIDb#PdjpmpOZsgRiKm3TXLNhFg)** —
and each script points at its folder with `--data_dir`. Train your own models
from them. The scripts expect:

**hSCC (cSCC patient P2)** — three ST replicate slides as `.h5ad` (coords in `.obsm['spatial']`):

```
<data_dir>/stP2.h5ad        ST1  ┐ training slides
           stP2rep2.h5ad     ST2  ┘
           stP2rep3.h5ad     ST3  → held out for inference (also the Stage-A "SC" domain)
```

**Mouse brain (simulated)** — one ST slide + a dissociated SC cohort, as gene×cell CSVs:

```
<data_dir>/simu_st3_counts_et.csv  + simu_st3_metadata_et.csv   ST3 spots  → training (coord_x, coord_y)
           simu_sc_counts.csv                                   ~10k dissociated SC cells → reconstruction target
           metadata.csv                                         SC ground-truth coords (x_global, y_global)
```

> **Where the mouse ST comes from.** The mouse ST slides are *simulated* from the
> SC cohort — the SC cells carry spatial coordinates, so a slide is just the SC
> binned onto a grid (each spot = the sum of the cells in its bin).
> [`scripts/simulate_mousebrain_st.py`](scripts/simulate_mousebrain_st.py)
> reproduces this (grid resolution / offset give ST1, ST2, …); the dataset ships
> the exact slides used for training.

The gene panel is the shared genes across slides (built by the script);
expression is `normalize_total` + `log1p`'d.

## Reproducing the reference runs

Each script runs the full pipeline end-to-end — Stage A encoder → Stage B
geometric targets → Stage C DDP diffusion training → held-out inference → metrics —
and writes everything (per-epoch checkpoints, final model, reconstruction, metrics
JSON, logs) into `--out_dir`.

### hSCC

```bash
PYTHONPATH=$PWD python scripts/train_hscc.py \
    --data_dir /path/to/hscc --out_dir hscc_run          # add --devices 2 for DDP
```

Trains on ST1+ST2, reconstructs the held-out ST3 (one-shot — it fits). Residual
diffusion + sigma-cap curriculum on; the Stage-A "SC" domain is ST3 itself
(transductive; ST3 coordinates are never used in training).

### Mouse brain

```bash
PYTHONPATH=$PWD python scripts/train_mousebrain_full.py \
    --data_dir /path/to/mousebrain --out_dir mousebrain_run   # add --devices 2 for DDP
```

Trains on the ST3 slide and reconstructs the ~10k dissociated SC cells
(patchwise — too large for one shot on a small GPU). Expression is
`normalize_total` + `log1p`'d.

### Common flags

| Flag | Purpose |
|------|---------|
| `--devices N` | Number of GPUs for Stage-C DDP (inference always runs on rank 0). |
| `--smoke` | Tiny end-to-end check (a few epochs, small samples) to validate the setup. |
| `--resume [--resume_ckpt PATH]` | Resume Stage-C training from a checkpoint (defaults to `{out_dir}/ckpt_latest.pt`). |
| `--stageA_epochs / --stageC_epochs / --num_st_samples` | Override training lengths. |

Checkpoints are written every epoch (`ckpt_latest.pt`, `ckpt_epoch_*.pt`), so a
run can be killed and resumed without losing progress.

## Outputs

In `--out_dir`:

- `gears_<dataset>.pt` — final model (encoder + context encoder + generator + score net).
- `ckpt_latest.pt`, `ckpt_epoch_*.pt` — resumable Stage-C checkpoints.
- `reconstruction_<slide>.pt` — predicted 2D coordinates + dense distance matrix.
- `metrics_<slide>.json` — full evaluation suite.
- `console.log`, `train.log` — run logs.

The evaluation reports global (Spearman / Pearson / Stress-1), local (Edge
ROC-AUC, balanced-AP, Shell-F1), neighborhood (Trustworthiness / Continuity@k),
and distribution (Sliced-Wasserstein, W1 on kNN-distances) metrics.

## Inference: one-shot vs patchwise

Reconstruction turns a bag of expression profiles (dissociated cells, or held-out
ST spots) into 2D coordinates. There are two modes:

- **One-shot** — encode every point and solve the whole set in a single pass.
  Tightest reconstruction, but the Set-Transformer generator and diffusion
  refiner hold all points in memory at once.
- **Patchwise** — split the set into many overlapping local patches, recover each
  patch's geometry independently, and stitch them into one global map by
  distance-geometry. Scales to sets that don't fit in memory.

Patchwise came first, out of necessity: the ~10k dissociated mouse-brain cells
were too large to reconstruct all at once on the GPUs we had, so we broke the set
up and stitched the pieces back together. It works well — but stitching always
sheds a little precision, so **when the whole set fits, one-shot is the better
choice.**

`GEARS.reconstruct` chooses for you — one-shot when the set is small enough,
patchwise otherwise:

```python
import torch
from gears import GEARS
from gears.eval import plot_reconstruction, score_reconstruction

model = GEARS(n_genes=n_genes, device="cuda").load("gears_mousebrain.pt")

out = model.reconstruct(expr)                    # auto: one-shot if it fits, else patchwise
coords = out["coords"].cpu().numpy()

score_reconstruction(coords, gt_coords, out["is_outlier"])           # full metric row
plot_reconstruction(coords, gt_coords, is_outlier=out["is_outlier"],
                    save_path="reconstruction.png")                   # GT vs prediction
```

Force the mode with `mode="single"` / `mode="patchwise"`; `single_patch_max` sets
the size cutoff for `"auto"`. On a large GPU, patchwise with a bigger
`patch_size` (384 / 576 / 1024) stitches fewer, larger pieces — or just use
`mode="single"` if it fits. The mouse-brain SC cohort (~10k) runs patchwise by
default; the hSCC held-out slide (~600 spots) runs one-shot.

## Using GEARS on your own data

**[`docs/USING_GEARS.md`](docs/USING_GEARS.md) is the guide** — data format, the
train→reconstruct shape, a per-stage cheat-sheet, and (most usefully, given how
many knobs there are) the handful of **signals that tell you it's actually
working**: `disc_acc → 0.5` in Stage A, Stage-C loss dropping, `sigma_data_resid`
settling at ~0.03–0.05, and one-shot ≈ patchwise at inference.

The three stages are exposed on a single `GEARS` object; a minimal single-GPU
copy-and-edit template is in [`scripts/run_gears.py`](scripts/run_gears.py) and
the API is documented in [`gears/README.md`](gears/README.md).

## Citation

If you use GEARS, please consider citing :)

```bibtex
@article{azim2026geometry,
  title={Geometry-First Generative Spatial Single-Cell Reconstruction},
  author={Azim, Ehtesamul and Alif, Muhtasim Noor and Hwang, Tae Hyun and Fu, Yanjie and Zhang, Wei},
  journal={arXiv preprint arXiv:2605.28200},
  year={2026}
}
```
