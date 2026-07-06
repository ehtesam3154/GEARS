# Running GEARS on your own data

This walks through setting GEARS up on a new dataset and — more importantly —
**how to tell, at each step, whether it is actually working**. There are a lot
of knobs; you mostly need to get a handful right and watch a few signals.

`scripts/run_gears.py` is a runnable single-GPU template; `scripts/train_mousebrain_full.py`
and `scripts/train_hscc.py` are the two full worked examples. Copy whichever is
closest and edit the data-loading block.

---

## 1. What you need

GEARS supervises geometry from **spatial transcriptomics (ST)** and reconstructs
positions for a **target set** (dissociated cells, or a held-out ST slide).

| | shape | notes |
|---|---|---|
| ST expression | `(n_spots, n_genes)` per training slide | `normalize_total` + `log1p` |
| ST coordinates | `(n_spots, 2)` per training slide | real spatial coords; canonicalized per slide |
| target expression | `(n_target, n_genes)` | same gene panel + normalization as ST |
| target GT coords | `(n_target, 2)` *(optional)* | only for scoring; not used in training |

**Get these three right or nothing downstream works:**
1. **Shared gene panel** — the exact same genes, in the same order, for every
   slide and the target. Build it once (`sorted(intersection)`), reuse everywhere.
2. **Same normalization** everywhere (`normalize_total` → `log1p`).
3. **Architecture matches between train and inference** — `isab_m`, `dist_bins`,
   generator/context/score block counts, `D_latent`, `c_dim`. Build the inference
   `GEARS(...)` with the identical arch you trained with (or auto-detect from the
   checkpoint).

---

## 2. Train → reconstruct (the shape of it)

```python
from gears import GEARS, EncoderConfig
from gears.training import StageCConfig

model = GEARS(n_genes=G, isab_m=128, dist_bins=24, gen_n_blocks=6, device="cuda")

model.train_encoder(st_expr, target_expr, st_slide_ids=slide_ids,     # Stage A
                    config=EncoderConfig(n_epochs=1000, lr=1e-3))
model.precompute_targets({0: st_coords_canon})                        # Stage B
model.train_geometry({0: st_expr}, n_min=96, n_max=<~slide size>,     # Stage C
                    config=StageCConfig(use_residual_diffusion=True,
                                        use_z_ln=True, curriculum_target_stage=3))

out = model.reconstruct(target_expr)          # auto one-shot / patchwise
```

Multi-slide training: pass `st_expr` = all spots concatenated with `st_slide_ids`
labelling them, and `precompute_targets` / `train_geometry` a dict keyed by slide.

---

## 3. Is it working? — the signals to watch

### Stage A (encoder) — per-epoch log line
Healthy:
- **`disc_acc` drifts toward ~0.5** as `α` ramps to 1.0. This is the whole point:
  the domain adversary can no longer tell ST spots from target cells → the
  embedding is domain-invariant. Starting ~0.9 and settling ~0.5–0.65 is good.
- **VICReg total decreasing**, `std_min` staying **> ~0.1** (no representation
  collapse).

Red flags:
- `disc_acc` stuck near 1.0 → not aligning; the target domain is separable.
  Increase `adv_weight`, lengthen `adv_ramp_epochs`, or check normalization/genes.
- `std_min` → 0 → collapse; lower `lr` or raise VICReg variance weight.

### Stage B (targets)
Just prints "targets computed" per slide. If a slide has very few spots the
geodesic graph can be degenerate — keep slides above a few hundred spots.

### Stage C (diffusion generator) — this is where most goes right or wrong
Healthy:
- **Total loss dropping**, **`score` ~0.002–0.01** after warmup.
- **Curriculum `stage` promotes** 0→1→2→3 and **`sigma_cap_eff` climbs** as it does.
- **`sigma_data_resid` settles ~0.03–0.05** after the step-3000 recompute.

**The single most important red flag:** `sigma_data_resid` coming out **much larger
(> ~0.07)**. That means the sigma-curriculum climbed to high stages *before* the
step-3000 residual-scale recompute, so the residual is measured at an inflated
scale — and the diffusion refiner then denoises from too-large a sigma at
inference and *adds* noise (rounded, scattered reconstructions). Fix by raising
`curriculum_min_epochs` (keep it a healthy fraction — ~60% — of `stageC_epochs`)
so the cap stays low through the recompute. This is the usual cause of a
reconstruction that looks like a round blob.

Checkpoints are written every epoch (`ckpt_latest.pt`) — kill and resume freely.

### Inference — the cheapest end-to-end check
Run **both** modes and compare them:

```python
one  = model.reconstruct(target_expr, mode="single")     # if it fits in memory
many = model.reconstruct(target_expr, mode="patchwise")
```

- **One-shot and patchwise should agree ~95–99%** (Spearman between their
  pairwise-distance structures). High agreement = a healthy checkpoint and a
  sound stitch. Big disagreement (patchwise much worse) points back at an
  inflated `sigma_data_resid` from Stage C.
- If you have GT coords, `gears.eval.score_reconstruction(coords, gt, out["is_outlier"])`
  gives the metric row and `plot_reconstruction(...)` the GT-vs-prediction figure.
- **Pearson ≪ Spearman** on the distance metrics almost always means one or two
  stray points blew up the scale — this is handled for you (metrics/plots run on
  the inlier set), so trust the plotted figure over the raw first glance.

---

## 4. Knob cheat-sheet

| Knob | Where | Sensible default | Tune when |
|---|---|---|---|
| gene panel / normalization | data prep | shared genes, `normalize_total`+`log1p` | never skip |
| `use_z_ln` | StageC + Infer | **must match** between train & inference | — |
| `curriculum_target_stage` | StageC | 3 | — |
| `curriculum_min_epochs` | StageC | ~60% of `stageC_epochs` | raise if `sigma_data_resid` inflates |
| `n_max` | StageC | ≈ your ST slide size (≤ ~640) | — |
| `stageA_epochs` / `stageC_epochs` | both | 1000 / 200–300 | shorter for a quick look |
| `mode` / `patch_size` | Infer | `auto` | force `single` when it fits; bigger patch on a big GPU |
| `sigma_data_resid` | Infer | taken from the checkpoint | — |

Everything else has a working default. Start from a worked example, change the
data loader, watch the four signals above (disc_acc → 0.5, Stage-C loss down,
`sigma_data_resid` ~0.03–0.05, one-shot ≈ patchwise), and you're in business.
