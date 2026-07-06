"""Reproduce the hSCC (cutaneous SCC, patient P2) checkpoint with the gears/ package.

Clean re-implementation of the recipe behind `model/gems_hscc_curr_new_v4/` (the
run recorded via `model/run_hscc_gems.py`). Patient P2 has three ST replicate
slides; we train on ST1 + ST2 and reconstruct the held-out ST3.

Two hSCC-specific facts, baked in:
  * The Stage-A domain-invariance "SC" domain is ST3 ITSELF (the held-out
    replicate) — transductive domain adaptation: ST3 *expression* aligns the
    encoder, but ST3 *coordinates* are never used in training. (The real
    dissociated scRNA, scP2.h5ad, was not used.)
  * Stage C is ST-only and trained across BOTH training slides (ST1 + ST2).

Stage-A loss config matches the mouse recipe EXCEPT for lighter augmentation and
a smaller discriminator (both are the gears EncoderConfig defaults); Stage C
mirrors the mouse recipe but with n_max=448 and use_z_ln=False (hSCC did not use
z-layernorm).

    PYTHONPATH=$PWD python scripts/train_hscc.py \
        --data_dir data/cSCC/processed --out_dir hscc_repro

Run on a box with the three P2 .h5ad slides. Single-GPU by default; `--devices 2`
matches the original DDP setup.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gears import GEARS
from gears.train_encoder import EncoderConfig
from gears.training import StageCConfig

# --------------------------------------------------------------------------- #
# Recipe constants (pinned to gems_hscc_curr_new_v4)
# --------------------------------------------------------------------------- #
SLIDES = {"ST1": "stP2.h5ad", "ST2": "stP2rep2.h5ad", "ST3": "stP2rep3.h5ad"}
EXPECT_N_GENES = 2000
EXPECT_SIGMA_DATA_RESID = 0.035
SIGMA_DATA_RESID_TOL = 0.012

ARCH = dict(
    n_embedding=[512, 256, 128], D_latent=32, c_dim=256, n_heads=4,
    isab_m=128, dist_bins=24, angle_bins=8, knn_k=12,
    ctx_n_blocks=3, gen_n_blocks=6, score_n_blocks=4,
)


def _logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lg = logging.getLogger("train_hscc")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(os.path.join(out_dir, "train_hscc.log")),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def canonicalize_coords(coords):
    """Center + per-point RMS-radius rescale (per slide)."""
    c = coords - coords.mean(dim=0)
    rms = c.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-8)
    return c / rms


def load_hscc(data_dir, log):
    """Load the three P2 slides, take the shared gene panel, normalize, and pull
    per-slide spatial coordinates. ST3 is returned separately as the transductive
    Stage-A domain + the inference target."""
    import scanpy as sc

    ad = {k: sc.read_h5ad(os.path.join(data_dir, v)) for k, v in SLIDES.items()}
    for k, a in ad.items():
        log.info(f"{k} ({SLIDES[k]}): {a.n_obs} spots, {a.n_vars} genes")

    genes = sorted(set(ad["ST1"].var_names) & set(ad["ST2"].var_names) & set(ad["ST3"].var_names))
    log.info(f"Shared gene panel (sorted ST1 n ST2 n ST3): {len(genes)}")
    assert len(genes) == EXPECT_N_GENES, f"gene panel {len(genes)} != {EXPECT_N_GENES}"

    def expr(a):
        b = a[:, genes].copy()
        sc.pp.normalize_total(b)     # library-size -> median (as in run_hscc_gems.py)
        sc.pp.log1p(b)
        X = b.X
        return torch.tensor(X.toarray() if hasattr(X, "toarray") else np.asarray(X), dtype=torch.float32)

    def coords(a):
        return torch.tensor(np.asarray(a.obsm["spatial"]), dtype=torch.float32)

    return dict(
        st1_expr=expr(ad["ST1"]), st2_expr=expr(ad["ST2"]), st3_expr=expr(ad["ST3"]),
        st1_coords=coords(ad["ST1"]), st2_coords=coords(ad["ST2"]), st3_coords=coords(ad["ST3"]),
        genes=genes,
    )


def build_encoder_config(args):
    # hSCC Stage A == gears EncoderConfig defaults (light aug 0.2/0.01/0.1,
    # disc 256/0.1, VICReg 25/25/1, adv 50, local-align 4.0/0.07, MMD/CORAL/kNN on)
    # with only the optimization knobs overridden.
    return EncoderConfig(n_epochs=args.stageA_epochs, batch_size=256, lr=1e-3, seed=args.seed)


def build_stageC_config(args):
    return StageCConfig(
        n_epochs=args.stageC_epochs, batch_size=8, lr=1e-4, ema_decay=0.999,
        sigma_min=0.01, sigma_max=3.0, use_edm=True, P_mean=-1.2, P_std=1.2,
        use_residual_diffusion=True, sigma_resid_recompute_step=3000,
        z_noise_std=0.0, z_dropout_rate=0.0, aug_prob=0.0,
        use_z_ln=False,                          # hSCC did NOT use z-layernorm
        curriculum_target_stage=3, curriculum_min_epochs=args.curriculum_min_epochs,
        curriculum_early_stop=True, sigma_cap_safe_mult=4.0, sigma_cap_abs_max=None,
        score_hi_boost=True, score_fx_hi=True,
        score_hi_boost_factor=4.0, score_hi_boost_ramp=200,
        score_hi_target_rate=0.25, score_fx_hi_weight=2.0,
        seed=args.seed, precision=args.precision, num_workers=args.num_workers,
    )


def main():
    ap = argparse.ArgumentParser(description="Reproduce hSCC gems_hscc_curr_new_v4 with gears/")
    ap.add_argument("--data_dir", default="data/cSCC/processed")
    ap.add_argument("--out_dir", default="hscc_repro")
    ap.add_argument("--stageA_epochs", type=int, default=1000)
    ap.add_argument("--stageC_epochs", type=int, default=200)
    ap.add_argument("--curriculum_min_epochs", type=int, default=120)
    ap.add_argument("--num_st_samples", type=int, default=4000)
    ap.add_argument("--n_min", type=int, default=96)
    ap.add_argument("--n_max", type=int, default=448)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_stageA", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = _logger(args.out_dir)
    log.info(f"=== hSCC reproduction (train ST1+ST2 -> infer ST3) | seed={args.seed} ===")

    fabric = None
    if args.devices > 1:
        import lightning as L
        fabric = L.Fabric(accelerator="cuda", devices=args.devices, precision=args.precision)
        fabric.launch()
        device = str(fabric.device)

    # ---- Data ----
    d = load_hscc(args.data_dir, log)
    n_genes = len(d["genes"])
    st1_expr, st2_expr, st3_expr = (d[k].to(device) for k in ("st1_expr", "st2_expr", "st3_expr"))
    st1_c = canonicalize_coords(d["st1_coords"]).to(device)
    st2_c = canonicalize_coords(d["st2_coords"]).to(device)

    # ---- Model ----
    model = GEARS(n_genes=n_genes, device=device, **ARCH)
    log.info(f"GEMS/GEARS init: enc {n_genes}->{ARCH['n_embedding']} isab_m={ARCH['isab_m']} "
             f"dist_bins={ARCH['dist_bins']} gen_blocks={ARCH['gen_n_blocks']}")

    # ---- Stage A: encoder — ST1+ST2 (domain 0) vs ST3 (domain 1, transductive) ----
    if not args.skip_stageA:
        log.info(f"=== Stage A: encoder ({args.stageA_epochs} ep, lr=1e-3) | "
                 f"ST1+ST2 vs held-out ST3 (no ST3 coords used) ===")
        st_expr = torch.cat([st1_expr, st2_expr], dim=0)
        st_slide_ids = torch.cat([
            torch.zeros(st1_expr.shape[0], dtype=torch.long),
            torch.ones(st2_expr.shape[0], dtype=torch.long)]).to(device)
        model.train_encoder(st_expr=st_expr, sc_expr=st3_expr, st_slide_ids=st_slide_ids,
                            config=build_encoder_config(args), out_dir=args.out_dir)
    else:
        log.info("=== Stage A: skipped ===")
    model.freeze_encoder()

    # ---- Stage B: pose-free targets on BOTH training slides ----
    log.info("=== Stage B: precompute geometric targets (ST1 + ST2) ===")
    model.precompute_targets({0: st1_c, 1: st2_c}, geodesic_k=15)

    # ---- Stage C: diffusion generator (ST-only, both training slides) ----
    log.info(f"=== Stage C: diffusion generator ST-only ({args.stageC_epochs} ep, batch=8, "
             f"lr=1e-4, n_max={args.n_max}, curriculum->3, z_ln=False, residual) ===")
    history = model.train_geometry(
        {0: st1_expr, 1: st2_expr}, config=build_stageC_config(args), out_dir=args.out_dir,
        num_samples=args.num_st_samples, n_min=args.n_min, n_max=args.n_max, fabric=fabric)

    sdr = model.sigma_data_resid
    log.info(f"Stage C done. sigma_data={model.sigma_data} sigma_data_resid={sdr}")
    if sdr is not None and abs(float(sdr) - EXPECT_SIGMA_DATA_RESID) > SIGMA_DATA_RESID_TOL:
        log.warning(f"[CHECK] sigma_data_resid={float(sdr):.4f} differs from the reference "
                    f"{EXPECT_SIGMA_DATA_RESID} (only meaningful after >3000 steps).")

    out_path = os.path.join(args.out_dir, "gears_hscc.pt")
    model.save(out_path)
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump({"n_genes": n_genes, "genes": d["genes"], "arch": ARCH,
                   "sigma_data": float(model.sigma_data) if model.sigma_data else None,
                   "sigma_data_resid": float(sdr) if sdr else None}, f, indent=2)
    log.info(f"Saved -> {out_path}")
    log.info("=== DONE. Reconstruct ST3: model.reconstruct(st3_expr) (auto -> one-shot, ~600 spots). ===")


if __name__ == "__main__":
    main()
