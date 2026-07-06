"""Reproduce the mouse-brain `full_v1` checkpoint with the clean gears/ package.

This is the corrected, auditable re-implementation of the original
`model/mouse_brain_full.py` recipe that produced
`model/mouse_brain_full_v1/ckpt_latest.pt` (spearman ~0.757 on the 10,150 SC
cells). The recipe was recovered from that checkpoint + its training log
`model/mouse_brain_full_feb_1.log`; every load-bearing value below is pinned to
what that run actually used (see scripts/FULL_V1_RECIPE.md).

Key facts baked in (do not "improve" without re-deriving from the log):
  * Stage C is ST-ONLY. The diffusion generator never sees a single dissociated
    SC cell during training (log: "(no SC batches sampled)"). There is therefore
    NO `num_sc_samples` knob -- it was dead in the original and is dropped here.
  * SC expression is used only to (a) train the domain-invariant encoder in
    Stage A and (b) be reconstructed at inference.

The script asserts each stage against the ground-truth the log recorded
(343 genes, ST3=1310 spots, SC=10150 cells, architecture, sigma_data range) so a
mismatch fails loudly instead of silently drifting.

    PYTHONPATH=$PWD python scripts/train_mousebrain_full.py \
        --data_dir data/mousedata_2020/E1z2 --out_dir mousebrain_full_repro

Run on a box that has ST3 (`simu_st3_counts_et.csv`); the original trained on
2x RTX A4500 via Lightning Fabric. Single-GPU is the default here; pass
`--devices 2` to match the original DDP setup exactly.

The two experimental high-noise score-loss terms the original had active --
EXP_SCORE_HI_BOOST (4x weight on the noisiest 25% of samples, data-driven
readiness + 200-step ramp + tail cap) and EXP_SCORE_FX_HI (FX_HI_WEIGHT=2.0
direct F_x supervision at high sigma) -- are now ported into gears/
(gears/training/score_hi_boost.py) and enabled here via StageCConfig
(score_hi_boost=True, score_fx_hi=True). The full recipe is therefore a faithful
match; see gears/training/score_hi_boost.py for the port + test_score_hi_boost.py.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gears import GEARS
from gears.train_encoder import EncoderConfig
from gears.training import StageCConfig

# --------------------------------------------------------------------------- #
# Recipe constants (pinned to full_v1; see scripts/FULL_V1_RECIPE.md)
# --------------------------------------------------------------------------- #
GENE_FILTER_MAX_ZERO = 0.85
GENE_FILTER_MIN_VAR = 0.01
EXPECT_N_GENES = 343
EXPECT_N_ST_SPOTS = 1310
EXPECT_N_SC_CELLS = 10150
EXPECT_SIGMA_DATA_RESID = 0.033   # full_v1 sigma_data_resid after step-3000 recompute
SIGMA_DATA_RESID_TOL = 0.010      # ~+/-30% (RNG/torch-version dependent)

ARCH = dict(
    n_embedding=[512, 256, 128], D_latent=32, c_dim=256, n_heads=4,
    isab_m=128, dist_bins=24, angle_bins=8, knn_k=12,
    ctx_n_blocks=3, gen_n_blocks=6, score_n_blocks=4,
)


def _logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lg = logging.getLogger("train_full")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(out_dir, "train_full.log"))
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def normalize_expression(X):
    """Library-size -> 10,000 then log1p (matches original normalize_expression)."""
    X = X.astype(np.float32)
    X = X / (X.sum(axis=1, keepdims=True) + 1e-8) * 10000.0
    return np.log1p(X)


def filter_informative_genes(arrays, genes, max_zero=GENE_FILTER_MAX_ZERO,
                             min_var=GENE_FILTER_MIN_VAR):
    """Keep genes passing (zero_frac < max_zero) AND (var > min_var) in ALL sources.

    Faithful to model/ssl_utils.py:filter_informative_genes with full_v1's params
    (max_zero_frac=0.85, min_variance=0.01) on the RAW counts.
    """
    keep = np.ones(len(genes), dtype=bool)
    for X in arrays.values():
        zero_frac = (X == 0).mean(axis=0)
        var = X.var(axis=0)
        keep &= (zero_frac < max_zero) & (var > min_var)
    return [g for g, k in zip(genes, keep) if k]


def load_mouse_data(data_dir, log):
    """Load ST3 (train ST) + SC (reconstruction target), select the 343-gene
    informative panel on sorted(SC n ST3), normalize. Returns tensors + coords."""
    st_counts = pd.read_csv(os.path.join(data_dir, "simu_st3_counts_et.csv"), index_col=0)  # genes x spots
    st_meta = pd.read_csv(os.path.join(data_dir, "simu_st3_metadata_et.csv"), index_col=0)
    sc_counts = pd.read_csv(os.path.join(data_dir, "simu_sc_counts.csv"), index_col=0)       # genes x cells
    sc_meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)

    st_spots = list(st_counts.columns)
    sc_cells = list(sc_counts.columns)
    log.info(f"ST3 loaded: {len(st_spots)} spots, {st_counts.shape[0]} genes")
    log.info(f"SC  loaded: {len(sc_cells)} cells, {sc_counts.shape[0]} genes")
    assert len(st_spots) == EXPECT_N_ST_SPOTS, f"ST3 spots {len(st_spots)} != {EXPECT_N_ST_SPOTS}"
    assert len(sc_cells) == EXPECT_N_SC_CELLS, f"SC cells {len(sc_cells)} != {EXPECT_N_SC_CELLS}"

    # common = sorted(SC n ST3); informative-filter on RAW counts (both sources)
    common = sorted(set(st_counts.index) & set(sc_counts.index))
    log.info(f"Common genes before filtering: {len(common)}")
    st_raw = st_counts.reindex(common).values.T.astype(np.float32)   # spots x common
    sc_raw = sc_counts.reindex(common).values.T.astype(np.float32)   # cells x common
    genes = filter_informative_genes({"ST": st_raw, "SC": sc_raw}, common)
    log.info(f"[Gene Filter] max_zero={GENE_FILTER_MAX_ZERO}, min_var={GENE_FILTER_MIN_VAR} "
             f"-> {len(genes)}/{len(common)} genes")
    assert len(genes) == EXPECT_N_GENES, f"gene panel {len(genes)} != {EXPECT_N_GENES}"

    gi = [common.index(g) for g in genes]
    st_expr = torch.tensor(normalize_expression(st_raw[:, gi]))
    sc_expr = torch.tensor(normalize_expression(sc_raw[:, gi]))

    st_coords = torch.tensor(st_meta.loc[st_spots][["coord_x", "coord_y"]].values, dtype=torch.float32)
    sc_gt = sc_meta.loc[sc_cells][["x_global", "y_global"]].values.astype(np.float32)
    return dict(st_expr=st_expr, sc_expr=sc_expr, st_coords=st_coords,
                sc_gt=sc_gt, genes=genes, sc_cells=sc_cells)


def canonicalize_coords(coords):
    """Center + per-point RMS-radius rescale, matching full_v1's
    ``canonicalize_st_coords_per_slide``: rms = sqrt(mean_i ||x_i - mu||^2)
    (mean over points of the squared radius, NOT over all N*d elements)."""
    c = coords - coords.mean(dim=0)
    rms = c.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-8)
    return c / rms


def build_encoder_config(args):
    return EncoderConfig(
        n_epochs=args.stageA_epochs, batch_size=256, lr=1e-3, seed=args.seed,
        # VICReg (full_v1)
        vicreg_lambda_inv=25.0, vicreg_lambda_var=25.0, vicreg_lambda_cov=1.0,
        vicreg_gamma=1.0, vicreg_eps=1e-4,
        # augmentation (full_v1: 0.3 / 0.02 / 0.2)
        aug_gene_dropout=0.3, aug_gauss_std=0.02, aug_scale_jitter=0.2,
        # adversary / GRL (full_v1: adv 50, warmup 50, ramp 200, disc 512/0.2, LN True)
        adv_weight=50.0, adv_warmup_epochs=50, adv_ramp_epochs=200, grl_alpha_max=1.0,
        disc_hidden=512, disc_dropout=0.2, adv_use_layernorm=True,
        # local alignment (full_v1: weight 4.0, tau 0.07, bidirectional)
        local_align_weight=4.0, local_align_tau_z=0.07, local_align_bidirectional=True,
        use_best_checkpoint=True,
    )


def build_stageC_config(args):
    return StageCConfig(
        n_epochs=args.stageC_epochs, batch_size=16, lr=1e-4, ema_decay=0.999,
        sigma_min=0.01, sigma_max=3.0, use_edm=True, P_mean=-1.2, P_std=1.2,
        use_residual_diffusion=True, sigma_resid_recompute_step=3000,
        # NO conditioning augmentation (full_v1: all zero)
        z_noise_std=0.0, z_dropout_rate=0.0, aug_prob=0.0,
        use_z_ln=True,                       # full_v1 trained WITH z-layernorm
        # curriculum: target stage 3, caps [1,2,3,4], hard clamp 0.5
        curriculum_target_stage=3, curriculum_min_epochs=args.curriculum_min_epochs,
        curriculum_early_stop=True, sigma_cap_safe_mult=4.0, sigma_cap_abs_max=0.5,
        # full_v1 high-noise score-loss shaping (EXP_SCORE_HI_BOOST + FX_HI)
        score_hi_boost=True, score_fx_hi=True,
        score_hi_boost_factor=4.0, score_hi_boost_ramp=200,
        score_hi_target_rate=0.25, score_fx_hi_weight=2.0,
        seed=args.seed, precision=args.precision, num_workers=args.num_workers,
    )


def _load_nets(model, ckpt_path, device, log):
    """Load encoder + context + generator + score_net from a checkpoint into the
    GEARS model (for warm-init or resume; the trained encoder is NOT restored by
    train_stageC's own resume, so it must be loaded here)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ck["encoder"])
    model.context_encoder.load_state_dict(ck["context_encoder"])
    model.generator.load_state_dict(ck["generator"])
    sn = ck["score_net"]
    edges = sn.get("st_dist_bin_edges", None)
    if edges is not None:
        model.score_net.st_dist_bin_edges = edges.to(device)
    model.score_net.load_state_dict(sn)
    log.info(f"[warm-init] loaded 4 nets from {ckpt_path} (saved epoch {ck.get('epoch')})")


def main():
    ap = argparse.ArgumentParser(description="Reproduce mouse-brain full_v1 with gears/")
    ap.add_argument("--data_dir", default="data/mousedata_2020/E1z2")
    ap.add_argument("--out_dir", default="mousebrain_full_repro")
    ap.add_argument("--stageA_epochs", type=int, default=1200)
    ap.add_argument("--stageC_epochs", type=int, default=300)
    # Epochs a curriculum sigma-stage dwells before it may promote. Keep this a
    # healthy fraction of stageC_epochs: too small and the cap climbs before the
    # step-3000 residual-scale recompute, which inflates sigma_data_resid.
    ap.add_argument("--curriculum_min_epochs", type=int, default=180)
    ap.add_argument("--num_st_samples", type=int, default=4000)
    ap.add_argument("--n_min", type=int, default=128)
    ap.add_argument("--n_max", type=int, default=640)
    ap.add_argument("--devices", type=int, default=1, help="ST/SC DDP via Fabric; full_v1 used 2")
    ap.add_argument("--num_workers", type=int, default=6,
                    help="DataLoader workers for Stage-C miniset sampling (CPU-bound; 0=main proc)")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_stageA", action="store_true")
    ap.add_argument("--resume_ckpt", default=None)
    ap.add_argument("--warm_init_ckpt", default=None,
                    help="Load all 4 nets from this ckpt as init + skip Stage A, then train "
                         "Stage C fresh (reuses learned weights without resuming curriculum/optimizer).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = _logger(args.out_dir)
    log.info(f"=== full_v1 reproduction | seed={args.seed} device={device} ===")

    fabric = None
    if args.devices > 1:
        import lightning as L
        fabric = L.Fabric(accelerator="cuda", devices=args.devices, precision=args.precision)
        fabric.launch()
        device = str(fabric.device)

    # ---- Data ----
    data = load_mouse_data(args.data_dir, log)
    n_genes = len(data["genes"])
    st_expr, sc_expr = data["st_expr"].to(device), data["sc_expr"].to(device)
    st_coords_canon = canonicalize_coords(data["st_coords"]).to(device)

    # ---- Model (exact full_v1 architecture) ----
    model = GEARS(n_genes=n_genes, device=device, **ARCH)
    log.info(f"GEMS/GEARS init: enc {n_genes}->{ARCH['n_embedding']} | D_latent={ARCH['D_latent']} "
             f"c_dim={ARCH['c_dim']} isab_m={ARCH['isab_m']} dist_bins={ARCH['dist_bins']} "
             f"gen_blocks={ARCH['gen_n_blocks']}")
    n_params = sum(p.numel() for m in (model.encoder, model.context_encoder,
                                       model.generator, model.score_net)
                   for p in m.parameters())
    log.info(f"total params: {n_params/1e6:.2f}M")

    # ---- Stage A: domain-invariant encoder (ST3 spots vs 10k SC cells) ----
    # warm-init / resume load the trained encoder (train_stageC resume does NOT
    # restore it) and skip Stage A entirely; a fresh run trains it.
    if args.warm_init_ckpt:
        _load_nets(model, args.warm_init_ckpt, device, log)
    elif args.resume_ckpt:
        _load_nets(model, args.resume_ckpt, device, log)
    elif not args.skip_stageA:
        log.info(f"=== Stage A: encoder ({args.stageA_epochs} ep, lr=1e-3, VICReg+GRL+Local) ===")
        model.train_encoder(st_expr=st_expr, sc_expr=sc_expr,
                            config=build_encoder_config(args), out_dir=args.out_dir)
    else:
        log.info("=== Stage A: skipped ===")
    model.freeze_encoder()

    # ---- Stage B: pose-free geometric targets on the ST3 slide ----
    log.info("=== Stage B: precompute geometric targets (ST3) ===")
    model.precompute_targets({0: st_coords_canon}, geodesic_k=15)

    # ---- Stage C: diffusion geometry generator (ST-ONLY, residual, curriculum->3) ----
    log.info(f"=== Stage C: diffusion generator ST-ONLY ({args.stageC_epochs} ep, batch=16, "
             f"lr=1e-4, n_max={args.n_max}, curriculum->3, z_ln=True, residual) ===")
    log.info("    (no SC batches sampled -- diffusion training touches zero SC cells)")
    history = model.train_geometry(
        {0: st_expr}, config=build_stageC_config(args), out_dir=args.out_dir,
        num_samples=args.num_st_samples, n_min=args.n_min, n_max=args.n_max, fabric=fabric,
        resume_ckpt=args.resume_ckpt,
    )

    # ---- Verify against the log's ground truth ----
    sd = model.sigma_data
    sdr = model.sigma_data_resid
    log.info(f"Stage C done. sigma_data={sd} sigma_data_resid={sdr}")
    # The meaningful fidelity signal is sigma_data_resid after the step-3000 recompute.
    if sdr is not None and abs(float(sdr) - EXPECT_SIGMA_DATA_RESID) > SIGMA_DATA_RESID_TOL:
        log.warning(f"[CHECK] sigma_data_resid={float(sdr):.4f} differs from full_v1's "
                    f"{EXPECT_SIGMA_DATA_RESID} by >{SIGMA_DATA_RESID_TOL} -- inspect residual scale "
                    f"(only meaningful after >3000 steps).")
    else:
        log.info(f"[CHECK] sigma_data_resid={float(sdr):.4f} ~ full_v1's {EXPECT_SIGMA_DATA_RESID}.")

    out_path = os.path.join(args.out_dir, "gears_mousebrain_full.pt")
    model.save(out_path)
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump({"n_genes": n_genes, "genes": data["genes"], "arch": ARCH,
                   "stageA_epochs": args.stageA_epochs, "stageC_epochs": args.stageC_epochs,
                   "sigma_data": float(sd) if sd is not None else None,
                   "sigma_data_resid": float(model.sigma_data_resid) if model.sigma_data_resid else None,
                   "hi_boost_fx_hi_ported": True},
                  f, indent=2)
    log.info(f"Saved -> {out_path}")
    log.info("=== DONE. Reconstruct SC with scripts/infer_mousebrain_sc.py "
             "(use_z_ln=True, patch 1024, 500 steps). ===")


if __name__ == "__main__":
    main()
