"""
Stage A — train the domain-invariant shared encoder.

The encoder is trained on augmented single expression profiles from both domains
(ST spots and dissociated SC cells) plus domain labels only; it does not use
spatial coordinates. After training it is frozen and reused by all later stages.

Objective (per mini-batch):

    L = L_vicreg
      + adv_weight       * L_adv          (gradient-reversal domain adversary)
      + coral_w          * L_coral        (ramped; ST vs SC mean/cov match)
      + mmd_w            * L_mmd          (ramped; ST vs SC RBF-MMD)
      + local_w          * L_local        (ramped; MNN InfoNCE, after warmup)
      + patient_coral_w  * L_patient_coral(only if sc_patient_ids given)
      + knn_weight       * L_knn          (global expression-neighborhood consistency)

Model selection: a domain-mixing kNN probe (see `_knn_domain_mixing`). Every
`probe_every` epochs we embed a balanced ST/SC subsample and measure how well a
kNN classifier can still tell the domains apart (balanced accuracy; 0.5 = fully
mixed, 1.0 = separable). The checkpoint with the best mixing (subject to a
variance-collapse guard) is restored at the end.

`EncoderConfig` defaults target the seqFISH / pseudo-Visium setting; the
multi-slide / multi-patient setting overrides a few knobs (noted inline) and
supplies `sc_slide_ids` / `sc_patient_ids`.

Example
-------
    from gears import SharedEncoder, EncoderConfig, train_encoder

    enc = SharedEncoder(n_genes=st_expr.shape[1])          # [n_genes]->[512,256,128]
    enc, history = train_encoder(
        enc, st_expr, sc_expr,
        st_slide_ids=slide_ids,                            # optional
        config=EncoderConfig(n_epochs=1000, lr=1e-4),
        device="cuda", out_dir="stageA_out",
    )
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import SharedEncoder, SlideDiscriminator
from .losses import (
    VICRegLoss,
    coral_loss,
    grad_reverse,
    grl_alpha_schedule,
    knn_consistency_loss,
    local_alignment_loss,
    mmd_rbf_loss,
)
from .sampling import (
    augment_expression,
    sample_balanced,
    sample_balanced_domain_and_slide,
)


@dataclass
class EncoderConfig:
    """Stage-A hyperparameters (defaults for the seqFISH / pseudo-Visium setting)."""

    # --- VICReg ---
    vicreg_lambda_inv: float = 25.0
    vicreg_lambda_var: float = 25.0
    vicreg_lambda_cov: float = 1.0
    vicreg_gamma: float = 1.0
    vicreg_eps: float = 1e-4

    # --- two-view augmentation ---   (multi-slide setting: 0.3 / 0.02 / 0.2)
    aug_gene_dropout: float = 0.2
    aug_gauss_std: float = 0.01
    aug_scale_jitter: float = 0.1

    # --- domain adversary (gradient reversal) ---   (multi-slide setting: adv 100, disc 512, LN False)
    adv_weight: float = 50.0
    adv_warmup_epochs: int = 50
    adv_ramp_epochs: int = 200
    grl_alpha_max: float = 1.0
    disc_hidden: int = 256
    disc_dropout: float = 0.1
    disc_steps: int = 10
    disc_lr_mult: float = 3.0          # discriminator LR = disc_lr_mult * lr
    adv_use_layernorm: bool = True

    # --- alignment terms ---
    mmd_weight: float = 20.0
    mmd_use_l2norm: bool = True
    mmd_ramp: bool = True              # ramp MMD on the same schedule as CORAL
    coral_raw_weight: float = 1.0      # extra CORAL on un-normalized embeddings
    local_align_weight: float = 4.0
    local_align_tau_z: float = 0.07
    local_align_bidirectional: bool = True
    local_align_warmup: int = 100
    patient_coral_weight: float = 50.0  # only active when sc_patient_ids is given (multi-patient)

    # --- global expression-neighborhood consistency ---
    knn_weight: float = 10.0
    knn_k: int = 15
    knn_precompute_k: int = 30
    knn_cache_update_freq: int = 50

    # --- optimization ---   (multi-slide setting: lr 1e-3)
    n_epochs: int = 1000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 1.0
    balanced_sampling: bool = True
    sc_inference_dropout_prob: float = 0.5  # multi-SC-slide only: randomly drop one SC slide per batch

    # --- model selection / early stop (domain-mixing kNN probe) ---
    use_best_checkpoint: bool = True
    probe_every: int = 100
    probe_k: int = 20
    probe_n_sample: int = 2000
    probe_min_std: float = 0.1          # reject variance-collapsed representations
    early_stop_margin: Optional[float] = None  # stop when bal_acc <= 0.5 + margin; None = off

    # --- reproducibility ---
    seed: Optional[int] = 42
    log_every: int = 50


@torch.no_grad()
def _knn_domain_mixing(
    encoder: SharedEncoder,
    X_ssl: torch.Tensor,
    n_st: int,
    n_sc: int,
    k: int,
    n_sample: int,
    device: str,
) -> Tuple[float, float]:
    """
    Measure how well a kNN classifier can still separate ST from SC in embedding
    space over a balanced subsample.

    Returns:
        (balanced_accuracy, min_std)
        balanced_accuracy in [0.5, 1.0]: 0.5 = perfectly mixed, 1.0 = separable.
        min_std: smallest per-dimension std of the embeddings (collapse guard).
    """
    was_training = encoder.training
    encoder.eval()

    n = min(n_sample, n_st, n_sc)
    st_idx = torch.randperm(n_st, device=device)[:n]
    sc_idx = n_st + torch.randperm(n_sc, device=device)[:n]
    idx = torch.cat([st_idx, sc_idx], dim=0)

    z_parts = [encoder(X_ssl[idx[i : i + 512]]) for i in range(0, idx.numel(), 512)]
    z = torch.cat(z_parts, dim=0)
    labels = torch.cat(
        [torch.zeros(n, device=device), torch.ones(n, device=device)]
    ).long()

    min_std = z.std(dim=0).min().item()

    zc = F.normalize(z, dim=1)
    S = zc @ zc.T
    S.fill_diagonal_(float("-inf"))
    knn = S.topk(k, dim=1).indices                 # (2n, k)
    frac_sc = labels[knn].float().mean(dim=1)      # fraction of SC among neighbors
    pred = (frac_sc > 0.5).long()
    acc0 = (pred[labels == 0] == 0).float().mean().item()
    acc1 = (pred[labels == 1] == 1).float().mean().item()

    if was_training:
        encoder.train()
    return 0.5 * (acc0 + acc1), min_std


def train_encoder(
    encoder: SharedEncoder,
    st_expr: torch.Tensor,
    sc_expr: torch.Tensor,
    st_slide_ids: Optional[torch.Tensor] = None,
    sc_slide_ids: Optional[torch.Tensor] = None,
    sc_patient_ids: Optional[torch.Tensor] = None,
    config: Optional[EncoderConfig] = None,
    device: str = "cuda",
    out_dir: Optional[str] = None,
) -> Tuple[SharedEncoder, Dict]:
    """
    Train the shared encoder (Stage A) and return (encoder, history).

    Args:
        encoder:        a SharedEncoder instance (built with the right n_genes).
        st_expr:        (n_st, n_genes) log1p ST spot expression.
        sc_expr:        (n_sc, n_genes) log1p dissociated SC expression.
        st_slide_ids:   (n_st,) ST slide ids, used only for hierarchical balancing
                        when SC also spans slides. Optional.
        sc_slide_ids:   (n_sc,) SC slide ids for hierarchical balancing (multi-slide SC).
        sc_patient_ids: (n_sc,) SC patient ids; enables patient-level CORAL within SC.
        config:         EncoderConfig (defaults for the seqFISH / pseudo-Visium setting).
        device:         torch device.
        out_dir:        if given, saves the final encoder + training history there.
    """
    cfg = config or EncoderConfig()

    if cfg.seed is not None:
        import random

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    encoder = encoder.to(device)
    st_expr = st_expr.to(device)
    sc_expr = sc_expr.to(device)
    if st_slide_ids is not None:
        st_slide_ids = st_slide_ids.to(device)
    if sc_slide_ids is not None:
        sc_slide_ids = sc_slide_ids.to(device)
    if sc_patient_ids is not None:
        sc_patient_ids = sc_patient_ids.to(device)

    # --- balance the two domains by subsampling the larger to match the smaller ---
    # Only needed for the random-batch fallback: with balanced per-batch sampling
    # (the default) the full pools are kept, so a much larger domain (e.g. ~10k
    # dissociated SC cells vs a few hundred ST spots) is covered over epochs
    # instead of being discarded down to the smaller domain's size on epoch 0.
    n_st, n_sc = st_expr.shape[0], sc_expr.shape[0]
    if not cfg.balanced_sampling:
        if n_sc > n_st:
            sel = torch.randperm(n_sc, device=device)[:n_st]
            sc_expr = sc_expr[sel]
            if sc_slide_ids is not None:
                sc_slide_ids = sc_slide_ids[sel]
            if sc_patient_ids is not None:
                sc_patient_ids = sc_patient_ids[sel]
            n_sc = n_st
        elif n_st > n_sc:
            sel = torch.randperm(n_st, device=device)[:n_sc]
            st_expr = st_expr[sel]
            if st_slide_ids is not None:
                st_slide_ids = st_slide_ids[sel]
            n_st = n_sc

    X_ssl = torch.cat([st_expr, sc_expr], dim=0)
    domain_ids = torch.cat(
        [torch.zeros(n_st, device=device), torch.ones(n_sc, device=device)]
    ).long()  # 0 = ST, 1 = SC

    # ST slide ids for hierarchical balancing (default: a single ST slide)
    st_ids_for_sampling = (
        st_slide_ids
        if st_slide_ids is not None
        else torch.zeros(n_st, dtype=torch.long, device=device)
    )

    h_dim = encoder.embedding_dim
    discriminator = SlideDiscriminator(
        h_dim, n_domains=2, hidden_dim=cfg.disc_hidden, dropout=cfg.disc_dropout
    ).to(device)

    vicreg = VICRegLoss(
        cfg.vicreg_lambda_inv,
        cfg.vicreg_lambda_var,
        cfg.vicreg_lambda_cov,
        cfg.vicreg_gamma,
        cfg.vicreg_eps,
    )

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    opt_disc = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.disc_lr_mult * cfg.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=cfg.n_epochs)

    # --- precompute global expression neighbors + embedding cache (kNN consistency) ---
    global_knn = None
    z_cache = None
    if cfg.knn_weight > 0:
        with torch.no_grad():
            X_norm = F.normalize(X_ssl, dim=1)
            n_total = X_ssl.shape[0]
            kk = cfg.knn_precompute_k
            global_knn = torch.zeros(n_total, kk, dtype=torch.long, device=device)
            for i in range(0, n_total, 1000):
                end = min(i + 1000, n_total)
                sim = X_norm[i:end] @ X_norm.T
                for j in range(i, end):
                    sim[j - i, j] = -float("inf")
                global_knn[i:end] = sim.topk(kk, dim=1).indices
            z_cache = torch.zeros(n_total, h_dim, device=device)
            for i in range(0, n_total, 512):
                end = min(i + 512, n_total)
                z_cache[i:end] = encoder(X_ssl[i:end])

    history: Dict[str, list] = {key: [] for key in (
        "epoch", "loss_total", "loss_vicreg", "loss_adv", "loss_coral", "loss_mmd",
        "loss_local", "loss_patient_coral", "loss_knn", "vicreg_inv", "vicreg_var",
        "vicreg_cov", "std_mean", "std_min", "disc_acc", "grl_alpha", "probe_bal_acc",
    )}

    best_bal_acc = float("inf")
    best_epoch = -1
    best_state = None
    probe_start = cfg.adv_warmup_epochs + cfg.adv_ramp_epochs

    print(f"[Stage A] Training encoder: {cfg.n_epochs} epochs, ST={n_st}, SC={n_sc}")
    for epoch in range(cfg.n_epochs):
        encoder.train()
        discriminator.train()

        # ---- balanced batch (hierarchical across slides when SC spans slides) ----
        if cfg.balanced_sampling:
            if sc_slide_ids is not None:
                idx = sample_balanced_domain_and_slide(
                    domain_ids, st_ids_for_sampling, sc_slide_ids, cfg.batch_size, device
                )
            else:
                idx = sample_balanced(domain_ids, cfg.batch_size, device)
        else:
            idx = torch.randperm(X_ssl.shape[0], device=device)[: cfg.batch_size]

        X_batch = X_ssl[idx]
        s_batch = domain_ids[idx]

        # multi-SC-slide robustness: randomly drop one SC slide's cells from the batch
        if (
            sc_slide_ids is not None
            and cfg.sc_inference_dropout_prob > 0
            and torch.rand(1).item() < cfg.sc_inference_dropout_prob
        ):
            sc_mask = idx >= n_st
            sc_local = idx[sc_mask] - n_st
            if sc_local.numel() > 0:
                present = torch.unique(sc_slide_ids[sc_local])
                if len(present) > 1:
                    drop = present[torch.randint(0, len(present), (1,), device=device)]
                    keep = torch.ones_like(s_batch, dtype=torch.bool)
                    keep[sc_mask] = sc_slide_ids[sc_local] != drop
                    idx = idx[keep]
                    X_batch = X_ssl[idx]
                    s_batch = domain_ids[idx]

        is_sc = idx >= n_st

        # ---- VICReg on two augmented views ----
        z1 = encoder(augment_expression(
            X_batch, cfg.aug_gene_dropout, cfg.aug_gauss_std, cfg.aug_scale_jitter))
        z2 = encoder(augment_expression(
            X_batch, cfg.aug_gene_dropout, cfg.aug_gauss_std, cfg.aug_scale_jitter))
        loss_vicreg, vstats = vicreg(z1, z2)

        alpha = grl_alpha_schedule(
            epoch, cfg.adv_warmup_epochs, cfg.adv_ramp_epochs, cfg.grl_alpha_max
        )

        # ---- clean embeddings (used by adversary / CORAL / MMD / local / kNN) ----
        z_clean = encoder(X_batch)
        z_cond = (
            F.layer_norm(z_clean, (z_clean.shape[1],))
            if cfg.adv_use_layernorm
            else z_clean
        )

        # kNN consistency (global expression neighborhoods)
        if cfg.knn_weight > 0 and global_knn is not None:
            loss_knn = knn_consistency_loss(idx, z_clean, global_knn, z_cache, k=cfg.knn_k)
        else:
            loss_knn = torch.tensor(0.0, device=device)

        # ---- (A) train discriminator on detached embeddings ----
        z_det = z_cond.detach()
        disc_acc = 0.5
        for _ in range(cfg.disc_steps):
            logits_d = discriminator(z_det)
            loss_disc = F.cross_entropy(logits_d, s_batch)
            opt_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            opt_disc.step()
            with torch.no_grad():
                disc_acc = (logits_d.argmax(1) == s_batch).float().mean().item()

        # ---- (B) train encoder to confuse the discriminator + align domains ----
        for p in discriminator.parameters():
            p.requires_grad_(False)

        z_st, z_sc = z_cond[~is_sc], z_cond[is_sc]
        loss_coral = torch.tensor(0.0, device=device)
        loss_mmd = torch.tensor(0.0, device=device)
        if z_st.shape[0] > 8 and z_sc.shape[0] > 8:
            loss_coral = coral_loss(z_st, z_sc) + cfg.coral_raw_weight * coral_loss(
                z_clean[~is_sc], z_clean[is_sc]
            )
            z_st_mmd = F.normalize(z_st, dim=1) if cfg.mmd_use_l2norm else z_st
            z_sc_mmd = F.normalize(z_sc, dim=1) if cfg.mmd_use_l2norm else z_sc
            loss_mmd = mmd_rbf_loss(z_st_mmd, z_sc_mmd)

        # patient-level CORAL within SC (multi-patient SC only)
        loss_patient_coral = torch.tensor(0.0, device=device)
        if sc_patient_ids is not None and is_sc.sum() > 16:
            batch_pid = sc_patient_ids[idx[is_sc] - n_st]
            unique_p = torch.unique(batch_pid)
            if len(unique_p) >= 2:
                z_sc_b = z_cond[is_sc]
                pair_losses = []
                for a in range(len(unique_p)):
                    for b in range(a + 1, len(unique_p)):
                        ma = batch_pid == unique_p[a]
                        mb = batch_pid == unique_p[b]
                        if ma.sum() > 4 and mb.sum() > 4:
                            pair_losses.append(coral_loss(z_sc_b[ma], z_sc_b[mb]))
                if pair_losses:
                    loss_patient_coral = torch.stack(pair_losses).mean()

        # domain adversary (gradient reversal)
        loss_adv = F.cross_entropy(discriminator(grad_reverse(z_cond, alpha)), s_batch)

        # local alignment (MNN InfoNCE), after its warmup
        loss_local = torch.tensor(0.0, device=device)
        local_w = 0.0
        if epoch >= cfg.local_align_warmup and z_st.shape[0] > 8 and z_sc.shape[0] > 8:
            loss_local = local_alignment_loss(
                z_clean[is_sc],
                z_clean[~is_sc],
                tau_z=cfg.local_align_tau_z,
                bidirectional=cfg.local_align_bidirectional,
            )
            local_w = cfg.local_align_weight * float(
                np.clip((epoch - cfg.local_align_warmup) / max(1, cfg.adv_ramp_epochs), 0.0, 1.0)
            )

        # ramps (CORAL / MMD / patient-CORAL share the adversary schedule)
        coral_w = float(
            np.clip((epoch - cfg.adv_warmup_epochs) / max(1, cfg.adv_ramp_epochs), 0.0, 1.0)
        )
        mmd_w = coral_w * cfg.mmd_weight if cfg.mmd_ramp else cfg.mmd_weight
        patient_w = coral_w * cfg.patient_coral_weight if sc_patient_ids is not None else 0.0

        loss_total = (
            loss_vicreg
            + cfg.adv_weight * loss_adv
            + coral_w * loss_coral
            + mmd_w * loss_mmd
            + local_w * loss_local
            + patient_w * loss_patient_coral
            + cfg.knn_weight * loss_knn
        )

        opt_enc.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=cfg.grad_clip)
        opt_enc.step()
        scheduler.step()

        for p in discriminator.parameters():
            p.requires_grad_(True)

        # refresh the kNN embedding cache periodically
        if cfg.knn_weight > 0 and z_cache is not None and epoch % cfg.knn_cache_update_freq == 0:
            with torch.no_grad():
                for i in range(0, X_ssl.shape[0], 512):
                    end = min(i + 512, X_ssl.shape[0])
                    z_cache[i:end] = encoder(X_ssl[i:end])

        # ---- history ----
        history["epoch"].append(epoch)
        history["loss_total"].append(loss_total.item())
        history["loss_vicreg"].append(loss_vicreg.item())
        history["loss_adv"].append(loss_adv.item())
        history["loss_coral"].append(loss_coral.item())
        history["loss_mmd"].append(loss_mmd.item())
        history["loss_local"].append(loss_local.item())
        history["loss_patient_coral"].append(loss_patient_coral.item())
        history["loss_knn"].append(loss_knn.item())
        history["vicreg_inv"].append(vstats["inv"])
        history["vicreg_var"].append(vstats["var"])
        history["vicreg_cov"].append(vstats["cov"])
        history["std_mean"].append(vstats["std_mean"])
        history["std_min"].append(vstats["std_min"])
        history["disc_acc"].append(disc_acc)
        history["grl_alpha"].append(alpha)

        # ---- domain-mixing probe + model selection / early stop ----
        bal_acc = None
        if epoch >= probe_start and epoch % cfg.probe_every == 0:
            bal_acc, min_std = _knn_domain_mixing(
                encoder, X_ssl, n_st, n_sc, cfg.probe_k, cfg.probe_n_sample, device
            )
            collapsed = min_std <= cfg.probe_min_std
            if not collapsed and bal_acc < best_bal_acc:
                best_bal_acc = bal_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                print(f"  [BEST] epoch {epoch}: domain-mixing kNN bal_acc={bal_acc:.4f} "
                      f"(0.5=mixed), min_std={min_std:.3f}")
            if (
                cfg.early_stop_margin is not None
                and not collapsed
                and bal_acc <= 0.5 + cfg.early_stop_margin
            ):
                print(f"  [EARLY STOP] epoch {epoch}: bal_acc={bal_acc:.4f} "
                      f"<= 0.5 + {cfg.early_stop_margin}")
                history["probe_bal_acc"].append(bal_acc)
                break
        history["probe_bal_acc"].append(bal_acc if bal_acc is not None else float("nan"))

        # ---- logging ----
        if epoch % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
            print(
                f"epoch {epoch}/{cfg.n_epochs} | total={loss_total.item():.3f} "
                f"(VIC={loss_vicreg.item():.2f}, adv={loss_adv.item():.3f}, "
                f"CORAL={loss_coral.item():.4f}, MMD={loss_mmd.item():.4f}, "
                f"local={loss_local.item():.3f}, kNN={loss_knn.item():.3f}) | "
                f"std_min={vstats['std_min']:.2f} disc_acc={disc_acc:.3f} α={alpha:.2f}"
            )

    # ---- restore best checkpoint ----
    if cfg.use_best_checkpoint and best_state is not None:
        print(f"[Stage A] Restoring best encoder from epoch {best_epoch} "
              f"(domain-mixing bal_acc={best_bal_acc:.4f})")
        encoder.load_state_dict(best_state)
    else:
        print("[Stage A] Using final-epoch encoder")

    if out_dir is not None:
        import json
        import os

        os.makedirs(out_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder_final.pt"))
        with open(os.path.join(out_dir, "stageA_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Stage A] Saved encoder + history to {out_dir}")

    encoder.eval()
    return encoder, history
