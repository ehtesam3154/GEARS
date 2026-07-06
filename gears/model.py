"""GEARS orchestrator: builds and wires the full pipeline.

    Stage A  encoder            train_encoder
    Stage B  geometric targets  precompute_targets
    Stage C  geometry + refiner train_geometry
    Inference (SC -> 2D)        reconstruct

The four networks (shared encoder, context encoder, generator, score net) are
held on one object; the frozen encoder feeds Stages B/C and inference.
"""

from typing import Dict, List, Optional

import torch

from .encoder import SharedEncoder
from .train_encoder import EncoderConfig, train_encoder
from .data import STStageBPrecomputer, STSetDataset
from .models import SetEncoderContext, MetricSetGenerator, DiffusionScoreNet
from .training import StageCConfig, train_stageC
from .inference import InferConfig, reconstruct_sc
from .inference.outliers import inlier_mask


class GEARS:
    def __init__(
        self,
        n_genes: int,
        n_embedding: List[int] = [512, 256, 128],
        D_latent: int = 32,
        c_dim: int = 256,
        n_heads: int = 4,
        isab_m: int = 64,
        dist_bins: int = 16,
        angle_bins: int = 8,
        knn_k: int = 12,
        ctx_n_blocks: int = 3,
        gen_n_blocks: int = 2,
        score_n_blocks: int = 4,
        device: str = "cuda",
    ):
        self.device = device
        self.D_latent = D_latent
        h = n_embedding[-1]

        self.encoder = SharedEncoder(n_genes, n_embedding).to(device)
        self.context_encoder = SetEncoderContext(
            h_dim=h, c_dim=c_dim, n_heads=n_heads, n_blocks=ctx_n_blocks, isab_m=isab_m).to(device)
        self.generator = MetricSetGenerator(
            c_dim=c_dim, D_latent=D_latent, n_heads=n_heads, n_blocks=gen_n_blocks, isab_m=isab_m).to(device)
        self.score_net = DiffusionScoreNet(
            D_latent=D_latent, c_dim=c_dim, n_heads=n_heads, n_blocks=score_n_blocks, isab_m=isab_m,
            dist_bins=dist_bins, angle_bins=angle_bins, knn_k=knn_k).to(device)

        self.targets = None
        self.sigma_data = None
        self.sigma_data_resid = None

    # ---- Stage A: domain-invariant encoder ----
    def train_encoder(
        self,
        st_expr: torch.Tensor,
        sc_expr: torch.Tensor,
        st_slide_ids: Optional[torch.Tensor] = None,
        sc_slide_ids: Optional[torch.Tensor] = None,
        sc_patient_ids: Optional[torch.Tensor] = None,
        config: Optional[EncoderConfig] = None,
        out_dir: Optional[str] = None,
    ):
        self.encoder, history = train_encoder(
            self.encoder, st_expr, sc_expr, st_slide_ids, sc_slide_ids, sc_patient_ids,
            config, self.device, out_dir)
        self.freeze_encoder()
        return history

    def freeze_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    # ---- Stage B: pose-free geometric targets ----
    def precompute_targets(
        self,
        slides: Dict[int, torch.Tensor],
        k: int = 20,
        use_geodesic_targets: bool = True,
        geodesic_k: int = 15,
    ):
        pre = STStageBPrecomputer(k=k, device=self.device)
        self.targets = pre.precompute(slides, use_geodesic_targets, geodesic_k)
        return self.targets

    # ---- Stage C: geometry generator + diffusion refiner ----
    def train_geometry(
        self,
        st_expr_dict: Dict[int, torch.Tensor],
        config: Optional[StageCConfig] = None,
        out_dir: str = "stageC_out",
        num_samples: int = 4000,
        n_min: int = 96,
        n_max: int = 384,
        pool_mult: float = 2.0,
        stochastic_tau: float = 1.0,
        fabric=None,
        resume_ckpt=None,
    ):
        if self.targets is None:
            raise RuntimeError("call precompute_targets(...) before train_geometry(...)")
        dataset = STSetDataset(
            self.targets, self.encoder, st_expr_dict, n_min=n_min, n_max=n_max,
            D_latent=self.D_latent, num_samples=num_samples, device=self.device,
            pool_mult=pool_mult, stochastic_tau=stochastic_tau)
        history = train_stageC(
            self.context_encoder, self.generator, self.score_net, dataset,
            encoder=self.encoder, config=config, device=self.device, out_dir=out_dir,
            resume_ckpt=resume_ckpt, fabric=fabric)
        self.sigma_data = history.get("sigma_data")
        self.sigma_data_resid = history.get("sigma_data_resid", self.sigma_data)
        return history

    # ---- Inference: expression -> 2D geometry ----
    def reconstruct(self, sc_expr: torch.Tensor, config: Optional[InferConfig] = None,
                    mode: str = "auto", single_patch_max: int = 2000):
        """Reconstruct 2D coordinates (+ dense distances) from expression.

        mode:
            "auto"      one shot (all points in a single patch) when
                        N <= single_patch_max, otherwise patchwise. Single-patch
                        is tighter when it fits in memory; patchwise scales to
                        large sets (e.g. the ~10k dissociated cells) that don't.
            "single"    force one-shot (patch covers every point).
            "patchwise" force patchwise stitching.
        Returns the reconstruct_sc dict plus 'mode' and an 'is_outlier' mask.
        """
        N = int(sc_expr.shape[0])
        if config is None:
            config = InferConfig(
                sigma_data=self.sigma_data or 0.5,
                sigma_data_resid=self.sigma_data_resid,
            )
        elif config.sigma_data_resid is None:
            config.sigma_data_resid = self.sigma_data_resid

        resolved = ("single" if N <= single_patch_max else "patchwise") if mode == "auto" else mode
        if resolved == "single":
            config.patch_size = N
            config.n_landmarks = min(config.n_landmarks, max(2, N - 1))

        out = reconstruct_sc(
            self.encoder, self.context_encoder, self.generator, self.score_net,
            sc_expr, config, self.device)
        out["mode"] = resolved
        out["is_outlier"] = ~inlier_mask(out["coords"].detach().cpu().numpy())
        return out

    # ---- persistence ----
    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "context_encoder": self.context_encoder.state_dict(),
            "generator": self.generator.state_dict(),
            "score_net": self.score_net.state_dict(),
            "sigma_data": self.sigma_data,
            "sigma_data_resid": self.sigma_data_resid,
            "D_latent": self.D_latent,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ck["encoder"])
        self.context_encoder.load_state_dict(ck["context_encoder"])
        self.generator.load_state_dict(ck["generator"])
        # score_net registers 'st_dist_bin_edges' as a None-valued buffer (absent
        # from a fresh state_dict); assign it first so the full dict loads strictly.
        sn = ck["score_net"]
        edges = sn.get("st_dist_bin_edges", None)
        if edges is not None:
            self.score_net.st_dist_bin_edges = edges.to(self.device)
        self.score_net.load_state_dict(sn)
        self.sigma_data = ck.get("sigma_data")
        self.sigma_data_resid = ck.get("sigma_data_resid")
        self.freeze_encoder()
        return self
