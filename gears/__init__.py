"""
GEARS — Geometry-First Generative Spatial Single-Cell Reconstruction.

Clean, self-contained reimplementation extracted from the research codebase.
This package is built up one stage at a time.

Stage A (this module set): the domain-invariant shared expression encoder that
aligns spatial-transcriptomics (ST) spots and dissociated single cells (SC) into
a common embedding space, used frozen by all downstream stages.

Public API:
    SharedEncoder      — the encoder backbone f_theta (frozen after Stage A).
    EncoderConfig      — all Stage-A hyperparameters (with sensible defaults).
    train_encoder      — train the shared encoder (VICReg + domain adversary + ...).
"""

from .encoder import SharedEncoder, SlideDiscriminator
from .train_encoder import EncoderConfig, train_encoder
from .model import GEARS

__all__ = [
    "GEARS",
    "SharedEncoder",
    "SlideDiscriminator",
    "EncoderConfig",
    "train_encoder",
]
