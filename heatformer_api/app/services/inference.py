"""
Inference service.

Manages a single HeatFormer model instance loaded at startup.
Provides thread-safe synchronous predict wrappers for use in
FastAPI route handlers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import get_settings
from app.models.heatformer import HeatFormer, load_model
from app.services.preprocessing import NormStats, normalize_satellite, \
    normalize_meteo, to_batch

log    = logging.getLogger(__name__)
_model: Optional[HeatFormer] = None
_stats: Optional[NormStats]  = None
_device: str                 = "cpu"


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def startup() -> None:
    """
    Load model + normalization stats.
    Called from FastAPI lifespan on startup.
    Gracefully falls back to a randomly-initialised model if no checkpoint found.
    """
    global _model, _stats, _device

    cfg = get_settings()

    # Safely determine the device — fall back to CPU if CUDA isn't available
    requested = cfg.DEVICE.lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            log.warning(
                "DEVICE=cuda requested but CUDA is not available. "
                "Falling back to CPU."
            )
            _device = "cpu"
    else:
        _device = "cpu"

    # Normalization stats
    _stats = NormStats.from_dir(cfg.NORM_STATS_DIR)

    # Model
    ckpt = Path(cfg.MODEL_PATH)
    if ckpt.exists():
        log.info("Loading HeatFormer from %s", ckpt)
        _model = load_model(str(ckpt), device=_device)
    else:
        log.warning(
            "Checkpoint not found at %s — "
            "using randomly-initialised HeatFormer.", ckpt
        )
        _model = HeatFormer().to(_device)
        _model.eval()

    log.info("HeatFormer ready on %s", _device)


def get_model() -> HeatFormer:
    if _model is None:
        raise RuntimeError("Model not initialised — call startup() first.")
    return _model


def get_stats() -> NormStats:
    if _stats is None:
        raise RuntimeError("Normalization stats not initialised.")
    return _stats


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_from_arrays(
    sat_raw: np.ndarray,    # (5, 64, 64)  unnormalized
    met_raw: np.ndarray,    # (8, 11)      unnormalized
) -> list[list[int]]:
    """
    Normalize + run forward pass.
    Returns a 64×64 risk map as a nested Python list.
    """
    model  = get_model()
    stats  = get_stats()

    sat_n  = normalize_satellite(sat_raw, stats)
    met_n  = normalize_meteo(met_raw, stats)
    sat_t, met_t = to_batch(sat_n, met_n, _device)

    risk_map, _ = model.predict(sat_t, met_t)
    return risk_map


def predict_from_normalized(
    sat_norm: np.ndarray,   # (5, 64, 64)  already normalized
    met_norm: np.ndarray,   # (8, 11)      already normalized
) -> list[list[int]]:
    """
    Skip normalization step — used when stats are already applied upstream
    (e.g. manual endpoint where user may supply pre-normalized data).
    """
    model = get_model()
    sat_t, met_t = to_batch(sat_norm, met_norm, _device)
    risk_map, _  = model.predict(sat_t, met_t)
    return risk_map


def predict_from_request(
    satellite_data: list,   # [5][64][64]
    meteo_data: list,       # [8][11]
) -> list[list[int]]:
    """
    Entry point for the manual inference endpoint.
    Applies normalization then runs the model.
    """
    sat_np = np.array(satellite_data, dtype=np.float32)
    met_np = np.array(meteo_data,     dtype=np.float32)
    return predict_from_arrays(sat_np, met_np)
