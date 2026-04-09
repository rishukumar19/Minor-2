"""Inference service lifecycle + prediction helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import get_settings
from app.models.heatformer import HeatFormer, load_model
from app.services.preprocessing import (
    NormStats,
    normalize_meteo,
    normalize_satellite,
    to_batch,
)

log = logging.getLogger(__name__)
_model: Optional[HeatFormer] = None
_stats: Optional[NormStats] = None
_device: str = "cpu"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_checkpoint_path(cfg) -> Optional[Path]:
    """Pick the first existing checkpoint from config + common defaults."""
    candidates: list[Path] = []

    explicit = Path(cfg.MODEL_PATH).expanduser()
    candidates.append(explicit)
    if not explicit.is_absolute():
        candidates.append(_project_root() / explicit)

    weights_dir = Path(cfg.NORM_STATS_DIR).expanduser()
    if not weights_dir.is_absolute():
        weights_dir = _project_root() / weights_dir

    for name in (
        "heatformer_final.pt",
        "heatformer_best.pth",
        "heatformer.pth",
        "model.pt",
        "model.pth",
    ):
        candidates.append(weights_dir / name)

    candidates.extend(sorted(weights_dir.glob("*.pt")))
    candidates.extend(sorted(weights_dir.glob("*.pth")))

    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None


def startup() -> None:
    """Load model + normalization statistics at API startup."""
    global _model, _stats, _device

    cfg = get_settings()

    requested = cfg.DEVICE.lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            log.warning(
                "DEVICE=cuda requested but CUDA is not available. Falling back to CPU."
            )
            _device = "cpu"
    else:
        _device = "cpu"

    _stats = NormStats.from_dir(cfg.NORM_STATS_DIR)

    ckpt = _resolve_checkpoint_path(cfg)
    if ckpt is not None:
        log.info("Loading HeatFormer from %s", ckpt)
        _model = load_model(str(ckpt), device=_device)
    else:
        log.warning(
            "Checkpoint not found (MODEL_PATH=%s). Using randomly initialised model.",
            cfg.MODEL_PATH,
        )
        _model = HeatFormer().to(_device)
        _model.eval()

    log.info("HeatFormer ready on %s", _device)


def get_model() -> HeatFormer:
    if _model is None:
        raise RuntimeError("Model not initialised - call startup() first.")
    return _model


def get_stats() -> NormStats:
    if _stats is None:
        raise RuntimeError("Normalization stats not initialised.")
    return _stats


def predict_from_arrays(
    sat_raw: np.ndarray,
    met_raw: np.ndarray,
) -> list[list[int]]:
    """Normalize arrays, run forward pass, and return a 64x64 class map."""
    model = get_model()
    stats = get_stats()

    sat_n = normalize_satellite(sat_raw, stats)
    met_n = normalize_meteo(met_raw, stats)
    sat_t, met_t = to_batch(sat_n, met_n, _device)

    risk_map, _ = model.predict(sat_t, met_t)
    return risk_map


def predict_from_normalized(
    sat_norm: np.ndarray,
    met_norm: np.ndarray,
) -> list[list[int]]:
    """Run forward pass on already-normalized arrays."""
    model = get_model()
    sat_t, met_t = to_batch(sat_norm, met_norm, _device)
    risk_map, _ = model.predict(sat_t, met_t)
    return risk_map


def predict_from_request(
    satellite_data: list,
    meteo_data: list,
) -> list[list[int]]:
    """Convert validated request arrays and run model prediction."""
    sat_np = np.array(satellite_data, dtype=np.float32)
    met_np = np.array(meteo_data, dtype=np.float32)
    return predict_from_arrays(sat_np, met_np)