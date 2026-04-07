"""
Preprocessing service.

Handles:
  1. Loading / caching normalization statistics from Notebook 2 artefacts.
  2. Normalizing satellite tensors    (5, 64, 64).
  3. Normalizing meteorological arrays (8, 11).
  4. Converting raw arrays to PyTorch tensors batched for inference.

If normalization stats are not found on disk, sensible per-channel defaults
derived from Raipur climatology are used as a fallback.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

# ── Raipur climatology defaults (used when .npy stats are unavailable) ────────
#   Derived from Notebook 2 outputs for 2019-2024 training data
_DEFAULT_SAT_MEAN = np.array([35.0, 0.35, 0.05, -0.10, 0.40], dtype=np.float32)
_DEFAULT_SAT_STD  = np.array([ 8.0, 0.15, 0.12,  0.15, 0.18], dtype=np.float32)

_DEFAULT_MET_MEAN = np.array([
    40.2, 24.8, 32.5, 52.3,  2.1,
     4.8,  2.4, 1006.0, 48.3, 38.7, 36.5,
], dtype=np.float32)

_DEFAULT_MET_STD  = np.array([
     5.1,  4.2,  4.6, 17.8,  1.2,
     2.3, 12.5,    4.2,  8.1,  6.4,  5.9,
], dtype=np.float32)


class NormStats:
    """Holds per-channel mean/std for satellite and meteorological data."""

    def __init__(
        self,
        sat_mean: np.ndarray,
        sat_std:  np.ndarray,
        met_mean: np.ndarray,
        met_std:  np.ndarray,
    ):
        self.sat_mean = sat_mean.astype(np.float32)
        self.sat_std  = np.where(sat_std < 1e-6, 1.0, sat_std).astype(np.float32)
        self.met_mean = met_mean.astype(np.float32)
        self.met_std  = np.where(met_std < 1e-6, 1.0, met_std).astype(np.float32)

    @classmethod
    def from_dir(cls, directory: str) -> "NormStats":
        """Load stats saved by Notebook 2.  Falls back to defaults on failure."""
        d = Path(directory)
        try:
            return cls(
                sat_mean=np.load(d / "SAT_MEAN.npy"),
                sat_std =np.load(d / "SAT_STD.npy"),
                met_mean=np.load(d / "MET_MEAN.npy"),
                met_std =np.load(d / "MET_STD.npy"),
            )
        except Exception as exc:
            log.warning(
                "Normalization stats not found in %s (%s). "
                "Using built-in Raipur defaults.", directory, exc
            )
            return cls(
                sat_mean=_DEFAULT_SAT_MEAN,
                sat_std =_DEFAULT_SAT_STD,
                met_mean=_DEFAULT_MET_MEAN,
                met_std =_DEFAULT_MET_STD,
            )

    @classmethod
    def defaults(cls) -> "NormStats":
        return cls(
            sat_mean=_DEFAULT_SAT_MEAN, sat_std=_DEFAULT_SAT_STD,
            met_mean=_DEFAULT_MET_MEAN, met_std=_DEFAULT_MET_STD,
        )


# ── Preprocessing functions ────────────────────────────────────────────────────

def normalize_satellite(
    sat: np.ndarray,
    stats: NormStats,
) -> np.ndarray:
    """
    Parameters
    ----------
    sat : (5, 64, 64) raw float32

    Returns
    -------
    (5, 64, 64) normalized float32
    """
    sat = sat.astype(np.float32)
    for c in range(5):
        sat[c] = (sat[c] - stats.sat_mean[c]) / stats.sat_std[c]
    return np.nan_to_num(sat, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_meteo(
    met: np.ndarray,
    stats: NormStats,
) -> np.ndarray:
    """
    Parameters
    ----------
    met : (8, 11) raw float32

    Returns
    -------
    (8, 11) normalized float32
    """
    met = met.astype(np.float32)
    met = (met - stats.met_mean) / stats.met_std
    return np.nan_to_num(met, nan=0.0, posinf=0.0, neginf=0.0)


def to_batch(
    sat: np.ndarray,
    met: np.ndarray,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap (5,64,64) + (8,11) arrays into (1,5,64,64) + (1,8,11) tensors.
    """
    sat_t = torch.tensor(sat[np.newaxis], dtype=torch.float32).to(device)
    met_t = torch.tensor(met[np.newaxis], dtype=torch.float32).to(device)
    return sat_t, met_t


def preprocess_manual(
    satellite_data: list,
    meteo_data: list,
    stats: NormStats,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert validated request body arrays to normalized tensors.

    Parameters
    ----------
    satellite_data : raw [5][64][64] Python list (from request)
    meteo_data     : raw [8][11] Python list     (from request)

    Returns
    -------
    sat_t : (1, 5, 64, 64) torch.Tensor on device
    met_t : (1, 8, 11)     torch.Tensor on device
    """
    sat_np = np.array(satellite_data, dtype=np.float32)   # (5, 64, 64)
    met_np = np.array(meteo_data,     dtype=np.float32)   # (8, 11)

    sat_norm = normalize_satellite(sat_np, stats)
    met_norm = normalize_meteo(met_np, stats)

    return to_batch(sat_norm, met_norm, device)
