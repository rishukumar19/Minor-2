"""
Google Earth Engine data fetcher.

Fetches:
  - MODIS MOD11A2  → LST_C           (B1)
  - Landsat 8/9 C2 → NDVI, NDBI, NDWI, SAVI  (B1–B4)

Uses ee.data.computePixels() for synchronous in-memory download
(avoids Drive export / polling loop required in the notebook).
"""

from __future__ import annotations

import io
import datetime
import logging
import numpy as np
from skimage.transform import resize

log = logging.getLogger(__name__)

# GEE initialisation is deferred so the module imports cleanly even without
# credentials (unit-test / mock scenarios).
_ee_ready = False


def _init_gee(project: str) -> None:
    global _ee_ready
    if _ee_ready:
        return
    import ee
    ee.Initialize(project=project)
    _ee_ready = True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_roi(lat: float, lon: float, delta: float):
    """Build a rectangular ROI centred on (lat, lon)."""
    import ee
    return ee.Geometry.Rectangle(
        [lon - delta, lat - delta, lon + delta, lat + delta]
    )


def _compute_pixels_npy(
    image,
    roi,
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:4326",
) -> np.ndarray:
    """
    Download an EE image as a structured NumPy array via computePixels.

    Returns shape (height, width, n_bands) — channels-last.
    """
    import ee

    bounds = roi.bounds().getInfo()["coordinates"][0]
    west   = min(p[0] for p in bounds)
    south  = min(p[1] for p in bounds)
    east   = max(p[0] for p in bounds)
    north  = max(p[1] for p in bounds)

    scale_x = (east  - west)  / width
    scale_y = (north - south) / height

    request = {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions":      {"width": width, "height": height},
            "affineTransform": {
                "scaleX":      scale_x,
                "shearX":      0,
                "translateX":  west,
                "shearY":      0,
                "scaleY":      -scale_y,
                "translateY":  north,
            },
            "crsCode": crs,
        },
    }
    raw    = ee.data.computePixels(request)
    arr    = np.load(io.BytesIO(raw))      # structured dtype
    # Convert structured array → plain float32 (bands as fields)
    fields = arr.dtype.names
    planes = [arr[f].astype(np.float32) for f in fields]
    return np.stack(planes, axis=-1)      # (H, W, C)


# ── MODIS LST ─────────────────────────────────────────────────────────────────

def _mask_modis_qa(img):
    import ee
    qa   = img.select("QC_Day")
    good = qa.bitwiseAnd(3).eq(0)
    return img.updateMask(good)


def _scale_lst(img):
    lst = (
        img.select("LST_Day_1km")
           .multiply(0.02)
           .subtract(273.15)
           .rename("LST_C")
    )
    return lst.copyProperties(img, ["system:time_start"])


def fetch_modis_lst(
    project: str,
    lat: float,
    lon: float,
    delta: float,
    year: int,
    month: int,
    tile: int = 64,
) -> np.ndarray:
    """
    Returns LST_C as a (tile, tile) float32 array.
    Invalid values filled with 0.
    """
    import ee
    _init_gee(project)
    roi = _make_roi(lat, lon, delta)

    start = f"{year}-{month:02d}-01"
    end   = ee.Date(start).advance(1, "month").format("YYYY-MM-dd").getInfo()

    col = (
        ee.ImageCollection("MODIS/061/MOD11A2")
          .filterDate(start, end)
          .filterBounds(roi)
          .map(_mask_modis_qa)
          .map(_scale_lst)
    )
    img = col.median().clip(roi)

    data = _compute_pixels_npy(img, roi, width=tile, height=tile)  # (T, T, 1)
    lst  = data[:, :, 0]

    lst[lst < -50] = np.nan
    lst[lst >  70] = np.nan
    lst = np.nan_to_num(lst, nan=0.0)
    return lst.astype(np.float32)


# ── Landsat 8/9 indices ───────────────────────────────────────────────────────

def _mask_clouds(img):
    import ee
    qa     = img.select("QA_PIXEL")
    cloud  = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return img.updateMask(cloud.And(shadow))


def _compute_indices(img):
    optical = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
    img     = img.addBands(optical, None, True)
    ndvi    = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi    = img.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    ndwi    = img.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    nir, red = img.select("SR_B5"), img.select("SR_B4")
    savi    = (
        nir.subtract(red)
           .divide(nir.add(red).add(0.5))
           .multiply(1.5)
           .rename("SAVI")
    )
    return img.addBands([ndvi, ndbi, ndwi, savi]).select(
        ["NDVI", "NDBI", "NDWI", "SAVI"]
    )


def _build_landsat_col(col_id: str, start: str, end: str, roi):
    import ee
    return (
        ee.ImageCollection(col_id)
          .filterDate(start, end)
          .filterBounds(roi)
          .map(_mask_clouds)
          .map(_compute_indices)
          .select(["NDVI", "NDBI", "NDWI", "SAVI"])
    )


def fetch_landsat_indices(
    project: str,
    lat: float,
    lon: float,
    delta: float,
    year: int,
    month: int,
    tile: int = 64,
) -> np.ndarray:
    """
    Returns spectral indices as a (4, tile, tile) float32 array.
    Channels: [NDVI, NDBI, NDWI, SAVI]
    """
    import ee
    _init_gee(project)
    roi = _make_roi(lat, lon, delta)

    start = f"{year}-{month:02d}-01"
    end   = ee.Date(start).advance(1, "month").format("YYYY-MM-dd").getInfo()

    merged = (
        _build_landsat_col("LANDSAT/LC08/C02/T1_L2", start, end, roi)
          .merge(_build_landsat_col("LANDSAT/LC09/C02/T1_L2", start, end, roi))
    )

    n = merged.size().getInfo()
    if n == 0:
        log.warning("No Landsat scenes for %d-%02d — returning zeros", year, month)
        return np.zeros((4, tile, tile), dtype=np.float32)

    img  = merged.median().toFloat().clip(roi)
    data = _compute_pixels_npy(img, roi, width=tile, height=tile)  # (T, T, 4)

    bands = []
    for i in range(4):
        b = data[:, :, i].astype(np.float32)
        b = np.clip(b, -1.5, 1.5)
        b[np.abs(b) > 1.4] = np.nan
        b = np.nan_to_num(b, nan=0.0)
        bands.append(b)

    return np.stack(bands, axis=0)   # (4, tile, tile)


# ── Combined satellite tensor ─────────────────────────────────────────────────

def fetch_satellite_tensor(
    project: str,
    lat: float,
    lon: float,
    delta: float,
    year: int,
    month: int,
    tile: int = 64,
) -> np.ndarray:
    """
    Full 5-channel satellite tensor for HeatFormer.

    Returns shape (5, tile, tile) float32:
      ch0 = LST_C   ch1 = NDVI   ch2 = NDBI   ch3 = NDWI   ch4 = SAVI
    """
    lst     = fetch_modis_lst(project, lat, lon, delta, year, month, tile)
    indices = fetch_landsat_indices(project, lat, lon, delta, year, month, tile)

    return np.concatenate(
        [lst[np.newaxis], indices], axis=0
    ).astype(np.float32)     # (5, tile, tile)
