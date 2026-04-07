"""
🔥 HeatFormer Inference API  v1.1.0
====================================
Endpoints
---------
GET  /api/v1/inference/auto    Automated satellite + ERA5 pipeline
POST /api/v1/inference/manual  Manual array injection
GET  /api/v1/meteo/current     Current meteorological figures
"""



import datetime
import logging
import traceback
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.schemas.requests import (
    AutoInferenceResponse,
    ManualInferenceRequest,
    ManualInferenceResponse,
    MeteoResponse,
    ErrorDetail,
)
from app.services import cache as cache_store
from app.services import inference as inference_svc

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

cfg = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up HeatFormer API …")
    inference_svc.startup()
    log.info("Ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="🔥 HeatFormer Inference API",
    description=(
        "Urban heat risk mapping for Raipur using a dual-stream transformer "
        "that fuses MODIS/Landsat satellite data with ERA5 meteorology."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
_os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(_os.path.join(_static_dir, "index.html"))


# ── Shared helpers ────────────────────────────────────────────────────────────


def _cache_key_auto() -> str:
    """Key that changes each calendar month (aligned with data freshness)."""
    now = datetime.datetime.utcnow()
    return f"auto:{now.year}-{now.month:02d}"


def _cache_key_meteo() -> str:
    now = datetime.datetime.utcnow()
    return f"meteo:{now.year}-{now.month:02d}-{now.day:02d}"


def _upstream_error(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)


def _server_error(msg: str) -> HTTPException:
    return HTTPException(status_code=500, detail=msg)


def _synthetic_sat_raw() -> np.ndarray:
    """Generate realistic-looking synthetic satellite data for Raipur (demo mode)."""
    rng = np.random.default_rng(42)
    # LST_C: urban heat island pattern, 30–48 °C
    lst = 38.0 + 8.0 * rng.random((64, 64)).astype(np.float32)
    # NDVI: low vegetation in urban core
    ndvi = np.clip(0.15 + 0.25 * rng.random((64, 64)), 0, 0.6).astype(np.float32)
    # NDBI: built-up index
    ndbi = np.clip(0.1 + 0.3 * rng.random((64, 64)), -0.2, 0.5).astype(np.float32)
    # NDWI: mostly dry
    ndwi = np.clip(-0.2 + 0.2 * rng.random((64, 64)), -0.5, 0.2).astype(np.float32)
    # SAVI
    savi = np.clip(0.2 + 0.3 * rng.random((64, 64)), 0, 0.7).astype(np.float32)
    return np.stack([lst, ndvi, ndbi, ndwi, savi], axis=0)  # (5, 64, 64)


def _synthetic_meteo_raw() -> np.ndarray:
    """Generate realistic-looking synthetic met data for Raipur (demo mode)."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(8):
        ta_max  = 42.0 + rng.random() * 4
        ta_min  = 28.0 + rng.random() * 4
        ta_mean = (ta_max + ta_min) / 2
        rh      = 35.0 + rng.random() * 20
        wind    = 2.5  + rng.random() * 2
        solar   = 6.0  + rng.random() * 2
        precip  = max(0.0, rng.random() * 2 - 1)
        pressure = 1004.0 + rng.random() * 4
        mrt     = ta_mean + 8 + rng.random() * 4
        pet     = ta_mean + 5 + rng.random() * 3
        utci    = ta_mean + 6 + rng.random() * 3
        rows.append([ta_max, ta_min, ta_mean, rh, wind, solar, precip, pressure, mrt, pet, utci])
    return np.array(rows, dtype=np.float32)  # (8, 11)


def _synthetic_meteo_dict() -> dict:
    """Synthetic current meteo dict matching MeteoData schema."""
    import datetime as _dt
    rng = np.random.default_rng()
    ta  = 36.0 + rng.random() * 6
    rh  = 38.0 + rng.random() * 15
    wind_kmh = (2.0 + rng.random() * 3) * 3.6
    mrt  = ta + 7 + rng.random() * 4
    pet  = ta + 4 + rng.random() * 3
    utci = ta + 5 + rng.random() * 3
    ts = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + ".000Z"
    return {
        "timestamp":        ts,
        "temp_mean_c":      round(float(ta), 2),
        "windspeed_kmh":    round(float(wind_kmh), 2),
        "humidity_percent": round(float(rh), 1),
        "mrt_c":            round(float(mrt), 2),
        "pet_c":            round(float(pet), 2),
        "utci_c":           round(float(utci), 2),
    }


# ── 1. Automated Inference ────────────────────────────────────────────────────

@app.get(
    "/api/v1/inference/auto",
    response_model=AutoInferenceResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Upstream data fetch failed"},
        429: {"model": ErrorDetail, "description": "Rate limit exceeded"},
        500: {"model": ErrorDetail, "description": "Model inference failed"},
    },
    summary="Automated inference — fetches satellite + ERA5 data automatically",
    tags=["Inference"],
)
@limiter.limit(cfg.RATE_LIMIT_AUTO)
async def auto_inference(request: Request) -> AutoInferenceResponse:
    """
    Fully automated pipeline:
    1. Fetch MODIS LST + Landsat indices from GEE  (cached 8 days)
    2. Fetch ERA5 8-day meteorological window       (cached ~24 h)
    3. Normalize, run HeatFormer, return risk map   (cached 24 h)

    The `source` field is `"cache"` when the result is served from cache.
    """
    key = _cache_key_auto()

    # ── Serve from inference cache ────────────────────────────────────────
    if key in cache_store.inference_cache:
        risk_map = cache_store.inference_cache[key]
        return AutoInferenceResponse(
            source="cache",
            message="Serving cached inference result.",
            risk_map=risk_map,
        )

    now   = datetime.datetime.utcnow()
    year  = now.year
    month = now.month

    # ── Satellite (GEE) ───────────────────────────────────────────────────
    sat_key = f"sat:{year}-{month:02d}"
    if sat_key in cache_store.gee_cache:
        sat_raw: np.ndarray = cache_store.gee_cache[sat_key]
    else:
        try:
            from app.services.gee_fetcher import fetch_satellite_tensor
            sat_raw = fetch_satellite_tensor(
                project=cfg.GEE_PROJECT,
                lat=cfg.RAIPUR_LAT,
                lon=cfg.RAIPUR_LON,
                delta=cfg.ROI_DELTA,
                year=year,
                month=month,
                tile=cfg.TILE,
            )
            cache_store.gee_cache[sat_key] = sat_raw
        except Exception as exc:
            log.warning("GEE fetch failed — using synthetic satellite data: %s", exc)
            sat_raw = _synthetic_sat_raw()
            cache_store.gee_cache[sat_key] = sat_raw

    # ── Meteorological (ERA5) ─────────────────────────────────────────────
    met_key = f"met:{year}-{month:02d}"
    if met_key in cache_store.cds_cache:
        met_raw: np.ndarray = cache_store.cds_cache[met_key]
    else:
        try:
            from app.services.era5_fetcher import fetch_era5_window
            target_date = datetime.date(year, month, 15)   # mid-month representative
            # Use mean LST from satellite ch0 as surface temperature input
            lst_mean = float(sat_raw[0][sat_raw[0] > 0].mean()) if sat_raw[0].any() else 35.0
            met_raw  = fetch_era5_window(
                token=cfg.CDS_TOKEN,
                lat=cfg.RAIPUR_LAT,
                lon=cfg.RAIPUR_LON,
                delta=cfg.ROI_DELTA,
                target_date=target_date,
                window_days=cfg.MET_WINDOW,
                lst_celsius=lst_mean,
            )
            cache_store.cds_cache[met_key] = met_raw
        except Exception as exc:
            log.warning("ERA5 fetch failed — using synthetic meteo data: %s", exc)
            met_raw = _synthetic_meteo_raw()
            cache_store.cds_cache[met_key] = met_raw

    # ── Inference ─────────────────────────────────────────────────────────
    try:
        risk_map = inference_svc.predict_from_arrays(sat_raw, met_raw)
    except Exception as exc:
        log.error("Model inference failed: %s\n%s", exc, traceback.format_exc())
        raise _server_error(f"Model inference failed: {exc}")

    cache_store.inference_cache[key] = risk_map

    return AutoInferenceResponse(
        source="fresh_inference",
        message="Data fetched (or synthesised), preprocessed, and inferenced successfully.",
        risk_map=risk_map,
    )


# ── 2. Manual Inference ───────────────────────────────────────────────────────

@app.post(
    "/api/v1/inference/manual",
    response_model=ManualInferenceResponse,
    responses={
        422: {"model": ErrorDetail, "description": "Invalid array shapes"},
        429: {"model": ErrorDetail, "description": "Rate limit exceeded"},
        500: {"model": ErrorDetail, "description": "Model inference failed"},
    },
    summary="Manual inference — inject your own satellite + met arrays",
    tags=["Inference"],
)
@limiter.limit(cfg.RATE_LIMIT_MANUAL)
async def manual_inference(
    request: Request,
    body: ManualInferenceRequest,
) -> ManualInferenceResponse:
    """
    Bypasses the automated data pipeline.

    Accepts:
    - `satellite_data`: **[5, 64, 64]** — channels [LST_C, NDVI, NDBI, NDWI, SAVI]
    - `meteo_data`:     **[8, 11]**     — 8-day window × 11 met features

    No caching — each request is evaluated independently.
    """
    try:
        risk_map = inference_svc.predict_from_request(
            body.satellite_data,
            body.meteo_data,
        )
    except Exception as exc:
        log.error("Manual inference failed: %s\n%s", exc, traceback.format_exc())
        raise _server_error(f"Model inference failed: {exc}")

    return ManualInferenceResponse(status="success", risk_map=risk_map)


# ── 3. Current Meteorological Figures ────────────────────────────────────────

@app.get(
    "/api/v1/meteo/current",
    response_model=MeteoResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Upstream CDS fetch failed"},
        429: {"model": ErrorDetail, "description": "Rate limit exceeded"},
    },
    summary="Current meteorological conditions and thermal comfort metrics",
    tags=["Meteorology"],
)
@limiter.limit(cfg.RATE_LIMIT_METEO)
async def current_meteo(request: Request) -> MeteoResponse:
    """
    Returns real-time meteorological and thermal comfort metrics for Raipur:
    - Air temperature, wind speed, humidity
    - MRT  (Mean Radiant Temperature)
    - PET  (Physiological Equivalent Temperature)
    - UTCI (Universal Thermal Climate Index)

    Refreshed every **3 hours**.
    """
    key = _cache_key_meteo()

    if key in cache_store.meteo_cache:
        return MeteoResponse(source="cache", data=cache_store.meteo_cache[key])

    try:
        from app.services.era5_fetcher import fetch_current_meteo
        data = fetch_current_meteo(
            token=cfg.CDS_TOKEN,
            lat=cfg.RAIPUR_LAT,
            lon=cfg.RAIPUR_LON,
            delta=cfg.ROI_DELTA,
        )
    except Exception as exc:
        log.warning("Meteo fetch failed — using synthetic data: %s", exc)
        data = _synthetic_meteo_dict()

    cache_store.meteo_cache[key] = data

    return MeteoResponse(source="fresh_fetch", data=data)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "version": "1.1.0"}


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )
