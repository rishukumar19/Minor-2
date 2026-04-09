"""Inference API router."""

import datetime as dt
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.schemas.requests import (
    AutoInferenceResponse,
    ManualInferenceRequest,
    ManualInferenceResponse,
)
from app.services.cache import inference_cache
from app.services.era5_fetcher import fetch_era5_window
from app.services.gee_fetcher import fetch_satellite_tensor
from app.services.inference import predict_from_arrays, predict_from_request

log = logging.getLogger(__name__)
_s = get_settings()

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])

_AUTO_CACHE_KEY = "inference:auto"


def _safe_model_path_note() -> str:
    return "model=" + (_s.MODEL_PATH or "<unset>")


def _assert_real_pipeline_configured() -> None:
    if not _s.GEE_PROJECT or _s.GEE_PROJECT == "your-gee-project-id":
        raise HTTPException(
            status_code=400,
            detail="GEE_PROJECT is not configured. Set a valid project in .env.",
        )
    if not _s.CDS_TOKEN:
        raise HTTPException(
            status_code=400,
            detail="CDS_TOKEN is not configured. Set a valid Copernicus token in .env.",
        )


def _shift_month(year: int, month: int, offset_back: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) - offset_back
    return total // 12, (total % 12) + 1


def _fetch_satellite_with_backoff(target_date: dt.date) -> tuple[np.ndarray, str]:
    """Try recent months until a valid satellite tensor is available."""
    errors: list[str] = []
    for offset in range(0, 6):
        year, month = _shift_month(target_date.year, target_date.month, offset)
        try:
            sat = fetch_satellite_tensor(
                project=_s.GEE_PROJECT,
                lat=_s.RAIPUR_LAT,
                lon=_s.RAIPUR_LON,
                delta=_s.ROI_DELTA,
                year=year,
                month=month,
                tile=_s.TILE,
            )
            return sat, f"{year:04d}-{month:02d}"
        except Exception as exc:
            errors.append(f"{year:04d}-{month:02d}: {exc}")

    raise RuntimeError(
        "Satellite fetch failed for recent months. "
        + " | ".join(errors[-3:])
    )


def _build_auto_inputs_real() -> tuple[np.ndarray, np.ndarray, dt.date, str]:
    """Fetch real satellite and ERA5 arrays for auto inference."""
    _assert_real_pipeline_configured()

    # ERA5 reanalysis is delayed by around 5 days.
    target_date = dt.date.today() - dt.timedelta(days=5)

    sat_raw, sat_month = _fetch_satellite_with_backoff(target_date)

    met_raw = fetch_era5_window(
        token=_s.CDS_TOKEN,
        lat=_s.RAIPUR_LAT,
        lon=_s.RAIPUR_LON,
        delta=_s.ROI_DELTA,
        target_date=target_date,
        window_days=_s.MET_WINDOW,
        lst_celsius=float(sat_raw[0].mean()),
    )

    return sat_raw.astype(np.float32), met_raw.astype(np.float32), target_date, sat_month


@router.get(
    "/auto",
    response_model=AutoInferenceResponse,
    summary="Run Fully Automated HeatFormer Inference",
)
@limiter.limit(_s.RATE_LIMIT_AUTO)
async def auto_inference(request: Request) -> AutoInferenceResponse:
    """Fetch real data sources, preprocess, and run inference."""
    if _AUTO_CACHE_KEY in inference_cache:
        return AutoInferenceResponse(
            source="cache",
            message="Cached real-data inference result served.",
            risk_map=inference_cache[_AUTO_CACHE_KEY],
        )

    try:
        sat_raw, met_raw, target_date, sat_month = _build_auto_inputs_real()
        risk_map = predict_from_arrays(sat_raw, met_raw)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Auto inference failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "Real-data auto inference failed. Check GEE project access and CDS token. "
                f"Reason: {exc} ({_safe_model_path_note()})"
            ),
        ) from exc

    inference_cache[_AUTO_CACHE_KEY] = risk_map

    return AutoInferenceResponse(
        source="fresh_inference",
        message=(
            "Real data fetched from GEE + ERA5 and inferenced successfully "
            f"(ERA5 target date: {target_date.isoformat()}, satellite month: {sat_month})."
        ),
        risk_map=risk_map,
    )


@router.post(
    "/manual",
    response_model=ManualInferenceResponse,
    summary="Run Manual HeatFormer Inference",
)
@limiter.limit(_s.RATE_LIMIT_MANUAL)
async def manual_inference(
    request: Request,
    payload: ManualInferenceRequest,
) -> ManualInferenceResponse:
    """Run inference from user-provided arrays."""
    try:
        risk_map = predict_from_request(
            satellite_data=payload.satellite_data,
            meteo_data=payload.meteo_data,
        )
    except Exception as exc:
        log.exception("Manual inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Manual inference failed: {exc}") from exc

    return ManualInferenceResponse(status="success", risk_map=risk_map)
