"""
Router: /api/v1/meteo/current

Provides real-time, human-centric meteorological and thermal comfort metrics.
Powered by Open-Meteo (https://open-meteo.com).

Rate limit : 60 requests/minute per IP  (via SlowAPI)
Cache      : 30min in-process TTLCache  (via cachetools)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.schemas.requests import MeteoData, MeteoResponse
from app.services.cache import meteo_cache
from app.services.openmeteo_fetcher import fetch_openmeteo_current

log = logging.getLogger(__name__)
_s  = get_settings()

# ── Rate limiter (shared key-function) ───────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1/meteo", tags=["Meteorological"])

_CACHE_KEY = "meteo:current"


@router.get(
    "/current",
    response_model=MeteoResponse,
    summary="Current Meteorological Figures",
    description=(
        "Returns real-time meteorological and thermal comfort metrics for Raipur "
        "sourced from the Open-Meteo API. Results are cached for 3 hours. "
        "Rate-limited to 60 requests per minute per IP."
    ),
    responses={
        200: {"description": "Successful response with current met data"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "Upstream Open-Meteo API error"},
    },
)
@limiter.limit(_s.RATE_LIMIT_METEO)
async def get_current_meteo(request: Request) -> MeteoResponse:
    """
    GET /api/v1/meteo/current

    - **No parameters required.**
    - Returns `source: "cache"` when the 3-hour cache is still valid.
    - Returns `source: "fresh_fetch"` after a cache miss.
    """
    # ── Cache hit ────────────────────────────────────────────────────────────
    if _CACHE_KEY in meteo_cache:
        log.debug("meteo/current: cache hit")
        cached: dict = meteo_cache[_CACHE_KEY]
        return MeteoResponse(source="cache", data=MeteoData(**cached))

    # ── Fresh fetch from Open-Meteo ──────────────────────────────────────────
    log.info("meteo/current: cache miss — fetching from Open-Meteo")
    try:
        data_dict = await fetch_openmeteo_current()
    except Exception as exc:
        log.exception("Open-Meteo fetch failed: %s", exc)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=502,
            detail=f"Upstream Open-Meteo API error: {exc}",
        ) from exc

    # Store in cache (TTL=3h is enforced by the TTLCache configuration)
    meteo_cache[_CACHE_KEY] = data_dict
    log.debug("meteo/current: stored in cache")

    return MeteoResponse(source="fresh_fetch", data=MeteoData(**data_dict))
