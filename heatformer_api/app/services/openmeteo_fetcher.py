"""
Open-Meteo meteorological data fetcher.

Fetches entirely live weather data for Raipur from the Open-Meteo API
(https://open-meteo.com) — no API key required.

All values used for thermal-comfort computation come directly from
Open-Meteo responses; nothing is assumed or hard-coded:

  temperature_2m          °C   → Ta (air temperature)
  windspeed_10m           km/h → wind
  relativehumidity_2m     %    → RH
  shortwave_radiation     W m⁻² × Δt → ssrd_day  (sum of today's hours)
  terrestrial_radiation   W m⁻² × Δt → strd_day  (sum of today's hours)
  soil_temperature_0cm    °C   → lst_celsius  (replaces any proxy/offset)

Thermal comfort indices are then computed from the above live fields:
  MRT   – compute_tmrt(ssrd_day, strd_day, lst_celsius, ta_mean, …)
  PET   – compute_pet(…)
  UTCI  – compute_utci(…)
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import httpx

from app.config import get_settings
from app.utils.thermal import compute_tmrt, compute_pet, compute_utci

log = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Hourly variables – all fetched live from Open-Meteo
_HOURLY_VARS = [
    "temperature_2m",            # Air temperature at 2 m  (°C)
    "relativehumidity_2m",       # Relative humidity       (%)
    "windspeed_10m",             # Wind speed at 10 m      (km/h)
    "shortwave_radiation",       # Global horizontal irradiance  (W m⁻²)
    "terrestrial_radiation",     # Downwelling longwave radiation (W m⁻²)
    "soil_temperature_0cm",      # Skin / surface temperature     (°C)
]


def _build_params(lat: float, lon: float) -> dict[str, Any]:
    """Construct Open-Meteo query parameters for today's hourly data."""
    today = datetime.date.today().isoformat()
    return {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(_HOURLY_VARS),
        "timezone":   "Asia/Kolkata",
        "start_date": today,
        "end_date":   today,
        "models":     "best_match",   # Open-Meteo auto-selects best model
        "wind_speed_unit": "kmh",
    }


def _current_hour_index(times: list[str]) -> int:
    """
    Return the index of the latest available (past or present) hourly slot.
    Walks backward to skip future slots that may contain NaN/None.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
    for i in range(len(times) - 1, -1, -1):
        if times[i] <= now_str:
            return i
    return 0


async def fetch_openmeteo_current(
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    """
    Async fetch of current meteorological figures from Open-Meteo.

    Every value in the returned dict is derived exclusively from live
    Open-Meteo data — no offsets, proxies, or fixed fallbacks.

    Returns
    -------
    dict matching the MeteoData schema:
        timestamp, temp_mean_c, windspeed_kmh, humidity_percent,
        mrt_c, pet_c, utci_c
    """
    s   = get_settings()
    lat = lat if lat is not None else s.RAIPUR_LAT
    lon = lon if lon is not None else s.RAIPUR_LON

    params = _build_params(lat, lon)

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(_OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        payload: dict = resp.json()

    hourly: dict     = payload["hourly"]
    times: list[str] = hourly["time"]
    idx: int         = _current_hour_index(times)

    log.debug("Open-Meteo: using slot index=%d  time=%s", idx, times[idx])

    # ── Live meteorological values at current hour ────────────────────────
    ta:       float = float(hourly["temperature_2m"][idx])
    rh:       float = float(hourly["relativehumidity_2m"][idx])
    wind_kmh: float = float(hourly["windspeed_10m"][idx])
    wind_ms:  float = wind_kmh / 3.6

    # Land Surface Temperature from Open-Meteo soil skin layer (no assumption)
    lst_c: float = float(hourly["soil_temperature_0cm"][idx])

    # ── Radiation: sum of all observed hours today up to current slot ─────
    # Open-Meteo gives instantaneous W m⁻².  Multiplying by 3600 converts
    # each hourly reading to J m⁻² for that hour; the sum gives the daily
    # accumulated energy actually measured, not estimated.
    def _daily_sum(key: str) -> float:
        return sum(
            float(v) * 3600.0
            for v in hourly[key][: idx + 1]
            if v is not None
        )

    ssrd_day: float = _daily_sum("shortwave_radiation")
    strd_day: float = _daily_sum("terrestrial_radiation")

    # ── Thermal comfort indices ───────────────────────────────────────────
    today = datetime.date.today()
    doy   = today.timetuple().tm_yday

    mrt_c:  float = compute_tmrt(ssrd_day, strd_day, lst_c, ta, doy=doy, lat=lat, lon=lon)
    pet_c:  float = compute_pet(ta, rh, wind_ms, ssrd_day, strd_day, lst_c, doy=doy)
    utci_c: float = compute_utci(ta, rh, wind_ms, mrt_c)

    # ── ISO 8601 timestamp (IST, UTC+5:30) ──────────────────────────────
    ts_naive = datetime.datetime.strptime(times[idx], "%Y-%m-%dT%H:%M")
    ts_ist   = ts_naive.replace(
        tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    )
    timestamp = ts_ist.isoformat()

    return {
        "timestamp":        timestamp,
        "temp_mean_c":      round(ta, 2),
        "windspeed_kmh":    round(wind_kmh, 2),
        "humidity_percent": round(rh, 1),
        "mrt_c":            round(mrt_c, 2),
        "pet_c":            round(pet_c, 2),
        "utci_c":           round(utci_c, 2),
    }
