"""
ERA5 meteorological data fetcher via the Copernicus CDS API.

Downloads ERA5 single-level reanalysis for the Raipur bounding box,
aggregates to daily values, and computes the 11 HeatFormer met features:

  [0]  temp_max      °C
  [1]  temp_min      °C
  [2]  temp_mean     °C
  [3]  humidity      %
  [4]  wind          m s⁻¹
  [5]  solar         kWh m⁻²
  [6]  precip        mm
  [7]  pressure      hPa
  [8]  mrt           °C   (Mean Radiant Temperature)
  [9]  pet           °C   (Physiological Equivalent Temperature)
  [10] utci          °C   (Universal Thermal Climate Index)
"""

from __future__ import annotations

import io
import logging
import os
import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ERA5 variables to request
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]

MET_FEATURE_NAMES = [
    "temp_max", "temp_min", "temp_mean",
    "humidity", "wind", "solar",
    "precip", "pressure", "mrt", "pet", "utci",
]


def _write_cdsapirc(token: str) -> None:
    """Write ~/.cdsapirc from the provided token."""
    rc = Path.home() / ".cdsapirc"
    rc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {token}\n")


def fetch_era5_window(
    token: str,
    lat: float,
    lon: float,
    delta: float,
    target_date: datetime.date,
    window_days: int = 8,
    lst_celsius: float = 35.0,  # fallback if MODIS LST unavailable
) -> np.ndarray:
    """
    Download ERA5 for [target_date - window_days + 1 … target_date] and
    return a (window_days, 11) float32 array of meteorological features.

    Parameters
    ----------
    token       : CDS API token
    lat, lon    : centre of study area
    delta       : half-width of bounding box in degrees
    target_date : last day of the window
    window_days : length of the rolling window (default 8)
    lst_celsius : MODIS LST for MRT/PET computation (°C)
    """
    import cdsapi
    import xarray as xr
    from app.utils.thermal import compute_tmrt, compute_pet, compute_utci

    _write_cdsapirc(token)

    start = target_date - datetime.timedelta(days=window_days - 1)
    end   = target_date

    years  = sorted({start.year, end.year})
    months = sorted({d.month for d in pd.date_range(start, end, freq="D")})
    days   = sorted({d.day   for d in pd.date_range(start, end, freq="D")})

    # ERA5 bounding box: [N, W, S, E]
    area = [lat + delta, lon - delta, lat - delta, lon + delta]

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        c = cdsapi.Client(quiet=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type":    ["reanalysis"],
                "variable":        ERA5_VARIABLES,
                "year":            [str(y) for y in years],
                "month":           [f"{m:02d}" for m in months],
                "day":             [f"{d:02d}" for d in days],
                "time":            ["00:00", "06:00", "12:00", "18:00"],
                "area":            area,
                "data_format":     "netcdf",
                "download_format": "unarchived",
            },
            tmp_path,
        )

        ds = xr.open_dataset(tmp_path, engine="netcdf4")
        # Nearest grid point to study centre
        pt = ds.sel(latitude=lat, longitude=lon, method="nearest")

        try:
            times = pd.to_datetime(pt["valid_time"].values)
        except Exception:
            times = pd.to_datetime(pt["time"].values)

        t2m  = pt["t2m"].values  - 273.15
        d2m  = pt["d2m"].values  - 273.15
        u10  = pt["u10"].values
        v10  = pt["v10"].values
        msl  = pt["msl"].values  / 100.0
        ssrd = pt["ssrd"].values
        tp   = pt["tp"].values   * 1000.0
        strd = pt["strd"].values
        ds.close()

        rows: list[list[float]] = []
        target_dates = pd.date_range(start, end, freq="D")

        for date in target_dates:
            mask = np.array([t.date() == date.date() for t in times])
            if mask.sum() == 0:
                rows.append([0.0] * 11)
                continue

            ta_max  = float(t2m[mask].max())
            ta_min  = float(t2m[mask].min())
            ta_mean = float(t2m[mask].mean())
            wind    = float(np.sqrt(u10[mask] ** 2 + v10[mask] ** 2).mean())

            # Relative humidity via Magnus formula
            es  = 6.112 * np.exp(17.67 * ta_mean / (ta_mean + 243.5))
            td  = float(d2m[mask].mean())
            ea  = 6.112 * np.exp(17.67 * td / (td + 243.5))
            rh  = float(np.clip(100 * ea / es, 0, 100))

            ssrd_day = float(ssrd[mask].sum())
            strd_day = float(strd[mask].sum())
            solar    = ssrd_day / 3.6e6        # J m⁻² → kWh m⁻²

            doy = date.day_of_year

            mrt_c  = compute_tmrt(ssrd_day, strd_day, lst_celsius, ta_mean, doy=doy,
                                  lat=lat, lon=lon)
            pet_c  = compute_pet(ta_mean, rh, wind, ssrd_day, strd_day,
                                 lst_celsius, doy=doy)
            utci_c = compute_utci(ta_mean, rh, wind, mrt_c)

            rows.append([
                ta_max, ta_min, ta_mean,
                rh, wind, solar,
                float(tp[mask].sum()),
                float(msl[mask].mean()),
                mrt_c, pet_c, utci_c,
            ])

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    arr = np.array(rows, dtype=np.float32)   # (window_days, 11)

    # Pad or trim to exactly window_days rows
    if arr.shape[0] < window_days:
        pad = np.zeros((window_days - arr.shape[0], 11), dtype=np.float32)
        arr = np.vstack([pad, arr])
    elif arr.shape[0] > window_days:
        arr = arr[-window_days:]

    return arr.astype(np.float32)


def fetch_current_meteo(
    token: str,
    lat: float,
    lon: float,
    delta: float,
    lst_celsius: float = 35.0,
) -> dict:
    """
    Fetch today's meteorological figures for the /meteo/current endpoint.

    Returns a dict matching the MeteoData schema.
    """
    import cdsapi
    import xarray as xr
    from app.utils.thermal import compute_tmrt, compute_pet, compute_utci

    _write_cdsapirc(token)

    today = datetime.date.today()
    # ERA5 has ~5-day latency — use yesterday to guarantee data availability
    target = today - datetime.timedelta(days=1)
    area   = [lat + delta, lon - delta, lat - delta, lon + delta]

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        c = cdsapi.Client(quiet=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type":    ["reanalysis"],
                "variable":        ERA5_VARIABLES,
                "year":            [str(target.year)],
                "month":           [f"{target.month:02d}"],
                "day":             [f"{target.day:02d}"],
                "time":            ["00:00", "06:00", "12:00", "18:00"],
                "area":            area,
                "data_format":     "netcdf",
                "download_format": "unarchived",
            },
            tmp_path,
        )

        ds  = xr.open_dataset(tmp_path, engine="netcdf4")
        pt  = ds.sel(latitude=lat, longitude=lon, method="nearest")
        try:
            times = pd.to_datetime(pt["valid_time"].values)
        except Exception:
            times = pd.to_datetime(pt["time"].values)

        t2m  = pt["t2m"].values  - 273.15
        d2m  = pt["d2m"].values  - 273.15
        u10  = pt["u10"].values
        v10  = pt["v10"].values
        msl  = pt["msl"].values  / 100.0
        ssrd = pt["ssrd"].values
        strd = pt["strd"].values
        ds.close()

        ta_mean = float(t2m.mean())
        wind_ms = float(np.sqrt(u10 ** 2 + v10 ** 2).mean())
        wind_kmh = wind_ms * 3.6

        es  = 6.112 * np.exp(17.67 * ta_mean / (ta_mean + 243.5))
        td  = float(d2m.mean())
        ea  = 6.112 * np.exp(17.67 * td / (td + 243.5))
        rh  = float(np.clip(100 * ea / es, 0, 100))

        ssrd_day = float(ssrd.sum())
        strd_day = float(strd.sum())

        doy    = target.timetuple().tm_yday
        mrt_c  = compute_tmrt(ssrd_day, strd_day, lst_celsius, ta_mean, doy=doy,
                               lat=lat, lon=lon)
        pet_c  = compute_pet(ta_mean, rh, wind_ms, ssrd_day, strd_day,
                              lst_celsius, doy=doy)
        utci_c = compute_utci(ta_mean, rh, wind_ms, mrt_c)

        timestamp = datetime.datetime.combine(
            target, datetime.time(12, 0, 0),
            tzinfo=datetime.timezone.utc,
        ).isoformat().replace("+00:00", ".000Z")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "timestamp":        timestamp,
        "temp_mean_c":      round(ta_mean, 2),
        "windspeed_kmh":    round(wind_kmh, 2),
        "humidity_percent": round(rh, 1),
        "mrt_c":            round(mrt_c, 2),
        "pet_c":            round(pet_c, 2),
        "utci_c":           round(utci_c, 2),
    }
