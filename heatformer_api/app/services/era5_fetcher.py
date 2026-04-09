"""ERA5 meteorological data fetcher via CDS API."""

from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

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
    "temp_max",
    "temp_min",
    "temp_mean",
    "humidity",
    "wind",
    "solar",
    "precip",
    "pressure",
    "mrt",
    "pet",
    "utci",
]


def _write_cdsapirc(token: str) -> None:
    """Write ~/.cdsapirc from the provided token."""
    rc = Path.home() / ".cdsapirc"
    rc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {token}\n")


def _latest_date_from_error(exc: Exception) -> datetime.date | None:
    """Parse latest-available date from CDS API error text."""
    msg = str(exc)
    m = re.search(r"latest date available[^\d]*(\d{4}-\d{2}-\d{2})", msg, re.IGNORECASE)
    if not m:
        return None
    try:
        return datetime.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def _date_fields(start: datetime.date, end: datetime.date) -> tuple[list[str], list[str], list[str]]:
    """Build year/month/day string lists for CDS requests."""
    rng = pd.date_range(start, end, freq="D")
    years = sorted({str(d.year) for d in rng})
    months = sorted({f"{d.month:02d}" for d in rng})
    days = sorted({f"{d.day:02d}" for d in rng})
    return years, months, days


def _open_dataset_with_fallback(xr, file_path: Path):
    """Open NetCDF using available engines."""
    last_exc: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy", None):
        try:
            if engine is None:
                return xr.open_dataset(file_path)
            return xr.open_dataset(file_path, engine=engine)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Unable to open NetCDF file: {file_path}") from last_exc


def _open_downloaded_era5(xr, download_path: str):
    """
    Open CDS response file.

    CDS may return either:
    - a plain .nc file
    - a .zip with one or more .nc files
    """
    path = Path(download_path)
    opened = []
    extract_dir: Path | None = None

    if zipfile.is_zipfile(path):
        extract_dir = Path(tempfile.mkdtemp(prefix="era5_extract_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)

        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"CDS zip response had no .nc files: {download_path}")

        for nc in nc_files:
            opened.append(_open_dataset_with_fallback(xr, nc))
        ds = xr.merge(opened, compat="override", join="outer")
    else:
        opened.append(_open_dataset_with_fallback(xr, path))
        ds = opened[0]

    def _cleanup() -> None:
        for d in opened:
            try:
                d.close()
            except Exception:
                pass
        if extract_dir and extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    return ds, _cleanup


def fetch_era5_window(
    token: str,
    lat: float,
    lon: float,
    delta: float,
    target_date: datetime.date,
    window_days: int = 8,
    lst_celsius: float = 35.0,
) -> np.ndarray:
    """
    Download ERA5 for [target_date - window_days + 1 ... target_date] and
    return a (window_days, 11) float32 array.
    """
    import cdsapi
    import xarray as xr
    from app.utils.thermal import compute_pet, compute_tmrt, compute_utci

    _write_cdsapirc(token)

    # ERA5 bounding box: [N, W, S, E]
    area = [lat + delta, lon - delta, lat - delta, lon + delta]

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    ds = None
    cleanup_ds: Callable[[], None] = lambda: None

    try:
        c = cdsapi.Client(quiet=True)
        req_end = target_date

        for _ in range(12):
            req_start = req_end - datetime.timedelta(days=window_days - 1)
            years, months, days = _date_fields(req_start, req_end)
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": ["reanalysis"],
                        "variable": ERA5_VARIABLES,
                        "year": years,
                        "month": months,
                        "day": days,
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area,
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    tmp_path,
                )
                break
            except Exception as exc:
                latest = _latest_date_from_error(exc)
                if latest is not None and latest < req_end:
                    log.warning(
                        "ERA5 requested end %s unavailable, retrying at latest %s",
                        req_end,
                        latest,
                    )
                    req_end = latest
                    continue

                msg = str(exc)
                if "None of the data you have requested is available yet" in msg and req_end > datetime.date(1980, 1, 1):
                    req_end = req_end - datetime.timedelta(days=1)
                    continue
                raise

        ds, cleanup_ds = _open_downloaded_era5(xr, tmp_path)

        pt = ds.sel(latitude=lat, longitude=lon, method="nearest")

        try:
            times = pd.to_datetime(pt["valid_time"].values)
        except Exception:
            times = pd.to_datetime(pt["time"].values)

        t2m = pt["t2m"].values - 273.15
        d2m = pt["d2m"].values - 273.15
        u10 = pt["u10"].values
        v10 = pt["v10"].values
        msl = pt["msl"].values / 100.0
        ssrd = pt["ssrd"].values
        tp = pt["tp"].values * 1000.0
        strd = pt["strd"].values

        rows: list[list[float]] = []
        target_dates = pd.date_range(req_end - datetime.timedelta(days=window_days - 1), req_end, freq="D")

        for date in target_dates:
            mask = np.array([t.date() == date.date() for t in times])
            if mask.sum() == 0:
                rows.append([0.0] * 11)
                continue

            ta_max = float(t2m[mask].max())
            ta_min = float(t2m[mask].min())
            ta_mean = float(t2m[mask].mean())
            wind = float(np.sqrt(u10[mask] ** 2 + v10[mask] ** 2).mean())

            es = 6.112 * np.exp(17.67 * ta_mean / (ta_mean + 243.5))
            td = float(d2m[mask].mean())
            ea = 6.112 * np.exp(17.67 * td / (td + 243.5))
            rh = float(np.clip(100 * ea / es, 0, 100))

            ssrd_day = float(ssrd[mask].sum())
            strd_day = float(strd[mask].sum())
            solar = ssrd_day / 3.6e6

            doy = date.day_of_year
            mrt_c = compute_tmrt(ssrd_day, strd_day, lst_celsius, ta_mean, doy=doy, lat=lat, lon=lon)
            pet_c = compute_pet(ta_mean, rh, wind, ssrd_day, strd_day, lst_celsius, doy=doy)
            utci_c = compute_utci(ta_mean, rh, wind, mrt_c)

            rows.append(
                [
                    ta_max,
                    ta_min,
                    ta_mean,
                    rh,
                    wind,
                    solar,
                    float(tp[mask].sum()),
                    float(msl[mask].mean()),
                    mrt_c,
                    pet_c,
                    utci_c,
                ]
            )

    finally:
        cleanup_ds()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    arr = np.array(rows, dtype=np.float32)

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
    """Fetch a recent available ERA5 meteorological snapshot."""
    import cdsapi
    import xarray as xr
    from app.utils.thermal import compute_pet, compute_tmrt, compute_utci

    _write_cdsapirc(token)

    # ERA5 is delayed by around 5 days, so start with a safer offset.
    req_day = datetime.date.today() - datetime.timedelta(days=5)
    area = [lat + delta, lon - delta, lat - delta, lon + delta]

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    ds = None
    cleanup_ds: Callable[[], None] = lambda: None

    try:
        c = cdsapi.Client(quiet=True)

        for _ in range(12):
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": ["reanalysis"],
                        "variable": ERA5_VARIABLES,
                        "year": [str(req_day.year)],
                        "month": [f"{req_day.month:02d}"],
                        "day": [f"{req_day.day:02d}"],
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area,
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    tmp_path,
                )
                break
            except Exception as exc:
                latest = _latest_date_from_error(exc)
                if latest is not None and latest < req_day:
                    req_day = latest
                    continue
                msg = str(exc)
                if "None of the data you have requested is available yet" in msg and req_day > datetime.date(1980, 1, 1):
                    req_day = req_day - datetime.timedelta(days=1)
                    continue
                raise

        ds, cleanup_ds = _open_downloaded_era5(xr, tmp_path)
        pt = ds.sel(latitude=lat, longitude=lon, method="nearest")

        t2m = pt["t2m"].values - 273.15
        d2m = pt["d2m"].values - 273.15
        u10 = pt["u10"].values
        v10 = pt["v10"].values
        ssrd = pt["ssrd"].values
        strd = pt["strd"].values

        ta_mean = float(t2m.mean())
        wind_ms = float(np.sqrt(u10 ** 2 + v10 ** 2).mean())
        wind_kmh = wind_ms * 3.6

        es = 6.112 * np.exp(17.67 * ta_mean / (ta_mean + 243.5))
        td = float(d2m.mean())
        ea = 6.112 * np.exp(17.67 * td / (td + 243.5))
        rh = float(np.clip(100 * ea / es, 0, 100))

        ssrd_day = float(ssrd.sum())
        strd_day = float(strd.sum())

        doy = req_day.timetuple().tm_yday
        mrt_c = compute_tmrt(ssrd_day, strd_day, lst_celsius, ta_mean, doy=doy, lat=lat, lon=lon)
        pet_c = compute_pet(ta_mean, rh, wind_ms, ssrd_day, strd_day, lst_celsius, doy=doy)
        utci_c = compute_utci(ta_mean, rh, wind_ms, mrt_c)

        timestamp = datetime.datetime.combine(
            req_day,
            datetime.time(12, 0, 0),
            tzinfo=datetime.timezone.utc,
        ).isoformat().replace("+00:00", ".000Z")

    finally:
        cleanup_ds()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "timestamp": timestamp,
        "temp_mean_c": round(ta_mean, 2),
        "windspeed_kmh": round(wind_kmh, 2),
        "humidity_percent": round(rh, 1),
        "mrt_c": round(mrt_c, 2),
        "pet_c": round(pet_c, 2),
        "utci_c": round(utci_c, 2),
    }
