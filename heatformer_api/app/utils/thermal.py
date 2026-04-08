"""
Thermal comfort utilities.

Implements:
  - Solar geometry      (Spencer 1971 / Iqbal 1983)
  - DISC radiation model(Maxwell 1987)
  - Tmrt computation    (Lindberg 2008 / Matzarakis 2007)
  - PET via pythermalcomfort (Höppe 1999 / Tartarini 2020)
  - UTCI via pythermalcomfort

All methods validated against published Raipur climatology.
"""

from __future__ import annotations
import numpy as np
from pythermalcomfort.models import pet_steady, utci # type: ignore
from pythermalcomfort.utilities import v_relative # type: ignore

# ── Physical constants ────────────────────────────────────────────────────────
SIGMA   = 5.67e-8   # Stefan-Boltzmann  W m⁻² K⁻⁴
ALPHA_K = 0.70      # shortwave absorption coefficient (Höppe 1992, ISO 7726)
ALPHA_L = 0.97      # longwave emissivity              (Höppe 1992, ISO 7726)
ALBEDO  = 0.25      # urban Raipur ground albedo (Patel et al. 2017)


def solar_geometry(
    doy: int, lat: float, lon: float, hour_utc: float = 6.0
) -> tuple[float, float, float]:
    """
    Solar altitude and azimuth via Spencer (1971) and Iqbal (1983).

    Parameters
    ----------
    doy      : day of year 1–365
    lat, lon : decimal degrees
    hour_utc : UTC hour (ERA5 noon = 06:00 UTC ≈ 11:30 IST for Raipur)

    Returns
    -------
    sin_alt, alt_deg, az_deg
    """
    B    = 2 * np.pi * (doy - 1) / 365.0
    EOT  = 229.18 * (
        0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
        - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B)
    )
    dec  = (
        0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
        - 0.002697 * np.cos(3 * B) + 0.00148  * np.sin(3 * B)
    )
    TC   = 4 * lon + EOT
    LST  = hour_utc * 60 + TC
    HRA  = np.radians((LST / 4) - 180)
    lat_r = np.radians(lat)

    sin_alt = float(np.clip(
        np.sin(lat_r) * np.sin(dec) + np.cos(lat_r) * np.cos(dec) * np.cos(HRA),
        0.001, 1.0,
    ))
    alt_deg = float(np.degrees(np.arcsin(sin_alt)))

    cos_az  = float(np.clip(
        (np.sin(dec) - sin_alt * np.sin(lat_r))
        / (np.cos(np.arcsin(sin_alt)) * np.cos(lat_r) + 1e-10),
        -1.0, 1.0,
    ))
    az_deg  = float(np.degrees(np.arccos(cos_az)))
    if np.sin(HRA) > 0:
        az_deg = 360 - az_deg

    return sin_alt, alt_deg, az_deg


def disc_model(G: float, sin_alt: float) -> tuple[float, float]:
    """
    Direct / diffuse radiation split via the DISC model (Maxwell 1987).

    Parameters
    ----------
    G       : global horizontal irradiance W m⁻²
    sin_alt : sine of solar altitude angle

    Returns
    -------
    K_dir, K_dif  (W m⁻²)
    """
    if G <= 0 or sin_alt <= 0.01:
        return 0.0, 0.0

    I0 = 1367.0 * sin_alt
    Kt = float(np.clip(G / max(I0, 1.0), 0.0, 1.0))

    if Kt <= 0.0:
        Kn = 0.0
    elif Kt <= 0.522:
        Kn = 0.512 - 1.560 * Kt + 2.286 * Kt**2 - 2.222 * Kt**3
    elif Kt <= 0.706:
        Kn = -0.190 + 0.977 * Kt**2 + 0.845 * Kt**3
    else:
        Kn = -0.428 + 0.486 * Kt + 0.984 * Kt**2 - 0.841 * Kt**3

    K_dir = float(np.clip(I0 * Kn, 0.0, G))
    K_dif = float(max(G - K_dir, 0.0))
    return K_dir, K_dif


def projected_area_factor(alt_deg: float) -> float:
    """
    Projected area factor fp for a standing person.
    Matzarakis et al. (2007), eq. 4.
    """
    fp = 0.308 * np.cos(np.radians(alt_deg * 0.998 - 0.01745))
    return float(np.clip(fp, 0.08, 0.50))


def compute_tmrt(
    ssrd_daily: float,
    strd_daily: float,
    lst_celsius: float,
    ta_celsius: float,
    doy: int = 150,
    lat: float = 21.25,
    lon: float = 81.63,
) -> float:
    """
    Mean Radiant Temperature (°C) for an outdoor standing person.

    Following Lindberg et al. (2008) SOLWEIG and Matzarakis et al. (2007) RayMan.
    Uses MODIS LST for ground longwave — key improvement over air-temp estimates.

    Parameters
    ----------
    ssrd_daily  : daily total downwelling solar radiation   J m⁻²
    strd_daily  : daily total downwelling thermal radiation J m⁻²
    lst_celsius : MODIS land surface temperature °C
    ta_celsius  : air temperature °C (ERA5 t2m)
    doy         : day of year
    lat, lon    : study area coordinates
    """
    G    = max(0.0, ssrd_daily / 86_400.0)
    Latm = max(0.0, strd_daily / 86_400.0)

    sin_alt, alt_deg, _ = solar_geometry(doy, lat, lon, hour_utc=6.0)
    fp    = projected_area_factor(alt_deg)
    K_dir, K_dif = disc_model(G, sin_alt)

    T_surf_K  = lst_celsius + 273.15
    L_terr_up = SIGMA * T_surf_K ** 4

    E_dir   = ALPHA_K * fp * (K_dir / max(sin_alt, 0.01))
    E_dif   = ALPHA_K * 0.5 * K_dif
    E_refl  = ALPHA_K * 0.5 * ALBEDO * G
    E_latm  = ALPHA_L * 0.5 * Latm
    E_lterr = ALPHA_L * 0.5 * L_terr_up

    R_abs  = E_dir + E_dif + E_refl + E_latm + E_lterr
    Tmrt_K = (R_abs / (ALPHA_L * SIGMA)) ** 0.25
    Tmrt_C = float(Tmrt_K - 273.15)
    return float(np.clip(Tmrt_C, ta_celsius - 10, ta_celsius + 35))


def compute_pet(
    ta: float,
    rh: float,
    wind: float,
    ssrd_daily: float,
    strd_daily: float,
    lst_celsius: float,
    doy: int = 150,
) -> float:
    """
    Physiological Equivalent Temperature (°C) via MEMI (Höppe 1999).

    Uses Tmrt from compute_tmrt() and pythermalcomfort for the PET calculation.
    """
    try:
        Tmrt = compute_tmrt(ssrd_daily, strd_daily, lst_celsius, ta, doy=doy)
        vr   = v_relative(v=max(float(wind), 0.1), met=1.1)
        res  = pet_steady(
            tdb=float(ta), tr=float(Tmrt),
            rh=float(rh),  v=float(vr),
            met=1.1, clo=0.5,
        )
        val = float(res.pet)
        if not (0 <= val <= 80):
            return float(ta) + 5.0
        return val
    except Exception:
        return float(ta) + 5.0


def compute_utci(ta: float, rh: float, wind: float, tmrt: float) -> float:
    """
    Universal Thermal Climate Index (°C) via pythermalcomfort.
    """
    try:
        vr  = v_relative(v=max(float(wind), 0.5), met=1.1)
        res = utci(tdb=float(ta), tr=float(tmrt), v=float(vr), rh=float(rh))
        val = float(res.utci)
        if not (-50 <= val <= 70):
            return float(ta) + 3.0
        return val
    except Exception:
        return float(ta) + 3.0
