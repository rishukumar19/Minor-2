from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── GEE ──────────────────────────────────────────────────────────────
    GEE_PROJECT: str = "your-gee-project-id"

    # ── CDS (Copernicus Climate Data Store) ──────────────────────────────
    CDS_TOKEN: str = ""

    # ── Model ────────────────────────────────────────────────────────────
    MODEL_PATH: str = "weights/heatformer.pth"
    # Normalization stats saved by Notebook 2
    NORM_STATS_DIR: str = "weights"

    # ── Study area (Raipur) ──────────────────────────────────────────────
    RAIPUR_LAT: float = 21.25
    RAIPUR_LON: float = 81.63
    ROI_DELTA: float = 0.27       # ±0.27° → ~60 km × 60 km rectangle
    TILE: int = 64

    # ── Inference ────────────────────────────────────────────────────────
    DEVICE: str = "cpu"           # "cuda" when GPU is available
    N_CLASSES: int = 4            # Low / Moderate / High / Extreme
    SAT_CHANNELS: int = 5         # LST_C, NDVI, NDBI, NDWI, SAVI
    MET_FEATURES: int = 11        # see preprocessing.py for full list
    MET_WINDOW: int = 8           # 8-day meteorological history window

    # ── Caching TTLs (seconds) ───────────────────────────────────────────
    GEE_CACHE_TTL: int = 8 * 86_400       # 8 days (MOD11A2 cycle)
    CDS_CACHE_TTL: int = 86_400           # until tomorrow 12:00 — approx 1 day
    INFERENCE_CACHE_TTL: int = 86_400     # 24 hours
    METEO_CACHE_TTL: int = 3 * 3_600     # 3 hours

    # ── API ──────────────────────────────────────────────────────────────
    RATE_LIMIT_AUTO: str = "10/minute"
    RATE_LIMIT_MANUAL: str = "30/minute"
    RATE_LIMIT_METEO: str = "60/minute"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
