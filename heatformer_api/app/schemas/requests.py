"""
Pydantic schemas for HeatFormer API request and response bodies.
"""

from typing import Annotated, List, Literal, Any
from pydantic import BaseModel, Field, model_validator


# ── Shared ────────────────────────────────────────────────────────────────────

RiskMap  = List[List[int]]    # 64×64, values in {0,1,2,3}
RiskClass = Literal[0, 1, 2, 3]

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}


# ── /api/v1/inference/auto  (GET) ─────────────────────────────────────────────

class AutoInferenceResponse(BaseModel):
    source: Literal["fresh_inference", "cache"]
    message: str
    risk_map: RiskMap

    model_config = {"json_schema_extra": {
        "example": {
            "source": "fresh_inference",
            "message": "Data fetched, preprocessed, and inferenced successfully.",
            "risk_map": [[0, 1, 2, 3, 0, "..."]],
        }
    }}


# ── /api/v1/inference/manual  (POST) ──────────────────────────────────────────

# Type aliases for readability
SatelliteData = Annotated[
    list[list[list[float]]],
    Field(description="5 channels × 64 rows × 64 cols. "
                      "Channels: [LST_C, NDVI, NDBI, NDWI, SAVI]"),
]
MeteoTimeSeries = Annotated[
    list[list[float]],
    Field(description="8 days × 11 features. "
                      "Features: [temp_max, temp_min, temp_mean, humidity, wind, "
                      "solar, precip, pressure, mrt, pet, utci]"),
]


class ManualInferenceRequest(BaseModel):
    satellite_data: SatelliteData
    meteo_data: MeteoTimeSeries

    @model_validator(mode="after")
    def validate_shapes(self) -> Any:
        sat = self.satellite_data
        met = self.meteo_data

        # satellite_data: [5, 64, 64]
        if len(sat) != 5:
            raise ValueError(
                f"satellite_data must have 5 channels, got {len(sat)}"
            )
        for ch_i, channel in enumerate(sat):
            if len(channel) != 64:
                raise ValueError(
                    f"satellite_data channel {ch_i} must have 64 rows, "
                    f"got {len(channel)}"
                )
            for row_i, row in enumerate(channel):
                if len(row) != 64:
                    raise ValueError(
                        f"satellite_data channel {ch_i}, row {row_i} must "
                        f"have 64 cols, got {len(row)}"
                    )

        # meteo_data: [8, 11]
        if len(met) != 8:
            raise ValueError(
                f"meteo_data must have 8 days (rows), got {len(met)}"
            )
        for day_i, day in enumerate(met):
            if len(day) != 11:
                raise ValueError(
                    f"meteo_data day {day_i} must have 11 features, "
                    f"got {len(day)}"
                )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                # 5 channels × 64 × 64 — realistic Raipur defaults
                "satellite_data": [
                    [[38.5] * 64] * 64,   # ch0: LST_C  (~38°C)
                    [[0.22] * 64] * 64,   # ch1: NDVI
                    [[0.15] * 64] * 64,   # ch2: NDBI
                    [[-0.10] * 64] * 64,  # ch3: NDWI
                    [[0.30] * 64] * 64,   # ch4: SAVI
                ],
                # 8 days × 11 features
                # [temp_max, temp_min, temp_mean, humidity, wind, solar,
                #  precip, pressure, mrt, pet, utci]
                "meteo_data": [
                    [44.1, 29.3, 36.7, 38.2, 3.1, 7.2, 0.0, 1005.4, 44.8, 41.5, 42.3],
                    [43.8, 28.9, 36.4, 39.0, 2.9, 7.0, 0.0, 1005.1, 44.2, 41.0, 42.0],
                    [44.5, 29.7, 37.1, 37.5, 3.3, 7.5, 0.0, 1004.8, 45.5, 42.0, 43.1],
                    [43.2, 28.5, 35.9, 40.1, 2.7, 6.8, 0.2, 1006.0, 43.5, 40.5, 41.5],
                    [44.8, 30.1, 37.5, 36.8, 3.5, 7.8, 0.0, 1004.5, 46.0, 42.5, 43.5],
                    [45.0, 30.5, 37.8, 35.9, 3.8, 8.0, 0.0, 1004.2, 46.5, 43.0, 44.0],
                    [44.3, 29.8, 37.0, 38.5, 3.2, 7.3, 0.0, 1005.0, 45.0, 41.8, 42.8],
                    [43.5, 29.0, 36.3, 39.8, 3.0, 7.1, 0.1, 1005.5, 44.0, 40.8, 41.8],
                ],
            }
        }
    }


class ManualInferenceResponse(BaseModel):
    status: Literal["success"]
    risk_map: RiskMap

    model_config = {"json_schema_extra": {
        "example": {
            "status": "success",
            "risk_map": [[1, 2, 2, 3, "..."]],
        }
    }}


# ── /api/v1/meteo/current  (GET) ──────────────────────────────────────────────

class MeteoData(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp of the data point")
    temp_mean_c: float      = Field(description="Mean air temperature (°C)")
    windspeed_kmh: float    = Field(description="Average wind speed (km/h)")
    humidity_percent: float = Field(description="Relative humidity (0–100)")
    mrt_c: float            = Field(description="Mean Radiant Temperature (°C)")
    pet_c: float            = Field(description="Physiological Equivalent Temperature (°C)")
    utci_c: float           = Field(description="Universal Thermal Climate Index (°C)")


class MeteoResponse(BaseModel):
    source: Literal["fresh_fetch", "cache"]
    data: MeteoData

    model_config = {"json_schema_extra": {
        "example": {
            "source": "cache",
            "data": {
                "timestamp": "2024-10-27T14:30:00.000Z",
                "temp_mean_c": 34.2,
                "windspeed_kmh": 12.5,
                "humidity_percent": 45.0,
                "mrt_c": 42.1,
                "pet_c": 39.5,
                "utci_c": 38.2,
            },
        }
    }}


# ── Error schema ──────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    detail: str
