"""
In-process TTL caches for the three data tiers.

GEE data     : 8 days   (matches MOD11A2 composite cycle)
CDS data     : ~24 h    (refreshed tomorrow 12:00 PM)
Inference    : 24 h     (risk map)
Meteo        : 3 h      (current conditions)
"""

from __future__ import annotations
from cachetools import TTLCache
from app.config import get_settings

_s = get_settings()

# One entry per tier; keyed by a descriptor string (e.g. "auto:2024-10")
gee_cache:       TTLCache = TTLCache(maxsize=64, ttl=_s.GEE_CACHE_TTL)
cds_cache:       TTLCache = TTLCache(maxsize=64, ttl=_s.CDS_CACHE_TTL)
inference_cache: TTLCache = TTLCache(maxsize=16, ttl=_s.INFERENCE_CACHE_TTL)
meteo_cache:     TTLCache = TTLCache(maxsize=8,  ttl=_s.METEO_CACHE_TTL)
