# 🔥 HeatFormer Inference API

FastAPI service that wraps the HeatFormer dual-stream transformer for urban heat
risk mapping over Raipur, India.

## Project structure

```
heatformer_api/
├── app/
│   ├── main.py                  ← FastAPI app & all three endpoints
│   ├── config.py                ← Pydantic settings (env / .env file)
│   ├── models/
│   │   └── heatformer.py        ← PyTorch model (ResNet-50 + LSTM + CrossAttn)
│   ├── schemas/
│   │   └── requests.py          ← Pydantic request / response schemas
│   ├── services/
│   │   ├── cache.py             ← TTL caches (GEE 8d / CDS 24h / infer 24h / meteo 3h)
│   │   ├── gee_fetcher.py       ← MODIS LST + Landsat indices via computePixels
│   │   ├── era5_fetcher.py      ← ERA5 8-day window + current conditions
│   │   ├── preprocessing.py     ← Normalization helpers
│   │   └── inference.py         ← Model singleton + predict wrappers
│   └── utils/
│       └── thermal.py           ← Tmrt / PET / UTCI calculations
├── weights/                     ← Place model + norm stats here (see below)
├── .env.example
├── requirements.txt
└── README.md
```

## Quick-start

### 1. Copy artefacts from your Colab training run

```bash
# From Drive/HeatFormer/models/
cp heatformer_best.pth  weights/

# From Drive/HeatFormer/processed/
cp SAT_MEAN.npy SAT_STD.npy MET_MEAN.npy MET_STD.npy  weights/
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — add your GEE project ID and CDS token
```

### 3. Authenticate GEE (one-time)

```bash
python -c "import ee; ee.Authenticate()"
```

### 4. Install dependencies & run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

---

## Endpoints

| Method | Path | Rate limit | Cache |
|--------|------|-----------|-------|
| `GET`  | `/api/v1/inference/auto`   | 10/min  | 24 h  |
| `POST` | `/api/v1/inference/manual` | 30/min  | none  |
| `GET`  | `/api/v1/meteo/current`    | 60/min  | 3 h   |

### GET /api/v1/inference/auto

Fully automated pipeline — fetches GEE satellite data (MODIS + Landsat) and
ERA5 meteorological data, preprocesses both, and runs HeatFormer.

```jsonc
// 200 OK
{
  "source": "fresh_inference",   // or "cache"
  "message": "Data fetched, preprocessed, and inferenced successfully.",
  "risk_map": [[0, 1, 2, 3, ...], ...]   // 64×64, values 0-3
}
```

Risk class mapping: `0=Low  1=Moderate  2=High  3=Extreme`

### POST /api/v1/inference/manual

Inject your own arrays. Useful for research / external pipelines.

```jsonc
// Request body
{
  "satellite_data": [ /* 5 × 64 × 64 float */ ],
  "meteo_data":     [ /* 8 × 11 float */ ]
}

// 200 OK
{ "status": "success", "risk_map": [[1, 2, ...], ...] }

// 422 Unprocessable Entity (shape mismatch)
{ "detail": "satellite_data must have 5 channels, got 4" }
```

Met feature order (11 total):
`temp_max, temp_min, temp_mean, humidity, wind, solar, precip, pressure, mrt, pet, utci`

### GET /api/v1/meteo/current

Real-time thermal comfort metrics (ERA5 + pythermalcomfort).

```jsonc
{
  "source": "cache",
  "data": {
    "timestamp": "2024-10-27T14:30:00.000Z",
    "temp_mean_c": 34.2,
    "windspeed_kmh": 12.5,
    "humidity_percent": 45.0,
    "mrt_c": 42.1,
    "pet_c": 39.5,
    "utci_c": 38.2
  }
}
```

---

## Error responses

All errors follow the standard schema:

```json
{ "detail": "String describing the specific error." }
```

| HTTP | Meaning |
|------|---------|
| 400  | Upstream fetch failure (GEE / CDS) |
| 422  | Invalid array shapes in manual request |
| 429  | Rate limit exceeded |
| 500  | PyTorch inference error / GPU OOM |

---

## Model architecture

```
Satellite (5, 64, 64)        Meteorology (8, 11)
        │                            │
  SpatialEncoder               TemporalEncoder
  (ResNet-50, 5-ch)           (2-layer LSTM, 128d)
        │                            │
        └──── CrossAttentionFusion ──┘
                      │
               SpatialDecoder
           (4× TransposedConv)
                      │
            Risk logits (4, 64, 64)
                      │
               argmax per pixel
                      │
            Risk map (64, 64) → {0,1,2,3}
```

## Deployment notes

- For **GPU inference**: set `DEVICE=cuda` in `.env` and install the CUDA
  wheel of PyTorch instead of the CPU wheel in `requirements.txt`.
- For **production**: run behind `nginx` + `gunicorn -k uvicorn.workers.UvicornWorker`.
- ERA5 has a ~5-day data latency; the fetcher automatically uses `today - 1`.
- The GEE `computePixels` call is synchronous and suitable for a 64×64 tile;
  for larger areas switch back to Drive export + polling.
