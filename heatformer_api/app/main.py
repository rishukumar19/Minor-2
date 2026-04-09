"""HeatFormer API application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.routers import inference as inference_router
from app.routers import meteo as meteo_router
from app.services import inference as inference_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
_APP_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _APP_DIR / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    inference_service.startup()
    yield


app = FastAPI(
    title="HeatFormer API",
    description=(
        "Urban heat island risk prediction and real-time meteorological data "
        "for Raipur, India."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(meteo_router.router)
app.include_router(inference_router.router)


@app.get("/", include_in_schema=False)
async def root():
    if _INDEX_HTML.exists():
        return FileResponse(_INDEX_HTML)
    return JSONResponse(
        {"status": "ok", "message": "HeatFormer API is running.", "docs": "/docs"}
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    icon_path = _STATIC_DIR / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path)
    return Response(status_code=204)


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok"}