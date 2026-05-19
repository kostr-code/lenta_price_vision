"""api/ml_app.py — FastAPI ML service application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import MLSettings, load_settings
from api.logging_setup import configure_logging
from api.routers.download import router as download_router
from api.routers.meta import router as meta_router
from api.routers.predict import router as predict_router

settings: MLSettings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    log = structlog.get_logger("startup")
    log.info("startup.begin",
             vlm=settings.vlm_model_id,
             weights_inside=settings.weights_inside,
             runs_dir=settings.runs_dir)

    from api import pipeline_bridge
    try:
        pipeline_bridge.load_models(settings)
    except Exception as exc:
        log.error("startup.models_failed", error=str(exc))
        # Don't abort startup — some endpoints (health, schema) work without models

    log.info("startup.done")
    yield
    log.info("shutdown")


app = FastAPI(
    title="Lenta Price Tag — ML Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta_router)
app.include_router(predict_router)
app.include_router(download_router)
