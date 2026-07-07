"""FastAPI application entry point.

Run locally with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

or:

    python -m app.main

Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import settings
from .services.logger import get_logger, setup_logging
from .services.predictor import init_predictor

log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle.

    Boots logging, then loads the model. The model load is heavyweight (timm +
    checkpoint) so it happens once at startup rather than per-request.
    """
    setup_logging()
    log.info("=" * 70)
    log.info("Starting %s", settings.model_name)
    log.info("API prefix: %s", settings.api_v1_prefix)
    log.info("Device: %s", settings.resolved_device)
    log.info("Checkpoint: %s", settings.model_checkpoint)

    try:
        init_predictor()
        log.info("Startup complete — ready to serve predictions.")
    except Exception:
        log.exception("Model failed to load at startup. /predict will return 503.")

    yield

    log.info("Shutting down %s", settings.model_name)
    log.info("=" * 70)


app = FastAPI(
    title=f"{settings.model_name} API",
    description="Skin lesion classification (DINOv2-B + ISIC 2019, 9 classes).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow all origins for local dev (frontend on :5173, backend on :8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root() -> dict:
    """Root redirect / info endpoint."""
    return {
        "name": settings.model_name,
        "docs": "/docs",
        "api": f"{settings.api_v1_prefix}/predict",
    }


def main() -> None:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
