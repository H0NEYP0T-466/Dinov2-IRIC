"""HTTP route handlers for the prediction API.

Endpoints (mounted under ``settings.api_v1_prefix``):

* ``POST /predict``  — upload an image, get multi-label predictions.
* ``GET  /health``   — server / model status.
* ``GET  /classes``  — the 43 BigEarthNet-S2 class names.

Every request is assigned a UUID and logged in full (file metadata,
preprocessing steps, inference details, response summary, errors with stack
traces).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..config import CLASSES_43, settings
from ..services.logger import get_logger
from ..services.predictor import get_predictor, to_dict
from ..utils.image_processing import preprocess

log = get_logger("app.api")

router = APIRouter(prefix=settings.api_v1_prefix, tags=["prediction"])

# Accepted upload content types. We accept both MIME form uploads and the
# common image/* superset.
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB hard cap


@router.get("/health")
async def health() -> dict:
    """Report server and model readiness."""
    predictor = get_predictor()
    loaded = predictor is not None and predictor.is_loaded
    log.info("Health check | model_loaded=%s", loaded)
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_name": settings.model_name if loaded else None,
        "num_classes": settings.num_classes,
        "device": settings.resolved_device,
    }


@router.get("/classes")
async def classes() -> dict:
    """Return the full 43-class nomenclature."""
    log.info("Classes requested | count=%d", len(CLASSES_43))
    return {"count": len(CLASSES_43), "classes": CLASSES_43}


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Accept an image upload and return multi-label predictions."""
    request_id = str(uuid.uuid4())
    filename = file.filename or "unknown"

    log.info("=" * 70)
    log.info(
        "POST /predict | request_id=%s | filename=%s | content_type=%s",
        request_id,
        filename,
        file.content_type,
    )

    # --- Validate content type -------------------------------------------
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        # Also allow generic "application/octet-stream" but rely on PIL decode
        # to reject non-images downstream.
        if content_type != "application/octet-stream":
            log.warning(
                "Rejected upload | request_id=%s | unsupported content_type=%s",
                request_id,
                content_type,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{content_type}'. Expected one of: "
                f"{', '.join(sorted(_ALLOWED_CONTENT_TYPES))}.",
            )

    # --- Validate size ----------------------------------------------------
    image_bytes = await file.read()
    size = len(image_bytes)
    log.info("Upload received | request_id=%s | size_bytes=%d", request_id, size)
    if size == 0:
        log.warning("Empty upload | request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if size > _MAX_FILE_BYTES:
        log.warning("Oversized upload | request_id=%s | size=%d", request_id, size)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size} bytes). Max {_MAX_FILE_BYTES} bytes.",
        )

    # --- Model availability ----------------------------------------------
    predictor = get_predictor()
    if predictor is None or not predictor.is_loaded:
        log.error("Model not loaded | request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check backend logs and the checkpoint path.",
        )

    # --- Preprocess + infer ----------------------------------------------
    try:
        tensor = preprocess(image_bytes)
    except ValueError as exc:
        log.error(
            "Preprocessing failed | request_id=%s | error=%s",
            request_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not process image: {exc}",
        )
    except Exception as exc:
        log.error(
            "Unexpected preprocessing error | request_id=%s", request_id, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during image preprocessing.",
        )

    try:
        result = predictor.predict(tensor)
    except Exception as exc:
        log.error(
            "Inference failed | request_id=%s | error=%s",
            request_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    payload = to_dict(result, request_id=request_id, filename=filename)
    log.info(
        "Response | request_id=%s | detected=%d | inference_ms=%.2f",
        request_id,
        len(payload["predictions"]),
        payload["inference_time_ms"],
    )
    log.info("=" * 70)
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
