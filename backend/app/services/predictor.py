"""Singleton predictor that owns the loaded model and runs inference.

Instantiated once at application startup (see :func:`get_predictor`) and shared
across requests. All inference details — raw logits, softmax probabilities,
predicted class, timing — are logged.

ISIC 2019 is a **single-label** classification task (8 classes). The model
outputs raw logits; softmax is applied here to get probabilities, and the
top-1 class is reported as the prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import ISIC_CLASSES, ISIC_CLASS_FULL_NAMES, settings
from ..models.dinov2 import MultiLabelDinoV2, load_model
from .logger import get_logger

log = get_logger("app.predictor")


@dataclass
class Prediction:
    """A single detected class above threshold."""

    class_name: str
    full_name: str
    confidence: float
    class_index: int


@dataclass
class PredictionResult:
    """Full response payload returned to the API layer."""

    predictions: list[Prediction]
    all_probabilities: dict[str, float]
    inference_time_ms: float
    threshold: float
    model: str
    top_class: str
    top_confidence: float
    raw_logits: list[float] = field(default_factory=list)


class Predictor:
    """Wraps the loaded model and exposes a ``predict`` method."""

    def __init__(self, model: MultiLabelDinoV2, device: str, classes: list[str]) -> None:
        self.model = model
        self.device = device
        self.threshold = settings.inference_threshold
        self.model_name = settings.model_name
        self.classes = classes
        self.num_classes = len(classes)
        # Build full-name lookup; fall back to abbreviation itself if missing.
        self.class_full_names: dict[str, str] = {
            c: ISIC_CLASS_FULL_NAMES.get(c, c) for c in classes
        }
        log.info(
            "Predictor ready | model=%s | device=%s | classes=%d (%s) | threshold=%.2f",
            self.model_name,
            self.device,
            self.num_classes,
            self.classes,
            self.threshold,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> PredictionResult:
        """Run a single forward pass and decode the single-label output.

        Args:
            input_tensor: Preprocessed batch of shape (1, 3, 224, 224).
        """
        if input_tensor.device.type != self.device.split(":")[0]:
            input_tensor = input_tensor.to(self.device)

        log.info("Running inference | input_shape=%s", tuple(input_tensor.shape))
        start = time.perf_counter()
        logits = self.model(input_tensor)  # (1, num_classes)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # (1, C) -> (C,)
        logits_vec = logits.squeeze(0).detach().cpu()
        probabilities = torch.softmax(logits_vec, dim=0).tolist()

        log.info("Inference complete | time_ms=%.2f", elapsed_ms)
        log.info(
            "Raw logits | min=%.4f | max=%.4f | mean=%.4f",
            min(logits_vec.tolist()),
            max(logits_vec.tolist()),
            float(logits_vec.mean().item()),
        )

        # Top-1 prediction
        top_idx = int(torch.argmax(logits_vec).item())
        top_class = self.classes[top_idx]
        top_conf = probabilities[top_idx]

        log.info(
            "Softmax probabilities | top_class=%s (%s) | confidence=%.4f",
            top_class,
            self.class_full_names.get(top_class, top_class),
            top_conf,
        )

        # Return all classes above threshold, sorted by confidence desc.
        detected = [
            (i, p)
            for i, p in enumerate(probabilities)
            if p >= self.threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)

        predictions = [
            Prediction(
                class_name=self.classes[i],
                full_name=self.class_full_names.get(self.classes[i], self.classes[i]),
                confidence=round(p, 4),
                class_index=i,
            )
            for i, p in detected
        ]

        all_probs = {self.classes[i]: round(p, 4) for i, p in enumerate(probabilities)}

        if predictions:
            top_names = ", ".join(
                f"{p.class_name} ({p.full_name})" for p in predictions[:5]
            )
            log.info(
                "Detected %d class(es) above threshold (top: %s)",
                len(predictions),
                top_names,
            )
        else:
            log.info("No classes above threshold %.2f.", self.threshold)

        return PredictionResult(
            predictions=predictions,
            all_probabilities=all_probs,
            inference_time_ms=round(elapsed_ms, 2),
            threshold=self.threshold,
            model=self.model_name,
            top_class=top_class,
            top_confidence=round(top_conf, 4),
            raw_logits=[round(v, 4) for v in logits_vec.tolist()],
        )


# --- Singleton accessor ---------------------------------------------------
_predictor: Predictor | None = None


def init_predictor() -> Predictor:
    """Load the model and create the global Predictor. Called at startup."""
    global _predictor
    model, device, classes = load_model()
    _predictor = Predictor(model, device, classes)
    return _predictor


def get_predictor() -> Predictor | None:
    """Return the global predictor (or None if not yet initialized)."""
    return _predictor


def to_dict(result: PredictionResult, request_id: str, filename: str, success: bool = True, error: str | None = None) -> dict[str, Any]:
    """Serialize a PredictionResult to the JSON shape expected by the frontend."""
    return {
        "request_id": request_id,
        "success": success,
        "predictions": [
            {
                "class_name": p.class_name,
                "full_name": p.full_name,
                "confidence": p.confidence,
                "class_index": p.class_index,
            }
            for p in result.predictions
        ],
        "all_probabilities": result.all_probabilities,
        "inference_time_ms": result.inference_time_ms,
        "threshold": result.threshold,
        "model": result.model,
        "top_class": result.top_class,
        "top_confidence": result.top_confidence,
        "filename": filename,
        "error": error,
    }
