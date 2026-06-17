"""Singleton predictor that owns the loaded model and runs inference.

Instantiated once at application startup (see :func:`get_predictor`) and shared
across requests. All inference details — raw logits, sigmoid probabilities,
applied threshold, predicted classes, timing — are logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import CLASSES_43, settings
from ..models.dinov2 import MultiLabelDinoV2, load_model
from .logger import get_logger

log = get_logger("app.predictor")


@dataclass
class Prediction:
    """A single detected class above threshold."""

    class_name: str
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
    raw_logits: list[float] = field(default_factory=list)


class Predictor:
    """Wraps the loaded model and exposes a ``predict`` method."""

    def __init__(self, model: MultiLabelDinoV2, device: str) -> None:
        self.model = model
        self.device = device
        self.threshold = settings.inference_threshold
        self.model_name = settings.model_name
        self.num_classes = settings.num_classes
        log.info(
            "Predictor ready | model=%s | device=%s | classes=%d | threshold=%.2f",
            self.model_name,
            self.device,
            self.num_classes,
            self.threshold,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> PredictionResult:
        """Run a single forward pass and decode the multi-label output.

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
        probabilities = torch.sigmoid(logits_vec).tolist()

        log.info("Inference complete | time_ms=%.2f", elapsed_ms)
        log.info(
            "Raw logits | min=%.4f | max=%.4f | mean=%.4f",
            min(logits_vec.tolist()),
            max(logits_vec.tolist()),
            float(logits_vec.mean().item()),
        )
        log.info(
            "Sigmoid probabilities | min=%.4f | max=%.4f | above_threshold=%d/%d",
            min(probabilities),
            max(probabilities),
            sum(1 for p in probabilities if p >= self.threshold),
            self.num_classes,
        )

        # Decode predictions above threshold, sorted by confidence desc.
        detected = [
            (i, p)
            for i, p in enumerate(probabilities)
            if p >= self.threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)

        predictions = [
            Prediction(
                class_name=CLASSES_43[i],
                confidence=round(p, 4),
                class_index=i,
            )
            for i, p in detected
        ]

        all_probs = {CLASSES_43[i]: round(p, 4) for i, p in enumerate(probabilities)}

        if predictions:
            top_names = ", ".join(p.class_name for p in predictions[:5])
            log.info(
                "Detected %d class(es) (top: %s)",
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
            raw_logits=[round(v, 4) for v in logits_vec.tolist()],
        )


# --- Singleton accessor ---------------------------------------------------
_predictor: Predictor | None = None


def init_predictor() -> Predictor:
    """Load the model and create the global Predictor. Called at startup."""
    global _predictor
    model, device = load_model()
    _predictor = Predictor(model, device)
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
            {"class_name": p.class_name, "confidence": p.confidence, "class_index": p.class_index}
            for p in result.predictions
        ],
        "all_probabilities": result.all_probabilities,
        "inference_time_ms": result.inference_time_ms,
        "threshold": result.threshold,
        "model": result.model,
        "filename": filename,
        "error": error,
    }
