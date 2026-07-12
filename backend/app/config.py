"""Application configuration and shared constants.

The 8-class nomenclature below is the single source of truth for the whole
system. It mirrors the official ISIC 2019 Challenge skin-lesion categories.
The same literal is duplicated in:

    - kaggle/trainCollab.py              (training ISIC_CLASSES)
    - src/types/index.ts                 (frontend CLASS_NAMES)

Keep these three copies in sync whenever the class list changes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# ISIC 2019 — 8-class skin-lesion nomenclature (UNK removed).
# Order matches the one-hot column order in ISIC_2019_Training_GroundTruth.csv
# and the model's output logit indices.
# ---------------------------------------------------------------------------
ISIC_CLASSES: list[str] = [
    "MEL",   # Melanoma
    "NV",    # Melanocytic nevus
    "BCC",   # Basal cell carcinoma
    "AK",    # Actinic keratosis
    "BKL",   # Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
    "DF",    # Dermatofibroma
    "VASC",  # Vascular lesion
    "SCC",   # Squamous cell carcinoma
]

ISIC_CLASS_FULL_NAMES: dict[str, str] = {
    "MEL":  "Melanoma",
    "NV":   "Melanocytic nevus",
    "BCC":  "Basal cell carcinoma",
    "AK":   "Actinic keratosis",
    "BKL":  "Benign keratosis",
    "DF":   "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC":  "Squamous cell carcinoma",
}

assert len(ISIC_CLASSES) == 8, f"Expected 8 classes, got {len(ISIC_CLASSES)}"

# ImageNet normalization statistics. DINOv2 was pretrained on ImageNet, so the
# same mean/std must be used at inference time to match the training
# distribution.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI backend.

    Values can be overridden via environment variables (case-insensitive), e.g.
    ``MODEL_CHECKPOINT=/data/model_best.pth``.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
    )

    # --- Model -------------------------------------------------------------
    model_name: str = "DINOv2-IRIC"
    backbone_name: str = "vit_base_patch14_dinov2.lvd142m"
    backbone_dim: int = 768
    num_classes: int = 8
    dropout: float = 0.3

    # Path to the best checkpoint produced by the training script.
    # Resolved relative to the backend/ directory.
    model_checkpoint: Path = Field(
        default=Path(__file__).resolve().parent.parent / "checkpoints" / "model_best.pth"
    )

    # --- Inference ---------------------------------------------------------
    device: str = "auto"  # "auto" picks CUDA if available, else CPU
    inference_threshold: float = 0.5  # min confidence to report a prediction
    image_size: int = 518

    # --- Server ------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    @property
    def resolved_device(self) -> str:
        """Resolve "auto" to a concrete torch device string lazily.

        Importing torch at module import time is expensive; defer until the
        property is first read (at startup).
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


settings = Settings()
