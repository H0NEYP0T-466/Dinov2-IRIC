"""Application configuration and shared constants.

The 43-class nomenclature below is the single source of truth for the whole
system. It mirrors the official BigEarthNet-S2 (Sentinel-2) multi-label class
set derived from CORINE land-cover codes (the same 43-class list used by
torchgeo's BigEarthNet dataset). The same literal is duplicated in:

    - kaggle/train_dinov2_bigearths2.ipynb   (training CFG.CLASSES_43)
    - src/types/index.ts                      (frontend CLASS_NAMES)

Keep these three copies in sync whenever the class list changes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Official BigEarthNet-S2 43-class multi-label nomenclature (CORINE-derived).
# Order matches the multi-hot vector indices used during training, so index i
# in a prediction vector corresponds to CLASSES_43[i].
# ---------------------------------------------------------------------------
CLASSES_43: list[str] = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Inland marshes",
    "Peatbogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]

assert len(CLASSES_43) == 43, f"Expected 43 classes, got {len(CLASSES_43)}"

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
    model_name: str = "DINOv2-BigEarthS2"
    backbone_name: str = "vit_base_patch14_dinov2.lvd142m"
    backbone_dim: int = 768
    num_classes: int = 43
    dropout: float = 0.3

    # Path to the best checkpoint produced by the Kaggle training notebook.
    # Resolved relative to the backend/ directory.
    model_checkpoint: Path = Field(
        default=Path(__file__).resolve().parent.parent / "checkpoints" / "model_best.pth"
    )

    # --- Inference ---------------------------------------------------------
    device: str = "auto"  # "auto" picks CUDA if available, else CPU
    inference_threshold: float = 0.5
    image_size: int = 224

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
