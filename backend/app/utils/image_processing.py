"""Image preprocessing for inference.

Mirrors the validation/test transform from the training pipeline (resize +
ImageNet normalize), but operates on raw uploaded bytes rather than a
torchvision sample. Every step is logged for observability, per the project's
"comprehensive logging" requirement.
"""

from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD, settings
from ..services.logger import get_logger

log = get_logger("app.preprocess")

# Deterministic validation transform (no augmentation at inference time).
# Applied to PIL images: Resize(224) -> ToTensor -> Normalize(imagenet).
_val_transform = transforms.Compose(
    [
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Decode uploaded bytes and produce a batched, normalized tensor.

    Args:
        image_bytes: Raw image file contents (PNG/JPEG/etc.).

    Returns:
        Tensor of shape (1, 3, 224, 224) on the configured inference device.
    """
    log.info("Preprocessing started | input_bytes=%d", len(image_bytes))

    # --- Decode -----------------------------------------------------------
    try:
        img = Image.open(io.BytesIO(image_bytes))
        log.info(
            "Decoded image | format=%s | mode=%s | size=%s",
            img.format,
            img.mode,
            img.size,
        )
    except Exception as exc:
        log.error("Failed to decode image bytes: %s", exc, exc_info=True)
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # --- Convert to RGB ---------------------------------------------------
    if img.mode != "RGB":
        log.info("Converting mode %s -> RGB", img.mode)
        img = img.convert("RGB")

    # Pixel statistics on the source image (0-255 range) for logging.
    extrema = img.getextrema()  # ((r_min, r_max), (g_min, g_max), (b_min, b_max))
    log.info("Source pixel ranges (per-channel 0-255): %s", extrema)

    # --- Transform --------------------------------------------------------
    tensor = _val_transform(img)  # (3, H, W)
    log.info(
        "Transformed tensor | shape=%s | dtype=%s",
        tuple(tensor.shape),
        tensor.dtype,
    )

    normalized_min = tensor.min().item()
    normalized_max = tensor.max().item()
    normalized_mean = tensor.mean().item()
    log.info(
        "Normalized tensor stats | min=%.4f | max=%.4f | mean=%.4f",
        normalized_min,
        normalized_max,
        normalized_mean,
    )

    # --- Batch + device ---------------------------------------------------
    batched = tensor.unsqueeze(0)  # (1, 3, 224, 224)
    device = settings.resolved_device
    batched = batched.to(device)
    log.info("Batched tensor ready | shape=%s | device=%s", tuple(batched.shape), device)

    return batched
