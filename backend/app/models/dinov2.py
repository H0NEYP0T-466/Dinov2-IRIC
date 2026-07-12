"""DINOv2-B single-label classification model for ISIC 2019.

The architecture here MUST match the training script exactly so that
checkpoints produced by ``kaggle/trainCollab.py`` load without remapping.
See :class:`MultiLabelDinoV2`.

Backbone: DINOv2-B (``vit_base_patch14_dinov2.lvd142m``), 86M parameters, with
``num_classes=0`` so timm returns the 768-d pooled features only. A custom head
then maps 768 -> 9 raw logits (no softmax in the forward pass — softmax is
applied at inference / CrossEntropyLoss at training).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import settings
from ..services.logger import get_logger

log = get_logger("app.model")


class MultiLabelDinoV2(nn.Module):
    """DINOv2-B backbone + custom 2-layer classification head.

    Head: Linear(768 -> 512) -> ReLU -> Dropout(0.3) -> Linear(512 -> 9).
    Output is raw logits; apply ``torch.softmax`` for probabilities.
    """

    def __init__(
        self,
        backbone_name: str = settings.backbone_name,
        backbone_dim: int = settings.backbone_dim,
        num_classes: int = settings.num_classes,
        dropout: float = settings.dropout,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "timm is required to build the DINOv2 backbone. "
                "Install it via `pip install timm`."
            ) from exc

        # num_classes=0 -> drop the built-in classification head; we attach our
        # own below. forward_features returns the CLS-token embedding (768-d).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Custom classification head. Order matters for checkpoint compatibility.
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(backbone_dim, 512)),
                    ("act", nn.ReLU(inplace=True)),
                    ("drop", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(512, num_classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 768)
        return self.classifier(features)  # (B, num_classes) raw logits


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the ``module.`` prefix added by ``nn.DataParallel``/``DistributedDataParallel``.

    Checkpoints saved while training with DataParallel have keys like
    ``module.backbone.patch_embed.proj.weight``; the inference model is a plain
    module and expects ``backbone.patch_embed.proj.weight``.
    """
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict

    log.info(
        "Checkpoint contains DataParallel 'module.' prefixes — stripping %d keys.",
        sum(1 for k in state_dict if k.startswith("module.")),
    )
    return {k[len("module."):]: v for k, v in state_dict.items()}


def build_model() -> MultiLabelDinoV2:
    """Construct the model with random weights (architecture only)."""
    model = MultiLabelDinoV2(pretrained=False)
    return model


def load_model(checkpoint_path: Path | str | None = None) -> tuple[MultiLabelDinoV2, str, list[str]]:
    """Build the model and load weights from the best checkpoint.

    The number of output classes is inferred from the checkpoint itself
    (via stored ``num_classes`` metadata or by inspecting the final
    classifier weight shape).  This avoids shape mismatches when the
    config default differs from the checkpoint.

    Args:
        checkpoint_path: Override for the configured checkpoint path. Defaults
            to ``settings.model_checkpoint``.

    Returns:
        (model, device, classes) — model is on the target device and in eval
        mode; *classes* is the ordered class-name list that matches the model's
        output indices.
    """
    from ..config import ISIC_CLASSES  # local import to avoid circular ref

    device = settings.resolved_device
    path = Path(checkpoint_path) if checkpoint_path else settings.model_checkpoint

    # ------------------------------------------------------------------
    # 1. Load the checkpoint first so we can discover num_classes.
    # ------------------------------------------------------------------
    checkpoint = None
    state_dict = None
    num_classes = settings.num_classes  # fallback
    classes: list[str] | None = None

    if path.exists():
        log.info("Loading checkpoint: %s", path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            log.info(
                "Checkpoint metadata | epoch=%s | keys=%s",
                checkpoint.get("epoch", "?"),
                ", ".join(k for k in checkpoint.keys() if k != "model_state"),
            )
            # Prefer the explicit num_classes stored by the training loop.
            if "num_classes" in checkpoint:
                num_classes = int(checkpoint["num_classes"])
                log.info(
                    "Using num_classes=%d from checkpoint metadata.", num_classes,
                )
            # Also grab the class-name list if the training loop saved it.
            if "classes" in checkpoint:
                classes = list(checkpoint["classes"])
                log.info("Using class list from checkpoint: %s", classes)
        else:
            state_dict = checkpoint

        state_dict = _strip_module_prefix(state_dict)

        # Fallback: infer from the final classifier weight shape.
        if num_classes == settings.num_classes:
            fc2_key = "classifier.fc2.weight"
            if fc2_key in state_dict:
                ckpt_num_classes = state_dict[fc2_key].shape[0]
                if ckpt_num_classes != num_classes:
                    log.info(
                        "Overriding num_classes: config=%d → checkpoint=%d "
                        "(inferred from %s shape %s).",
                        num_classes,
                        ckpt_num_classes,
                        fc2_key,
                        list(state_dict[fc2_key].shape),
                    )
                    num_classes = ckpt_num_classes
    else:
        log.warning(
            "Checkpoint not found at %s — serving an UNTRAINED model. "
            "Place a trained model_best.pth in backend/checkpoints/ for real predictions.",
            path,
        )

    # If no class list was found in the checkpoint, fall back to config but
    # truncate to match num_classes.
    if classes is None:
        classes = ISIC_CLASSES[:num_classes]
        log.info("Falling back to config class list (first %d): %s", num_classes, classes)

    # ------------------------------------------------------------------
    # 2. Build model with the correct num_classes.
    # ------------------------------------------------------------------
    log.info(
        "Loading model architecture | backbone=%s | num_classes=%d",
        settings.backbone_name,
        num_classes,
    )
    model = MultiLabelDinoV2(pretrained=False, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model built | total_params=%s | trainable=%s | device=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
        device,
    )

    # ------------------------------------------------------------------
    # 3. Load the state dict into the model.
    # ------------------------------------------------------------------
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys when loading checkpoint: %s", missing[:10])
        if unexpected:
            log.warning("Unexpected keys when loading checkpoint: %s", unexpected[:10])
        log.info("Checkpoint loaded successfully.")

    model = model.to(device)
    model.eval()
    log.info("Model set to eval mode on device '%s'.", device)
    return model, device, classes

