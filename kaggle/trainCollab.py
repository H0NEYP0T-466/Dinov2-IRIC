#!/usr/bin/env python3
"""
Dinov2-IRIC — Google Colab Training Pipeline (8-class folder-based dataset).

Single-file script for fine-tuning DINOv2-B on a heavily imbalanced 8-class
skin-lesion classification dataset.  The dataset is a simple folder structure:

    /content/data/
        AK/     (867 images)
        BCC/    (3323 images)
        BKL/    (2624 images)
        DF/     (239 images)     ← extreme minority
        MEL/    (4522 images)
        NV/     (12875 images)   ← dominant class
        SCC/    (628 images)
        VASC/   (253 images)     ← extreme minority

===============================================================================
IMBALANCE HANDLING STRATEGY
===============================================================================
    ✓ Heavy augmentation for minority classes (DF, VASC, SCC, AK)
    ✓ Moderate augmentation for mid-tier classes (BKL, BCC)
    ✓ Standard augmentation for majority classes (MEL, NV)
    ✓ Focal Loss (γ=2) — down-weights easy/majority examples
    ✓ Inverse-frequency class weights on loss
    ✓ WeightedRandomSampler — oversamples minority classes
    ✓ Mixup (α=0.4) + CutMix (α=1.0) — regularisation + virtual augmentation
    ✓ Label smoothing (ε=0.1)
    ✓ Cosine Annealing with Warmup scheduler
    ✓ Three-phase progressive unfreezing
    ✓ Gradient clipping + accumulation
    ✓ Mixed precision (AMP)
    ✓ Early stopping on balanced F1-macro (not accuracy)
    ✓ Stratified train/val split
    ✓ Per-class metrics tracking
    ✓ Test-Time Augmentation (TTA) at final evaluation

===============================================================================
USAGE (Google Colab):
===============================================================================

    # 1. Upload dataset to /content/data/ with 8 class folders

    # 2. Install requirements
    !pip install timm scikit-learn matplotlib pillow tensorboard albumentations

    # 3. Run training
    !python trainCollab.py

Architecture MUST match backend/app/models/dinov2.py and backend/app/config.py.
The trained model_best.pth drops straight into backend/checkpoints/.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42

# --- Dataset paths (Colab defaults) ----------------------------------------
# The dataset is a simple folder structure: /content/data/<CLASS_NAME>/*.jpg
if Path("/content/data").exists():
    DATA_ROOT = Path("/content/data")
elif Path("X:/file/FAST_API/Dinov2-IRIC/backend/dataset/data").exists():
    DATA_ROOT = Path("X:/file/FAST_API/Dinov2-IRIC/backend/dataset/data")
else:
    DATA_ROOT = Path("./data")  # local dev fallback

# --- Output root -----------------------------------------------------------
if Path("/content/drive/MyDrive").exists():
    OUTPUT_DIR = Path("/content/drive/MyDrive/Dinov2-IRIC-output")
elif Path("/content").exists():
    OUTPUT_DIR = Path("/content/Dinov2-IRIC-output")
else:
    OUTPUT_DIR = Path("./output")

# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()

# --- 8 classes (UNK removed) ---
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
NUM_CLASSES = 8

FULL_NAMES = {
    "AK":   "Actinic keratosis",
    "BCC":  "Basal cell carcinoma",
    "BKL":  "Benign keratosis",
    "DF":   "Dermatofibroma",
    "MEL":  "Melanoma",
    "NV":   "Melanocytic nevus",
    "SCC":  "Squamous cell carcinoma",
    "VASC": "Vascular lesion",
}

CFG = {
    # --- Model ---
    "backbone":       "vit_base_patch14_dinov2.lvd142m",
    "backbone_dim":   768,
    "num_classes":    NUM_CLASSES,
    "head_dropout":   0.3,
    "image_size":     224,

    # --- Data ---
    "batch_size":     32 if N_GPUS >= 1 else 16,
    "num_workers":    2,       # Colab has limited CPU cores
    "grad_accum_steps": 4,    # effective batch = 32 * 4 = 128
    "val_split":      0.2,    # 80/20 stratified split

    # --- Three-phase fine-tuning ---
    "phases": [
        {"name": "head_only",     "epochs": 5,  "lr": 3e-4, "unfreeze": "head"},
        {"name": "last_4_blocks", "epochs": 15, "lr": 5e-5, "unfreeze": "last_4_blocks"},
        {"name": "full_finetune", "epochs": 20, "lr": 1e-5, "unfreeze": "full"},
    ],

    # --- Warmup ---
    "warmup_epochs":   2,      # cosine annealing warmup epochs per phase

    # --- Optimiser ---
    "weight_decay":    0.05,
    "betas":           (0.9, 0.999),

    # --- Early stopping (on F1 macro — balanced metric) ---
    "es_patience":     10,
    "es_min_delta":    0.001,

    # --- Mixed precision ---
    "use_amp":         torch.cuda.is_available(),

    # --- Focal Loss ---
    "focal_gamma":     2.0,

    # --- Label smoothing ---
    "label_smoothing": 0.1,

    # --- Mixup / CutMix ---
    "mixup_alpha":     0.4,
    "cutmix_alpha":    1.0,
    "mix_prob":        0.5,     # probability of applying mixup or cutmix

    # --- ImageNet normalisation (DINOv2 was pretrained with these) ---
    "imagenet_mean":   [0.485, 0.456, 0.406],
    "imagenet_std":    [0.229, 0.224, 0.225],

    # --- Epoch image / curve save interval ---
    "save_images_every": 5,
    "save_curves_every": 5,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging helpers
# ═══════════════════════════════════════════════════════════════════════════════

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_dir: Path) -> logging.Logger:
    """Configure a root logger with console + file output."""
    logger = logging.getLogger("dinov2_iric")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU memory monitoring
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str() -> str:
    """Return a readable GPU memory usage string."""
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    return f"GPU mem: {alloc:.1f}G alloc / {reserved:.1f}G reserved / {total:.1f}G total"


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS-AWARE AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
#  Minority classes get MUCH heavier augmentation than majority classes.
#  This acts as a form of "virtual oversampling" — when the sampler repeatedly
#  draws minority samples, each pass looks significantly different.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_class_tier(class_name: str) -> str:
    """Classify each class into an augmentation intensity tier.

    Based on the actual class distribution:
        NV:   12875  → majority
        MEL:  4522   → majority
        BCC:  3323   → mid-tier
        BKL:  2624   → mid-tier
        AK:   867    → minority
        SCC:  628    → minority
        VASC: 253    → extreme minority
        DF:   239    → extreme minority
    """
    extreme_minority = {"DF", "VASC"}       # < 300 samples
    minority          = {"SCC", "AK"}        # < 900 samples
    mid_tier          = {"BCC", "BKL"}       # 2000–4000 samples
    # majority         = {"MEL", "NV"}       # > 4000 samples

    if class_name in extreme_minority:
        return "extreme"
    elif class_name in minority:
        return "minority"
    elif class_name in mid_tier:
        return "mid"
    else:
        return "majority"


def get_train_transforms(class_name: str, image_size: int = 224):
    """Return class-specific augmentation pipeline using torchvision.

    Extreme minority → very aggressive augmentation
    Minority          → aggressive augmentation
    Mid-tier          → moderate augmentation
    Majority          → standard augmentation
    """
    import torchvision.transforms as T

    tier = _get_class_tier(class_name)

    # --- Common base ---
    normalize = T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"])

    if tier == "extreme":
        # EXTREME MINORITY (DF, VASC) — pull out all the stops
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            T.RandomAffine(
                degrees=30, translate=(0.15, 0.15),
                scale=(0.7, 1.3), shear=(-20, 20, -20, 20),
            ),
            T.RandomPerspective(distortion_scale=0.3, p=0.4),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
            T.RandomGrayscale(p=0.15),
            T.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.2),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ])

    elif tier == "minority":
        # MINORITY (SCC, AK) — aggressive but not insane
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=120),
            T.RandomAffine(
                degrees=20, translate=(0.12, 0.12),
                scale=(0.8, 1.2), shear=(-15, 15),
            ),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.12),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomAutocontrast(p=0.25),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.35, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])

    elif tier == "mid":
        # MID-TIER (BCC, BKL) — moderate augmentation
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    else:
        # MAJORITY (MEL, NV) — standard augmentation, lighter
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=45),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms — deterministic resize + normalise."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"]),
    ])


def get_tta_transforms(image_size: int = 224):
    """Test-Time Augmentation transforms — 5 views per image."""
    import torchvision.transforms as T
    normalize = T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"])
    return [
        # Original
        T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(), normalize,
        ]),
        # Horizontal flip
        T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(), normalize,
        ]),
        # Vertical flip
        T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(), normalize,
        ]),
        # Center crop at slightly larger size
        T.Compose([
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),
            T.ToTensor(), normalize,
        ]),
        # Rotation 90°
        T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=(90, 90)),
            T.ToTensor(), normalize,
        ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — folder-based with class-aware augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class FolderDataset(Dataset):
    """Dataset that loads from a folder structure:
        data_root/CLASS_NAME/image.jpg

    Supports class-specific augmentation for imbalanced datasets.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        class_names: list[str],
        transform_fn=None,
        is_train: bool = True,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.is_train = is_train
        self.transform_fn = transform_fn  # Function: class_name -> transform

        # Pre-build transforms per class for efficiency
        if is_train and transform_fn is not None:
            self._class_transforms = {
                cls: transform_fn(cls) for cls in class_names
            }
        else:
            self._class_transforms = None

        # Fallback val transform
        self._val_transform = get_val_transforms(CFG["image_size"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback: return a black image
            img = Image.new("RGB", (CFG["image_size"], CFG["image_size"]), (0, 0, 0))

        # Apply class-specific augmentation for training
        if self.is_train and self._class_transforms is not None:
            class_name = self.class_names[label]
            transform = self._class_transforms[class_name]
        else:
            transform = self._val_transform

        img = transform(img)
        return img, label


class TTADataset(Dataset):
    """Wrapper for Test-Time Augmentation — returns multiple views per image."""

    def __init__(self, image_paths: list[str], labels: list[int]):
        self.image_paths = image_paths
        self.labels = labels
        self.tta_transforms = get_tta_transforms(CFG["image_size"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["image_size"], CFG["image_size"]), (0, 0, 0))

        # Apply all TTA transforms and stack
        views = [t(img) for t in self.tta_transforms]
        views = torch.stack(views)  # (num_tta, C, H, W)
        return views, label


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal Loss — handles class imbalance by down-weighting easy examples
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    When γ > 0, easy examples (high p_t) are down-weighted, forcing the model
    to focus on hard / minority examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) raw logits.
            targets: (B,) integer class labels.
        """
        num_classes = inputs.size(1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            # Convert to one-hot then smooth
            one_hot = F.one_hot(targets, num_classes=num_classes).float()
            smooth = self.label_smoothing / num_classes
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth
        else:
            one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute log-softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-class weighting
        if self.weight is not None:
            class_weight = self.weight[targets].unsqueeze(1)  # (B, 1)
            focal_weight = focal_weight * class_weight

        # Focal cross-entropy
        loss = -focal_weight * one_hot * log_probs
        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Mixup / CutMix — data mixing regularisation
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.4):
    """Apply Mixup: linearly combine random pairs of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix: paste a random patch from one image onto another."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape

    # Random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for both mixup and cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cosine Annealing with Warmup
# ═══════════════════════════════════════════════════════════════════════════════

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing schedule with linear warmup.

    During warmup, LR linearly increases from 0 to base_lr.
    After warmup, LR follows a cosine decay to eta_min.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Build dataloaders from folder structure
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_dataset(data_root: Path, log: logging.Logger):
    """Scan folder structure and return image paths, labels, and class info."""
    from sklearn.model_selection import train_test_split

    image_paths = []
    labels = []
    class_counts = {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_root / class_name
        if not class_dir.exists():
            log.warning(f"Class directory not found: {class_dir}")
            continue

        count = 0
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in valid_extensions:
                image_paths.append(str(f))
                labels.append(class_idx)
                count += 1

        class_counts[class_name] = count

    log.info(f"Total images found: {len(image_paths)}")
    log.info("Class distribution:")
    total = len(image_paths)
    for cls_name in CLASSES:
        cnt = class_counts.get(cls_name, 0)
        pct = 100.0 * cnt / total if total > 0 else 0
        tier = _get_class_tier(cls_name)
        log.info(f"  {cls_name:5s} ({FULL_NAMES[cls_name]:30s}): {cnt:6d} ({pct:5.1f}%)  [{tier}]")

    return image_paths, labels, class_counts


def _compute_class_weights(labels: list[int], log: logging.Logger) -> torch.Tensor:
    """Compute inverse-frequency class weights with effective number of samples.

    Uses the "effective number" scheme from the paper:
    "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019).

    This gives even more aggressive up-weighting to rare classes compared
    to simple inverse frequency.
    """
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)

    # Effective number of samples: E_n = (1 - β^n) / (1 - β)
    # β close to 1.0 → more balanced weights; β close to 0 → inverse frequency
    beta = 0.9999  # aggressive balancing for highly imbalanced data

    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    weights = 1.0 / (effective_num + 1e-8)

    # Normalise so weights sum to NUM_CLASSES (average weight = 1.0)
    weights = weights / weights.sum() * NUM_CLASSES

    log.info(f"Class weights (effective number, β={beta}):")
    for i, cls_name in enumerate(CLASSES):
        log.info(f"  {cls_name:5s}: count={class_counts[i]:6.0f} → weight={weights[i]:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def _build_dataloaders(log: logging.Logger):
    """Load folder-based dataset, return (train_loader, val_loader, class_weights, train_steps)."""
    from sklearn.model_selection import train_test_split

    # ---- Scan dataset ----
    image_paths, labels, class_counts = _scan_dataset(DATA_ROOT, log)

    # ---- Class weights ----
    class_weights = _compute_class_weights(labels, log)

    # ---- Stratified train/val split ----
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=CFG["val_split"],
        stratify=labels,
        random_state=SEED,
    )
    log.info(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # Log per-class split
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    val_counts = np.bincount(val_labels, minlength=NUM_CLASSES)
    for i, cls_name in enumerate(CLASSES):
        log.info(f"  {cls_name:5s}: train={train_counts[i]:5d} | val={val_counts[i]:5d}")

    # ---- Create datasets with class-aware augmentation ----
    train_ds = FolderDataset(
        train_paths, train_labels, CLASSES,
        transform_fn=lambda cls: get_train_transforms(cls, CFG["image_size"]),
        is_train=True,
    )
    val_ds = FolderDataset(
        val_paths, val_labels, CLASSES,
        transform_fn=None,
        is_train=False,
    )

    # ---- WeightedRandomSampler for class imbalance ----
    # Each sample's weight = 1/class_count → minority samples drawn more often
    sample_weights = np.array([1.0 / max(1, train_counts[l]) for l in train_labels])
    # Normalise
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    train_steps = len(train_loader)
    log.info(f"Train steps/epoch: {train_steps} | Val batches: {len(val_loader)}")

    # Sanity check
    x, y = next(iter(train_loader))
    log.info(f"Batch check: images {tuple(x.shape)} {x.dtype} | labels {tuple(y.shape)} {y.dtype}")

    # Check batch class distribution (should be roughly balanced due to sampler)
    batch_dist = np.bincount(y.numpy(), minlength=NUM_CLASSES)
    log.info(f"First batch class distribution: {dict(zip(CLASSES, batch_dist.tolist()))}")

    return train_loader, val_loader, class_weights, train_steps, val_paths, val_labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """DINOv2-B backbone + 2-layer classification head.

    Architecture MUST match backend/app/models/dinov2.py exactly.
    Head: Linear(768→512) → ReLU → Dropout(0.3) → Linear(512→num_classes).
    Forward returns raw logits (no softmax).
    """

    def __init__(self, backbone=CFG["backbone"], dim=CFG["backbone_dim"],
                 num_classes=CFG["num_classes"], dropout=CFG["head_dropout"],
                 pretrained=True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(dim, 512)),
            ("act", nn.ReLU(inplace=True)),
            ("drop", nn.Dropout(dropout)),
            ("fc2", nn.Linear(512, num_classes)),
        ]))

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n=4):
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            return
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
        # Also unfreeze the backbone's norm layer
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def unfreeze_full(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def _build_model(log: logging.Logger):
    log.info("Building DINOv2-B model with pretrained backbone...")
    model = DINOv2Classifier(pretrained=True).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Total params: {total:,} (~{total / 1e6:.0f}M)")
    log.info(gpu_mem_str())
    if N_GPUS > 1:
        model = nn.DataParallel(model)
        log.info(f"Wrapped in DataParallel across {N_GPUS} GPUs.")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """Compute accuracy, F1 macro, F1 weighted, per-class recall."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    preds = torch.argmax(logits, dim=1).cpu().numpy()
    tgt = targets.cpu().numpy()

    acc = accuracy_score(tgt, preds)
    f1_macro = f1_score(tgt, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(tgt, preds, average="weighted", zero_division=0)

    # Per-class recall (sensitivity) — critical for minority classes
    per_class_recall = recall_score(tgt, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    per_class_precision = precision_score(tgt, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

    # Balanced accuracy (mean of per-class recalls)
    balanced_acc = np.mean(per_class_recall)

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_recall": per_class_recall.tolist(),
        "per_class_precision": per_class_precision.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping based on F1 Macro (balanced metric for imbalanced data)."""

    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.triggered = False
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        improved = (value - self.best > self.min_delta) if self.mode == "max" \
            else (self.best - value > self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, epoch, optimizer, scheduler, train_m, val_m,
                    is_best, ckpt_dir, log):
    """Save model checkpoint."""
    raw = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "epoch": epoch,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "config": {k: v for k, v in CFG.items() if not callable(v)},
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
    }

    # Save every epoch
    epoch_path = ckpt_dir / f"epoch{epoch}.pth"
    torch.save(payload, epoch_path)
    log.info(f"  Saved checkpoint: {epoch_path.name}")

    if is_best:
        best_path = ckpt_dir / "model_best.pth"
        torch.save(payload, best_path)
        log.info(f"  ★ New best model! (val_f1_macro={val_m['f1_macro']:.4f}) → model_best.pth")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — training curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curves(history, epoch, curve_dir, log):
    """Plot 2×3 training curves grid with per-class metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    ep = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0, 0].plot(ep, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(ep, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy (normal + balanced)
    axes[0, 1].plot(ep, history["val_accuracy"], "g-", label="Accuracy", linewidth=2)
    axes[0, 1].plot(ep, history["val_balanced_accuracy"], "m--", label="Balanced Acc", linewidth=2)
    axes[0, 1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Scores
    axes[0, 2].plot(ep, history["val_f1_macro"], "m-", label="F1 Macro", linewidth=2)
    axes[0, 2].plot(ep, history["val_f1_weighted"], "c-", label="F1 Weighted", linewidth=2)
    axes[0, 2].set_title("F1 Scores", fontsize=14, fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(ep, history["lr"], "k-", linewidth=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    # Per-class Recall over epochs (critical for minority tracking!)
    if "per_class_recall" in history and len(history["per_class_recall"]) > 0:
        recalls = np.array(history["per_class_recall"])  # (epochs, num_classes)
        colours = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        for i, cls_name in enumerate(CLASSES):
            tier = _get_class_tier(cls_name)
            style = "--" if tier in ("extreme", "minority") else "-"
            lw = 2.5 if tier in ("extreme", "minority") else 1.5
            axes[1, 1].plot(ep, recalls[:, i], style, color=colours[i],
                          label=f"{cls_name} [{tier}]", linewidth=lw)
        axes[1, 1].set_title("Per-Class Recall (↑ minority is key)", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend(fontsize=7, ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.05)

    # Per-class Precision over epochs
    if "per_class_precision" in history and len(history["per_class_precision"]) > 0:
        precisions = np.array(history["per_class_precision"])  # (epochs, num_classes)
        colours = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        for i, cls_name in enumerate(CLASSES):
            tier = _get_class_tier(cls_name)
            style = "--" if tier in ("extreme", "minority") else "-"
            lw = 2.5 if tier in ("extreme", "minority") else 1.5
            axes[1, 2].plot(ep, precisions[:, i], style, color=colours[i],
                          label=f"{cls_name} [{tier}]", linewidth=lw)
        axes[1, 2].set_title("Per-Class Precision", fontsize=14, fontweight="bold")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Precision")
        axes[1, 2].legend(fontsize=7, ncol=2)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1.05)

    plt.suptitle(f"Dinov2-IRIC Training — Epoch {epoch} (8 classes, imbalance-aware)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = curve_dir / f"curves_epoch_{epoch}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved training curves → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — sample prediction images
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_prediction_samples(model, val_loader, epoch, sample_dir, log):
    """Save a grid of sample predictions vs ground truth labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    images, labels = next(iter(val_loader))
    images_gpu = images.to(DEVICE)
    logits = model(images_gpu)
    preds = torch.argmax(logits, dim=1).cpu()
    probs = F.softmax(logits, dim=1).cpu()

    # Denormalize for visualisation
    mean = torch.tensor(CFG["imagenet_mean"]).view(3, 1, 1)
    std = torch.tensor(CFG["imagenet_std"]).view(3, 1, 1)

    n_show = min(16, len(images))
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    for i in range(n_show):
        ax = axes[i // cols, i % cols]
        img = images[i] * std + mean  # denormalize
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        true_cls = CLASSES[labels[i].item()]
        pred_cls = CLASSES[preds[i].item()]
        conf = probs[i, preds[i].item()].item()
        correct = labels[i].item() == preds[i].item()
        tier = _get_class_tier(true_cls)

        ax.imshow(img)
        colour = "green" if correct else "red"
        ax.set_title(
            f"True: {true_cls} [{tier}]\nPred: {pred_cls} ({conf:.1%})",
            fontsize=9, fontweight="bold", color=colour,
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(n_show, rows * cols):
        axes[i // cols, i % cols].axis("off")

    plt.suptitle(
        f"Dinov2-IRIC — Sample Predictions (Epoch {epoch})",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    out = sample_dir / f"predictions_epoch_{epoch}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved prediction samples → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_confusion_matrix(model, val_loader, out_dir, log, title_suffix=""):
    """Generate and save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # Normalised confusion matrix (per-class recall)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Plot both raw and normalised
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.figure.colorbar(im1, ax=ax1)
    ax1.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=CLASSES, yticklabels=CLASSES,
        ylabel="True label", xlabel="Predicted label",
    )
    ax1.set_title(f"Confusion Matrix (Counts){title_suffix}", fontsize=14, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)

    # Normalised (recall per class)
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=CLASSES, yticklabels=CLASSES,
        ylabel="True label", xlabel="Predicted label",
    )
    ax2.set_title(f"Normalised (Per-Class Recall){title_suffix}", fontsize=14, fontweight="bold")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

    plt.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower() if title_suffix else ""
    out = out_dir / f"confusion_matrix{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix → {out}")

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    log.info(f"\n{'='*60}\n  CLASSIFICATION REPORT{title_suffix}\n{'='*60}\n{report}")

    report_path = out_dir / f"classification_report{suffix}.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Saved classification report → {report_path}")

    return cm, report


# ═══════════════════════════════════════════════════════════════════════════════
#  Test-Time Augmentation evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_with_tta(model, val_paths, val_labels, out_dir, log):
    """Run TTA evaluation — average predictions over multiple augmented views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    log.info("Running Test-Time Augmentation (5 views per image)...")
    model.eval()

    tta_ds = TTADataset(val_paths, val_labels)
    tta_loader = DataLoader(tta_ds, batch_size=8, shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=True)

    all_preds = []
    all_labels_list = []

    for views, labels in tta_loader:
        # views: (B, num_tta, C, H, W)
        B, N, C, H, W = views.shape
        views_flat = views.view(B * N, C, H, W).to(DEVICE)

        with autocast(enabled=CFG["use_amp"]):
            logits = model(views_flat)  # (B*N, num_classes)

        logits = logits.view(B, N, -1)       # (B, N, num_classes)
        avg_logits = logits.mean(dim=1)       # (B, num_classes) — average over TTA views
        preds = torch.argmax(avg_logits, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels_list.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels_arr = np.array(all_labels_list)

    # Classification report
    report = classification_report(
        all_labels_arr, all_preds,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    log.info(f"\n{'='*60}\n  TTA CLASSIFICATION REPORT\n{'='*60}\n{report}")

    report_path = out_dir / "classification_report_tta.txt"
    report_path.write_text(report, encoding="utf-8")

    # TTA confusion matrix
    cm = confusion_matrix(all_labels_arr, all_preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=CLASSES, yticklabels=CLASSES,
        ylabel="True label", xlabel="Predicted label",
    )
    ax.set_title("TTA Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

    plt.tight_layout()
    out = out_dir / "confusion_matrix_tta.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved TTA confusion matrix → {out}")

    return cm, report


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — class distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(labels, out_dir, log):
    """Plot and save the class distribution bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.bincount(labels, minlength=NUM_CLASSES)

    # Color by tier
    tier_colours = {
        "extreme":  "#ff4444",   # red
        "minority": "#ff8800",   # orange
        "mid":      "#ffcc00",   # yellow
        "majority": "#44bb44",   # green
    }
    colours = [tier_colours[_get_class_tier(cls)] for cls in CLASSES]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(CLASSES, counts, color=colours, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / len(labels)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff4444", label="Extreme minority (< 300)"),
        Patch(facecolor="#ff8800", label="Minority (< 900)"),
        Patch(facecolor="#ffcc00", label="Mid-tier (2000-4000)"),
        Patch(facecolor="#44bb44", label="Majority (> 4000)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_title("Dataset — Class Distribution (8 classes)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Skin Lesion Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = out_dir / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved class distribution chart → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — three-phase training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Output directories ──────────────────────────────────────────────────
    work = OUTPUT_DIR
    ckpt_dir    = work / "checkpoints"
    curve_dir   = work / "training_curves"
    sample_dir  = work / "prediction_samples"
    log_dir     = work / "logs"
    for d in (ckpt_dir, curve_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logger(log_dir)

    # ── Banner ──────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  Dinov2-IRIC  —  8-Class Imbalance-Aware Training Pipeline")
    log.info("=" * 70)
    log.info(f"  Python  : {sys.version.split()[0]}")
    log.info(f"  PyTorch : {torch.__version__}")
    log.info(f"  Device  : {DEVICE}  |  GPUs: {N_GPUS}")
    if torch.cuda.is_available():
        log.info(f"  GPU     : {torch.cuda.get_device_name(0)}")
    log.info(f"  Batch   : {CFG['batch_size']}  (effective ~{CFG['batch_size'] * CFG['grad_accum_steps']})")
    log.info(f"  AMP     : {CFG['use_amp']}")
    log.info(f"  Classes : {NUM_CLASSES} ({', '.join(CLASSES)})")
    log.info(f"  Dataset : {DATA_ROOT}")
    log.info(f"  Outputs : {work}")
    log.info("")
    log.info("  IMBALANCE STRATEGY:")
    log.info(f"    Focal Loss (γ={CFG['focal_gamma']})")
    log.info(f"    Label Smoothing (ε={CFG['label_smoothing']})")
    log.info(f"    Mixup (α={CFG['mixup_alpha']}) + CutMix (α={CFG['cutmix_alpha']})")
    log.info(f"    Class-aware augmentation (extreme/minority/mid/majority tiers)")
    log.info(f"    WeightedRandomSampler (inverse class frequency)")
    log.info(f"    Effective Number class weights (β=0.9999)")
    log.info(f"    Early stopping on F1 Macro (patience={CFG['es_patience']})")
    log.info(f"    Cosine Annealing + Warmup ({CFG['warmup_epochs']} epochs)")
    log.info("=" * 70)

    # ── Dataset ─────────────────────────────────────────────────────────────
    log.info("\n[1/6] Loading dataset …")
    train_loader, val_loader, class_weights, train_steps, val_paths, val_labels = _build_dataloaders(log)

    # Plot class distribution
    all_labels = []
    for _, labels in val_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    # Get all labels from the full dataset
    full_labels = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATA_ROOT / class_name
        if class_dir.exists():
            count = sum(1 for f in class_dir.iterdir() if f.is_file())
            full_labels.extend([class_idx] * count)
    plot_class_distribution(full_labels, work, log)

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("\n[2/6] Building model …")
    import timm  # noqa
    model = _build_model(log)

    # Focal Loss with class weights and label smoothing
    criterion = FocalLoss(
        gamma=CFG["focal_gamma"],
        weight=class_weights,
        label_smoothing=CFG["label_smoothing"],
    )
    log.info(f"Loss: Focal Loss (γ={CFG['focal_gamma']}, label_smooth={CFG['label_smoothing']})")
    log.info(f"Class weights: {class_weights.cpu().numpy().round(4).tolist()}")

    # ── Training setup ──────────────────────────────────────────────────────
    log.info("\n[3/6] Starting training …")
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    history = {k: [] for k in (
        "train_loss", "val_loss",
        "val_accuracy", "val_balanced_accuracy",
        "val_f1_macro", "val_f1_weighted",
        "lr", "epoch_time",
        "per_class_recall", "per_class_precision",
    )}

    # Early stopping monitors F1 Macro — the key balanced metric
    early = EarlyStopping(
        patience=CFG["es_patience"],
        min_delta=CFG["es_min_delta"],
        mode="max",
    )
    best_f1_macro = -1.0
    global_epoch = 0

    def _unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    # ── Phase loop ──────────────────────────────────────────────────────────
    for phase in CFG["phases"]:
        pname = phase["name"]
        n_epochs = phase["epochs"]
        lr = phase["lr"]
        unfreeze = phase["unfreeze"]

        log.info(f"\n{'═' * 70}")
        log.info(f"  PHASE: {pname}  |  epochs={n_epochs}  |  lr={lr}  |  unfreeze={unfreeze}")
        log.info(f"{'═' * 70}")

        # Freeze / unfreeze
        raw = _unwrap(model)
        raw.freeze_all()
        if unfreeze == "last_4_blocks":
            raw.unfreeze_last_n_blocks(4)
        elif unfreeze == "full":
            raw.unfreeze_full()
        for p in raw.classifier.parameters():
            p.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        log.info(f"  Trainable params: {n_trainable:,}")
        log.info(f"  {gpu_mem_str()}")

        optimizer = torch.optim.AdamW(
            trainable, lr=lr, betas=CFG["betas"], weight_decay=CFG["weight_decay"],
        )

        # Cosine Annealing with Warmup
        scheduler = CosineAnnealingWarmup(
            optimizer,
            warmup_epochs=CFG["warmup_epochs"],
            total_epochs=n_epochs,
            eta_min=1e-7,
        )

        scaler = GradScaler(enabled=CFG["use_amp"])

        for epoch_in_phase in range(1, n_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            model.train()

            # ──── Train ────
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (imgs, labels) in enumerate(train_loader, 1):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # Apply Mixup or CutMix with probability
                use_mix = (random.random() < CFG["mix_prob"]) and (global_epoch > CFG["warmup_epochs"])
                if use_mix:
                    if random.random() < 0.5:
                        imgs, labels_a, labels_b, lam = mixup_data(
                            imgs, labels, alpha=CFG["mixup_alpha"])
                    else:
                        imgs, labels_a, labels_b, lam = cutmix_data(
                            imgs, labels, alpha=CFG["cutmix_alpha"])

                with autocast(enabled=CFG["use_amp"]):
                    logits = model(imgs)
                    if use_mix:
                        loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / CFG["grad_accum_steps"]
                    else:
                        loss = criterion(logits, labels) / CFG["grad_accum_steps"]

                scaler.scale(loss).backward()
                running_loss += loss.item() * CFG["grad_accum_steps"]

                # Track train accuracy (use original labels for mixup)
                preds = torch.argmax(logits, dim=1)
                if use_mix:
                    running_correct += (lam * (preds == labels_a).sum().item() +
                                       (1 - lam) * (preds == labels_b).sum().item())
                else:
                    running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)

                accum += 1
                if accum >= CFG["grad_accum_steps"]:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0

                if step % 50 == 0 or step == train_steps:
                    train_acc_so_far = running_correct / max(1, running_total)
                    log.info(
                        f"  [Phase {pname}] Epoch {global_epoch} "
                        f"Step {step}/{train_steps} "
                        f"loss={loss.item() * CFG['grad_accum_steps']:.4f} "
                        f"acc={train_acc_so_far:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            train_loss = running_loss / max(1, train_steps)
            train_acc = running_correct / max(1, running_total)

            # Step the cosine scheduler (per epoch)
            scheduler.step()

            # ──── Validate ────
            model.eval()
            v_loss = 0.0
            all_logits, all_tgt = [], []

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    with autocast(enabled=CFG["use_amp"]):
                        logits = model(imgs)
                        v_loss += criterion(logits, labels).item()

                    all_logits.append(logits.float().cpu())
                    all_tgt.append(labels.cpu())

            val_loss = v_loss / max(1, len(val_loader))
            logits_cat = torch.cat(all_logits)
            tgt_cat = torch.cat(all_tgt)
            metrics = compute_metrics(logits_cat, tgt_cat)

            cur_lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # ──── Record ────
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(metrics["accuracy"])
            history["val_balanced_accuracy"].append(metrics["balanced_accuracy"])
            history["val_f1_macro"].append(metrics["f1_macro"])
            history["val_f1_weighted"].append(metrics["f1_weighted"])
            history["lr"].append(cur_lr)
            history["epoch_time"].append(dt)
            history["per_class_recall"].append(metrics["per_class_recall"])
            history["per_class_precision"].append(metrics["per_class_precision"])

            writer.add_scalar("Loss/train", train_loss, global_epoch)
            writer.add_scalar("Loss/val", val_loss, global_epoch)
            writer.add_scalar("Accuracy/train", train_acc, global_epoch)
            writer.add_scalar("Accuracy/val", metrics["accuracy"], global_epoch)
            writer.add_scalar("Accuracy/balanced", metrics["balanced_accuracy"], global_epoch)
            writer.add_scalar("F1/macro", metrics["f1_macro"], global_epoch)
            writer.add_scalar("F1/weighted", metrics["f1_weighted"], global_epoch)
            writer.add_scalar("LR", cur_lr, global_epoch)

            # Log per-class recall to TensorBoard
            for i, cls_name in enumerate(CLASSES):
                writer.add_scalar(f"Recall/{cls_name}", metrics["per_class_recall"][i], global_epoch)
                writer.add_scalar(f"Precision/{cls_name}", metrics["per_class_precision"][i], global_epoch)

            # Highlight minority class performance
            minority_recalls = {
                cls: metrics["per_class_recall"][i]
                for i, cls in enumerate(CLASSES)
                if _get_class_tier(cls) in ("extreme", "minority")
            }

            log.info(
                f"\n  ┌─ Epoch {global_epoch} Summary {'─' * 50}"
                f"\n  │ Phase           : {pname}"
                f"\n  │ Train Loss      : {train_loss:.4f}"
                f"\n  │ Train Acc       : {train_acc:.4f}"
                f"\n  │ Val Loss        : {val_loss:.4f}"
                f"\n  │ Val Accuracy    : {metrics['accuracy']:.4f}"
                f"\n  │ Val Balanced Acc: {metrics['balanced_accuracy']:.4f}"
                f"\n  │ Val F1 Macro    : {metrics['f1_macro']:.4f}  ← monitored"
                f"\n  │ Val F1 Wgt      : {metrics['f1_weighted']:.4f}"
                f"\n  │ LR              : {cur_lr:.2e}"
                f"\n  │ Time            : {dt:.1f}s"
                f"\n  │ {gpu_mem_str()}"
                f"\n  │"
                f"\n  │ MINORITY CLASS RECALLS:"
            )
            for cls, recall in minority_recalls.items():
                status = "✓" if recall > 0.5 else "⚠" if recall > 0.3 else "✗"
                log.info(f"  │   {status} {cls:5s}: {recall:.4f}")

            log.info(f"  └{'─' * 60}")

            # ──── Checkpoint (every epoch) ────
            is_best = metrics["f1_macro"] > best_f1_macro + CFG["es_min_delta"]
            if is_best:
                best_f1_macro = metrics["f1_macro"]

            save_checkpoint(
                model, global_epoch, optimizer, scheduler,
                {"loss": train_loss, "accuracy": train_acc},
                {"loss": val_loss, **metrics},
                is_best, ckpt_dir, log,
            )

            # ──── Prediction samples + curves every N epochs ────
            if global_epoch % CFG["save_images_every"] == 0:
                save_prediction_samples(model, val_loader, global_epoch, sample_dir, log)

            if global_epoch % CFG["save_curves_every"] == 0:
                plot_curves(history, global_epoch, curve_dir, log)

            # ──── Early stopping on F1 Macro ────
            if early.step(metrics["f1_macro"], global_epoch):
                log.info(
                    f"\n  ⚠ EARLY STOPPING triggered! "
                    f"No improvement for {CFG['es_patience']} epochs. "
                    f"Best F1 Macro: {early.best:.4f} at epoch {early.best_epoch}"
                )
                break

            # Clean up GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if early.triggered:
            break

    writer.close()

    # ── Final evaluation ────────────────────────────────────────────────────
    log.info("\n[4/6] Final evaluation …")

    # Load best model for final evaluation
    best_ckpt = ckpt_dir / "model_best.pth"
    if best_ckpt.exists():
        log.info(f"Loading best model from {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        raw = _unwrap(model)
        raw.load_state_dict(checkpoint["model_state"])
        log.info(f"Best model loaded (epoch {checkpoint['epoch']})")

    plot_confusion_matrix(model, val_loader, work, log)
    save_prediction_samples(model, val_loader, global_epoch, sample_dir, log)

    # ── TTA Evaluation ──────────────────────────────────────────────────────
    log.info("\n[5/6] Test-Time Augmentation evaluation …")
    evaluate_with_tta(model, val_paths, val_labels, work, log)

    # ── Export history ──────────────────────────────────────────────────────
    log.info("\n[6/6] Exporting results …")

    # Save training history
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Saved → {log_dir / 'training_history.json'}")

    # Final curves
    plot_curves(history, global_epoch, curve_dir, log)

    # List checkpoints
    log.info("\n── Checkpoints ──")
    for p in sorted(ckpt_dir.glob("*.pth")):
        size_mb = p.stat().st_size / 1e6
        log.info(f"  {p.name:30s}  ({size_mb:.1f} MB)")

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info(f"\n{'═' * 70}")
    log.info(f"  TRAINING COMPLETE")
    log.info(f"{'═' * 70}")
    log.info(f"  Total epochs      : {global_epoch}")
    log.info(f"  Best F1 Macro     : {best_f1_macro:.4f}")
    log.info(f"  Best epoch        : {early.best_epoch}")
    log.info(f"  Early stopped     : {early.triggered}")
    log.info(f"  Output directory  : {work}")
    log.info(f"  Best model        : {ckpt_dir / 'model_best.pth'}")
    log.info(f"{'═' * 70}")
    log.info("  ⚠  NOTE: This model has 8 classes (UNK removed).")
    log.info("  ⚠  Update backend config.py: num_classes=8, ISIC_CLASSES list")
    log.info("  ⚠  before deploying model_best.pth.")
    log.info(f"{'═' * 70}")


if __name__ == "__main__":
    main()
