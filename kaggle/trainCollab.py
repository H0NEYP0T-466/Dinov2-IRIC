#!/usr/bin/env python3
"""
Dinov2-IRIC — Google Colab Training Pipeline (ISIC 2019).

Single-file script that orchestrates the full training pipeline for skin-lesion
classification on the ISIC 2019 dataset using a DINOv2-B backbone.

===============================================================================
USAGE (Google Colab):
===============================================================================

    # 1. Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # 2. Upload / place dataset in Drive:
    #    /content/drive/MyDrive/dataset/ISIC_2019_Training_GroundTruth.csv
    #    /content/drive/MyDrive/dataset/ISIC_2019_Training_Input/  (images)

    # 3. Install requirements
    !pip install timm scikit-learn matplotlib pandas pillow tensorboard

    # 4. Run training
    !python trainCollab.py

===============================================================================
FEATURES:
===============================================================================
    ✓ ISIC 2019 — 9-class single-label skin-lesion classification
    ✓ DINOv2-B backbone (86M params) + 2-layer classification head
    ✓ Three-phase fine-tuning: head → last 2 blocks → full backbone
    ✓ CrossEntropyLoss with class-weight balancing (handles NV dominance)
    ✓ ReduceLROnPlateau scheduler
    ✓ Early stopping with configurable patience
    ✓ Mixed precision (AMP) training
    ✓ Gradient accumulation
    ✓ Stratified train/val split (80/20)
    ✓ Rich data augmentation (flip, rotate, color jitter, random erasing)
    ✓ Model saved every epoch as epoch{N}.pth
    ✓ Best model saved as model_best.pth
    ✓ Sample prediction images saved every 5 epochs
    ✓ Training curves plotted every 5 epochs
    ✓ Confusion matrix at end of training
    ✓ Per-class accuracy report
    ✓ TensorBoard logging
    ✓ training_history.json export
    ✓ GPU memory monitoring
    ✓ Full reproducibility (seed everything)
    ✓ Colab-optimised paths

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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to taste
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42

# --- Dataset paths (Colab defaults) ----------------------------------------
# Auto-detect: Colab Drive → local fallback
if Path("/content/drive/MyDrive").exists():
    _DATA_ROOT = Path("/content/drive/MyDrive/dataset")
elif Path("/content/dataset").exists():
    _DATA_ROOT = Path("/content/dataset")
else:
    _DATA_ROOT = Path(".")  # local dev fallback

# Override these if your layout is different:
DATASET_CSV   = _DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv"
DATASET_IMAGES = _DATA_ROOT / "ISIC_2019_Training_Input"

# If images are double-nested (archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input/)
# we auto-detect and fix below.

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

CFG = {
    # --- Model ---
    "backbone":       "vit_base_patch14_dinov2.lvd142m",
    "backbone_dim":   768,
    "num_classes":    9,
    "head_dropout":   0.3,
    "image_size":     224,

    # --- Data ---
    "batch_size":     64 if N_GPUS >= 1 else 32,
    "num_workers":    2,  # Colab has limited CPU cores
    "grad_accum_steps": 2,
    "val_split":      0.2,  # 80/20 stratified split

    # --- Three-phase fine-tuning ---
    "phases": [
        {"name": "head",           "epochs": 5,  "lr": 1e-3, "unfreeze": "head"},
        {"name": "last_2_blocks",  "epochs": 10, "lr": 1e-4, "unfreeze": "last_2_blocks"},
        {"name": "full",           "epochs": 15, "lr": 1e-5, "unfreeze": "full"},
    ],

    # --- Optimiser / scheduler ---
    "weight_decay":    0.01,
    "betas":           (0.9, 0.999),
    "sched_mode":      "max",       # monitor val accuracy (higher = better)
    "sched_factor":    0.5,
    "sched_patience":  3,
    "sched_min_lr":    1e-7,

    # --- Early stopping ---
    "es_patience":     7,
    "es_min_delta":    0.001,

    # --- Mixed precision ---
    "use_amp":         torch.cuda.is_available(),

    # --- ImageNet normalisation (DINOv2 was pretrained with these) ---
    "imagenet_mean":   [0.485, 0.456, 0.406],
    "imagenet_std":    [0.229, 0.224, 0.225],

    # --- Epoch image / curve save interval ---
    "save_images_every": 5,
    "save_curves_every": 5,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ISIC 2019 — 9-class nomenclature
# ═══════════════════════════════════════════════════════════════════════════════

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

ISIC_FULL_NAMES = {
    "MEL":  "Melanoma",
    "NV":   "Melanocytic nevus",
    "BCC":  "Basal cell carcinoma",
    "AK":   "Actinic keratosis",
    "BKL":  "Benign keratosis",
    "DF":   "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC":  "Squamous cell carcinoma",
    "UNK":  "Unknown",
}

assert len(ISIC_CLASSES) == CFG["num_classes"]


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
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class ISICDataset(Dataset):
    """ISIC 2019 dataset — loads images from disk with on-the-fly transforms.

    Args:
        image_ids: List of ISIC image IDs (e.g. ["ISIC_0000000", ...]).
        labels:    Corresponding integer class labels (0–8).
        image_dir: Path to the directory containing .jpg images.
        transform: torchvision transform pipeline.
    """

    def __init__(self, image_ids: list[str], labels: list[int],
                 image_dir: Path, transform):
        from PIL import Image  # noqa
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform
        self._Image = Image

        # Build a lookup for actual filenames (some have _downsampled suffix)
        self._file_map: dict[str, Path] = {}
        self._build_file_map()

    def _build_file_map(self):
        """Scan the image directory once and map image IDs to file paths."""
        # Check for double-nested structure
        nested = self.image_dir / "ISIC_2019_Training_Input"
        scan_dir = nested if nested.is_dir() else self.image_dir

        extensions = (".jpg", ".jpeg", ".png")
        for f in scan_dir.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                # Extract the base ISIC ID: ISIC_0000017_downsampled.jpg -> ISIC_0000017
                name = f.stem
                base_id = name.replace("_downsampled", "")
                self._file_map[base_id] = f

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        img_path = self._file_map.get(img_id)
        if img_path is None:
            # Fallback: try standard naming
            nested = self.image_dir / "ISIC_2019_Training_Input"
            scan_dir = nested if nested.is_dir() else self.image_dir
            for ext in (".jpg", ".jpeg", ".png"):
                p = scan_dir / f"{img_id}{ext}"
                if p.exists():
                    img_path = p
                    break
                p = scan_dir / f"{img_id}_downsampled{ext}"
                if p.exists():
                    img_path = p
                    break

        if img_path is None or not img_path.exists():
            # Return a black image as fallback (shouldn't happen with proper data)
            img = self._Image.new("RGB", (CFG["image_size"], CFG["image_size"]), (0, 0, 0))
        else:
            img = self._Image.open(img_path).convert("RGB")

        img = self.transform(img)
        return img, label


def _build_dataloaders(log: logging.Logger):
    """Load ISIC 2019 from CSV + image folder, return (train_loader, val_loader, class_weights)."""
    import torchvision.transforms as T
    from sklearn.model_selection import train_test_split

    # ---- Read CSV ----
    log.info(f"Reading CSV: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    log.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    # CSV columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    # One-hot encoded, single label → convert to integer class index
    label_cols = ISIC_CLASSES  # ["MEL", "NV", "BCC", ...]
    image_ids = df["image"].tolist()
    labels_onehot = df[label_cols].values  # (N, 9)
    labels = np.argmax(labels_onehot, axis=1).tolist()  # integer class indices

    # ---- Auto-detect image directory ----
    img_dir = DATASET_IMAGES
    # Handle double-nested: archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input/
    nested = img_dir / "ISIC_2019_Training_Input"
    actual_dir = nested if nested.is_dir() else img_dir
    # Quick check
    sample_files = list(actual_dir.iterdir())[:5]
    log.info(f"Image directory: {img_dir}")
    log.info(f"Actual scan directory: {actual_dir}")
    log.info(f"Sample files: {[f.name for f in sample_files]}")

    # ---- Class distribution ----
    class_counts = np.bincount(labels, minlength=CFG["num_classes"])
    log.info("Class distribution:")
    for i, cls_name in enumerate(ISIC_CLASSES):
        pct = 100.0 * class_counts[i] / len(labels)
        log.info(f"  {cls_name:5s} ({ISIC_FULL_NAMES[cls_name]:30s}): {class_counts[i]:6d} ({pct:5.1f}%)")

    # ---- Class weights for imbalanced data ----
    total_samples = len(labels)
    class_weights = total_samples / (CFG["num_classes"] * class_counts.astype(np.float64))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    log.info(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    # ---- Stratified train/val split ----
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        image_ids, labels,
        test_size=CFG["val_split"],
        stratify=labels,
        random_state=SEED,
    )
    log.info(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

    # ---- Transforms ----
    train_tf = T.Compose([
        T.RandomResizedCrop(CFG["image_size"], scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = T.Compose([
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"]),
    ])

    train_ds = ISICDataset(train_ids, train_labels, img_dir, train_tf)
    val_ds   = ISICDataset(val_ids, val_labels, img_dir, val_tf)

    # ---- Weighted random sampler for class imbalance ----
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

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

    return train_loader, val_loader, class_weights, train_steps


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """DINOv2-B backbone + 2-layer classification head.

    Architecture MUST match backend/app/models/dinov2.py exactly.
    Head: Linear(768→512) → ReLU → Dropout(0.3) → Linear(512→9).
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

    def unfreeze_last_n_blocks(self, n=2):
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            return
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
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
    """Compute accuracy, F1 macro, F1 weighted for single-label classification.

    Returns dict with: accuracy, f1_macro, f1_weighted, top1.
    """
    from sklearn.metrics import accuracy_score, f1_score

    preds = torch.argmax(logits, dim=1).cpu().numpy()
    tgt = targets.cpu().numpy()

    acc = accuracy_score(tgt, preds)
    f1_macro = f1_score(tgt, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(tgt, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping based on a monitored metric.

    Args:
        patience:  Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode:      "max" for accuracy/F1 (higher=better), "min" for loss.
    """

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
    """Save model checkpoint.

    Every epoch: epoch{N}.pth
    Best model:  model_best.pth
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "epoch": epoch,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "config": {k: v for k, v in CFG.items() if not callable(v)},
        "classes": ISIC_CLASSES,
    }

    # Save every epoch
    epoch_path = ckpt_dir / f"epoch{epoch}.pth"
    torch.save(payload, epoch_path)
    log.info(f"  Saved checkpoint: {epoch_path.name}")

    if is_best:
        best_path = ckpt_dir / "model_best.pth"
        torch.save(payload, best_path)
        log.info(f"  ★ New best model! (val_acc={val_m['accuracy']:.4f}) → model_best.pth")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — training curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curves(history, epoch, curve_dir, log):
    """Plot 2×2 training curves grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ep = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0, 0].plot(ep, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(ep, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(ep, history["val_accuracy"], "g-", label="Val Accuracy", linewidth=2)
    axes[0, 1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Scores
    axes[1, 0].plot(ep, history["val_f1_macro"], "m-", label="F1 Macro", linewidth=2)
    axes[1, 0].plot(ep, history["val_f1_weighted"], "c-", label="F1 Weighted", linewidth=2)
    axes[1, 0].set_title("F1 Scores", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(ep, history["lr"], "k-", linewidth=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning Rate", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Dinov2-IRIC Training — Epoch {epoch}", fontsize=16, fontweight="bold")
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

        true_cls = ISIC_CLASSES[labels[i].item()]
        pred_cls = ISIC_CLASSES[preds[i].item()]
        correct = labels[i].item() == preds[i].item()

        ax.imshow(img)
        colour = "green" if correct else "red"
        ax.set_title(
            f"True: {true_cls}\nPred: {pred_cls}",
            fontsize=10, fontweight="bold", color=colour,
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
def plot_confusion_matrix(model, val_loader, out_dir, log):
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

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(CFG["num_classes"])))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(CFG["num_classes"]),
        yticks=range(CFG["num_classes"]),
        xticklabels=ISIC_CLASSES,
        yticklabels=ISIC_CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title("Confusion Matrix — Dinov2-IRIC", fontsize=16, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    out = out_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix → {out}")

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=ISIC_CLASSES,
        digits=4,
        zero_division=0,
    )
    log.info(f"\n{'='*60}\n  CLASSIFICATION REPORT\n{'='*60}\n{report}")

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Saved classification report → {report_path}")

    return cm, report


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation — class distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(labels, out_dir, log):
    """Plot and save the ISIC class distribution bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.bincount(labels, minlength=CFG["num_classes"])
    colours = plt.cm.Set3(np.linspace(0, 1, CFG["num_classes"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(ISIC_CLASSES, counts, color=colours, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / len(labels)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title("ISIC 2019 — Class Distribution", fontsize=16, fontweight="bold")
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
    log.info("  Dinov2-IRIC  —  ISIC 2019 Training Pipeline")
    log.info("=" * 70)
    log.info(f"  Python  : {sys.version.split()[0]}")
    log.info(f"  PyTorch : {torch.__version__}")
    log.info(f"  Device  : {DEVICE}  |  GPUs: {N_GPUS}")
    if torch.cuda.is_available():
        log.info(f"  GPU     : {torch.cuda.get_device_name(0)}")
    log.info(f"  Batch   : {CFG['batch_size']}  (effective ~{CFG['batch_size'] * CFG['grad_accum_steps']})")
    log.info(f"  AMP     : {CFG['use_amp']}")
    log.info(f"  Classes : {CFG['num_classes']} ({', '.join(ISIC_CLASSES)})")
    log.info(f"  Outputs : {work}")
    log.info(f"  Dataset : {DATASET_CSV}")
    log.info(f"  Images  : {DATASET_IMAGES}")
    log.info("=" * 70)

    # ── Dataset ─────────────────────────────────────────────────────────────
    log.info("\n[1/5] Loading ISIC 2019 dataset …")
    train_loader, val_loader, class_weights, train_steps = _build_dataloaders(log)

    # Plot class distribution
    all_labels = []
    df = pd.read_csv(DATASET_CSV)
    labels_onehot = df[ISIC_CLASSES].values
    all_labels = np.argmax(labels_onehot, axis=1)
    plot_class_distribution(all_labels, work, log)

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("\n[2/5] Building model …")
    import timm  # noqa
    model = _build_model(log)

    # CrossEntropyLoss with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Loss: CrossEntropyLoss with class weights")

    # ── Training setup ──────────────────────────────────────────────────────
    log.info("\n[3/5] Starting training …")
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    history = {k: [] for k in (
        "train_loss", "val_loss",
        "val_accuracy", "val_f1_macro", "val_f1_weighted",
        "lr", "epoch_time",
    )}

    early = EarlyStopping(
        patience=CFG["es_patience"],
        min_delta=CFG["es_min_delta"],
        mode="max",  # monitor val accuracy
    )
    best_accuracy = -1.0
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
        if unfreeze == "last_2_blocks":
            raw.unfreeze_last_n_blocks(2)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=CFG["sched_mode"], factor=CFG["sched_factor"],
            patience=CFG["sched_patience"], min_lr=CFG["sched_min_lr"],
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

                with autocast(enabled=CFG["use_amp"]):
                    logits = model(imgs)
                    loss = criterion(logits, labels) / CFG["grad_accum_steps"]

                scaler.scale(loss).backward()
                running_loss += loss.item() * CFG["grad_accum_steps"]

                # Track train accuracy
                preds = torch.argmax(logits, dim=1)
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
                        f"acc={train_acc_so_far:.4f}"
                    )

            train_loss = running_loss / max(1, train_steps)
            train_acc = running_correct / max(1, running_total)

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
            history["val_f1_macro"].append(metrics["f1_macro"])
            history["val_f1_weighted"].append(metrics["f1_weighted"])
            history["lr"].append(cur_lr)
            history["epoch_time"].append(dt)

            writer.add_scalar("Loss/train", train_loss, global_epoch)
            writer.add_scalar("Loss/val", val_loss, global_epoch)
            writer.add_scalar("Accuracy/train", train_acc, global_epoch)
            writer.add_scalar("Accuracy/val", metrics["accuracy"], global_epoch)
            writer.add_scalar("F1/macro", metrics["f1_macro"], global_epoch)
            writer.add_scalar("F1/weighted", metrics["f1_weighted"], global_epoch)
            writer.add_scalar("LR", cur_lr, global_epoch)

            log.info(
                f"\n  ┌─ Epoch {global_epoch} Summary ({'─' * 50})"
                f"\n  │ Phase        : {pname}"
                f"\n  │ Train Loss   : {train_loss:.4f}"
                f"\n  │ Train Acc    : {train_acc:.4f}"
                f"\n  │ Val Loss     : {val_loss:.4f}"
                f"\n  │ Val Accuracy : {metrics['accuracy']:.4f}"
                f"\n  │ Val F1 Macro : {metrics['f1_macro']:.4f}"
                f"\n  │ Val F1 Wgt   : {metrics['f1_weighted']:.4f}"
                f"\n  │ LR           : {cur_lr:.2e}"
                f"\n  │ Time         : {dt:.1f}s"
                f"\n  │ {gpu_mem_str()}"
                f"\n  └{'─' * 60}"
            )

            # ──── Scheduler ────
            scheduler.step(metrics["accuracy"])

            # ──── Checkpoint (every epoch) ────
            is_best = metrics["accuracy"] > best_accuracy + CFG["es_min_delta"]
            if is_best:
                best_accuracy = metrics["accuracy"]

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

            # ──── Early stopping ────
            if early.step(metrics["accuracy"], global_epoch):
                log.info(
                    f"\n  ⚠ EARLY STOPPING triggered! "
                    f"No improvement for {CFG['es_patience']} epochs. "
                    f"Best accuracy: {early.best:.4f} at epoch {early.best_epoch}"
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
    log.info("\n[4/5] Final evaluation …")

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

    # ── Export history ──────────────────────────────────────────────────────
    log.info("\n[5/5] Exporting results …")

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
    log.info(f"  Total epochs     : {global_epoch}")
    log.info(f"  Best val accuracy: {best_accuracy:.4f}")
    log.info(f"  Best epoch       : {early.best_epoch}")
    log.info(f"  Early stopped    : {early.triggered}")
    log.info(f"  Output directory : {work}")
    log.info(f"  Best model       : {ckpt_dir / 'model_best.pth'}")
    log.info(f"{'═' * 70}")
    log.info("  Copy model_best.pth to backend/checkpoints/ for deployment.")
    log.info(f"{'═' * 70}")


if __name__ == "__main__":
    main()
