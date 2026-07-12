"""
DINOv2-IRIC -- Kaggle Test-Set Evaluation (2x T4 GPU)
======================================================
Self-contained script for evaluating the trained DINOv2-B checkpoint on the
ISIC 2019 test set.  Optimised for Kaggle's 2x T4 GPU environment.

Expected layout:
    /kaggle/working/model_best.pth
    /kaggle/working/data/
        ISIC_2019_Test_GroundTruth.csv
        ISIC_2019_Test_Input/
            ISIC_0034321.jpg
            ...

Usage:
    python test_kaggle.py
    python test_kaggle.py --limit 500       # quick test on 500 samples
    python test_kaggle.py --batch-size 64   # larger batches for T4
    python test_kaggle.py --tta             # enable Test-Time Augmentation
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (auto-detect Kaggle vs local)
# ---------------------------------------------------------------------------
if Path("/kaggle/working").exists():
    CHECKPOINT = Path("/kaggle/working/model_best.pth")
    DATA_DIR   = Path("/kaggle/working/data")
    OUTPUT_DIR = Path("/kaggle/working/test_results")
else:
    # Local fallback
    BACKEND_DIR = Path(__file__).resolve().parent
    CHECKPOINT  = BACKEND_DIR / "checkpoints" / "model_best.pth"
    DATA_DIR    = BACKEND_DIR / "dataset" / "ISIC_2019_Test_Input"
    OUTPUT_DIR  = BACKEND_DIR / "test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 518  # DINOv2-B patch14: 14 * 37 = 518

ALL_CSV_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

FULL_NAMES = {
    "AK":   "Actinic keratosis",
    "BCC":  "Basal cell carcinoma",
    "BKL":  "Benign keratosis",
    "DF":   "Dermatofibroma",
    "MEL":  "Melanoma",
    "NV":   "Melanocytic nevus",
    "SCC":  "Squamous cell carcinoma",
    "VASC": "Vascular lesion",
    "UNK":  "Unknown",
}


# ===========================================================================
#  MODEL ARCHITECTURE (must match training exactly)
# ===========================================================================

class MultiLabelDinoV2(nn.Module):
    """DINOv2-B backbone + custom 2-layer classification head.

    Head: Linear(768 -> 512) -> ReLU -> Dropout(0.3) -> Linear(512 -> C).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        backbone_dim: int = 768,
        num_classes: int = 8,
        dropout: float = 0.3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.classifier = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(backbone_dim, 512)),
                ("act", nn.ReLU(inplace=True)),
                ("drop", nn.Dropout(dropout)),
                ("fc2", nn.Linear(512, num_classes)),
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ===========================================================================
#  DATASET
# ===========================================================================

class ISICTestDataset(Dataset):
    """Map-style dataset for ISIC test images."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image']}.jpg"
        label = int(row["true_label"])

        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img)
            return tensor, label, True
        except Exception:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label, False


class ISICTTADataset(Dataset):
    """Dataset that returns multiple augmented views per image for TTA."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, tta_transforms: list):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tta_transforms = tta_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image']}.jpg"
        label = int(row["true_label"])

        try:
            img = Image.open(image_path).convert("RGB")
            views = torch.stack([t(img) for t in self.tta_transforms])
            return views, label, True
        except Exception:
            n_views = len(self.tta_transforms)
            return torch.zeros(n_views, 3, IMAGE_SIZE, IMAGE_SIZE), label, False


# ===========================================================================
#  HELPERS
# ===========================================================================

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms():
    """5 deterministic views for Test-Time Augmentation."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return [
        # 1. Original
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), normalize,
        ]),
        # 2. Horizontal flip
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), normalize,
        ]),
        # 3. Vertical flip
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(), normalize,
        ]),
        # 4. Center crop at slightly larger size
        transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(), normalize,
        ]),
        # 5. 90-degree rotation
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(), normalize,
        ]),
    ]


def collate_filter(batch):
    """Custom collate that filters out invalid (failed) samples."""
    tensors, labels, valids = zip(*batch)
    valid_t = [t for t, v in zip(tensors, valids) if v]
    valid_l = [l for l, v in zip(labels, valids) if v]
    n_bad = sum(1 for v in valids if not v)
    if not valid_t:
        return None, None, n_bad
    return torch.stack(valid_t), torch.tensor(valid_l, dtype=torch.long), n_bad


def collate_tta(batch):
    """Collate for TTA dataset (each item has multiple views)."""
    views_list, labels, valids = zip(*batch)
    valid_v = [v for v, ok in zip(views_list, valids) if ok]
    valid_l = [l for l, ok in zip(labels, valids) if ok]
    n_bad = sum(1 for v in valids if not v)
    if not valid_v:
        return None, None, n_bad
    return torch.stack(valid_v), torch.tensor(valid_l, dtype=torch.long), n_bad


def gpu_info():
    """Print GPU info."""
    n = torch.cuda.device_count()
    if n == 0:
        print("  No GPU detected, running on CPU")
        return
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / 1e9
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")


# ===========================================================================
#  LOAD MODEL
# ===========================================================================

def load_checkpoint(checkpoint_path: Path, device: str):
    """Load checkpoint, build model with correct num_classes, return (model, classes)."""
    print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict & metadata
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        epoch = ckpt.get("epoch", "?")
        ckpt_classes = list(ckpt["classes"]) if "classes" in ckpt else None
        ckpt_num_classes = int(ckpt["num_classes"]) if "num_classes" in ckpt else None
        print(f"       Checkpoint epoch: {epoch}")
    else:
        state_dict = ckpt
        ckpt_classes = None
        ckpt_num_classes = None

    # Strip DataParallel prefixes
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Determine num_classes
    if ckpt_num_classes is not None:
        num_classes = ckpt_num_classes
    elif "classifier.fc2.weight" in state_dict:
        num_classes = state_dict["classifier.fc2.weight"].shape[0]
    else:
        num_classes = 8

    # Determine class list
    classes = ckpt_classes if ckpt_classes else ALL_CSV_CLASSES[:num_classes]

    print(f"       num_classes: {num_classes}")
    print(f"       classes: {classes}")

    # Build model
    model = MultiLabelDinoV2(pretrained=False, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # -------------------------------------------------------------------
    # Multi-GPU: use DataParallel if 2+ GPUs available
    # -------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        print(f"       Using DataParallel across {n_gpus} GPUs")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(device)

    model.eval()
    print(f"       Model loaded on '{device}' [OK]")
    return model, classes


# ===========================================================================
#  LOAD GROUND TRUTH
# ===========================================================================

def find_csv_and_images(data_dir: Path):
    """Auto-find the CSV and image directory under data_dir."""
    # Look for the CSV
    csv_path = None
    for candidate in [
        data_dir / "ISIC_2019_Test_GroundTruth.csv",
        data_dir / "ISIC_2019_Test_Input" / "ISIC_2019_Test_GroundTruth.csv",
    ]:
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        # Recursive search
        csvs = list(data_dir.rglob("*GroundTruth*.csv"))
        if csvs:
            csv_path = csvs[0]

    # Look for images directory
    image_dir = None
    for candidate in [
        data_dir / "ISIC_2019_Test_Input" / "ISIC_2019_Test_Input",
        data_dir / "ISIC_2019_Test_Input",
        data_dir,
    ]:
        if candidate.exists() and any(candidate.glob("*.jpg")):
            image_dir = candidate
            break
    if image_dir is None:
        # Look for any dir with .jpg files
        for d in data_dir.rglob("*.jpg"):
            image_dir = d.parent
            break

    return csv_path, image_dir


def load_ground_truth(csv_path: Path, model_classes: list[str]):
    """Read CSV, convert one-hot to labels, filter to model classes."""
    print(f"\n[2/4] Loading ground truth: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"       Total samples in CSV: {len(df)}")

    label_cols = [c for c in ALL_CSV_CLASSES if c in df.columns]
    df["true_class"] = df[label_cols].idxmax(axis=1)

    # Filter to model classes only
    mask = df["true_class"].isin(model_classes)
    skipped = (~mask).sum()
    if skipped > 0:
        skipped_info = df.loc[~mask, "true_class"].value_counts().to_dict()
        print(f"  [WARN] Skipping {skipped} samples not in model classes: {skipped_info}")

    df = df[mask].reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(model_classes)}
    df["true_label"] = df["true_class"].map(class_to_idx)

    print(f"       Evaluating on {len(df)} samples across {len(model_classes)} classes")
    print(f"       Class distribution:")
    for cls in model_classes:
        count = (df["true_class"] == cls).sum()
        print(f"         {cls:>5s}: {count:>5d}  ({100 * count / len(df):.1f}%)")

    return df


# ===========================================================================
#  INFERENCE
# ===========================================================================

def run_inference(model, df, image_dir: Path, device: str,
                  batch_size: int = 32, num_workers: int = 4, use_tta: bool = False):
    """Run batched inference with optional TTA, optimised for multi-GPU."""

    mode = "TTA (5 views)" if use_tta else "standard"
    print(f"\n[3/4] Running {mode} inference on {len(df)} images "
          f"(batch_size={batch_size}, workers={num_workers})...")

    if use_tta:
        dataset = ISICTTADataset(df, image_dir, get_tta_transforms())
        collate_fn = collate_tta
    else:
        dataset = ISICTestDataset(df, image_dir, get_val_transform())
        collate_fn = collate_filter

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    y_true = []
    y_pred = []
    y_prob = []
    errors = 0

    # Use AMP for faster inference on GPU
    use_amp = torch.cuda.is_available()

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_data, batch_labels, n_invalid in tqdm(loader, desc="Predicting", unit="batch"):
            errors += n_invalid
            if batch_data is None:
                continue

            if use_tta:
                # batch_data shape: (B, N_views, C, H, W)
                B, N, C, H, W = batch_data.shape
                # Flatten to (B*N, C, H, W) for a single forward pass
                flat = batch_data.view(B * N, C, H, W).to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(flat)  # (B*N, num_classes)

                # Reshape back and average
                logits = logits.view(B, N, -1).mean(dim=1)  # (B, num_classes)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                batch_data = batch_data.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(batch_data)

                probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_indices = np.argmax(probs, axis=1)
            y_true.extend(batch_labels.numpy().tolist())
            y_pred.extend(pred_indices.tolist())
            y_prob.extend(probs)

    elapsed = time.perf_counter() - start_time
    imgs_per_sec = len(y_true) / elapsed if elapsed > 0 else 0

    print(f"\n       Inference complete!")
    print(f"       Processed: {len(y_true)} images in {elapsed:.1f}s ({imgs_per_sec:.1f} img/s)")
    if errors:
        print(f"  [WARN] Skipped {errors} images due to errors")

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ===========================================================================
#  METRICS & VISUALISATION
# ===========================================================================

def compute_and_display_metrics(y_true, y_pred, y_prob, classes: list[str]):
    """Compute all metrics, print them, and save confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        top_k_accuracy_score,
    )

    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)

    # --- Overall Metrics ---
    acc        = accuracy_score(y_true, y_pred)
    bal_acc    = balanced_accuracy_score(y_true, y_pred)
    kappa      = cohen_kappa_score(y_true, y_pred)
    mcc        = matthews_corrcoef(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_wt    = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_wt     = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_wt      = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'-' * 42}")
    print(f"  {'Accuracy':<30s} {acc:>10.4f}")
    print(f"  {'Balanced Accuracy':<30s} {bal_acc:>10.4f}")
    print(f"  {'Cohen Kappa':<30s} {kappa:>10.4f}")
    print(f"  {'Matthews Corr. Coeff. (MCC)':<30s} {mcc:>10.4f}")
    print(f"  {'-' * 42}")
    print(f"  {'Precision (macro)':<30s} {prec_macro:>10.4f}")
    print(f"  {'Recall (macro)':<30s} {rec_macro:>10.4f}")
    print(f"  {'F1 Score (macro)':<30s} {f1_macro:>10.4f}")
    print(f"  {'-' * 42}")
    print(f"  {'Precision (weighted)':<30s} {prec_wt:>10.4f}")
    print(f"  {'Recall (weighted)':<30s} {rec_wt:>10.4f}")
    print(f"  {'F1 Score (weighted)':<30s} {f1_wt:>10.4f}")

    # Top-k accuracy
    if len(classes) > 2 and y_prob is not None:
        for k in [3, 5]:
            if k <= len(classes):
                topk = top_k_accuracy_score(y_true, y_prob, k=k, labels=range(len(classes)))
                print(f"  {'Top-' + str(k) + ' Accuracy':<30s} {topk:>10.4f}")

    # --- Per-Class Report ---
    print(f"\n{'-' * 70}")
    print("  Per-Class Classification Report")
    print(f"{'-' * 70}")
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    print(f"{'-' * 70}")
    print("  Confusion Matrix (rows=true, cols=predicted)")
    print(f"{'-' * 70}")
    header = "        " + "  ".join(f"{c:>6s}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {classes[i]:>5s} {row_str}")

    # --- Save confusion matrix plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        # Raw counts confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5, linecolor="white", ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax.set_ylabel("True", fontsize=13, fontweight="bold")
        ax.set_title(
            f"DINOv2-IRIC Confusion Matrix\n"
            f"Accuracy={acc:.4f}  |  F1(macro)={f1_macro:.4f}  |  Kappa={kappa:.4f}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"\n  [OK] Confusion matrix saved -> {cm_path}")

        # Normalised confusion matrix
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5, linecolor="white", ax=ax2,
        )
        ax2.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax2.set_ylabel("True", fontsize=13, fontweight="bold")
        ax2.set_title(
            f"DINOv2-IRIC Normalised Confusion Matrix\n"
            f"Balanced Acc={bal_acc:.4f}  |  F1(weighted)={f1_wt:.4f}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        cm_norm_path = OUTPUT_DIR / "confusion_matrix_normalised.png"
        fig2.savefig(cm_norm_path, dpi=150)
        plt.close(fig2)
        print(f"  [OK] Normalised confusion matrix saved -> {cm_norm_path}")

        # Per-class F1 bar chart
        report_dict = classification_report(
            y_true, y_pred, target_names=classes,
            digits=4, zero_division=0, output_dict=True,
        )
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        per_class_f1 = [report_dict[c]["f1-score"] for c in classes]
        colors = plt.cm.RdYlGn([f / max(per_class_f1) if max(per_class_f1) > 0 else 0 for f in per_class_f1])
        bars = ax3.bar(classes, per_class_f1, color=colors, edgecolor="black", linewidth=0.5)
        ax3.axhline(y=f1_macro, color="red", linestyle="--", linewidth=1.5, label=f"Macro F1 = {f1_macro:.4f}")
        ax3.set_ylabel("F1 Score", fontsize=12)
        ax3.set_title("Per-Class F1 Score", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.set_ylim(0, 1.05)
        for bar, val in zip(bars, per_class_f1):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        f1_path = OUTPUT_DIR / "per_class_f1.png"
        fig3.savefig(f1_path, dpi=150)
        plt.close(fig3)
        print(f"  [OK] Per-class F1 chart saved -> {f1_path}")

    except ImportError:
        print("\n  [WARN] matplotlib/seaborn not installed -- skipping plots.")

    # --- Save metrics to CSV ---
    metrics_dict = {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa, "mcc": mcc,
        "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
        "precision_weighted": prec_wt, "recall_weighted": rec_wt, "f1_weighted": f1_wt,
    }
    pd.DataFrame([metrics_dict]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"  [OK] Metrics CSV saved -> {OUTPUT_DIR / 'metrics.csv'}")

    # Per-class report CSV
    report_dict = classification_report(
        y_true, y_pred, target_names=classes,
        digits=4, zero_division=0, output_dict=True,
    )
    pd.DataFrame(report_dict).T.to_csv(OUTPUT_DIR / "per_class_report.csv")
    print(f"  [OK] Per-class report saved -> {OUTPUT_DIR / 'per_class_report.csv'}")

    # Save raw predictions
    pred_df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": y_pred,
        "true_class": [classes[i] for i in y_true],
        "pred_class": [classes[i] for i in y_pred],
        "correct": (y_true == y_pred).astype(int),
    })
    # Add per-class probabilities
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = [p[i] for p in y_prob]
    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"  [OK] Predictions CSV saved -> {OUTPUT_DIR / 'predictions.csv'}")

    print(f"\n{'=' * 70}")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv2-IRIC Kaggle Test Evaluation")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Limit to N samples (stratified). Default: all.")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size. Default: 32 (good for T4 16GB).")
    parser.add_argument("--num-workers", "-w", type=int, default=4,
                        help="DataLoader workers. Default: 4.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (5 views, slower but better).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory path.")
    args = parser.parse_args()

    # Override paths if provided
    checkpoint = Path(args.checkpoint) if args.checkpoint else CHECKPOINT
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    print("\n" + "=" * 70)
    print("  DINOv2-IRIC -- Kaggle Test Set Evaluation")
    print("=" * 70)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    gpu_info()

    if device == "cuda":
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.tta:
        print(f"  TTA: ENABLED (5 views per image)")
        # TTA uses 5x memory per image, so reduce effective batch size
        effective_bs = max(1, args.batch_size // 5)
        print(f"  Effective batch size for TTA: {effective_bs} (5 views each)")
    else:
        effective_bs = args.batch_size
        print(f"  TTA: disabled")

    # Verify paths
    if not checkpoint.exists():
        sys.exit(f"  [FAIL] Checkpoint not found: {checkpoint}")

    # Auto-find CSV and image dir
    csv_path, image_dir = find_csv_and_images(data_dir)
    if csv_path is None:
        sys.exit(f"  [FAIL] Ground truth CSV not found under: {data_dir}")
    if image_dir is None:
        sys.exit(f"  [FAIL] No image directory with .jpg files found under: {data_dir}")

    print(f"  CSV: {csv_path}")
    print(f"  Images: {image_dir}")

    # Load model
    model, classes = load_checkpoint(checkpoint, device)

    # Load ground truth
    df = load_ground_truth(csv_path, classes)

    # Apply limit
    if args.limit and args.limit < len(df):
        df = (
            df.groupby("true_class", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), max(1, int(args.limit * len(x) / len(df)))),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
        print(f"  [INFO] Limited to {len(df)} samples (stratified)")

    # Run inference
    y_true, y_pred, y_prob = run_inference(
        model, df, image_dir, device,
        batch_size=effective_bs,
        num_workers=args.num_workers,
        use_tta=args.tta,
    )

    if len(y_true) == 0:
        sys.exit("  [FAIL] No images processed!")

    # Compute and display metrics
    compute_and_display_metrics(y_true, y_pred, y_prob, classes)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
