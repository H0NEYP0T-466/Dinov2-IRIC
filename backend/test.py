"""
DINOv2-IRIC — Test-Set Evaluation Script
=========================================
Loads the trained checkpoint, runs inference on every image in the ISIC 2019
test set, compares against ground-truth labels, and reports:

  • Accuracy, Balanced Accuracy
  • Precision, Recall, F1 (macro / weighted / per-class)
  • Cohen's Kappa, Matthews Correlation Coefficient (MCC)
  • Classification report table
  • Confusion matrix (saved as PNG)

Usage:
    cd backend
    python test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths  (adjust if your layout differs)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DATASET_DIR = BACKEND_DIR / "dataset" / "ISIC_2019_Test_Input"
CSV_PATH    = DATASET_DIR / "ISIC_2019_Test_GroundTruth.csv"
IMAGE_DIR   = DATASET_DIR / "ISIC_2019_Test_Input"
CHECKPOINT  = BACKEND_DIR / "checkpoints" / "model_best.pth"
OUTPUT_DIR  = BACKEND_DIR / "test_results"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ImageNet normalisation (must match training pipeline)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# All 9 ISIC-2019 class abbreviations (column order in the CSV)
ALL_CSV_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def load_checkpoint(checkpoint_path: Path, device: str):
    """Load checkpoint, build model with matching num_classes, return (model, classes)."""
    # We need timm for the backbone
    try:
        import timm  # noqa: F401
    except ImportError:
        sys.exit("timm is required. Install via: pip install timm")

    print(f"[1/4] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Strip DataParallel prefixes if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Determine num_classes from checkpoint
    if ckpt_num_classes is not None:
        num_classes = ckpt_num_classes
    elif "classifier.fc2.weight" in state_dict:
        num_classes = state_dict["classifier.fc2.weight"].shape[0]
    else:
        num_classes = 9  # fallback

    # Determine class list
    if ckpt_classes is not None:
        classes = ckpt_classes
    else:
        classes = ALL_CSV_CLASSES[:num_classes]

    print(f"       num_classes: {num_classes}")
    print(f"       classes: {classes}")

    # Build model
    from app.models.dinov2 import MultiLabelDinoV2
    model = MultiLabelDinoV2(pretrained=False, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN]  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN]  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()
    print(f"       Model loaded on '{device}' [OK]")

    # Determine image size from timm model's config
    image_size = 518  # DINOv2-B default
    try:
        data_cfg = timm.data.resolve_model_data_config(model.backbone)
        image_size = data_cfg.get("input_size", (3, 518, 518))[-1]
    except Exception:
        pass
    print(f"       Input size: {image_size}×{image_size}")

    return model, classes, image_size


def build_transform(image_size: int):
    """Deterministic val/test transform matching training pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_ground_truth(csv_path: Path, model_classes: list[str]):
    """
    Read the CSV, convert one-hot to integer labels, and filter to only the
    classes the model was trained on.

    Returns:
        df with columns: image, true_label (int index into model_classes),
                         true_class (str)
        Also returns a mask of rows that belong to model_classes.
    """
    print(f"[2/4] Loading ground truth: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"       Total samples in CSV: {len(df)}")

    # Get the true class for each row (column with 1.0)
    label_cols = [c for c in ALL_CSV_CLASSES if c in df.columns]
    df["true_class"] = df[label_cols].idxmax(axis=1)

    # Filter to only classes the model knows about
    mask = df["true_class"].isin(model_classes)
    skipped = (~mask).sum()
    if skipped > 0:
        skipped_classes = df.loc[~mask, "true_class"].value_counts().to_dict()
        print(f"  [WARN]  Skipping {skipped} samples with classes not in model: {skipped_classes}")

    df = df[mask].reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(model_classes)}
    df["true_label"] = df["true_class"].map(class_to_idx)

    print(f"       Evaluating on {len(df)} samples across {len(model_classes)} classes")
    print(f"       Class distribution:")
    for cls in model_classes:
        count = (df["true_class"] == cls).sum()
        print(f"         {cls:>5s}: {count:>5d}  ({100 * count / len(df):.1f}%)")

    return df


def run_inference(model, df, transform, image_dir: Path, device: str):
    """Run inference on all images and return arrays of true/predicted labels."""
    print(f"\n[3/4] Running inference on {len(df)} images...")

    y_true = []
    y_pred = []
    y_prob = []
    errors = 0

    start_time = time.perf_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting", unit="img"):
        image_name = row["image"]
        image_path = image_dir / f"{image_name}.jpg"

        if not image_path.exists():
            errors += 1
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            y_true.append(row["true_label"])
            y_pred.append(pred_idx)
            y_prob.append(probs)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [WARN]  Error processing {image_name}: {e}")

    elapsed = time.perf_counter() - start_time
    imgs_per_sec = len(y_true) / elapsed if elapsed > 0 else 0

    print(f"\n       Inference complete!")
    print(f"       Processed: {len(y_true)} images in {elapsed:.1f}s ({imgs_per_sec:.1f} img/s)")
    if errors:
        print(f"  [WARN]  Skipped {errors} images due to errors")

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


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

    # --- Overall Metrics ---------------------------------------------------
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
    print(f"  {'─' * 42}")
    print(f"  {'Accuracy':<30s} {acc:>10.4f}")
    print(f"  {'Balanced Accuracy':<30s} {bal_acc:>10.4f}")
    print(f"  {'Cohen Kappa':<30s} {kappa:>10.4f}")
    print(f"  {'Matthews Corr. Coeff. (MCC)':<30s} {mcc:>10.4f}")
    print(f"  {'─' * 42}")
    print(f"  {'Precision (macro)':<30s} {prec_macro:>10.4f}")
    print(f"  {'Recall (macro)':<30s} {rec_macro:>10.4f}")
    print(f"  {'F1 Score (macro)':<30s} {f1_macro:>10.4f}")
    print(f"  {'─' * 42}")
    print(f"  {'Precision (weighted)':<30s} {prec_wt:>10.4f}")
    print(f"  {'Recall (weighted)':<30s} {rec_wt:>10.4f}")
    print(f"  {'F1 Score (weighted)':<30s} {f1_wt:>10.4f}")

    # Top-k accuracy (if more than 2 classes)
    if len(classes) > 2 and y_prob is not None:
        for k in [3, 5]:
            if k <= len(classes):
                topk = top_k_accuracy_score(y_true, y_prob, k=k, labels=range(len(classes)))
                print(f"  {'Top-' + str(k) + ' Accuracy':<30s} {topk:>10.4f}")

    # --- Per-Class Classification Report -----------------------------------
    print(f"\n{'─' * 70}")
    print("  Per-Class Classification Report")
    print(f"{'─' * 70}")
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    print(report)

    # --- Confusion Matrix --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    print(f"{'─' * 70}")
    print("  Confusion Matrix (rows=true, cols=predicted)")
    print(f"{'─' * 70}")

    # Print as a nice table
    header = "        " + "  ".join(f"{c:>6s}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {classes[i]:>5s} {row_str}")

    # --- Save confusion matrix as image ------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalised confusion matrix (percentages)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm,
            annot=cm,           # show raw counts
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax.set_ylabel("True", fontsize=13, fontweight="bold")
        ax.set_title(
            f"DINOv2-IRIC Confusion Matrix\n"
            f"Accuracy={acc:.4f}  |  F1(macro)={f1_macro:.4f}  |  Kappa={kappa:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"\n  [OK] Confusion matrix saved -> {cm_path}")

        # Also save normalised version
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5,
            linecolor="white",
            ax=ax2,
        )
        ax2.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax2.set_ylabel("True", fontsize=13, fontweight="bold")
        ax2.set_title(
            f"DINOv2-IRIC Normalised Confusion Matrix\n"
            f"Balanced Acc={bal_acc:.4f}  |  F1(weighted)={f1_wt:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        cm_norm_path = OUTPUT_DIR / "confusion_matrix_normalised.png"
        fig2.savefig(cm_norm_path, dpi=150)
        plt.close(fig2)
        print(f"  [OK] Normalised confusion matrix saved -> {cm_norm_path}")

    except ImportError:
        print("\n  [WARN]  matplotlib/seaborn not installed — skipping confusion matrix image.")
        print("     Install via: pip install matplotlib seaborn")

    # --- Save metrics to CSV -----------------------------------------------
    metrics_dict = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "mcc": mcc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_wt,
        "recall_weighted": rec_wt,
        "f1_weighted": f1_wt,
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  [OK] Metrics CSV saved -> {metrics_path}")

    # Save per-class report as CSV
    report_dict = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T
    report_path = OUTPUT_DIR / "per_class_report.csv"
    report_df.to_csv(report_path)
    print(f"  [OK] Per-class report saved -> {report_path}")

    print(f"\n{'=' * 70}")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  DINOv2-IRIC  —  Test Set Evaluation")
    print("=" * 70 + "\n")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Verify paths
    if not CHECKPOINT.exists():
        sys.exit(f"  [FAIL] Checkpoint not found: {CHECKPOINT}")
    if not CSV_PATH.exists():
        sys.exit(f"  [FAIL] Ground truth CSV not found: {CSV_PATH}")
    if not IMAGE_DIR.exists():
        sys.exit(f"  [FAIL] Image directory not found: {IMAGE_DIR}")

    # Load model
    model, classes, image_size = load_checkpoint(CHECKPOINT, device)

    # Load ground truth
    df = load_ground_truth(CSV_PATH, classes)

    # Build transform
    transform = build_transform(image_size)

    # Run inference
    y_true, y_pred, y_prob = run_inference(model, df, transform, IMAGE_DIR, device)

    if len(y_true) == 0:
        sys.exit("  [FAIL] No images were successfully processed!")

    # Compute and display metrics
    compute_and_display_metrics(y_true, y_pred, y_prob, classes)


if __name__ == "__main__":
    # Add backend dir to path so we can import the app package
    sys.path.insert(0, str(BACKEND_DIR))
    main()
