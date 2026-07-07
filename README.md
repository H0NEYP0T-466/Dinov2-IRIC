# Dinov2-IRIC

**Skin lesion classification** with a DINOv2-B backbone (86M params) on **ISIC 2019** (9 classes). Three components:

1. **Google Colab training script** — phased fine-tuning, checkpointing, curves, confusion matrix.
2. **FastAPI backend** — inference server with comprehensive logging.
3. **React + TypeScript frontend** — upload an image, get predictions.

Single-label classification (9 skin-lesion categories). RGB dermoscopic images.

---

## ISIC 2019 Classes

| Code | Full Name | Description |
|------|-----------|-------------|
| MEL  | Melanoma | Malignant skin tumour from melanocytes |
| NV   | Melanocytic nevus | Benign mole |
| BCC  | Basal cell carcinoma | Common skin cancer |
| AK   | Actinic keratosis | Pre-cancerous rough patch |
| BKL  | Benign keratosis | Seborrheic keratosis / solar lentigo |
| DF   | Dermatofibroma | Benign fibrous nodule |
| VASC | Vascular lesion | Blood vessel related |
| SCC  | Squamous cell carcinoma | Skin cancer from squamous cells |
| UNK  | Unknown | None of the above |

---

## Repository layout

```
Dinov2-IRIC/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point (lifespan, CORS, router)
│   │   ├── config.py            # Settings + shared ISIC_CLASSES list
│   │   ├── api/routes.py        # /predict, /health, /classes
│   │   ├── models/dinov2.py     # DINOv2Classifier + checkpoint loader
│   │   ├── services/predictor.py
│   │   ├── services/logger.py
│   │   └── utils/image_processing.py
│   ├── checkpoints/             # <- place model_best.pth here
│   ├── dataset/                 # ISIC 2019 dataset (not committed)
│   ├── logs/                    # server log files written here
│   ├── requirements.txt
│   └── Dockerfile
├── kaggle/
│   ├── trainCollab.py           # single-file Colab training script
│   └── train_requirements.txt
├── src/                         # React frontend (repo root)
│   ├── App.tsx
│   ├── components/
│   ├── services/api.ts
│   └── types/index.ts           # shared types + CLASS_NAMES
├── package.json, vite.config.ts, tsconfig*.json
└── README.md
```

> The 9-class nomenclature is the **single source of truth** and is duplicated in three places that must stay in sync: `backend/app/config.py` (`ISIC_CLASSES`), `kaggle/trainCollab.py` (`ISIC_CLASSES`), and `src/types/index.ts` (`CLASS_NAMES`).

---

## Part 1 — Training (Google Colab)

A single Python script — upload to Colab and run.

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install timm scikit-learn matplotlib pandas pillow tensorboard

# Run training
!python trainCollab.py
```

**Dataset setup**: Place the ISIC 2019 dataset in your Google Drive:
```
/content/drive/MyDrive/dataset/
├── ISIC_2019_Training_GroundTruth.csv
└── ISIC_2019_Training_Input/
    ├── ISIC_0000000.jpg
    ├── ISIC_0000001.jpg
    └── ...
```

The script features:

- Reads ISIC 2019 CSV + images from Google Drive (auto-detects paths).
- 9-class single-label classification with **CrossEntropyLoss** + class weights.
- 3-phase fine-tuning: **head** (5 ep, lr 1e-3) → **last 2 blocks** (10 ep, lr 1e-4) → **full** (≤15 ep, lr 1e-5).
- AdamW (wd 0.01), `ReduceLROnPlateau` (factor 0.5, patience 3), early stopping (patience 7).
- FP16 mixed precision, gradient accumulation, gradient clipping.
- Stratified 80/20 train/val split with weighted random sampling.
- Rich data augmentation: random crop, flip, rotate, colour jitter, erasing.
- **Saves `epoch{N}.pth` every epoch** + `model_best.pth` on val accuracy improvement.
- **Sample prediction images every 5 epochs** (4×4 grid with true/pred labels).
- Training curves (2×2 grid) every 5 epochs, TensorBoard logs.
- Confusion matrix + per-class classification report at end.
- Class distribution visualisation.
- GPU memory monitoring.
- `training_history.json` export.

Outputs land in `/content/drive/MyDrive/Dinov2-IRIC-output/`. **Copy `model_best.pth`** into `backend/checkpoints/`.

---

## Part 2 — Backend (FastAPI)

### Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# Place the trained model:
#   model_best.pth  ->  backend/checkpoints/model_best.pth

python -m app.main              # or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at **http://localhost:8000**, Swagger UI at **http://localhost:8000/docs**.

### Endpoints

| Method | Path                       | Description                          |
|--------|----------------------------|--------------------------------------|
| POST   | `/api/v1/predict`          | Upload an image → skin lesion prediction |
| GET    | `/api/v1/health`           | Server/model status                  |
| GET    | `/api/v1/classes`          | The 9 ISIC class names               |
| GET    | `/`                        | Info + doc links                     |

### Logging

Every operation is logged to both the console (INFO) and a timestamped file under `backend/logs/` (DEBUG): server startup, model load (param count, device, arch), each request (id, filename, size, content-type), every preprocessing step (original size/mode/pixel range → tensor shape/normalized range), inference details (raw logits, softmax probs, top-1 class, time), and full stack traces on errors.

### Docker (optional)

```bash
docker build -t dinov2-iric-backend ./backend
# CPU-only:
docker run -p 8000:8000 -v $(pwd)/backend/checkpoints:/app/checkpoints dinov2-iric-backend
# With GPU:
docker run --gpus all -p 8000:8000 -v $(pwd)/backend/checkpoints:/app/checkpoints dinov2-iric-backend
```

---

## Part 3 — Frontend (React + TS + Vite)

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` → `http://localhost:8000` (see `vite.config.ts`). For a production build pointed elsewhere, set `VITE_API_BASE` (see `.env.example`).

UI: drag-and-drop or browse a dermoscopic image → preview → **Classify** → predicted skin lesion class with confidence bars, a metadata strip (model, inference ms, threshold, request id), and an expandable view of **all 9 class probabilities**. Dark theme, responsive.

```bash
npm run build      # typecheck + production build -> dist/
npm run preview    # serve the production build
```

---

## End-to-end test

1. Start the backend (with `model_best.pth` in place).
2. Start the frontend (`npm run dev`).
3. Open http://localhost:5173, upload a dermoscopic image, click **Classify**.
4. Verify the predicted skin lesion class renders with confidence.
5. Watch the backend terminal — every preprocessing and inference step is logged.
6. Try the `/predict` endpoint directly via the Swagger UI at `/docs` or:

```bash
curl -X POST http://localhost:8000/api/v1/predict -F "file=@test.jpg"
```

---

## Model architecture

```
Input (1, 3, 224, 224)
   │
   ├── DINOv2-B backbone  (timm: vit_base_patch14_dinov2.lvd142m, 768-d, ~86M params)
   │       └── num_classes=0  → pooled 768-d feature
   │
   └── Head:
           Linear(768 → 512) → ReLU → Dropout(0.3) → Linear(512 → 9)
   │
   └── raw logits (1, 9)   # softmax applied at inference / CrossEntropyLoss at training
```

No softmax inside the model — probabilities come from `torch.softmax` at inference, and `CrossEntropyLoss` applies it internally during training. Identical architecture across training, backend loader, and checkpoint format.

---

## Configuration overrides

Backend settings can be overridden via environment variables (see `backend/app/config.py`):

| Var                   | Default                                   | Notes                          |
|-----------------------|-------------------------------------------|--------------------------------|
| `MODEL_CHECKPOINT`    | `backend/checkpoints/model_best.pth`      | Path to trained weights        |
| `DEVICE`              | `auto` (cuda if available, else cpu)      | Force a device                 |
| `INFERENCE_THRESHOLD` | `0.5`                                     | Min confidence to report       |
| `CORS_ORIGINS`        | `["*"]`                                   | Restrict in production         |
