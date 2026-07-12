/**
 * Shared TypeScript types for the Dinov2-IRIC frontend.
 *
 * The `CLASS_NAMES` array below MUST mirror `ISIC_CLASSES` in
 * `backend/app/config.py` and `ISIC_CLASSES` in the training script. Keep all
 * three copies in sync when the class list changes.
 */

/** A single class predicted above threshold. */
export interface Prediction {
  class_name: string;
  full_name: string;
  confidence: number;
  class_index: number;
}

/** Full response payload from `POST /api/v1/predict`. */
export interface PredictionResponse {
  request_id: string;
  success: boolean;
  predictions: Prediction[];
  /** All 8 class names -> probability. */
  all_probabilities: Record<string, number>;
  inference_time_ms: number;
  threshold: number;
  model: string;
  top_class: string;
  top_confidence: number;
  filename: string;
  error?: string | null;
}

/** Server / model status from `GET /api/v1/health`. */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  num_classes: number;
  device: string;
}

/** Class list from `GET /api/v1/classes`. */
export interface ClassesResponse {
  count: number;
  classes: string[];
}

/**
 * ISIC 2019 — 8-class skin-lesion nomenclature, matching the backend
 * `ISIC_CLASSES` and the training script. Used for the full
 * probability-distribution view in the UI.
 */
export const CLASS_NAMES: readonly string[] = [
  'MEL',   // Melanoma
  'NV',    // Melanocytic nevus
  'BCC',   // Basal cell carcinoma
  'AK',    // Actinic keratosis
  'BKL',   // Benign keratosis
  'DF',    // Dermatofibroma
  'VASC',  // Vascular lesion
  'SCC',   // Squamous cell carcinoma
] as const;

/** Full descriptive names for each ISIC class abbreviation. */
export const CLASS_FULL_NAMES: Record<string, string> = {
  MEL:  'Melanoma',
  NV:   'Melanocytic nevus',
  BCC:  'Basal cell carcinoma',
  AK:   'Actinic keratosis',
  BKL:  'Benign keratosis',
  DF:   'Dermatofibroma',
  VASC: 'Vascular lesion',
  SCC:  'Squamous cell carcinoma',
};

/** Runtime guard: CLASS_NAMES stays in sync with the backend's 8 classes. */
if (CLASS_NAMES.length !== 8) {
  throw new Error(
    `CLASS_NAMES must contain 8 entries (got ${CLASS_NAMES.length}). ` +
      'Sync with backend/app/config.py ISIC_CLASSES.',
  );
}

/** Error thrown by the API service for non-2xx responses. */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}
