/**
 * Shared TypeScript types for the Dinov2-BigEarthS2 frontend.
 *
 * The `CLASS_NAMES` array below MUST mirror `CLASSES_43` in
 * `backend/app/config.py` and `CFG.CLASSES_43` in the Kaggle notebook. Keep all
 * three copies in sync when the class list changes.
 */

/** A single class predicted above threshold. */
export interface Prediction {
  class_name: string;
  confidence: number;
  class_index: number;
}

/** Full response payload from `POST /api/v1/predict`. */
export interface PredictionResponse {
  request_id: string;
  success: boolean;
  predictions: Prediction[];
  /** All 43 class names -> probability. */
  all_probabilities: Record<string, number>;
  inference_time_ms: number;
  threshold: number;
  model: string;
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
 * Official BigEarthNet-S2 43-class nomenclature (CORINE-derived), matching
 * torchgeo's `class_sets[43]` and the backend `CLASSES_43`. Used for the full
 * probability-distribution view in the UI.
 */
export const CLASS_NAMES: readonly string[] = [
  'Continuous urban fabric',
  'Discontinuous urban fabric',
  'Industrial or commercial units',
  'Road and rail networks and associated land',
  'Port areas',
  'Airports',
  'Mineral extraction sites',
  'Dump sites',
  'Construction sites',
  'Green urban areas',
  'Sport and leisure facilities',
  'Non-irrigated arable land',
  'Permanently irrigated land',
  'Rice fields',
  'Vineyards',
  'Fruit trees and berry plantations',
  'Olive groves',
  'Pastures',
  'Annual crops associated with permanent crops',
  'Complex cultivation patterns',
  'Land principally occupied by agriculture, with significant areas of natural vegetation',
  'Agro-forestry areas',
  'Broad-leaved forest',
  'Coniferous forest',
  'Mixed forest',
  'Natural grassland',
  'Moors and heathland',
  'Sclerophyllous vegetation',
  'Transitional woodland/shrub',
  'Beaches, dunes, sands',
  'Bare rock',
  'Sparsely vegetated areas',
  'Burnt areas',
  'Inland marshes',
  'Peatbogs',
  'Salt marshes',
  'Salines',
  'Intertidal flats',
  'Water courses',
  'Water bodies',
  'Coastal lagoons',
  'Estuaries',
  'Sea and ocean',
] as const;

/** Runtime guard: CLASS_NAMES stays in sync with the backend's 43 classes. */
if (CLASS_NAMES.length !== 43) {
  throw new Error(
    `CLASS_NAMES must contain 43 entries (got ${CLASS_NAMES.length}). ` +
      'Sync with backend/app/config.py CLASSES_43.',
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
