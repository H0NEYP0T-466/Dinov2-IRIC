/**
 * API client for the Dinov2-BigEarthS2 backend.
 *
 * Base URL is configurable via the `VITE_API_BASE` env var (see `.env`),
 * defaulting to the Vite dev proxy path `/api/v1` so the browser and backend
 * can share an origin during development.
 */

import { ApiError, type ClassesResponse, type HealthResponse, type PredictionResponse } from '../types';

const API_BASE: string = 'http://localhost:8011/api/v1';

/** Accepted image MIME types, mirrors the backend allow-list. */
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return body?.detail ?? body?.message ?? `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status} ${res.statusText})`;
  }
}

/**
 * Upload an image and return multi-label predictions.
 * @throws {ApiError} on non-2xx responses or network failure.
 */
export async function predictImage(file: File): Promise<PredictionResponse> {
  if (!ACCEPTED_TYPES.includes(file.type) && file.type !== 'application/octet-stream') {
    throw new ApiError(
      `Unsupported file type "${file.type}". Use one of: ${ACCEPTED_TYPES.join(', ')}.`,
      400,
    );
  }

  const formData = new FormData();
  formData.append('file', file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE}. Is the server running?`,
      0,
    );
  }

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }

  return (await res.json()) as PredictionResponse;
}

/** Fetch server + model health. */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return (await res.json()) as HealthResponse;
}

/** Fetch the 43-class list. */
export async function fetchClasses(): Promise<ClassesResponse> {
  const res = await fetch(`${API_BASE}/classes`);
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return (await res.json()) as ClassesResponse;
}
