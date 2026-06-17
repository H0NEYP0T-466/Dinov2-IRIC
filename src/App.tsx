/**
 * Dinov2-BigEarthS2 — single-page satellite image classifier.
 *
 * Upload an image -> POST to backend -> display multi-label predictions.
 */

import { useState } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { PredictionResults } from './components/PredictionResults';
import { predictImage } from './services/api';
import type { PredictionResponse } from './types';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictImage(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>Dinov2-BigEarthS2</h1>
        <p className="app__subtitle">
          Multi-label satellite image classification · DINOv2-B · 43 classes
        </p>
      </header>

      <main className="app__main">
        <section className="panel">
          <h2 className="panel__title">1 · Upload image</h2>
          <ImageUploader onClassify={handleClassify} loading={loading} />
        </section>

        <section className="panel">
          <h2 className="panel__title">2 · Predictions</h2>
          {loading && (
            <div className="spinner-wrap">
              <div className="spinner" />
              <p>Running inference…</p>
            </div>
          )}
          {!loading && error && (
            <div className="alert alert--error">
              <strong>Error:</strong> {error}
            </div>
          )}
          {!loading && !error && !result && (
            <p className="placeholder">
              Upload an image and click <em>Classify</em> to see predictions.
            </p>
          )}
          {!loading && result && <PredictionResults result={result} />}
        </section>
      </main>

      <footer className="app__footer">
        <span>DINOv2-B · BigEarthNet-S2 · local deployment</span>
      </footer>
    </div>
  );
}

export default App;
