/**
 * Prediction results display.
 *
 * Shows detected classes (above threshold) sorted by confidence, a metadata
 * strip, and an expandable section listing ALL 43 class probabilities.
 */

import { useMemo, useState } from 'react';
import { CLASS_NAMES, type PredictionResponse } from '../types';
import { ConfidenceBar } from './ConfidenceBar';

interface PredictionResultsProps {
  result: PredictionResponse;
}

export function PredictionResults({ result }: PredictionResultsProps) {
  const [showAll, setShowAll] = useState(false);

  // Sort the full distribution by probability desc for the expanded view.
  const sortedAll = useMemo(() => {
    return CLASS_NAMES.map((name) => ({
      name,
      value: result.all_probabilities[name] ?? 0,
    })).sort((a, b) => b.value - a.value);
  }, [result.all_probabilities]);

  return (
    <div className="results">
      <div className="results__header">
        <h2>
          {result.predictions.length > 0
            ? `${result.predictions.length} class${result.predictions.length > 1 ? 'es' : ''} detected`
            : 'No classes above threshold'}
        </h2>
        <div className="results__metadata">
          <span>{result.model}</span>
          <span>{result.inference_time_ms.toFixed(1)} ms</span>
          <span>threshold {result.threshold}</span>
          <span title={result.request_id}>id {result.request_id.slice(0, 8)}</span>
        </div>
      </div>

      {result.predictions.length > 0 ? (
        <div className="results__detected">
          {result.predictions.map((p) => (
            <ConfidenceBar
              key={p.class_index}
              name={p.class_name}
              value={p.confidence}
              threshold={result.threshold}
            />
          ))}
        </div>
      ) : (
        <p className="results__empty">
          The model found no class above the {result.threshold} threshold. See the full
          distribution below.
        </p>
      )}

      <details className="results__all" open={showAll} onToggle={(e) => setShowAll(e.currentTarget.open)}>
        <summary>All 43 class probabilities</summary>
        <div className="results__all-grid">
          {sortedAll.map(({ name, value }) => (
            <ConfidenceBar
              key={name}
              name={name}
              value={value}
              threshold={result.threshold}
              dim
            />
          ))}
        </div>
      </details>
    </div>
  );
}
