/**
 * A reusable horizontal confidence bar for a single class.
 *
 * - Detected (value >= threshold): green fill.
 * - Not detected: gray fill.
 */

interface ConfidenceBarProps {
  name: string;
  value: number; // 0..1
  threshold: number;
  /** Hide very-low bars in the full distribution to reduce noise. */
  dim?: boolean;
}

export function ConfidenceBar({ name, value, threshold, dim = false }: ConfidenceBarProps) {
  const detected = value >= threshold;
  const pct = Math.max(0, Math.min(100, value * 100));

  return (
    <div className={`confidence-bar ${dim ? 'confidence-bar--dim' : ''}`}>
      <div className="confidence-bar__label">
        <span className="confidence-bar__name" title={name}>
          {name}
        </span>
        <span className={`confidence-bar__value ${detected ? 'is-detected' : ''}`}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="confidence-bar__track">
        <div
          className={`confidence-bar__fill ${detected ? 'is-detected' : ''}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
