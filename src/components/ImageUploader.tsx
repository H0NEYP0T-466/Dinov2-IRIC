/**
 * Image upload component.
 *
 * Supports drag-and-drop and click-to-browse, shows a preview, the file name
 * and size, and a "Classify" button that triggers prediction. The component is
 * disabled (with a spinner) while a prediction is in flight.
 */

import { useCallback, useRef, useState } from 'react';

interface ImageUploaderProps {
  onClassify: (file: File) => void;
  loading: boolean;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function ImageUploader({ onClassify, loading }: ImageUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectFile = useCallback((f: File | null) => {
    setError(null);
    if (!f) return;
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError(`Unsupported file type "${f.type}". Use JPG, PNG, WEBP, BMP, or TIFF.`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0] ?? null;
      selectFile(f);
    },
    [selectFile],
  );

  const handleClassify = () => {
    if (file) onClassify(file);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="uploader">
      <div
        className={`uploader__dropzone ${dragging ? 'is-dragging' : ''} ${loading ? 'is-disabled' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        {preview ? (
          <img src={preview} alt="preview" className="uploader__preview" />
        ) : (
          <div className="uploader__hint">
            <div className="uploader__hint-icon">⬆</div>
            <p>Drag &amp; drop a skin lesion image here</p>
            <p className="uploader__hint-sub">Dermoscopy image · JPG, PNG, WEBP, BMP, or TIFF</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          className="uploader__input"
          onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
          disabled={loading}
        />
      </div>

      {error && <p className="uploader__error">{error}</p>}

      {file && (
        <div className="uploader__meta">
          <div className="uploader__file">
            <span className="uploader__filename" title={file.name}>
              {file.name}
            </span>
            <span className="uploader__filesize">{formatSize(file.size)}</span>
          </div>
          <div className="uploader__actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleClear}
              disabled={loading}
            >
              Clear
            </button>
            <button
              type="button"
              className="btn btn--primary"
              onClick={handleClassify}
              disabled={loading || !file}
            >
              {loading ? 'Classifying…' : 'Classify'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
