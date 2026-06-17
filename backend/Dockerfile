# Dinov2-BigEarthS2 backend — local containerized deployment.
#
# Build:  docker build -t dinov2-bigearths2-backend ./backend
# Run:    docker run --gpus all -p 8000:8000 -v $(pwd)/backend/checkpoints:/app/checkpoints dinov2-bigearths2-backend
#
# CPU-only: drop the --gpus flag.

FROM python:3.11-slim

# Avoid Python writing .pyc files and buffer stdout for clean logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code.
COPY app/ ./app/

# Create runtime dirs (checkpoint is expected to be volume-mounted).
RUN mkdir -p /app/checkpoints /app/logs

EXPOSE 8000

# Reload disabled in container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
