# ── Base image ────────────────────────────────────────────────────────────────
# CPU-only torch — keeps image size reasonable on HF free tier
FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app files ────────────────────────────────────────────────────────────
COPY main.py .
COPY pipeline_3.py .
COPY offside_2.py .
COPY validator.py .
COPY index.html .

# ── Copy model weights ────────────────────────────────────────────────────────
COPY football-pitch-detection.pt .
COPY football-player-detection.pt .

# ── Create runtime dirs ───────────────────────────────────────────────────────
RUN mkdir -p uploads outputs

# ── HuggingFace Spaces runs as non-root user 1000 ────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Expose port (HF Spaces expects 7860) ─────────────────────────────────────
EXPOSE 7860

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
