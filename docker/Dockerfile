# Multi-stage build for the Heart Disease prediction API
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Optional proxy support for building behind a corporate proxy. These are
# build-time only (passed via --build-arg) and are NOT persisted to the image.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=""

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY} \
    pip install --upgrade pip \
    && http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY} \
    pip install -r requirements.txt

# Copy source and the trained model.
COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Python-based healthcheck avoids needing curl / extra OS packages.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else sys.exit(1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
