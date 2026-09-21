# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    FRONTEND_DIST=/app/frontend/dist \
    DATA_DIR=/var/data
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements-render.txt ./requirements-render.txt
RUN python -m pip install -r requirements-render.txt
COPY backend/ ./
# Download models into the image, not Git. Builds fail if preparation fails.
RUN python scripts/prepare_face_models.py \
    && python scripts/check_face_runtime.py \
    && python scripts/prepare_audio_model.py \
    && python scripts/prepare_ocr_models.py
COPY --from=frontend /build/frontend/dist /app/frontend/dist
EXPOSE 10000
CMD ["python", "scripts/start_server.py"]
