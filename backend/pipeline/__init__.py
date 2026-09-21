"""Video ingestion, frame processing, OCR, and export pipeline."""

from utils.windows_runtime import prepare_native_runtime

# Also protect scripts that import a pipeline directly instead of main:app.
prepare_native_runtime()
