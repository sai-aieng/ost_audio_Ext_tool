# Video OCR Pipeline

**Deploy on Render:** see [DEPLOYMENT.md](DEPLOYMENT.md). Docker and the Render
Blueprint serve the frontend and API together.

A two-application video text extraction system:

- **backend/** contains the FastAPI service, PaddleOCR pipeline, tests, and
  runtime storage.
- **frontend/** contains the React 19 and Vite user interface.

Local development uses separate frontend and backend servers. The frontend calls
**http://127.0.0.1:8001/api/v1**. On Render, FastAPI serves the built frontend
and the browser calls the same-origin **/api/v1** API.

## Architecture

~~~text
.
├── backend/
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── api/
│   ├── pipeline/
│   ├── storage/
│   ├── utils/
│   ├── tests/
│   ├── temp/
│   └── output/
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
~~~

The processing sequence is:

~~~text
upload → validate/probe → sample frames → pHash deduplicate
       → OpenCV preprocess → PaddleOCR → clean/deduplicate → export
~~~

## Prerequisites

- Python compatible with the pinned packages in **backend/requirements.txt**
- Node.js and npm
- The FFmpeg and ffprobe executables available on PATH
- Sufficient disk space for uploaded videos and extracted frames

PaddleOCR downloads its language model the first time OCR actually runs. GPU use
is disabled by default.

## Start the backend

From the repository root:

~~~powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
~~~

Open **http://localhost:8001/docs** for the generated OpenAPI interface. A
readiness endpoint is available at **GET /health**.

## Start the frontend

In a second terminal:

~~~powershell
cd frontend
npm install
npm run dev
~~~

Open **http://localhost:5174**. The frontend API URL is defined in
**frontend/src/api/client.js**.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/v1/upload | Validate and store one multipart video |
| POST | /api/v1/process/{job_id} | Start background OCR processing |
| GET | /api/v1/status/{job_id} | Poll status, counters, and progress |
| GET | /api/v1/results/{job_id} | Return extraction results |
| GET | /api/v1/results/{job_id}?format=json | Return a format download URL |
| GET | /api/v1/results/{job_id}/download?format=json | Download an artifact |
| DELETE | /api/v1/jobs/{job_id} | Remove a finished job and its files |

Supported upload extensions are MP4, AVI, MKV, and MOV. The default upload limit
is 500 MB.

Example processing override:

~~~json
{
  "processing_mode": "fast_cpu",
  "sample_rate_fps": 10.0,
  "confidence_threshold": 0.6,
  "output_formats": ["json", "csv", "txt"]
}
~~~

Two processing modes are available:

- **accuracy** uses 960px preprocessing, denoising, angle classification,
  whole-frame change detection, and a one-second safety check.
- **fast_cpu** uses a 960px OCR image to retain more small-text detail than the
  previous 640px profile. It skips denoising and angle classification, compares
  320px thumbnails in an 8-by-8 grid, and has a ten-second safety check.

Both modes compare against the last OCR-confirmed image and inspect known text
regions at source resolution to catch small changes and cumulative drift. A
bounded per-job cache reuses OCR for identical decoded images, including at safety
checks; changed images never match this exact-image cache. Source coordinates are
scaled before rounding and clipped to the source image. Merged detections retain
the highest-confidence wording, which is not a guarantee of correct spelling.

The frontend selects **fast_cpu** by default. At 10 FPS the sampling interval is
0.1 seconds, not a guarantee of boundary accuracy: skipped detections are carried
forward estimates. Subtle new text outside known text regions may remain unseen
until a safety check, and fuzzy grouping can merge similar real text changes.
True source-frame timestamp refinement is not implemented. The existing 0.15s
merge window remains unchanged. Higher OCR resolution and stricter change checks
trade some speed for detail. Sub-three-minute processing is not yet benchmarked;
benefits depend on repeated content. Logs now separate preprocessing/inference
time and report exact-cache hits and carried samples.

## Optional ONNX CPU engine

Pipeline settings now includes an independent **Inference engine** selector.
Paddle remains the default. Select **ONNX Runtime CPU (experimental)** to compare
the same OCR model names, 960px images, and sampling/change-detection settings.
No accuracy or speed equivalence has yet been established.

Prepare once from the backend directory:

~~~powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-onnx.txt
.\.venv\Scripts\python.exe scripts/prepare_onnx_models.py
~~~

Preparation downloads official PaddlePaddle exports at pinned revisions, verifies
their SHA-256 checksums and preprocessing/postprocessing configuration against
cached Paddle models, and stores them in git-ignored models/onnx directories.
It does not run inference. Source manifests record the downloaded revisions.
The local converter failed with a Windows DLL incompatibility, so official
exports are used instead. Numerical equivalence to the cached weights is still
unverified. Optional --convert-cached requires a working Paddle2ONNX installation.

Restart the backend and refresh the frontend. Select the same processing mode
for both runs, changing only the inference engine. Job status shows the actual
engine and any fallback warning. On ONNX initialization or prediction errors,
the job retries with Paddle and stays on Paddle for its remaining OCR calls.
Set ocr.onnx_fallback_to_paddle to false to fail instead of falling back.
The models for text-line orientation are included so Accuracy mode retains it.

Logs include requested/effective engines and refresh-reason counts; DEBUG logs
include each refresh frame and timestamp. This does not relax the change guard.
Tests and video benchmarks for this engine integration are left to manual testing.

## Audio transcription engine

Audio now defaults to **Faster-Whisper 1.2.1**, using the multilingual **base**
model, **float32** computation, two CPU threads, and one model worker. OCR and
audio still run in parallel; scheduling and OCR thresholds are unchanged.
No measured speedup or identical text/timestamp output is claimed yet.

After installing backend/requirements.txt, prepare the model once from backend:

~~~powershell
.\.venv\Scripts\python.exe scripts/prepare_audio_model.py
~~~

This downloads the converted base model from Systran/faster-whisper-base and
records its revision in models/faster-whisper/base/source.json. Runtime uses local
files only, so model-download time is excluded from manual processing comparisons.
The original Whisper package remains installed: set audio_transcription.engine
to whisper in backend/config.yaml and restart to use the old engine. The new
engine does not silently fall back; initialization failures show as audio errors.

The adapter explicitly keeps segment timestamps, automatic language detection
unless configured, previous-text context, and the temperature fallback schedule.
It uses beam_size=1 and best_of=1 to match the previous direct API's single
candidate behavior. Silence filtering and word-level timestamp alignment are
disabled, as before. It consumes the complete lazy segment iterator before
exporting results or stopping the audio timer.

The Audio panel shows its engine and independent timing. Logs distinguish
audio_model_load_start, audio_model_loaded, audio_transcribed, and audio_complete.
Transcription timing includes audio decoding, feature extraction, and generation.
The timer includes model loading and export, but excludes upload time.

## Configuration

Job status displays the local processing start/finish timestamps and a live
elapsed estimate. On completion or failure it shows the backend-measured OCR
duration (HH:MM:SS.mmm), including model initialization/fallback, extraction, and
export. Upload time and the independently running audio task are excluded.
The timer is available for new runs; older records have no retroactive timing.
The Audio transcription panel has its own independent start/finish timestamps
and live/final duration, including Whisper initialization, transcription, and
transcript export. It stops on success or failure independently of OCR.

All backend tuning is in **backend/config.yaml**:

- upload limits and runtime directories
- frame sampling rate
- selectable Accuracy and Fast CPU processing profiles
- perceptual-hash threshold
- OpenCV resize, contrast, denoise, and sharpening
- PaddleOCR language, angle classification, GPU, and confidence
- output formats
- rotating log size and retention

Relative runtime paths resolve from the backend directory, regardless of the
shell's current working directory.

## Tests

After installing backend dependencies:

~~~powershell
cd backend
python -m pytest -q
~~~

The tests mock ffmpeg probing/execution and PaddleOCR inference, so they do not
require a real video or model download.

To verify a frontend production build:

~~~powershell
cd frontend
npm run build
~~~

## Runtime and production notes

### Standalone presenter-face validation

POST /api/v1/faces/extract uploads a video for body-movement-based presenter face
selection. A separate frontend panel provides upload, status and image galleries.
It does not start the OCR/audio workflow. MediaPipe is included in the existing
backend requirements. See [face API setup and testing](backend/FACES.md).

- On Windows, startup selects a compatible installed Microsoft C++ runtime before
  importing native ML libraries. This avoids Anaconda's old MSVCP140.dll causing
  a CTranslate2 model-load access violation (0xC0000005). No system/Anaconda files
  are replaced. If runtime validation fails, update the Microsoft Visual C++ x64
  Redistributable and restart the backend before importing ML libraries.
- Audio status requests time out and retry on connection failures; a missing job
  after a backend restart is shown explicitly and requires a fresh upload.
- Jobs are held in a process-local dictionary. Use Redis or a database before
  running multiple Uvicorn workers or replicas.
- FastAPI BackgroundTasks runs OCR in the API process. A durable queue is the
  appropriate next step for long-running or high-volume workloads.
- CORS defaults to local frontend origins. Set CORS_ORIGINS for a separate
  frontend; the Render deployment uses the same origin.
- Uploaded videos remain until the job is deleted. Extracted frame directories
  are removed after successful processing.
- Logs rotate under **backend/logs/**; result files are stored under
  **backend/output/{job_id}/**.
