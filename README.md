# Video OCR Pipeline

A two-application video text extraction system:

- **backend/** contains the FastAPI service, PaddleOCR pipeline, tests, and
  runtime storage.
- **frontend/** contains the React 18 and Vite user interface.

The applications are intentionally independent. The frontend communicates with
the backend over HTTP at **http://localhost:8000/api/v1**; FastAPI does not build
or serve the frontend.

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
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
~~~

Open **http://localhost:8000/docs** for the generated OpenAPI interface. A
readiness endpoint is available at **GET /health**.

## Start the frontend

In a second terminal:

~~~powershell
cd frontend
npm install
npm run dev
~~~

Open **http://localhost:5173**. The frontend API URL is defined in
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
  "sample_rate_fps": 1.0,
  "confidence_threshold": 0.6,
  "hash_threshold": 10,
  "output_formats": ["json", "csv", "txt"]
}
~~~

## Configuration

All backend tuning is in **backend/config.yaml**:

- upload limits and runtime directories
- frame sampling rate
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

- Jobs are held in a process-local dictionary. Use Redis or a database before
  running multiple Uvicorn workers or replicas.
- FastAPI BackgroundTasks runs OCR in the API process. A durable queue is the
  appropriate next step for long-running or high-volume workloads.
- CORS permits every origin for local development. Restrict origins before
  deployment.
- Uploaded videos remain until the job is deleted. Extracted frame directories
  are removed after successful processing.
- Logs rotate under **backend/logs/**; result files are stored under
  **backend/output/{job_id}/**.
