# Render deployment

1. In Render choose **New > Blueprint**.
2. Connect `sai-aieng/ost_audio_Ext_tool`, branch **main**.
3. Review `render.yaml`: one paid **2c-4g** web service (2 CPU / 4 GB RAM)
   and a **10 GB** persistent disk. Review charges before deploying.
4. Deploy. Docker builds the frontend, installs FFmpeg and CPU libraries,
   and downloads the face, Faster-Whisper base, and PaddleOCR models.
5. Open the assigned HTTPS URL. API docs: `/docs`; health check: `/health`.
   Test a short video for faces and OCR/audio before trying long videos.

For manual **New > Web Service**, choose Docker, leave Root Directory blank,
use `./Dockerfile`, leave Docker Command blank, set health check `/health`,
and attach a disk at `/var/data`. Frontend and API run in the same service.

## Runtime settings

- `PORT` is supplied by Render; startup binds `0.0.0.0:$PORT`.
- `DATA_DIR=/var/data` stores uploads, outputs, and logs on the mounted disk.
- `FRONTEND_DIST=/app/frontend/dist` serves the built UI (set in Docker).
- Production frontend calls `/api/v1`; local Vite calls port 8001.
- Optional `VITE_API_BASE_URL` overrides the API URL at frontend build time.
- Optional `CORS_ORIGINS` sets comma-separated allowed frontend origins.
- Keep one instance and one worker: jobs/queues are process-local.

Active jobs cannot resume after restart. Disk-backed files persist, but OCR/audio
live job IDs are not restored. Saved OCR exports can be reopened through the
saved-results selector. Finish processing before redeploying. Monitor disk use;
there is no automatic retention policy. The current app has no login system.

## Dependencies and validation

`backend/requirements-runtime.txt` is the runtime input;
`backend/requirements-render.txt` locks Linux/Python 3.10 dependencies for Docker.
Regenerate with:

```sh
uv pip compile backend/requirements-runtime.txt --python-version 3.10 --python-platform x86_64-manylinux_2_28 --only-binary :all: --no-annotate --no-header --output-file backend/requirements-render.txt
```

Docker uses Faster-Whisper. The optional original Whisper/PyTorch engine is
included only in local development requirements. Experimental ONNX OCR models
are not bundled; selecting ONNX uses the configured Paddle fallback unless
those models are separately prepared.

The free 512 MB plan is not a suitable default for the combined ML workload;
it also sleeps on idle and cannot attach a persistent disk. The paid size is a
starting point, not a performance guarantee.

Validation: 69 backend tests, 10 frontend tests, frontend production build,
and Linux binary dependency resolution passed. Docker is unavailable locally,
so verify the full image build/model preparation in Render and test a short video.

References: [Docker](https://render.com/docs/docker),
[compute plans](https://render.com/docs/compute-plans),
[disks](https://render.com/docs/disks).

The Docker build checks MediaPipe shared-library dependencies with `ldd`; it
does not initialize MediaPipe models inside the build sandbox. Native TCMalloc
CPU detection aborted during model initialization in the Render builder. Model
loading is still required at face-job runtime. For a full check on a runtime
host, run `python scripts/check_face_runtime.py` from the backend directory.
A passing build does not establish inference compatibility or available memory.

## Gemini video analysis

The separate Gemini workspace uses `gemini-3-flash-preview` by default. Set
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in Render > Environment. A local
`backend/.env` is loaded for development and is intentionally excluded from Git
and Docker. Set `GEMINI_MODEL` only to override the default model ID.

The original face and OCR/audio tools remain available. Gemini analysis uploads
video to Google's Files API, analyzes 60-second intervals, and creates the
five-column scene report as CSV/JSON. Whisper provides dialogue when selected;
uncheck it for a visual-only API test without loading Whisper on Render.
Face photos use Gemini boxes on up to 24 representative scene frames, with local
cropping and conservative duplicate filtering. Observation times are not full
appearance intervals or confirmed narrator identities. Scene timing/OCR require
review; context is an interpretation. Partial face failure preserves the report
and records a warning. Dialogue segments spanning a scene boundary appear in
both rows, unchanged, with original segment times in JSON.

Limits: 500 MB upload (existing configuration), 20 minutes per Gemini test,
one Gemini job at a time per process. Keep one worker/instance. Uploaded Google
files are deleted after processing where possible; local source video is removed
after the run. Results use the configured output storage and free-tier restarts
can remove them. Running jobs do not resume after a server restart. Existing
local model downloads are retained for the original tools. Whisper still needs
local memory; API offloading does not establish compatibility with 512 MB RAM.
