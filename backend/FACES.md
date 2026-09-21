# Standalone presenter-face API

## Categorized outputs (new runs)

All usable detected face tracks now save one best JPEG, including tracks that
do not qualify as presenters. New output layout:

~~~text
output/faces/{face_job_id}/
  output_YYYY-MM-DD_HH-MM-SS_microseconds_UTC/
    narrator/
      track-0001.jpg
    static/
      track-0002.jpg
  faces.json
  status.json
  worker.log
~~~

The folder timestamp is UTC export time, not a video timestamp. Existing runs
are not moved or reprocessed. The JSON contains faces (all saved tracks),
narrators, static_faces, and output_folder. The legacy presenters array remains
top-N; max_presenters does not discard crops or classify excess narrators static.

Narrator means a likely presenter based on body movement, not confirmed speech.
Static means no qualifying body-movement evidence; classification_evidence
distinguishes low movement from insufficient body evidence. Independent face
scanning also includes faces with no detectable body, labeled uncertain/static.
This is not guaranteed detection of every face or one image per unique identity.
Quality thresholds and sampling still apply. Both categories use the image API.

This feature lives in the existing backend and uses the existing Python
environment and requirements.txt. It does not start OCR/audio, alter their
settings or scheduling, or integrate with the frontend yet.

## Start and test

From the backend directory:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
~~~

Open http://127.0.0.1:8001/docs and expand **presenter faces (standalone
validation)**. Use POST /api/v1/faces/extract, click Try it out, choose the
video, and execute. Default sampling is 2 FPS and up to 3 presenter tracks.

Alternatively:

~~~powershell
curl.exe -X POST "http://127.0.0.1:8001/api/v1/faces/extract" -F "file=@C:/Users/pjaja/Downloads/DM_IN_ENG_MCN_2018_L1T5.mp4" -F "sample_rate_fps=2" -F "max_presenters=3"
~~~

The 202 response contains face_job_id, status_url, and results_url. Poll the
status URL until completed or failed, then open the results URL. A queued job
waits for earlier FACE jobs only. For speed comparisons on this CPU, do not run
OCR/audio simultaneously. No CPU scheduling integration has been added.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | /api/v1/faces/extract | Multipart video upload; start faces only |
| GET | /api/v1/faces/status/{face_job_id} | Queue/stage, progress, errors, timestamps and duration |
| GET | /api/v1/faces/results/{face_job_id} | Ranked tracks, image URLs and metadata |
| GET | /api/v1/faces/images/{face_job_id}/{track_id} | Selected JPEG crop |
| GET | /api/v1/faces/download/{face_job_id} | Download JSON |
| DELETE | /api/v1/faces/jobs/{face_job_id} | Delete a terminal face job's upload, crops and metadata |

Status includes started_at, completed_at, processing_duration_sec and
frames_processed. Runtime includes child-process startup, model loading,
decoding, inference and exports. Upload and queue wait are excluded.
The processing duration is finalized on success or failure; during processing,
clients can calculate an approximate timer from started_at.

## Models and selection

- MediaPipe Pose Landmarker Lite: body landmarks in full-frame and overlapping
  half-frame views, improving recall for small presenter insets.
- BlazeFace short-range: face detection on a source-resolution head-region
  crop, rather than downscaling the entire frame for face detection.
- Spatial association tracks nearby observations within scenes; **no identity
  recognition, names, or cross-scene face matching**.
- Body-joint displacement is corrected for an estimated global camera transform.
  A pixel-change check suppresses pose-estimation jitter on static photographs.
- A track needs repeated movement, a minimum moving fraction and visible time,
  and a usable face crop. Eligible tracks are ranked primarily by visible time.
- Crop selection prefers detection confidence, sharpness and face pixel area.
  Tiny faces and very blurred crops are rejected. No face enhancement is applied.

Face boxes use original-frame pixels and [left, top, right, bottom], with right
and bottom exclusive. crop_bbox_xyxy includes padding and exactly describes
the saved crop. timestamp_sec is the selected decoded frame's presentation
timestamp relative to video-stream start, not the extraction's wall-clock time.
Visible time and intervals are estimates from sampled body detections.

## Setup on another machine

~~~powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_face_models.py
~~~

MediaPipe 1.0.1 is in the main requirements file. There is no separate face
environment. The model-preparation script downloads pinned version-1 official
assets and records their URLs, sizes and SHA256 hashes in models/faces/source.json.
Runtime uses local files only.

Settings are in face_config.json, separate from the existing OCR/audio config.
The endpoint accepts sample_rate_fps from 0.5 to 5 and max_presenters from 1 to 10.

## Isolation, limits and retention

The API launches faces/worker.py with the current sys.executable in a child
process. A native model crash becomes a failed face job rather than terminating
the API worker. One face job runs at a time; at most four are admitted, including
uploads and queued jobs. Additional requests receive 429.

The worker time limit defaults to 1800 seconds. Shutdown/timeout stops only the
owned face child process tree. Unfinished jobs are reported as interrupted after
a backend restart. Completed face artifacts remain readable.

Use one Uvicorn API worker, as with the existing application. This is a local
validation feature, not a distributed or authenticated production job service.

Uploads: temp/faces/{face_job_id}/

Artifacts, status, configuration and worker.log: output/faces/{face_job_id}/

Face images are personal data. They remain locally until deleted through the
face DELETE endpoint. That deletes only this face job, not any OCR/audio data.
No automatic retention policy or external identity service has been added.

## Validation and limitations

- 12 face-specific tests plus 2 existing API tests passed (14 total).
- The actual main application's API schema includes all six face routes and
  retains the existing OCR endpoint.
- Real model loading and face detection passed on a source-video frame.
- A roughly six-second clip of the supplied video sampled 13 frames and selected
  one crop of the woman presenter, excluding the static background faces.
  Worker-measured stage time: 9.916 seconds (not full API end-to-end runtime).
- Full-video accuracy, overall speed and track continuity remain for manual
  validation. The short clip produced four tracks and selected one with four
  seconds of observed visibility; this is not a six-second identity guarantee.
- Body movement identifies likely presenters, **not confirmed narrators**.
  Other moving people can qualify, while nearly motionless narrators may not.
- Misses, occlusion, large movement and scene transitions can split one person
  into multiple tracks. Global camera/scene-change compensation is heuristic.
  Cross-scene identity merging is explicitly deferred.
- If nothing meets the criteria, results contain an empty presenters list and
  a warning, not a fabricated face.

Focused tests (no video/model inference):

~~~powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_face_tracking.py tests/test_faces_api.py
~~~

Sources: [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
and [Face Detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector).


### Duplicate photos

New face extractions suppress identical and conservatively matched near-duplicate
photos across both categories within the job. Matching uses an exact byte hash,
or perceptual hash plus close RGB pixels and compatible aspect ratio. Low-detail
images use exact matching only. Narrator evidence takes precedence, then crop
quality. Only retained images are saved and shown; JSON reports
`duplicate_photos_removed` and each retained image lists `duplicate_track_ids`.
Times and movement evidence remain those of the retained track; they are not
merged across people or scenes. Different poses may remain. This is photo
similarity, not identity recognition. Existing results are unchanged.
