import { useEffect, useRef, useState } from "react";
import { getFaceDownloadUrl, getFaceImageUrl, uploadFaceVideo } from "../api/client";
import { useFacePoller } from "../hooks/useFacePoller";

function seconds(value) {
  return value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(2) + " s" : "—";
}

function FaceGallery({ title, faces, jobId }) {
  return (
    <section>
      <h3>{title} ({faces.length})</h3>
      {!faces.length && <p>No faces in this category.</p>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        {faces.map((face) => (
          <figure key={face.track_id} style={{ margin: 0, maxWidth: 280 }}>
            <a href={getFaceImageUrl(jobId, face.track_id)} target="_blank" rel="noreferrer">
              <img src={getFaceImageUrl(jobId, face.track_id)} alt={title + ": " + face.track_id}
                loading="lazy" style={{ width: "100%", height: 180, objectFit: "contain", background: "#F1F5F9", borderRadius: 8 }} />
            </a>
            <figcaption>
              <strong>{face.track_id}</strong><br />
              Video timestamp: {seconds(face.timestamp_sec)}<br />
              Visible time: {seconds(face.visible_time_sec)}<br />
              <a href={getFaceImageUrl(jobId, face.track_id)} target="_blank" rel="noreferrer">Open full image</a>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

export function FaceExtractionPanel() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [now, setNow] = useState(Date.now());
  const uploadLock = useRef(false);
  const { status, results, isPolling, error } = useFacePoller(jobId);
  const busy = submitting || isPolling;
  useEffect(() => {
    if (!isPolling) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [isPolling]);

  function selectFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!/\.(mp4|avi|mkv|mov)$/i.test(selected.name) || selected.size === 0) {
      setFile(null);
      setUploadError("Choose a non-empty MP4, AVI, MKV, or MOV video.");
      event.target.value = "";
      return;
    }
    setFile(selected);
    setJobId(null);
    setUploadError("");
  }

  async function start() {
    if (!file || busy || uploadLock.current) return;
    uploadLock.current = true;
    setSubmitting(true);
    setUploadError("");
    setJobId(null);
    try {
      const accepted = await uploadFaceVideo(file);
      if (!accepted.face_job_id) throw new Error("The API did not return a face job ID.");
      setJobId(accepted.face_job_id);
    } catch (failure) {
      setUploadError(failure instanceof Error ? failure.message : "Face upload failed.");
    } finally {
      setSubmitting(false);
      uploadLock.current = false;
    }
  }

  const started = Date.parse(status?.started_at || "");
  const ended = Date.parse(status?.completed_at || "");
  const elapsed = status?.processing_duration_sec ?? (Number.isFinite(started)
    ? Math.max(0, ((Number.isFinite(ended) ? ended : now) - started) / 1000) : 0);
  const progress = Math.min(100, Math.max(0, Number(status?.progress_pct) || 0));
  const narrators = results?.narrators ?? results?.presenters ?? [];
  const staticFaces = results?.static_faces ?? [];
  return (
    <section className="card upload-card" aria-labelledby="face-heading">
      <div className="section-heading"><div>
        <p className="eyebrow">Separate face extraction</p>
        <h2 id="face-heading">Upload video for faces</h2>
      </div></div>
      <p>Extract padded face images into narrator and static folders. This upload does not start OCR or audio.</p>
      <label className="slider-field" htmlFor="face-video">
        <span>Choose face-extraction video</span>
        <input id="face-video" type="file" accept=".mp4,.avi,.mkv,.mov" disabled={busy} onChange={selectFile} />
      </label>
      {file && <p>{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</p>}
      <button className="primary-button" type="button" disabled={!file || busy} onClick={() => void start()} aria-busy={busy}>
        {submitting ? "Uploading face video…" : isPolling ? "Extracting faces…" : "Upload & extract faces"}
      </button>
      {(uploadError || error) && <div className="error-message" role="alert">{uploadError || error}</div>}
      {jobId && <div>
        <p role="status">Face extraction: {status?.status || "queued"}{status?.stage ? " · " + status.stage : ""}</p>
        <progress aria-label="Face extraction progress" value={progress} max="100" style={{ width: "100%" }} />
        <p>{progress.toFixed(0)}% · Processing time: {seconds(elapsed)} · Frames checked: {status?.frames_processed ?? 0}</p>
        <small>Processing time excludes upload and queue wait. Job: {jobId}</small>
      </div>}
      {results && <div style={{ display: "grid", gap: 20 }}>
        <p>Saved folder: <code style={{ overflowWrap: "anywhere" }}>{results.output_folder || jobId}</code><br />
          <a href={getFaceDownloadUrl(jobId)}>Download face results (JSON)</a>
        </p>
        <p>One best image per usable track, not every frame or necessarily every person. Narrators are movement-based candidates; static/uncertain faces may include motionless narrators.</p>
        {results.duplicate_photos_removed > 0 && <p>Duplicate photos removed: {results.duplicate_photos_removed}</p>}
        {(results.warnings || []).map((warning, index) => <p key={index}>{warning}</p>)}
        <FaceGallery title="Narrator candidates" faces={narrators} jobId={jobId} />
        <FaceGallery title="Static / uncertain faces" faces={staticFaces} jobId={jobId} />
      </div>}
    </section>
  );
}
