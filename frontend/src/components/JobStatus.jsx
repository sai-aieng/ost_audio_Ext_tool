import { useEffect, useState } from "react";

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const milliseconds = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor(milliseconds / 60000) % 60;
  const wholeSeconds = Math.floor(milliseconds / 1000) % 60;
  return (
    [hours, minutes, wholeSeconds].map((value) => String(value).padStart(2, "0")).join(":") +
    "." + String(milliseconds % 1000).padStart(3, "0")
  );
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : "—";
}

const COLORS = {
  uploaded: "#3B8BD4",
  processing: "#EF9F27",
  completed: "#1D9E75",
  failed: "#E24B4A",
  track: "#E2E8F0",
  text: "#0F172A",
  muted: "#64748B",
  stat: "#F8FAFC",
};

export function JobStatus({ status, isPolling }) {
  const [now, setNow] = useState(Date.now);
  const running = status?.status === "processing";
  const startedAt = status?.started_at;

  useEffect(() => {
    setNow(Date.now());
    if (!running || !startedAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [running, startedAt, status?.job_id]);

  if (!status) {
    return null;
  }

  const statusColor = COLORS[status.status] || COLORS.muted;
  const startMillis = startedAt ? Date.parse(startedAt) : NaN;
  const endMillis = status.completed_at ? Date.parse(status.completed_at) : NaN;
  const measuredSeconds = status.processing_duration_sec;
  const elapsedSeconds = Number.isFinite(measuredSeconds)
    ? measuredSeconds
    : (Number.isFinite(endMillis) ? endMillis - startMillis : running ? now - startMillis : NaN) / 1000;
  const timings = [
    ["Started (local time)", formatTime(startedAt)],
    ["Finished (local time)", formatTime(status.completed_at)],
    [running ? "Elapsed (live estimate)" : "OCR processing time", formatDuration(elapsedSeconds)],
  ];
  const stats = [
    ["Extracted", status.frames_extracted ?? 0],
    ["Unique", status.frames_deduplicated ?? 0],
    ["Processed", status.frames_processed ?? 0],
    ["Texts found", status.texts_found ?? 0],
  ];

  return (
    <section className="card status-card" aria-live="polite">
      <div className="status-heading">
        <div>
          <p className="eyebrow" style={{ color: COLORS.muted }}>
            Job status
          </p>
          <h2 style={{ color: COLORS.text }}>Processing overview</h2>
        </div>
        <span
          className="status-badge"
          style={{ background: statusColor }}
        >
          {status.status}
        </span>
      </div>
      <div
        className="progress-track"
        style={{ background: COLORS.track }}
        aria-label={"Progress " + Math.round(status.progress_pct || 0) + " percent"}
      >
        <div
          className={
            "progress-value" +
            (status.status === "processing" && isPolling ? " is-processing" : "")
          }
          style={{
            background: statusColor,
            width: Math.max(0, Math.min(100, status.progress_pct || 0)) + "%",
          }}
        />
      </div>
      <div className="progress-label" style={{ color: COLORS.muted }}>
        <span>{Math.round(status.progress_pct || 0)}% complete</span>
        <span>{status.job_id}</span>
      </div>
      {(status.inference_engine || status.requested_inference_engine) && (
        <p style={{ color: COLORS.muted }}>
          OCR engine: {status.inference_engine ?? "initializing"}
          {status.requested_inference_engine &&
            status.requested_inference_engine !== status.inference_engine &&
            " (requested " + status.requested_inference_engine + ")"}
        </p>
      )}
      {status.inference_fallback_reason && (
        <p role="status" style={{ color: COLORS.processing }}>
          {status.inference_fallback_reason}
        </p>
      )}
      <div
        aria-live="off"
        style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "12px", marginBottom: "16px",
        }}
      >
        {timings.map(([label, value]) => (
          <div className="stat-box" style={{ background: COLORS.stat }} key={label}>
            <strong style={{ color: COLORS.text, fontSize: "0.95rem", fontVariantNumeric: "tabular-nums" }}>
              {value}
            </strong>
            <span style={{ color: COLORS.muted }}>{label}</span>
          </div>
        ))}
      </div>
      <p style={{ color: COLORS.muted, fontSize: "0.8rem" }}>
        OCR processing only; includes model loading and fallback. Upload time and the separate audio job are excluded.
      </p>
      <div className="stats-grid">
        {stats.map(([label, number]) => (
          <div className="stat-box" style={{ background: COLORS.stat }} key={label}>
            <strong style={{ color: COLORS.text }}>{number}</strong>
            <span style={{ color: COLORS.muted }}>{label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
