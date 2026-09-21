import { useEffect, useState } from "react";

const COLORS = { text: "#0F172A", muted: "#64748B" };

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  const ms = Math.max(0, Math.round(seconds * 1000));
  return [Math.floor(ms / 3600000), Math.floor(ms / 60000) % 60, Math.floor(ms / 1000) % 60]
    .map((part) => String(part).padStart(2, "0")).join(":") +
    "." + String(ms % 1000).padStart(3, "0");
}

function formatTimestamp(value) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : "--";
}

function time(value) {
  return Number(value).toFixed(2) + "s";
}

export function TranscriptPanel({ status, transcript, error }) {
  const [now, setNow] = useState(Date.now);
  const running = status?.status === "processing";
  const startedAt = status?.started_at;
  useEffect(() => {
    setNow(Date.now());
    if (!running || !startedAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [running, startedAt, status?.job_id]);

  if (!status) return error ? (
    <section className="card output-card">
      <h2 style={{ color: COLORS.text }}>Audio status unavailable</h2>
      <p role="alert" style={{ color: "#B91C1C" }}>{error}</p>
    </section>
  ) : null;
  const start = startedAt ? Date.parse(startedAt) : NaN;
  const end = status.completed_at ? Date.parse(status.completed_at) : NaN;
  const elapsed = Number.isFinite(status.processing_duration_sec)
    ? status.processing_duration_sec
    : (Number.isFinite(end) ? end - start : running ? now - start : NaN) / 1000;
  const timings = [
    ["Started (local time)", formatTimestamp(startedAt)],
    ["Finished (local time)", formatTimestamp(status.completed_at)],
    [running ? "Audio elapsed (live estimate)" : "Audio processing time", formatDuration(elapsed)],
  ];
  return (
    <section className="card output-card">
      <p className="eyebrow" style={{ color: COLORS.muted }}>Audio transcription</p>
      <h2 style={{ color: COLORS.text }}>
        {status.status === "completed" ? "Audio transcript"
          : status.status === "failed" ? "Audio transcription failed"
          : running ? "Transcribing audio..." : "Audio transcription queued"}
      </h2>
      {error && status.status !== "failed" && (
        <p role="alert" style={{ color: "#B91C1C" }}>{error}</p>
      )}
      {status.inference_engine && (
        <p style={{ color: COLORS.muted }}>
          Audio engine: {status.inference_engine === "faster_whisper" ? "Faster-Whisper" : status.inference_engine}
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
          <div className="stat-box" style={{ background: "#F8FAFC" }} key={label}>
            <strong style={{ color: COLORS.text, fontSize: "0.95rem", fontVariantNumeric: "tabular-nums" }}>
              {value}
            </strong>
            <span style={{ color: COLORS.muted }}>{label}</span>
          </div>
        ))}
      </div>
      <p style={{ color: COLORS.muted, fontSize: "0.8rem" }}>
        Audio processing only; includes model loading, transcription, and export. Upload and OCR completion time are excluded.
      </p>
      {status.status === "failed" && status.error && (
        <p role="alert" style={{ color: "#B91C1C" }}>{status.error}</p>
      )}
      {transcript && <div className="result-table">
        {transcript.segments.map((segment, index) => (
          <div className="result-row" key={index}>
            <span>{time(segment.start_sec)} - {time(segment.end_sec)}</span>
            <span>{segment.text}</span>
          </div>
        ))}
      </div>}
    </section>
  );
}
