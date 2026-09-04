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
  if (!status) {
    return null;
  }

  const statusColor = COLORS[status.status] || COLORS.muted;
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
