const COLORS = { text: "#0F172A", muted: "#64748B" };

function time(value) {
  return Number(value).toFixed(2) + "s";
}

export function TranscriptPanel({ status, transcript }) {
  if (!status) return null;
  return (
    <section className="card output-card">
      <p className="eyebrow" style={{ color: COLORS.muted }}>Audio transcription</p>
      <h2 style={{ color: COLORS.text }}>
        {status.status === "completed" ? "Audio transcript" : "Transcribing audio..."}
      </h2>
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
