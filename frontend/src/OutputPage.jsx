import { JobStatus } from "./components/JobStatus";
import { OutputTabs } from "./components/OutputTabs";
import { TranscriptPanel } from "./components/TranscriptPanel";

const COLORS = { text: "#0F172A", muted: "#64748B", accent: "#2563EB", border: "#E2E8F0", surface: "#FFFFFF" };

const savedStatus = {
  job_id: "saved-run-2026-08-31-0600", status: "completed", progress_pct: 100,
  frames_extracted: 1025, frames_deduplicated: 201, frames_processed: 201, texts_found: 13,
};

const savedTextResults = [
  [0, 0, 102.5, "Let's now take a look at some useful insurance concepts.", 0.9999745488],
  [0, 0, 102.5, "What are some key terms I", 0.9993398189],
  [0, 0, 102.5, "must know?", 0.9845754504],
  [9, 4.5, 8, "4", 0.9173906446],
  [11, 5.5, 16.5, "Beneficiary", 0.999990046],
  [34, 17, 28.5, "Cash Value", 0.9998986125],
  [59, 29.5, 43, "Free Look Period", 0.9999130964],
  [87, 43.5, 49.5, "Mortality", 0.9999936223],
  [100, 50, 57, "Premium", 0.999996841],
  [115, 57.5, 71.5, "Riders", 0.9999970794],
  [144, 72, 79.5, "Sum Assured", 0.9998390079],
  [160, 80, 88.5, "Surrender Value", 0.995151341],
  [179, 89.5, 102.5, "Underwriting", 0.999994278],
].map(([frame_index, start, end, text, confidence]) => ({
  frame_index, timestamp_sec: start, timestamp_start_sec: start, timestamp_end_sec: end, text, confidence,
  bbox: [[416, 58], [888, 60], [888, 83], [416, 81]], bbox_end: [[416, 58], [887, 60], [887, 83], [416, 81]],
  bbox_xyxy: [416, 58, 888, 83], bbox_end_xyxy: [416, 58, 887, 83],
}));

const savedAudio = { segments: [
  { start_sec: 0, end_sec: 9.8, text: "Let's now take a look at some useful insurance concepts." },
  { start_sec: 9.8, end_sec: 19.4, text: "What are some key terms I must know?" },
  { start_sec: 19.4, end_sec: 31.3, text: "A beneficiary is the person or organization that receives the benefit from an insurance policy." },
  { start_sec: 31.3, end_sec: 43.2, text: "Cash value is the savings element of a permanent life insurance policy." },
  { start_sec: 43.2, end_sec: 54.5, text: "A free look period gives you time to review your policy after purchase." },
  { start_sec: 54.5, end_sec: 66.7, text: "Premiums keep the policy active, while riders can provide additional coverage options." },
  { start_sec: 66.7, end_sec: 79.8, text: "Sum assured is the guaranteed amount paid to the beneficiary under the policy terms." },
  { start_sec: 79.8, end_sec: 91.5, text: "Surrender value is the amount available when a policy is ended early." },
  { start_sec: 91.5, end_sec: 102.5, text: "Underwriting helps an insurer evaluate risk before issuing a policy." },
] };

export function OutputPage() {
  return (
    <div className="app-shell" style={{ background: "#F8FAFC" }}>
      <header className="app-header">
        <div className="brand-mark" style={{ background: COLORS.accent }}>
          <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" strokeWidth="1.8" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 9h5M7 13h10M7 17h7" /></svg>
        </div>
        <div><h1 style={{ color: COLORS.text }}>Video OCR Tester</h1><p style={{ color: COLORS.muted }}>Extract timestamped text from video frames.</p></div>
      </header>
      <main className="main-content">
        <section className="card upload-card" style={{ background: COLORS.surface }}>
          <div className="section-heading">
            <div><p className="eyebrow" style={{ color: COLORS.muted }}>Completed extraction</p><h2 style={{ color: COLORS.text }}>Video results</h2></div>
            <a href="/" className="secondary-button" style={{ borderColor: COLORS.border, color: COLORS.text, textDecoration: "none" }}>New extraction</a>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "15px 16px", border: `1px solid ${COLORS.border}`, borderRadius: 11, background: "#F8FAFC" }}>
            <svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke={COLORS.accent} strokeWidth="1.7" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 9h5M7 13h10" /></svg>
            <div><strong style={{ display: "block", color: COLORS.text, fontSize: "0.9rem" }}>DM_IN_ENG_MCN_2018_L1T5.mp4</strong><span style={{ color: COLORS.muted, fontSize: "0.82rem" }}>Saved completed video</span></div>
          </div>
        </section>
        <JobStatus status={savedStatus} isPolling={false} />
        <section className="card output-card">
          <p className="eyebrow" style={{ color: COLORS.muted }}>Audio transcription</p>
          <h2 style={{ color: COLORS.text }}>Full audio transcript</h2>
          <div style={{ marginTop: 14, borderTop: `1px solid ${COLORS.border}` }}>
            {savedAudio.segments.map((segment, index) => (
              <div key={index} style={{ display: "grid", gridTemplateColumns: "130px minmax(0, 1fr)", gap: 12, padding: "13px 4px", borderBottom: `1px solid ${COLORS.border}`, alignItems: "start" }}>
                <span style={{ color: COLORS.muted, fontSize: "0.82rem", fontVariantNumeric: "tabular-nums" }}>{Number(segment.start_sec).toFixed(2)}s - {Number(segment.end_sec).toFixed(2)}s</span>
                <span style={{ color: COLORS.text, fontSize: "0.9rem", lineHeight: 1.55, whiteSpace: "normal", overflowWrap: "anywhere" }}>{segment.text}</span>
              </div>
            ))}
          </div>
        </section>
        <OutputTabs results={savedTextResults} jobId={null} showDownloads={false} />
      </main>
    </div>
  );
}