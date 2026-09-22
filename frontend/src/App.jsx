import { useEffect, useState } from "react";

import {
  deleteJob,
  getSavedExtraction,
  getSavedExtractions,
  startProcessing,
  uploadVideo,
} from "./api/client";
import { DropZone } from "./components/DropZone";
import { FaceExtractionPanel } from "./components/FaceExtractionPanel";
import { JobStatus } from "./components/JobStatus";
import { OutputTabs } from "./components/OutputTabs";
import { PipelineConfig } from "./components/PipelineConfig";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { useAudioPoller } from "./hooks/useAudioPoller";
import { useJobPoller } from "./hooks/useJobPoller";

const COLORS = {
  background: "#F8FAFC",
  surface: "#FFFFFF",
  text: "#0F172A",
  muted: "#64748B",
  accent: "#2563EB",
  accentHover: "#1D4ED8",
  border: "#E2E8F0",
  error: "#B91C1C",
  errorSurface: "#FEF2F2",
  errorBorder: "#FECACA",
};

const DEFAULT_CONFIG = {
  processing_mode: "fast_cpu",
  inference_engine: "paddle",
  sample_rate_fps: 10,
  confidence_threshold: 0.6,
};
const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"];

export function App() {
  const [activeWorkflow, setActiveWorkflow] = useState("faces");
  const [selectedFile, setSelectedFile] = useState(null);
  const [savedRuns, setSavedRuns] = useState([]);
  const [selectedSavedRun, setSelectedSavedRun] = useState("");
  const [savedResults, setSavedResults] = useState([]);
  const [pipelineConfig, setPipelineConfig] = useState(DEFAULT_CONFIG);
  const [jobId, setJobId] = useState(null);
  const [audioJobId, setAudioJobId] = useState(null);
  const [uploadedStatus, setUploadedStatus] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const {
    jobStatus,
    results,
    isPolling,
    error: pollingError,
  } = useJobPoller(jobId);
  const { status: audioStatus, transcript, error: audioError } = useAudioPoller(audioJobId);

  useEffect(() => {
    void getSavedExtractions()
      .then(async (runs) => {
        setSavedRuns(runs);
        if (runs.length) {
          setSelectedSavedRun(runs[0].job_id);
          setSavedResults(await getSavedExtraction(runs[0].job_id));
        }
      })
      .catch(() => {});
  }, []);

  async function selectSavedRun(event) {
    const jobId = event.target.value;
    setSelectedSavedRun(jobId);
    if (jobId) setSavedResults(await getSavedExtraction(jobId));
  }

  function selectFile(file) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setSelectedFile(null);
      setError("Choose an MP4, AVI, MKV, or MOV video.");
      return;
    }
    setSelectedFile(file);
    setError("");
  }

  async function uploadAndRun() {
    if (!selectedFile) {
      setError("Select a video before starting.");
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const uploaded = await uploadVideo(selectedFile);
      const initialStatus = {
        job_id: uploaded.job_id,
        status: "uploaded",
        frames_extracted: 0,
        frames_deduplicated: 0,
        frames_processed: 0,
        texts_found: 0,
        progress_pct: 0,
      };
      setUploadedStatus(initialStatus);
      setJobId(uploaded.job_id);
      const started = await startProcessing(uploaded.job_id, {
        ...pipelineConfig,
        output_formats: ["json", "csv", "txt"],
      });
      setAudioJobId(started.audio_job_id ?? null);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The video could not be submitted.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function reset() {
    const previousJobId = jobId;
    setJobId(null);
    setAudioJobId(null);
    setUploadedStatus(null);
    setSelectedFile(null);
    setPipelineConfig(DEFAULT_CONFIG);
    setError("");
    if (previousJobId) {
      try {
        await deleteJob(previousJobId);
      } catch (deleteError) {
        setError(
          deleteError instanceof Error
            ? deleteError.message
            : "The server-side job could not be deleted.",
        );
      }
    }
  }

  const displayedStatus = jobStatus || uploadedStatus;
  const displayedError = error || pollingError;
  const controlsDisabled = isSubmitting || Boolean(jobId);

  return (
    <div className="app-shell" style={{ background: COLORS.background }}>
      <header className="app-header">
        <div className="brand-mark" style={{ background: COLORS.accent }}>
          <svg
            width="25"
            height="25"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#FFFFFF"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <path d="M7 9h5M7 13h10M7 17h7" />
          </svg>
        </div>
        <div>
          <h1 style={{ color: COLORS.text }}>Video Extraction Studio</h1>
          <p style={{ color: COLORS.muted }}>
            Choose face images or extract on-screen text and audio from your video.
          </p>
        </div>
      </header>

      <main className="main-content">
        <nav className="workflow-navigation" aria-label="Choose extraction tool">
          <button type="button" className="workflow-choice" aria-pressed={activeWorkflow === "faces"}
            aria-controls="faces-workflow" onClick={() => setActiveWorkflow("faces")}>
            <span className="workflow-choice-title">Face extraction</span>
            <span>Save face photos from your video</span>
            <small>Output: face images + JSON</small>
          </button>
          <button type="button" className="workflow-choice" aria-pressed={activeWorkflow === "ocr"}
            aria-controls="ocr-workflow" onClick={() => setActiveWorkflow("ocr")}>
            <span className="workflow-choice-title">Text extraction (OCR) &amp; audio</span>
            <span>Read on-screen text and transcribe speech</span>
            <small>Output: timestamped text + transcript</small>
          </button>
        </nav>
        <p className="workflow-switch-hint">Each tool has its own video upload and results. Switch tools without losing your current selection or progress.</p>
        <div id="faces-workflow" className="workflow-panel" hidden={activeWorkflow !== "faces"}>
          <FaceExtractionPanel />
        </div>
        <div id="ocr-workflow" className="workflow-panel" hidden={activeWorkflow !== "ocr"}>
        <section className="card upload-card" style={{ background: COLORS.surface }}>
          <div className="section-heading">
            <div>
              <p className="eyebrow" style={{ color: COLORS.muted }}>
                Text & audio tool
              </p>
              <h2 style={{ color: COLORS.text }}>Extract text &amp; audio from video</h2>
            </div>
            {jobId && (
              <button
                type="button"
                className="secondary-button"
                onClick={() => void reset()}
                style={{ borderColor: COLORS.border, color: COLORS.text }}
              >
                Reset
              </button>
            )}
          </div>
          <p className="workflow-description">Get on-screen text with timestamps and an audio transcript. Download text results as JSON, CSV, or TXT.</p>
          <p className="upload-step">1. Choose a video for text &amp; audio</p>
          <DropZone
            label="Choose video for OCR & audio"
            selectedFile={selectedFile}
            onFileSelect={selectFile}
            disabled={controlsDisabled}
          />
          <PipelineConfig
            value={pipelineConfig}
            onChange={setPipelineConfig}
            disabled={controlsDisabled}
          />
          <button
            type="button"
            className="primary-button"
            disabled={!selectedFile || controlsDisabled}
            onClick={() => void uploadAndRun()}
            aria-busy={isSubmitting}
            style={{
              background: isSubmitting ? COLORS.accentHover : COLORS.accent,
            }}
          >
            {isSubmitting && <span className="button-spinner" aria-hidden="true" />}
            {isSubmitting ? "Uploading text & audio video..." : "Upload & extract text + audio"}
          </button>
          {displayedError && (
            <div
              className="error-message"
              role="alert"
              style={{
                color: COLORS.error,
                background: COLORS.errorSurface,
                borderColor: COLORS.errorBorder,
              }}
            >
              {displayedError}
            </div>
          )}
        </section>

        <JobStatus status={displayedStatus} isPolling={isPolling} />
        <TranscriptPanel status={audioStatus} transcript={transcript} error={audioError} />
        {!jobId && savedRuns.length > 0 && (
          <label className="slider-field">
            <span style={{ color: COLORS.text }}>Show saved extraction</span>
            <select value={selectedSavedRun} onChange={(event) => void selectSavedRun(event)}>
              {savedRuns.map((run) => (
                <option value={run.job_id} key={run.job_id}>
                  {new Date(run.completed_at * 1000).toLocaleString()} ({run.result_count} text results)
                </option>
              ))}
            </select>
          </label>
        )}
        <OutputTabs
          results={jobId ? (results ?? []) : savedResults}
          jobId={jobId}
          showDownloads={Boolean(jobId)}
        />
        </div>
      </main>
    </div>
  );
}
