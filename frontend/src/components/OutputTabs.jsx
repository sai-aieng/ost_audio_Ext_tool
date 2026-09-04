import { useMemo, useState } from "react";

import { getDownloadUrl } from "../api/client";
import { ResultsTable } from "./ResultsTable";

const COLORS = {
  text: "#0F172A",
  muted: "#64748B",
  border: "#E2E8F0",
  accent: "#2563EB",
  activeSurface: "#EFF6FF",
  codeSurface: "#0F172A",
  codeText: "#E2E8F0",
};

const TABS = [
  ["results", "Results"],
  ["json", "Raw JSON"],
  ["csv", "CSV"],
  ["txt", "TXT"],
];
const PREVIEW_LIMIT = 15;

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return '"' + text.replaceAll('"', '""') + '"';
}

function timeRange(result) {
  const start = Number(result.timestamp_start_sec ?? result.timestamp_sec);
  const end = Number(result.timestamp_end_sec ?? result.timestamp_sec);
  return start.toFixed(2) + "s - " + end.toFixed(2) + "s";
}

function buildPreview(results, format) {
  const visible = results.slice(0, PREVIEW_LIMIT);
  const remaining = results.length - visible.length;
  let text = "";

  if (format === "json") {
    text = JSON.stringify(visible, null, 2);
  } else if (format === "csv") {
    const header = "frame_index,timestamp_start_sec,timestamp_end_sec,text,confidence,bbox_start,bbox_end,bbox_start_xyxy,bbox_end_xyxy,source_frame_width,source_frame_height";
    const rows = visible.map((result) =>
      [
        result.frame_index,
        Number(result.timestamp_start_sec ?? result.timestamp_sec).toFixed(3),
        Number(result.timestamp_end_sec ?? result.timestamp_sec).toFixed(3),
        result.text,
        Number(result.confidence).toFixed(4),
        JSON.stringify(result.bbox),
        JSON.stringify(result.bbox_end),
        JSON.stringify(result.bbox_xyxy),
        JSON.stringify(result.bbox_end_xyxy),
        result.source_frame_width,
        result.source_frame_height,
      ]
        .map(csvCell)
        .join(","),
    );
    text = [header, ...rows].join("\n");
  } else {
    text = visible
      .map(
        (result) =>
          "[" + timeRange(result) + "] " +
          result.text +
          " (confidence: " +
          Number(result.confidence).toFixed(3) +
          "; coordinates: " + JSON.stringify(result.bbox) + ")",
      )
      .join("\n");
  }

  if (remaining > 0) {
    text += "\n\n…and " + remaining + " more";
  }
  return text;
}

export function OutputTabs({ results, jobId, showDownloads = true }) {
  const [activeTab, setActiveTab] = useState("results");
  const preview = useMemo(
    () => (activeTab === "results" ? "" : buildPreview(results, activeTab)),
    [activeTab, results],
  );

  function download(format) {
    window.open(getDownloadUrl(jobId, format), "_blank", "noopener,noreferrer");
  }

  return (
    <section className="card output-card">
      <div className="output-heading">
        <div>
          <p className="eyebrow" style={{ color: COLORS.muted }}>
            Extraction output
          </p>
          <h2 style={{ color: COLORS.text }}>
            {results.length} text {results.length === 1 ? "result" : "results"}
          </h2>
        </div>
        {showDownloads && (
          <div className="download-actions" aria-label="Download result files">
            {["json", "csv", "txt"].map((format) => (
            <button
              type="button"
              className="secondary-button download-button"
              key={format}
              onClick={() => download(format)}
              style={{ borderColor: COLORS.border, color: COLORS.text }}
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
              >
                <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14" />
              </svg>
              .{format}
            </button>
            ))}
          </div>
        )}
      </div>
      <div className="tabs" role="tablist" style={{ borderColor: COLORS.border }}>
        {TABS.map(([id, label]) => (
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === id}
            className="tab-button"
            key={id}
            onClick={() => setActiveTab(id)}
            style={{
              color: activeTab === id ? COLORS.accent : COLORS.muted,
              background: activeTab === id ? COLORS.activeSurface : "transparent",
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel">
        {activeTab === "results" ? (
          <ResultsTable results={results} />
        ) : (
          <pre
            className="preview-code"
            style={{ background: COLORS.codeSurface, color: COLORS.codeText }}
          >
            {preview || "No rows to preview."}
          </pre>
        )}
      </div>
    </section>
  );
}
