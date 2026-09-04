import { useEffect, useState } from "react";

const COLORS = {
  text: "#0F172A",
  muted: "#64748B",
  border: "#E2E8F0",
  green: "#1D9E75",
  amber: "#D97706",
  red: "#E24B4A",
  expanded: "#F8FAFC",
  accent: "#2563EB",
};

const WINDOW_SIZE = 50;

function confidenceColor(confidence) {
  if (confidence >= 0.8) {
    return COLORS.green;
  }
  if (confidence >= 0.6) {
    return COLORS.amber;
  }
  return COLORS.red;
}

function timeRange(result) {
  const start = Number(result.timestamp_start_sec ?? result.timestamp_sec);
  const end = Number(result.timestamp_end_sec ?? result.timestamp_sec);
  return start === end ? start.toFixed(2) + "s" : start.toFixed(2) + "s - " + end.toFixed(2) + "s";
}

function sourceFrameLabel(result) {
  if (result.source_frame_width && result.source_frame_height) {
    return "Original video frame: " + result.source_frame_width + " x " + result.source_frame_height + " pixels";
  }
  return "Original video-frame pixel coordinates";
}

export function ResultsTable({ results }) {
  const [visibleCount, setVisibleCount] = useState(WINDOW_SIZE);
  const [expandedRow, setExpandedRow] = useState(null);

  useEffect(() => {
    setVisibleCount(WINDOW_SIZE);
    setExpandedRow(null);
  }, [results]);

  if (!results.length) {
    return <p className="empty-state">No text detections met the confidence threshold.</p>;
  }

  const visibleResults = results.slice(0, visibleCount);
  return (
    <div>
      <div className="results-list" role="table" aria-label="OCR results">
        <div className="result-header" role="row">
          <span>Time range</span>
          <span>Text</span>
          <span>Confidence</span>
        </div>
        {visibleResults.map((result, index) => {
          const rowKey = result.frame_index + "-" + index;
          const isExpanded = expandedRow === rowKey;
          return (
            <div
              className="result-row-group"
              key={rowKey}
              style={{ borderColor: COLORS.border }}
            >
              <button
                className="result-row"
                type="button"
                role="row"
                onClick={() => setExpandedRow(isExpanded ? null : rowKey)}
                aria-expanded={isExpanded}
                title="Show exact coordinates"
              >
                <span style={{ color: COLORS.muted }}>
                  {timeRange(result)}
                </span>
                <span className="result-text" style={{ color: COLORS.text }}>
                  {result.text}
                </span>
                <strong style={{ color: confidenceColor(result.confidence) }}>
                  {(Number(result.confidence) * 100).toFixed(1)}%
                </strong>
              </button>
              {isExpanded && (
                <div className="bbox-panel" style={{ background: COLORS.expanded }}>
                  <span style={{ color: COLORS.muted }}>{sourceFrameLabel(result)}</span>
                  <span style={{ color: COLORS.muted }}>Start coordinates (x, y polygon)</span>
                  <code style={{ color: COLORS.text }}>
                    {result.bbox ? JSON.stringify(result.bbox) : "Not included"}
                  </code>
                  <span style={{ color: COLORS.muted }}>End coordinates (x, y polygon)</span>
                  <code style={{ color: COLORS.text }}>
                    {result.bbox_end ? JSON.stringify(result.bbox_end) : "Not included"}
                  </code>
                  <span style={{ color: COLORS.muted }}>Start rectangle [x-min, y-min, x-max, y-max]</span>
                  <code style={{ color: COLORS.text }}>
                    {result.bbox_xyxy ? JSON.stringify(result.bbox_xyxy) : "Not included"}
                  </code>
                  <span style={{ color: COLORS.muted }}>End rectangle [x-min, y-min, x-max, y-max]</span>
                  <code style={{ color: COLORS.text }}>
                    {result.bbox_end_xyxy ? JSON.stringify(result.bbox_end_xyxy) : "Not included"}
                  </code>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {visibleCount < results.length && (
        <button
          className="load-more"
          type="button"
          style={{ color: COLORS.accent, borderColor: COLORS.border }}
          onClick={() => setVisibleCount((count) => count + WINDOW_SIZE)}
        >
          Load 50 more
        </button>
      )}
    </div>
  );
}
