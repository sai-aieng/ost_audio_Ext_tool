import { useRef, useState } from "react";

const COLORS = {
  border: "#CBD5E1",
  borderActive: "#3B8BD4",
  surface: "#FFFFFF",
  surfaceActive: "#EFF6FF",
  text: "#0F172A",
  muted: "#64748B",
  accent: "#2563EB",
};

const ACCEPTED_TYPES = ".mp4,.avi,.mkv,.mov";

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(1) + " KB";
  }
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export function DropZone({ selectedFile, displayName, onFileSelect, disabled }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  function chooseFile() {
    if (!disabled) {
      inputRef.current?.click();
    }
  }

  function handleFiles(files) {
    const [file] = Array.from(files || []);
    if (file) {
      onFileSelect(file);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (!disabled) {
      handleFiles(event.dataTransfer.files);
    }
  }

  return (
    <div
      className="drop-zone"
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={chooseFile}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          chooseFile();
        }
      }}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) {
          setIsDragging(true);
        }
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setIsDragging(false);
        }
      }}
      onDrop={handleDrop}
      style={{
        borderColor: isDragging ? COLORS.borderActive : COLORS.border,
        background: isDragging ? COLORS.surfaceActive : COLORS.surface,
        color: COLORS.text,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.65 : 1,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        hidden
        disabled={disabled}
        onChange={(event) => handleFiles(event.target.files)}
      />
      <svg
        width="34"
        height="34"
        viewBox="0 0 24 24"
        fill="none"
        stroke={COLORS.accent}
        strokeWidth="1.7"
        aria-hidden="true"
      >
        <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
        <path d="M5 13v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" />
      </svg>
      {selectedFile || displayName ? (
        <div className="selected-file">
          <strong>{selectedFile?.name ?? displayName}</strong>
          <span style={{ color: COLORS.muted }}>
            {selectedFile ? formatBytes(selectedFile.size) : "Saved uploaded video"}
          </span>
        </div>
      ) : (
        <div>
          <strong>Drop a video here or click to browse</strong>
          <span className="drop-zone-hint" style={{ color: COLORS.muted }}>
            MP4, AVI, MKV, or MOV
          </span>
        </div>
      )}
    </div>
  );
}
