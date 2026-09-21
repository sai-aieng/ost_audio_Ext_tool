import { useState } from "react";

const COLORS = {
  text: "#0F172A",
  muted: "#64748B",
  border: "#E2E8F0",
  accent: "#2563EB",
  surface: "#F8FAFC",
};

const SETTINGS = [
  {
    key: "sample_rate_fps",
    label: "Sampling rate",
    min: 0.1,
    max: 30,
    step: 0.1,
    suffix: " fps",
  },
  {
    key: "confidence_threshold",
    label: "Min confidence",
    min: 0.1,
    max: 1,
    step: 0.05,
    suffix: "",
  },
];

const MODES = [
  {
    value: "fast_cpu",
    label: "Fast CPU",
    description:
      "Uses 960px OCR, checks text-region changes, and reuses identical frames. Safety check every 10 seconds.",
  },
  {
    value: "accuracy",
    label: "Accuracy",
    description:
      "Uses 960px OCR with denoising, angle classification, and a safety check every second. Identical frames are reused.",
  },
];

export function PipelineConfig({ value, onChange, disabled }) {
  const [isOpen, setIsOpen] = useState(false);
  const activeMode =
    MODES.find((mode) => mode.value === value.processing_mode) ?? MODES[0];

  return (
    <section className="config-section" style={{ borderColor: COLORS.border }}>
      <button
        className="config-toggle"
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
        style={{ color: COLORS.text }}
      >
        <span>
          Pipeline settings
          <small className="config-mode-badge">{activeMode.label}</small>
        </span>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke={COLORS.muted}
          strokeWidth="2"
          aria-hidden="true"
          style={{ transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {isOpen && (
        <div className="config-fields" style={{ background: COLORS.surface }}>
          <label className="mode-field">
            <span>Processing mode</span>
            <select
              value={activeMode.value}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...value,
                  processing_mode: event.target.value,
                })
              }
            >
              {MODES.map((mode) => (
                <option value={mode.value} key={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
            <small>{activeMode.description}</small>
          </label>
          <label className="mode-field">
            <span>Inference engine</span>
            <select
              value={value.inference_engine ?? "paddle"}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...value, inference_engine: event.target.value })
              }
            >
              <option value="paddle">Paddle (baseline)</option>
              <option value="onnxruntime">ONNX Runtime CPU (experimental)</option>
            </select>
            <small>
              Same models and resolution. ONNX falls back to Paddle if unavailable.
              Job status shows the actual engine. Compare speed and accuracy manually.
            </small>
          </label>
          {SETTINGS.map((setting) => (
            <label className="slider-field" key={setting.key}>
              <span>
                {setting.label}
                <output style={{ color: COLORS.accent }}>
                  {Number(value[setting.key]).toFixed(
                    setting.step < 0.1 ? 2 : setting.step < 1 ? 1 : 0,
                  )}
                  {setting.suffix}
                </output>
              </span>
              <input
                type="range"
                min={setting.min}
                max={setting.max}
                step={setting.step}
                value={value[setting.key]}
                disabled={disabled}
                onChange={(event) =>
                  onChange({
                    ...value,
                    [setting.key]: Number(event.target.value),
                  })
                }
              />
            </label>
          ))}
        </div>
      )}
    </section>
  );
}
