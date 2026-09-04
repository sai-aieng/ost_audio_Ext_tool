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
    label: "Time precision",
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

export function PipelineConfig({ value, onChange, disabled }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <section className="config-section" style={{ borderColor: COLORS.border }}>
      <button
        className="config-toggle"
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
        style={{ color: COLORS.text }}
      >
        <span>Pipeline settings</span>
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
