"""Request payload schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessOptions(BaseModel):
    """Per-job overrides accepted when starting OCR processing."""

    model_config = ConfigDict(extra="forbid")

    processing_mode: Literal["accuracy", "fast_cpu"] | None = None
    inference_engine: Literal["paddle", "onnxruntime"] | None = None
    sample_rate_fps: float | None = Field(default=None, gt=0, le=30)
    output_formats: list[Literal["json", "csv", "txt"]] | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("output_formats")
    @classmethod
    def output_formats_must_not_be_empty(
        cls, value: list[str] | None
    ) -> list[str] | None:
        """Reject an explicitly empty format selection."""

        if value == []:
            raise ValueError("output_formats must include at least one format")
        return value
