"""Response and internal result schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """One OCR detection associated with its source video frame."""

    frame_index: int
    timestamp_sec: float
    timestamp_start_sec: float | None = None
    timestamp_end_sec: float | None = None
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[list[int]] | None = None
    bbox_end: list[list[int]] | None = None
    bbox_xyxy: list[int] | None = None
    bbox_end_xyxy: list[int] | None = None
    source_frame_width: int | None = None
    source_frame_height: int | None = None


class JobStatus(BaseModel):
    """Public progress and lifecycle state for one processing job."""

    job_id: str
    status: Literal["uploaded", "processing", "completed", "failed"]
    frames_extracted: int = 0
    frames_deduplicated: int = 0
    frames_processed: int = 0
    texts_found: int = 0
    requested_inference_engine: str | None = None
    inference_engine: str | None = None
    inference_fallback_reason: str | None = None
    progress_pct: float = Field(default=0, ge=0, le=100)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    processing_duration_sec: float | None = Field(default=None, ge=0)
    completed_at: datetime | None = None


class UploadResponse(BaseModel):
    """Metadata returned after a video has been accepted and stored."""

    job_id: str
    filename: str
    duration_sec: float
    fps: float
    resolution: str
    status: Literal["uploaded"]


class ProcessResponse(BaseModel):
    """Acknowledgement returned when background processing starts."""

    job_id: str
    status: Literal["processing"]
    audio_job_id: str | None = None


class TranscriptSegment(BaseModel):
    """One Whisper transcript segment in source-video time."""

    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    text: str


class TranscriptResponse(BaseModel):
    """Timestamped transcript returned by the independent audio job."""

    job_id: str
    language: str | None = None
    segments: list[TranscriptSegment]


class DownloadLinkResponse(BaseModel):
    """A format-specific result download link."""

    job_id: str
    format: Literal["json", "csv", "txt"]
    download_url: str


class DeleteResponse(BaseModel):
    """Confirmation returned after deleting a job and its artifacts."""

    job_id: str
    status: Literal["deleted"]
