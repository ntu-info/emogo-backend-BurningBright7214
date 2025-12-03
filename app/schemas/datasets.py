from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str | None = Field(default=None, alias="_id")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO timestamp recorded when the entry was created on the device.",
    )
    notes: str | None = Field(default=None, description="Optional user-provided notes.")


class SentimentEntry(BaseEntry):
    mood_score: int = Field(
        ge=0,
        le=10,
        description="Mood score provided by the user (0-10).",
        serialization_alias="moodScore",
        validation_alias="moodScore",
    )
    mood_label: str | None = Field(
        default=None,
        description="Optional mood label (e.g., happy, sad).",
        serialization_alias="moodLabel",
        validation_alias="moodLabel",
    )


class VlogEntry(SentimentEntry):
    video_uri: str = Field(
        description="Publicly accessible URL or storage reference for the recorded vlog.",
        serialization_alias="videoUri",
        validation_alias="videoUri",
    )
    video_storage_id: str | None = Field(
        default=None,
        serialization_alias="videoStorageId",
        validation_alias="videoStorageId",
        description="GridFS file identifier for backend-hosted videos.",
    )
    video_filename: str | None = Field(
        default=None,
        serialization_alias="videoFilename",
        validation_alias="videoFilename",
        description="Original filename for download convenience.",
    )
    latitude: float
    longitude: float


class GPSEntry(BaseEntry):
    latitude: float
    longitude: float
    accuracy: float | None = Field(default=None, description="GPS accuracy in meters, if available.")


class DatasetResponse(BaseModel):
    dataset: Literal["vlogs", "sentiments", "gps"]
    total: int
    items: list[dict[str, Any]]

