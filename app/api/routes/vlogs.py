from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import VlogEntry
from app.services.storage import save_upload_file

router = APIRouter(prefix="/vlogs", tags=["vlogs"])


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp format") from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_vlog(
    file: UploadFile = File(..., description="Vlog video file"),
    timestamp: str = Form(..., description="ISO timestamp captured on device"),
    moodScore: int = Form(..., ge=0, le=10),
    moodLabel: str | None = Form(default=None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: str | None = Form(default=None),
) -> dict:
    """Upload a vlog (video + metadata) and persist it together with a GridFS file."""

    parsed_timestamp = parse_timestamp(timestamp)
    metadata = {
        "content_type": file.content_type,
        "original_filename": file.filename,
    }
    file_id = await save_upload_file(file, metadata=metadata)
    video_uri = f"/media/{file_id}"

    entry_payload = {
        "timestamp": parsed_timestamp,
        "moodScore": moodScore,
        "moodLabel": moodLabel,
        "latitude": latitude,
        "longitude": longitude,
        "notes": notes,
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
    entry = VlogEntry(**entry_payload)

    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
    collection = get_collection("vlogs")
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document

