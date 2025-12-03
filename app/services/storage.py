from typing import AsyncIterator

from bson import ObjectId
from fastapi import UploadFile
from gridfs.errors import NoFile

from app.db.mongo import get_gridfs_bucket


async def save_upload_file(file: UploadFile, metadata: dict | None = None) -> str:
    """Store an UploadFile into GridFS and return its ObjectId as a string."""

    bucket = get_gridfs_bucket()
    file_bytes = await file.read()
    gridfs_metadata = metadata or {}
    gridfs_metadata.setdefault("content_type", file.content_type or "application/octet-stream")
    gridfs_metadata.setdefault("original_filename", file.filename or "upload.bin")
    file_id = await bucket.upload_from_stream(
        file.filename or "upload.bin",
        file_bytes,
        metadata=gridfs_metadata,
    )
    await file.seek(0)
    return str(file_id)


async def open_file(file_id: ObjectId):
    bucket = get_gridfs_bucket()
    try:
        return await bucket.open_download_stream(file_id)
    except NoFile as exc:
        raise FileNotFoundError("Requested file not found in storage") from exc


async def delete_file(file_id: ObjectId) -> None:
    bucket = get_gridfs_bucket()
    await bucket.delete(file_id)


async def iter_gridfs_file(grid_out) -> AsyncIterator[bytes]:
    """Yield chunks from a GridFS file for StreamingResponse."""

    while True:
        chunk = await grid_out.readchunk()
        if not chunk:
            break
        yield chunk

