from typing import AsyncIterator

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Path, Query, Response, status
from fastapi.responses import StreamingResponse

from app.core.security import enforce_export_token
from app.services.storage import iter_gridfs_file, open_file

router = APIRouter(tags=["media"])


@router.get("/media/{file_id}", name="download_media")
async def download_media(
    file_id: str = Path(..., description="GridFS file identifier returned when uploading."),
    token: str | None = Query(default=None),
) -> Response:
    enforce_export_token(token)
    try:
        file_id_obj = ObjectId(file_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file identifier") from exc

    try:
        grid_out = await open_file(file_id_obj)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found") from exc
    media_type = grid_out.metadata.get("content_type", "application/octet-stream") if grid_out.metadata else "application/octet-stream"
    filename = grid_out.metadata.get("original_filename") if grid_out.metadata else grid_out.filename

    return StreamingResponse(
        iter_gridfs_file(grid_out),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

