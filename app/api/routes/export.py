from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.security import enforce_export_token
from app.db.mongo import get_collection
from app.services.exporter import documents_to_csv, fetch_all_documents

router = APIRouter(tags=["export"])
templates = Jinja2Templates(directory="app/templates")

DATASETS = ("vlogs", "sentiments", "gps")


@router.get("/export/{dataset}")
async def export_dataset(
    dataset: str,
    request: Request,
    format: Literal["json", "csv"] = Query(default="json"),
    token: str | None = Query(default=None),
) -> Response:
    enforce_export_token(token)
    if dataset not in DATASETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown dataset")

    collection = get_collection(dataset)
    documents = await fetch_all_documents(collection)
    if dataset == "vlogs":
        for document in documents:
            storage_id = document.get("videoStorageId") or document.get("video_storage_id")
            if storage_id:
                download_url = request.url_for("download_media", file_id=str(storage_id))
                if token:
                    download_url = f"{download_url}?token={token}"
                document["videoUri"] = download_url
    if format == "csv":
        csv_payload = documents_to_csv(documents)
        filename = f"{dataset}-{datetime.utcnow().date().isoformat()}.csv"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=csv_payload, media_type="text/csv", headers=headers)

    return JSONResponse(
        content=documents,
        headers={"X-Total-Count": str(len(documents)), "Cache-Control": "no-store"},
    )


@router.get("/data-export", response_class=HTMLResponse)
async def data_export_page(request: Request, token: str | None = Query(default=None)) -> HTMLResponse:
    enforce_export_token(token)

    dataset_stats: list[dict[str, str]] = []
    token_suffix = f"&token={token}" if token else ""
    for name in DATASETS:
        collection = get_collection(name)
        total = await collection.count_documents({})
        latest = await collection.find().sort("timestamp", -1).limit(1).to_list(1)
        timestamp_value = latest[0].get("timestamp") if latest else None
        last_timestamp = (
            timestamp_value.isoformat() if isinstance(timestamp_value, datetime) else str(timestamp_value or "N/A")
        )
        base_url = request.url_for("export_dataset", dataset=name)
        dataset_stats.append(
            {
                "name": name,
                "total": total,
                "last_timestamp": last_timestamp,
                "json_url": f"{base_url}?format=json{token_suffix}",
                "csv_url": f"{base_url}?format=csv{token_suffix}",
            }
        )

    recent_vlogs = await get_collection("vlogs").find().sort("timestamp", -1).limit(10).to_list(10)
    recent_vlog_rows: list[dict[str, str]] = []
    for doc in recent_vlogs:
        video_id = doc.get("videoStorageId") or doc.get("video_storage_id")
        if not video_id:
            continue
        download_url = request.url_for("download_media", file_id=str(video_id))
        if token:
            download_url = f"{download_url}?token={token}"
        recent_vlog_rows.append(
            {
                "timestamp": str(doc.get("timestamp")),
                "moodScore": str(doc.get("moodScore") or doc.get("mood_score")),
                "moodLabel": doc.get("moodLabel") or doc.get("mood_label") or "",
                "download_url": download_url,
                "filename": doc.get("videoFilename") or doc.get("video_filename") or "vlog.mp4",
            }
        )

    return templates.TemplateResponse(
        "data_export.html",
        {
            "request": request,
            "project_name": settings.project_name,
            "dataset_stats": dataset_stats,
            "token": token,
            "token_enabled": settings.export_token is not None,
            "recent_vlogs": recent_vlog_rows,
        },
    )

