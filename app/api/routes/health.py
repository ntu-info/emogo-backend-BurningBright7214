from datetime import datetime

from fastapi import APIRouter

from app import settings
from app.db.mongo import ping_database

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def healthcheck() -> dict[str, str]:
    status = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
    try:
        await ping_database()
        status["database"] = "reachable"
    except Exception:  # pragma: no cover - we just report failure
        status["database"] = "unreachable"
    status["service"] = settings.project_name
    return status

