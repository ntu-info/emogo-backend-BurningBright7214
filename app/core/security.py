from fastapi import HTTPException, status

from app.core.config import settings


def enforce_export_token(token: str | None) -> None:
    """Check whether the provided token matches settings.EXPORT_TOKEN (if set)."""

    if settings.export_token is None:
        return
    if token != settings.export_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token")

