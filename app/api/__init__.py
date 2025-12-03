"""FastAPI routers registration helpers."""

from fastapi import APIRouter

from app.api.routes import datasets, health, vlogs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router, prefix="/datasets")
api_router.include_router(vlogs.router)

