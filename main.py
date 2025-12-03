from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.api import api_router
from app.api.routes import export as export_routes
from app.api.routes import media as media_routes
from app.db.mongo import close_mongo_connection, connect_to_mongo

app = FastAPI(title=settings.project_name)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def on_startup() -> None:
    await connect_to_mongo()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_mongo_connection()


@app.get("/")
async def root():
    return {"service": settings.project_name, "docs": "/docs", "export_page": "/data-export"}


app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(export_routes.router)
app.include_router(media_routes.router)