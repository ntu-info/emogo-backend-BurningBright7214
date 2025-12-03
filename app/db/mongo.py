from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

from app.core.config import settings

mongo_client: AsyncIOMotorClient | None = None
mongo_db: AsyncIOMotorDatabase | None = None
gridfs_bucket: AsyncIOMotorGridFSBucket | None = None


async def connect_to_mongo() -> None:
    """Create a single shared MongoDB client for the FastAPI app lifecycle."""

    global mongo_client, mongo_db, gridfs_bucket

    if mongo_client is not None:
        return

    mongo_client = AsyncIOMotorClient(settings.mongodb_uri, uuidRepresentation="standard")
    mongo_db = mongo_client[settings.database_name]
    gridfs_bucket = AsyncIOMotorGridFSBucket(mongo_db, bucket_name="vlog_files")


async def close_mongo_connection() -> None:
    """Gracefully close the MongoDB client when the app shuts down."""

    global mongo_client, mongo_db, gridfs_bucket

    if mongo_client is None:
        return

    mongo_client.close()
    mongo_client = None
    mongo_db = None
    gridfs_bucket = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the active MongoDB database instance."""

    if mongo_db is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database not initialized")
    return mongo_db


def get_collection(name: str):
    """Convenience helper to fetch a Mongo collection by name."""

    db = get_database()
    return db[name]


def get_gridfs_bucket() -> AsyncIOMotorGridFSBucket:
    if gridfs_bucket is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Storage not initialized")
    return gridfs_bucket


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncIOMotorDatabase]:
    """Async context manager useful in background tasks and scripts."""

    try:
        yield get_database()
    finally:
        # Nothing to clean up here because Motor handles pooling internally.
        ...


async def ping_database() -> dict[str, Any]:
    """Ping the MongoDB server to ensure connectivity."""

    db = get_database()
    result = await db.command("ping")
    return result

