import csv
import io
from typing import Any, Iterable

from bson import ObjectId
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorCollection


async def fetch_all_documents(collection: AsyncIOMotorCollection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    cursor = collection.find().sort("timestamp", -1)
    async for doc in cursor:
        documents.append(jsonable_encoder(doc, custom_encoder={ObjectId: str}))
    return documents


def documents_to_csv(documents: Iterable[dict[str, Any]]) -> str:
    """Convert a list of flat dictionaries to CSV format."""

    rows = list(documents)
    if not rows:
        return ""

    headers = sorted({key for row in rows for key in row.keys()})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()

