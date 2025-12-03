from typing import Any, Dict, Type

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import BaseEntry, DatasetResponse, GPSEntry, SentimentEntry, VlogEntry

router = APIRouter(tags=["datasets"])

DATASET_SCHEMAS: Dict[str, Type[BaseEntry]] = {
    "sentiments": SentimentEntry,
    "vlogs": VlogEntry,
    "gps": GPSEntry,
}


def get_schema(dataset: str) -> Type[BaseEntry]:
    try:
        return DATASET_SCHEMAS[dataset]
    except KeyError as exc:
        valid = ", ".join(DATASET_SCHEMAS.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset}' not found. Valid options: {valid}",
        ) from exc


def serialize_document(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize Mongo documents (ObjectId, datetime) into JSON serializable dicts."""

    return jsonable_encoder(document)


@router.get("/{dataset}", response_model=DatasetResponse)
async def list_entries(
    dataset: str,
    limit: int = Query(100, ge=1, le=5000),
    skip: int = Query(0, ge=0),
) -> DatasetResponse:
    collection = get_collection(dataset)
    cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
    items: list[dict[str, Any]] = []
    async for document in cursor:
        items.append(serialize_document(document))

    return DatasetResponse(dataset=dataset, total=len(items), items=items)


@router.post("/{dataset}", status_code=status.HTTP_201_CREATED)
async def create_entry(dataset: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    schema = get_schema(dataset)
    entry = schema(**payload)
    collection = get_collection(dataset)

    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document

