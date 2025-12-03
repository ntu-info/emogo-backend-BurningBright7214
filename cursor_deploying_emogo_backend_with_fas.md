# Deploying EmoGo backend with FastAPI
_Exported on 12/3/2025 at 22:15:20 GMT+8 from Cursor (2.1.32)_

---

**User**

Follow the steps in the shared whiteboard for deploying your FastAPI & MongoDB: https://reurl.cc/Dbo3lNLinks to an external site.



Submit your GitHub repo to NTU COOL by 12/4(Thu) 8 PM.

[Goal] Making an EmoGo backend on a public server using FastAPI+MongoDB.

[Required] List the URI of the data-exporting/downloading page of your backend in the README.md of your GitHub repo. In there, TAs & Tren should be able to see/download all three types of data collected by the EmoGo frontend (i.e., vlogs, sentiments, & GPS coordinates).

[Optional] Revise your EmoGo frontend to close the loop of making a complete product.





先跟我確認需求，期中前端的code在emogo-frontend。

---

**Cursor**

了解目前需求如下，請幫我確認是否正確，或有補充：

- 依照學習單上指示，將 EmoGo 的 FastAPI + MongoDB 後端部署到公開伺服器。
- 在 `README.md` 中列出後端資料匯出/下載頁面的 URI，讓助教與 Tren 能下載 vlog、sentiment、GPS 三類資料。
- 完成後需於 12/4（四）20:00 前提交 GitHub repo 連結到 NTU COOL。
- （可選）更新 EmoGo 前端（`emogo-frontend`）以串接新的後端。

若我理解正確，下一步會先確認目前的後端狀態與部署需求細節（例如目標雲端環境、白板中的具體指示內容、是否已有 MongoDB 服務、資料格式要求等）。請告訴我是否還有其他條件或限制，或提供需要遵循的白板重點，之後我再開始著手處理。

---

**User**

ok請你計劃要做的事，越詳細越好，你可以先閱讀所有檔案(不要修改)

---

**Cursor**

以下是詳細作業計劃（目前僅閱讀專案檔案，尚未修改）：

## 1. 需求與規格釐清
- 取得並整理白板內容：確認部署流程細節、必要環境變數、任何指定 API/頁面格式，確保完全符合課程要求。
- 與前端 `emogo-frontend` 對齊：盤點前端期中版已送出的資料格式／欄位，確保後端資料庫 schema 與 API 合拍，並確認是否要在本輪同步調整前端（若選擇完成 optional 項目）。
- 確認 MongoDB 來源：決定使用 Render 內建 Mongo plugin、MongoDB Atlas，或既有的 Mongo 實例，並蒐集連線 URI、帳號、網路白名單需求等。

## 2. 後端功能設計與實作藍圖
- 專案結構擴充：
  - 拆分 `main.py` 為 `app` 目錄、`routers`/`services`/`models`/`schemas` 等模組。
  - 建立 `config.py`（或 `settings.py`）讀取環境變數（Mongo URI、資料庫名稱、認證 token 等）。
- 資料模型與 schema：
  - 為 vlog、sentiment、GPS 建立 Pydantic schema 與 Mongo 集合結構，決定索引（例如 userId、timestamp）。
  - 若前端已有固定欄位，先定義對應欄位與型別轉換邏輯。
- Mongo 連線管理：
  - 加入 motor/asyncio 客戶端，封裝資料庫連線存取，提供 CRUD/查詢方法。
- API 規劃：
  - 基本健康檢查與版本資訊端點。
  - 資料寫入端點（若需要直接供前端串接）—含驗證。
  - 資料查詢／下載端點：支援依時間、使用者等條件篩選；輸出 JSON/CSV 供下載。
- 資料匯出頁面（README 需放 URI）：
  - 以 FastAPI 的 `Jinja2Templates` 或 `fastapi.responses.HTMLResponse` 建立頁面，列出三類資料的下載連結（REST 端點或即時生成 CSV）。
  - 規畫最小認證（即可讀取 token / basic auth / query key）以避免公開資料外洩，同時確保助教可輕鬆使用。
- README 更新：
  - 補上部署服務 URL、資料下載頁面 URI、使用方法、環境變數列表、啟動指令、資料格式說明。

## 3. 部署與環境作業
- 本地測試：
  - 建立 `.env`/`.env.example`，使用 `uvicorn` 驗證 API 與資料庫操作。
  - 撰寫基本測試（pytest 或內建測試路由）確保資料處理正確。
- Render 設定：
  - 若沿用現有 `render.yaml`，補上所需環境變數、部署指令（含安裝 motor/pydantic 等依賴）。
  - 在 Render 後台建立 Mongo（若 Render 提供）或設定外部 Mongo URI。
  - 確保 `autoDeploy` 策略與 Start Command 正確，並在部署後記錄公開 URL。
- 部署驗證：
  - 線上服務健康檢查。
  - 測試資料寫入 & 匯出頁面下載三類資料。
  - 若有認證 token，確認助教所需憑證的提供方式（README 或另行提供）。

## 4. 驗收與提交
- 撰寫操作/測試紀錄：列出本地測試、線上 Smoke Test、已知限制。
- 最終檔案檢查：READ ME、程式、`requirements.txt`（增列 motor、pydantic[email?], python-dotenv 等）確認無遺漏。
- 推送 GitHub：確保 main 分支包含部署後最新程式與文件。
- 於 NTU COOL 提交 repo 連結與資料下載頁面 URI（依課程規定）。

## 5. Optional：前端串接完善
- 若要完成前端改版：
  - 更新 `emogo-frontend` 的 API base URL、鑑權處理、錯誤處理。
  - 新增資料下載連結或後台檢視入口。
  - 驗證完整資料回傳流程與 UI 回饋。

若上述計劃沒問題，下一步我會先整理白板需求與 Mongo/資料格式細節，再開始調整專案結構與後端功能。請告知是否要補充或調整任何環節。

---

**User**

1.白板需求為:

[Deploy a FastAPI App on Render]



Follow this official tutorial: https://render.com/docs/deploy-fastapi

[Deployment of MongoDB]

Follow this Chinese tutorial: https://askstw.medium.com/re-mongodb-atlas-b331acd3d7c

Use “0.0.0.0” for the allowed IP for your Mongo DB. This will allow all IPs to access your Mongo DB. 

At the final step, click [Connect]:

And choose MongoDB Compass to get a connection URI like this:

mongodb+srv://tren:psychinfo@info.62iz1ix.mongodb.net/

In your MongoDB Compass, click the plus sign [+] to add a new connection:

Put in your connection string & hit [Save & Connect]:

Use MongoDB Compass to create some random data for later testing!

In the connection URI, tren / psychinfo are the account / password and info.62iz1ix.mongodb.net is the server domain name. Your backend needs this URI to connect to your Mongo DB.

Specifically, “pip install motor[srv]” & here is a FastAPI example:

from fastapi import FastAPI

from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URI = "mongodb+srv://tren:psychinfo@info.62iz1ix.mongodb.net/" # your URI

DB_NAME = "mydatabase"  # your DB name	

app = FastAPI()

@app.on_event("startup")

async def startup_db_client():

    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)

    app.mongodb = app.mongodb_client[DB_NAME]

@app.on_event("shutdown")S

async def shutdown_db_client():

    app.mongodb_client.close()

@app.get("/items")

async def get_items():

    items = await app.mongodb["items"].find().to_list(100)

    return items







2.你有看到前端的code嗎

---

**Cursor**

1. 已完整閱讀白板需求：  
- 後端部署流程需依照 Render 官方 FastAPI 教學執行。  
- MongoDB 需依照指定中文教學使用 MongoDB Atlas，允許 IP 設為 `0.0.0.0`，並透過 MongoDB Compass 建立連線 URI、測試資料。  
- FastAPI 需使用 `motor[srv]` 串接 MongoDB，啟動/關閉事件中建立與關閉連線。  

2. 目前尚未開啟或閱讀 `emogo-frontend`；我只檢視了 backend 目錄中的檔案（`main.py`、`render.yaml`、`README.md`、`requirements.txt`）。若需要查看或整合前端代碼，可再指示我進行。

---

**User**

所以這份作業在幹嘛你知道嗎

---

**Cursor**

是的，我已經釐清這份作業的目標：把 EmoGo 的後端（FastAPI + MongoDB）部署到公開雲端服務（依白板要求走 Render + MongoDB Atlas），讓助教能在 README 中找到資料下載頁面的 URI，並能從該頁面下載由前端蒐集的三類資料（vlog、sentiment、GPS）。完成後要把 GitHub repo 連結提交到 NTU COOL。若需要，同時可以選擇更新前端 `emogo-frontend` 以串接此後端，形成完整產品。

---

**User**

所以你需要寫什麼程式

---

**Cursor**

需要開發的部分可以分幾塊：

- **後端應用本體**：把現在的 `main.py` 擴充成完整的 FastAPI 專案，加入路由、資料模型、服務層，包含健康檢查、資料寫入/查詢 API。
- **MongoDB 連線與資料存取**：用 `motor[srv]` 建立 AsyncIOMotorClient，實作 vlog/sentiment/GPS 三類資料的 CRUD，確保 schema 對應前端欄位，並能透過查詢條件導出資料。
- **資料匯出頁面**：新增一個供 TA/Tren 使用的前端頁面（或 HTML 介面），列出三種資料的下載連結，支援 JSON/CSV 下載；其 URI 需在 README 中明示。
- **設定與部署腳本**：整理 `requirements.txt`、環境變數載入、`render.yaml` 或 Render 後台設定，確保服務可在 Render + MongoDB Atlas 正常啟動並連線。
- **README/文件**：補上部署後的公開 URL、資料下載頁面 URI、使用指南、環境變數與 Mongo 連線說明、測試步驟等。

若有 optional 需求，再額外修改 `emogo-frontend`，把 API base URL 換成新後端並測試整體流程。

---

**User**

好的請開始，我還沒用任何mongodb或fastapi，所以你需要我做什麼都隨時提出(白板跟readme都只是範例而已)，另外前端程式碼已經放在資料夾中了，你需要可以看。

---

**Cursor**

接下來會依這份詳細計劃展開實作；若有需要您配合（例如提供 Mongo 連線 URI、Render 環境變數設定）會再即時提醒。

### Backend 開發與結構
- 重構為 `app/` 模組化架構：`main.py` 只做啟動，實際程式拆到 `app/core/config.py`（環境變數 & Mongo URI）、`app/db/mongo.py`（Motor 連線池）、`app/models`/`schemas`（vlog/sentiment/gps 的 Pydantic schema）、`app/routers`（資料寫入、查詢、匯出 API）、`app/services`（資料處理）。
- 依照白板示例引入 `motor[srv]`，在 FastAPI 的 startup/shutdown 建立/關閉 `AsyncIOMotorClient`，並封裝 `db.client`, `db.database`, `db.get_collection(...)`。
- 增加必要依賴到 `requirements.txt`：`motor[srv]`, `python-dotenv`, `pydantic[email]?`, `jinja2`（若用模板）等；同步新增 `requirements-dev.txt`（可選）以利本地開發。
- 建立 `.env.example`（包含 `MONGODB_URI`, `DATABASE_NAME`, `EXPORT_TOKEN` 等），程式使用 Pydantic `BaseSettings` 載入。
- 根據前端 `emogo-frontend-BurningBright7214/utils/database.js` 的資料結構設計三個 collection：
  - `vlogs`: 影片 metadata + GridFS/Object Storage 參考（暫定存 `video_url` or GridFS file id`，同時記錄與 sentiment/gps 的關聯 id)。
  - `sentiments`: `timestamp`, `moodScore`, `moodLabel`, `notes`。
  - `gps_tracks`: `timestamp`, `latitude`, `longitude`, `accuracy` optional。
- API 規劃（第一版）：
  - `GET /health`：回應版本、DB 狀態。
  - `POST /api/v1/vlogs`：multipart 上傳影片（`UploadFile`）+ metadata；使用 GridFS 或暫時存本地/雲端後回寫 URL，再寫 Mongo document。
  - `POST /api/v1/sentiments`、`/api/v1/gps`：接收 JSON。若未來要直接從前端送資料可沿用這些路由。
  - `GET /api/v1/export/{dataset}?format=json|csv`：回傳相對應集合的資料；CSV 以 `StreamingResponse` 產生。
- 撰寫 `app/services/exporter.py` 將 Mongo cursor 轉 CSV/JSON，並可設定最大筆數或日期篩選。
- 簡易驗證：為匯出頁面與 API 增加 query token 或 header token；透過 `.env` 提供 `EXPORT_TOKEN`，在路由中檢查。

### 資料匯出頁面
- 新增 `app/routers/ui.py` 使用 `Jinja2Templates` 或直接 `HTMLResponse` 輸出一個頁面：
  - 顯示三種資料的下載連結（JSON/CSV）。
  - 可展示資料筆數、最後更新時間等統計。
  - 此頁 URI 會寫進 `README.md`（例如 `/data-export`），部署後改成完整 `https://<render-app>.onrender.com/data-export`。
- 如果要直接讓 TA 預覽資料，可在頁面中嵌入小表格（例如載入前 50 筆 JSON）或以 `fetch` 呼叫 API。

### 測試與驗證
- 撰寫最低限度的單元測試/整合測試（`pytest`）：mock Mongo、測 `GET /health`、`GET /export` 等。
- 本地用 `uvicorn app.main:app --reload` + `.env` 連 Mongo Atlas（需您先依白板教學建立 cluster、帳號、資料庫，並產生連線字串）。
- 使用 MongoDB Compass 建幾筆測試資料，驗證 `GET /export` 與 HTML 頁面可下載。

### 部署與文件
- 更新 `render.yaml`：加入 `envVars`（或在 Render 後台設定）、Plan 選 free or starter、`buildCommand` & `startCommand`；若使用 `poetry` 則調整對應命令。
- README 改為專案說明：目的、架構、如何在本地執行、需要的環境變數、部署流程、資料匯出頁面 URI、測試方式，以及 TA 如何下載三類資料的步驟（含截圖/指令）。
- 於 README 附上前端 repo/資料夾說明與（可選）如何串接後端 API（目前 `emogo-frontend-BurningBright7214` 已在同一資料夾，可在 README 標註路徑）。

### 您需要協助準備的事項（有即可提供、沒關係也可等我催）
- 建立 MongoDB Atlas 專案與 cluster，並回傳可用的 `mongodb+srv://...` connection string（含帳號/密碼/資料庫名稱）或將資訊存成 `.env`，讓我可以連線並測試。
- Render 帳號與服務建立（若需要我提供打包後的 repo，您可自行部署；或給我 temporary access? 一般只需提醒要設定的環境變數：`MONGODB_URI`, `DATABASE_NAME`, `EXPORT_TOKEN`）。
- 如果希望我也調整前端，請確認現在的 Expo/React Native 環境需求（例如使用 `eas build` or `expo start`），以及是否要接新 API。

我會先在本地把後端完整實作完畢，再視需要協助您部署到 Render。任何額外需求請再告訴我！

---

**User**

好的請告訴我有需要我協助註冊什麼網站提供給你東西嗎?或是你可以直接開始

---

**Cursor**

目前需要您協助的只有兩件事，其他我可以直接開始實作：

1. **MongoDB Atlas 連線資訊**  
   - 依白板教學建立 cluster、帳號與資料庫（允許 IP 設 `0.0.0.0`）。  
   - 在 Compass 產出正式的 connection string（`mongodb+srv://帳號:密碼@.../`），告訴我：  
     - `MONGODB_URI`（含使用者/密碼）  
     - `DATABASE_NAME`（您想放 EmoGo 資料的 DB 名稱）  
   我會用它來測試與部署。若您擔心安全，可改給我一組只用於此專案的臨時帳號與密碼。

2. **（稍後部署時）Render 環境變數**  
   - 等我把程式完成後，您只需在 Render 建立 Web Service，並把 `MONGODB_URI`、`DATABASE_NAME`、`EXPORT_TOKEN` 等環境變數貼上即可；目前不用先處理，等我交付再請您設定。

以上資訊一旦提供，我就能繼續實作（結構重構、API、資料匯出頁面等）並在本地測試。若有任何限制（例如不能把影片檔傳到後端，只存 metadata 等），也請一併告知。

---

**User**

好的我已經用了1，帳號為:mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/

---

**Cursor**

收到，感謝提供 URI！目前還需要知道您希望使用的資料庫名稱（`DATABASE_NAME`），例如 `emogo` 或其他名稱。確定後我會把它加入設定，剩下就可以直接開始開發囉。

---

**User**

好emogo

---

**Cursor**

```pip-requirements
fastapi[all]
uvicorn[standard]
motor[srv]
pydantic-settings>=2.0.3
python-dotenv
jinja2
```

```python
"""
EmoGo backend FastAPI application package.

The heavy lifting is done in submodules such as:
- core: configuration and shared settings
- db: database connection helpers
- api: route registrations
"""

from .core.config import settings

__all__ = ["settings"]

```

```python
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration loaded from environment variables."""

    api_prefix: str = "/api/v1"
    project_name: str = "EmoGo Backend"
    mongodb_uri: str
    database_name: str = "emogo"
    export_token: str | None = None
    cors_allow_origins: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("mongodb_uri")
    @classmethod
    def validate_mongo_uri(cls, value: str) -> str:
        if not value:
            msg = "MONGODB_URI is required. Please set it in your environment or .env file."
            raise ValueError(msg)
        return value

    @property
    def cors_origins(self) -> List[str]:
        if not self.cors_allow_origins:
            return []
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()


settings = get_settings()

```

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

mongo_client: AsyncIOMotorClient | None = None
mongo_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    """Create a single shared MongoDB client for the FastAPI app lifecycle."""

    global mongo_client, mongo_db

    if mongo_client is not None:
        return

    mongo_client = AsyncIOMotorClient(settings.mongodb_uri, uuidRepresentation="standard")
    mongo_db = mongo_client[settings.database_name]


async def close_mongo_connection() -> None:
    """Gracefully close the MongoDB client when the app shuts down."""

    global mongo_client, mongo_db

    if mongo_client is None:
        return

    mongo_client.close()
    mongo_client = None
    mongo_db = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the active MongoDB database instance."""

    if mongo_db is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database not initialized")
    return mongo_db


def get_collection(name: str):
    """Convenience helper to fetch a Mongo collection by name."""

    db = get_database()
    return db[name]


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

```

```python
from .datasets import DatasetResponse, GPSEntry, SentimentEntry, VlogEntry

__all__ = [
    "GPSEntry",
    "SentimentEntry",
    "VlogEntry",
    "DatasetResponse",
]

```

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class BaseEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str | None = Field(default=None, alias="_id")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO timestamp recorded when the entry was created on the device.",
    )
    notes: str | None = Field(default=None, description="Optional user-provided notes.")


class GeoPoint(BaseModel):
    latitude: float
    longitude: float


class SentimentEntry(BaseEntry):
    mood_score: int = Field(ge=0, le=10, description="Mood score provided by the user (0-10).")
    mood_label: str | None = Field(default=None, description="Optional mood label (e.g., happy, sad).")


class VlogEntry(SentimentEntry):
    video_url: HttpUrl | str = Field(
        description="Publicly accessible URL or storage reference for the recorded vlog."
    )
    location: GeoPoint


class GPSEntry(BaseEntry):
    location: GeoPoint
    accuracy: float | None = Field(default=None, description="GPS accuracy in meters, if available.")


class DatasetResponse(BaseModel):
    dataset: Literal["vlogs", "sentiments", "gps"]
    total: int
    items: list[dict[str, Any]]

```

```python
"""FastAPI routers registration helpers."""

from fastapi import APIRouter

from app.api.routes import datasets, export, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router, prefix="/datasets")
api_router.include_router(export.router)

```

```python
from . import datasets, export, health

__all__ = ["datasets", "export", "health"]

```

```python
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

```

```python
from typing import Any, Callable, Dict, Type

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import DatasetResponse, GPSEntry, SentimentEntry, VlogEntry

router = APIRouter(tags=["datasets"])

DATASET_SCHEMAS: Dict[str, Type[SentimentEntry | VlogEntry | GPSEntry]] = {
    "sentiments": SentimentEntry,
    "vlogs": VlogEntry,
    "gps": GPSEntry,
}


def get_schema(dataset: str) -> Type[SentimentEntry | VlogEntry | GPSEntry]:
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

```

```python
from typing import Any, Dict, Type

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import DatasetResponse, GPSEntry, SentimentEntry, VlogEntry

router = APIRouter(tags=["datasets"])

DATASET_SCHEMAS: Dict[str, Type[SentimentEntry | VlogEntry | GPSEntry]] = {
    "sentiments": SentimentEntry,
    "vlogs": VlogEntry,
    "gps": GPSEntry,
}


def get_schema(dataset: str) -> Type[SentimentEntry | VlogEntry | GPSEntry]:
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


```

```python
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


```

```python
import csv
import io
from typing import Any, Iterable

from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorCollection


async def fetch_all_documents(collection: AsyncIOMotorCollection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    cursor = collection.find().sort("timestamp", -1)
    async for doc in cursor:
        documents.append(jsonable_encoder(doc))
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

```

```python
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.db.mongo import get_collection
from app.services.exporter import documents_to_csv, fetch_all_documents
from app.schemas.datasets import DatasetResponse

router = APIRouter(tags=["export"])
templates = Jinja2Templates(directory="app/templates")

DATASETS = ("vlogs", "sentiments", "gps")


def ensure_token(token: str | None) -> None:
    """Simple shared-secret check for export-only endpoints."""

    if settings.export_token is None:
        return
    if token != settings.export_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token")


@router.get("/export/{dataset}")
async def export_dataset(
    dataset: str,
    request: Request,
    format: Literal["json", "csv"] = Query(default="json"),
    token: str | None = Query(default=None),
) -> Response:
    ensure_token(token)
    if dataset not in DATASETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown dataset")

    documents = await fetch_all_documents(get_collection(dataset))
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
    ensure_token(token)

    dataset_stats: list[dict[str, str]] = []
    for name in DATASETS:
        collection = get_collection(name)
        total = await collection.count_documents({})
        latest = await collection.find().sort("timestamp", -1).limit(1).to_list(1)
        last_timestamp = latest[0].get("timestamp").isoformat() if latest else "N/A"
        dataset_stats.append(
            {
                "name": name,
                "total": total,
                "last_timestamp": last_timestamp,
                "json_url": request.url_for("export_dataset", dataset=name) + "?format=json",
                "csv_url": request.url_for("export_dataset", dataset=name) + "?format=csv",
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
        },
    )

```

```python
*** Begin Patch
*** Update File: app/api/routes/export.py
@@
-from datetime import datetime
-from typing import Literal
+from datetime import datetime
+from typing import Literal
 
-from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
+from fastapi import APIRouter, HTTPException, Query, Request, Response, status
 from fastapi.responses import HTMLResponse, JSONResponse
@@
-    dataset_stats: list[dict[str, str]] = []
+    dataset_stats: list[dict[str, str]] = []
+    token_suffix = f"&token={token}" if token else ""
+    for name in DATASETS:
+        collection = get_collection(name)
+        total = await collection.count_documents({})
+        latest = await collection.find().sort("timestamp", -1).limit(1).to_list(1)
+        timestamp_value = latest[0].get("timestamp") if latest else None
+        formatted_timestamp = (
+            timestamp_value.isoformat() if isinstance(timestamp_value, datetime) else str(timestamp_value or "N/A")
+        )
 
-    for name in DATASETS:
-        collection = get_collection(name)
-        total = await collection.count_documents({})
-        latest = await collection.find().sort("timestamp", -1).limit(1).to_list(1)
-        last_timestamp = latest[0].get("timestamp").isoformat() if latest else "N/A"
+        base_url = request.url_for("export_dataset", dataset=name)
         dataset_stats.append(
             {
                 "name": name,
                 "total": total,
-                "last_timestamp": last_timestamp,
-                "json_url": request.url_for("export_dataset", dataset=name) + "?format=json",
-                "csv_url": request.url_for("export_dataset", dataset=name) + "?format=csv",
+                "last_timestamp": formatted_timestamp,
+                "json_url": f"{base_url}?format=json{token_suffix}",
+                "csv_url": f"{base_url}?format=csv{token_suffix}",
             }
         )
*** End Patch
```

```python
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.db.mongo import get_collection
from app.services.exporter import documents_to_csv, fetch_all_documents
from app.schemas.datasets import DatasetResponse

router = APIRouter(tags=["export"])
templates = Jinja2Templates(directory="app/templates")

DATASETS = ("vlogs", "sentiments", "gps")


def ensure_token(token: str | None) -> None:
    """Simple shared-secret check for export-only endpoints."""

    if settings.export_token is None:
        return
    if token != settings.export_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token")


@router.get("/export/{dataset}")
async def export_dataset(
    dataset: str,
    request: Request,
    format: Literal["json", "csv"] = Query(default="json"),
    token: str | None = Query(default=None),
) -> Response:
    ensure_token(token)
    if dataset not in DATASETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown dataset")

    documents = await fetch_all_documents(get_collection(dataset))
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
    ensure_token(token)

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

    return templates.TemplateResponse(
        "data_export.html",
        {
            "request": request,
            "project_name": settings.project_name,
            "dataset_stats": dataset_stats,
            "token": token,
            "token_enabled": settings.export_token is not None,
        },
    )


```

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{ project_name }} · Data Export</title>
    <style>
      :root {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        color: #142748;
        background-color: #f6f7fb;
      }
      body {
        margin: 0;
        padding: 2rem;
      }
      h1 {
        margin-bottom: 0.25rem;
      }
      .subtitle {
        color: #506283;
        margin-bottom: 2rem;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(20, 39, 72, 0.08);
      }
      thead {
        background: #f0f3ff;
      }
      th,
      td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid #e4e8f3;
      }
      th {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c7ca4;
      }
      tr:last-child td {
        border-bottom: none;
      }
      .badge {
        display: inline-flex;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background: #eef3ff;
        color: #41509b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .actions a {
        display: inline-block;
        margin-right: 0.5rem;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: background 0.2s ease;
      }
      .actions a.json {
        background: #ebf8ff;
        color: #0a6fbf;
      }
      .actions a.csv {
        background: #fff3e6;
        color: #cc6a00;
      }
      .actions a:hover {
        filter: brightness(0.95);
      }
      .token-info {
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: #fff8d8;
        border: 1px solid #f7e5a0;
        border-radius: 10px;
        color: #8a6d1e;
      }
    </style>
  </head>
  <body>
    <h1>{{ project_name }} Export Portal</h1>
    <p class="subtitle">下載 EmoGo 前端蒐集的所有資料類型。</p>

    {% if token_enabled %}
    <div class="token-info">
      目前已啟用下載 Token 驗證。若看到「Unauthorized」請於網址列加上
      <code>?token=YOUR_TOKEN</code>，或在下方按鈕後自行補上 <code>&token=YOUR_TOKEN</code>。
    </div>
    {% endif %}

    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Total Records</th>
          <th>Last Entry</th>
          <th>Download</th>
        </tr>
      </thead>
      <tbody>
        {% for dataset in dataset_stats %}
        <tr>
          <td>
            <span class="badge">{{ dataset.name }}</span>
          </td>
          <td>{{ dataset.total }}</td>
          <td>{{ dataset.last_timestamp }}</td>
          <td class="actions">
            <a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
            <a class="csv" href="{{ dataset.csv_url }}">CSV</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </body>
</html>

```

```python
"""FastAPI routers registration helpers."""

from fastapi import APIRouter

from app.api.routes import datasets, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router, prefix="/datasets")


```

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.api import api_router
from app.api.routes import export as export_routes
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
```

```plaintext
# Copy this file to ".env" and fill in your credentials before running locally.
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.example.mongodb.net/
DATABASE_NAME=emogo
EXPORT_TOKEN=please-change-me
# CORS_ALLOW_ORIGINS=https://emogo.example.com,https://studio.apollographql.com

```

```yaml
services:
  # A Docker web service
  - type: web
    name: fastapi-example
    runtime: python
    plan: free
    autoDeploy: false
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: MONGODB_URI
        sync: false
      - key: DATABASE_NAME
        value: emogo
      - key: EXPORT_TOKEN
        sync: false

```

```markdown
# EmoGo Backend (FastAPI + MongoDB)

這是一個部署在 Render 的 FastAPI 服務，串接 MongoDB Atlas 以提供 EmoGo 前端（`emogo-frontend-BurningBright7214/`）上傳的 vlog、情緒問卷與 GPS 資料。專案滿足課程要求：提供線上資料下載頁面，支援 JSON/CSV 匯出，並可於 README 中找到公開 URI。

## 立即重點

- **資料匯出頁面 URI（請在部署後更新成實際網址）**
  - 本地開發：`http://127.0.0.1:8000/data-export`
  - Render 正式站：`https://<your-render-app>.onrender.com/data-export`
  - 若設定 `EXPORT_TOKEN`，請在網址後加上 `?token=YOUR_TOKEN`
- **三種資料型別**：`/api/v1/datasets/vlogs`、`/sentiments`、`/gps`
- **健康檢查**：`/api/v1/health`

> ✅ 請務必在部署完成後，把正式站的 `https://.../data-export` URI 更新回本段，並於 NTU COOL 提交 GitHub 連結與此 URI。

## 專案結構

```
.
├── app/
│   ├── api/                 # FastAPI routers
│   ├── core/                # 設定檔與環境變數
│   ├── db/                  # Mongo 連線管理
│   ├── schemas/             # Pydantic schema
│   ├── services/            # 資料匯出工具
│   └── templates/           # 匯出頁面 (Jinja2)
├── emogo-frontend-BurningBright7214/   # 期中前端程式
├── main.py
├── render.yaml
├── requirements.txt
└── env.example
```

## 環境變數

| 變數名稱          | 說明                                               |
| ----------------- | -------------------------------------------------- |
| `MONGODB_URI`     | MongoDB Atlas 連線字串（白板指定：允許 IP = 0.0.0.0） |
| `DATABASE_NAME`   | Mongo 資料庫名稱，預設 `emogo`                     |
| `EXPORT_TOKEN`    | (可選) 匯出頁面密鑰。設定後需用 `?token=...` 存取   |
| `CORS_ALLOW_ORIGINS` | (可選) 允許的前端網域，逗號分隔                   |

> 建議：將 `env.example` 複製為 `.env` 後填上以上值，再啟動本地服務。

## 本地開發

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
setx MONGODB_URI "<your-connection-uri>"    # 或使用 .env
uvicorn main:app --reload
```

啟動後即可：
- `http://127.0.0.1:8000/docs`：互動式 API 說明
- `http://127.0.0.1:8000/data-export`：資料下載頁面（有 token 就加上 query）

## 資料匯出 / 測試流程

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料。
2. 確認後端啟動，造訪 `/data-export` 頁面，應可看到三種資料的筆數與最後一筆時間。
3. 點選 JSON/CSV 即可下載所有資料；header 會顯示 `X-Total-Count`。

## 部署到 Render

1. 到 Render 建立 **Web Service**，連結此 GitHub repo。
2. Build / Start 指令已寫在 `render.yaml`：
   - `pip install -r requirements.txt`
   - `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. 於 Render 的 Environment variables 區填入：
   - `MONGODB_URI`（請勿直接硬編碼到 repo）
   - `DATABASE_NAME=emogo`
   - `EXPORT_TOKEN=<自訂安全 token>`
4. 部署完成後，瀏覽 `https://<render-app>.onrender.com/data-export`，確認可以下載三大資料集。

## 前端整合（可選）

`emogo-frontend-BurningBright7214/` 內含 Expo App 原始碼。若要完整串接本後端，可：
- 在 APP 內新增設定畫面，填寫後端 base URL。
- 於拍攝 vlog 後呼叫 `POST /api/v1/datasets/vlogs` 上傳 metadata（影片檔可另存雲端，於 `video_url` 存取）。
- 用 `expo-location` 取得 GPS，送至 `/api/v1/datasets/gps`。

## 待辦 / 交付清單

- [ ] 在 README 中更新正式部署的 `data-export` URI。
- [ ] Render 後端可正常啟動並連線 MongoDB。
- [ ] MongoDB 中預先放入測試資料供 TA 驗證下載。
- [ ] 於 NTU COOL 提交 GitHub repo 連結與資料匯出 URI（deadline：12/4 Thu 20:00）。
```

```python
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.db.mongo import get_collection
from app.services.exporter import documents_to_csv, fetch_all_documents

router = APIRouter(tags=["export"])
templates = Jinja2Templates(directory="app/templates")

DATASETS = ("vlogs", "sentiments", "gps")


def ensure_token(token: str | None) -> None:
    """Simple shared-secret check for export-only endpoints."""

    if settings.export_token is None:
        return
    if token != settings.export_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token")


@router.get("/export/{dataset}")
async def export_dataset(
    dataset: str,
    request: Request,
    format: Literal["json", "csv"] = Query(default="json"),
    token: str | None = Query(default=None),
) -> Response:
    ensure_token(token)
    if dataset not in DATASETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown dataset")

    documents = await fetch_all_documents(get_collection(dataset))
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
    ensure_token(token)

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

    return templates.TemplateResponse(
        "data_export.html",
        {
            "request": request,
            "project_name": settings.project_name,
            "dataset_stats": dataset_stats,
            "token": token,
            "token_enabled": settings.export_token is not None,
        },
    )


```

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str | None = Field(default=None, alias="_id")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO timestamp recorded when the entry was created on the device.",
    )
    notes: str | None = Field(default=None, description="Optional user-provided notes.")


class SentimentEntry(BaseEntry):
    mood_score: int = Field(
        ge=0,
        le=10,
        description="Mood score provided by the user (0-10).",
        serialization_alias="moodScore",
        validation_alias="moodScore",
    )
    mood_label: str | None = Field(
        default=None,
        description="Optional mood label (e.g., happy, sad).",
        serialization_alias="moodLabel",
        validation_alias="moodLabel",
    )


class VlogEntry(SentimentEntry):
    video_uri: str = Field(
        description="Publicly accessible URL or storage reference for the recorded vlog.",
        serialization_alias="videoUri",
        validation_alias="videoUri",
    )
    latitude: float
    longitude: float


class GPSEntry(BaseEntry):
    latitude: float
    longitude: float
    accuracy: float | None = Field(default=None, description="GPS accuracy in meters, if available.")


class DatasetResponse(BaseModel):
    dataset: Literal["vlogs", "sentiments", "gps"]
    total: int
    items: list[dict[str, Any]]


```

## 變更重點
- 建立完整 FastAPI 啟動流程、CORS 與 MongoDB 生命週期，之後只要準備 `.env`（或環境變數）即可直接 `uvicorn main:app --reload` 啟動；首頁會帶出 `/docs` 與 `/data-export` 入口。  
```1:37:main.py
from fastapi import FastAPI
...
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(export_routes.router)
```

- 針對 `vlogs`、`sentiments`、`gps` 設計 schema 與共用路由，支援 `POST /api/v1/datasets/{dataset}` 新增資料、`GET` 查詢資料並回傳筆數；欄位 alias 與 EmoGo 前端（如 `moodScore`, `videoUri`, `latitude`) 對齊，方便之後直接串接。  
```1:59:app/api/routes/datasets.py
@router.get("/{dataset}", response_model=DatasetResponse)
async def list_entries(...):
    cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
...
@router.post("/{dataset}", status_code=status.HTTP_201_CREATED)
async def create_entry(...):
    entry = schema(**payload)
    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
```

- 新增 `/export/{dataset}` JSON/CSV 下載與 `/data-export` Jinja2 頁面，並可選擇以 `EXPORT_TOKEN` 保護。頁面會顯示三類資料的筆數、最後資料時間與直接下載按鈕。  
```27:85:app/api/routes/export.py
@router.get("/export/{dataset}")
async def export_dataset(...):
    documents = await fetch_all_documents(get_collection(dataset))
    if format == "csv":
        ...
@router.get("/data-export", response_class=HTMLResponse)
async def data_export_page(...):
    token_suffix = f"&token={token}" if token else ""
    ...
    return templates.TemplateResponse("data_export.html", {...})
```
```91:124:app/templates/data_export.html
<h1>{{ project_name }} Export Portal</h1>
<p class="subtitle">下載 EmoGo 前端蒐集的所有資料類型。</p>
...
<a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
<a class="csv" href="{{ dataset.csv_url }}">CSV</a>
```

- 補齊 `requirements.txt`、`render.yaml`（加入環境變數）、`env.example` 與全新 README，記載 Mongo/Render 設定、匯出頁面 URI、三類資料下載方式，TA 只需開啟 `/data-export` 即可取得 JSON/CSV。

## 測試
- `python -m compileall main.py app`

## 待您配合
- 依照 `env.example` 建立 `.env`（或在 PowerShell `setx`）並填入您提供的 `MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/`、`DATABASE_NAME=emogo`，再自訂 `EXPORT_TOKEN`（給 TA 的共享密鑰）。
- 用 MongoDB Compass 在 `vlogs`、`sentiments`、`gps` 三個 collection 建幾筆測試資料，方便驗證下載頁面。
- 本地測試：`pip install -r requirements.txt` → `uvicorn main:app --reload` → 開瀏覽器到 `http://127.0.0.1:8000/data-export?token=YOUR_TOKEN`。
- 部署到 Render 時，把同樣的三個環境變數填進服務設定，啟動完成後確認 `https://<render-app>.onrender.com/data-export` 可下載資料；最後回 README 的「資料匯出頁面 URI」欄位填上這個實際網址並在 NTU COOL 繳交。
- 若日後要讓 Expo 前端直接上傳，記得將 API base URL 指向新後端並呼叫 `POST /api/v1/datasets/{dataset}`，目前 schema 已與前端欄位命名對齊。

```python
from fastapi import FastAPI
...
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(export_routes.router)
```

```python
@router.get("/{dataset}", response_model=DatasetResponse)
async def list_entries(...):
    cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
...
@router.post("/{dataset}", status_code=status.HTTP_201_CREATED)
async def create_entry(...):
    entry = schema(**payload)
    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
```

```python
@router.get("/export/{dataset}")
async def export_dataset(...):
    documents = await fetch_all_documents(get_collection(dataset))
    if format == "csv":
        ...
@router.get("/data-export", response_class=HTMLResponse)
async def data_export_page(...):
    token_suffix = f"&token={token}" if token else ""
    ...
    return templates.TemplateResponse("data_export.html", {...})
```

```html
<h1>{{ project_name }} Export Portal</h1>
<p class="subtitle">下載 EmoGo 前端蒐集的所有資料類型。</p>
...
<a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
<a class="csv" href="{{ dataset.csv_url }}">CSV</a>
```

---

**User**

現在這個code有什麼功能

---

**Cursor**

- `main.py`：建立 FastAPI 應用並掛上 MongoDB 連線生命週期、CORS、根路由與 API/匯出路由，啟動後首頁會回傳服務名稱與 `/data-export` 入口。  
```1:37:main.py
app = FastAPI(title=settings.project_name)
...
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(export_routes.router)
```

- `app/api/routes/datasets.py`：提供 `GET/POST /api/v1/datasets/{vlogs|sentiments|gps}`，用 Pydantic schema 驗證前端送來的 vlog/心情/GPS 資料，寫入 Mongo，再支援分頁查詢與 ObjectId/json 序列化。  
```1:59:app/api/routes/datasets.py
cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
...
result = await collection.insert_one(document)
```

- `app/api/routes/export.py` + `app/templates/data_export.html`：實作 `/export/{dataset}?format=json|csv` 下載 API 與 `/data-export` 頁面，會列出三種資料的筆數、最後時間並提供 JSON/CSV 連結；若設定 `EXPORT_TOKEN` 會要求 query token。  
```1:85:app/api/routes/export.py
if format == "csv":
    ...
return JSONResponse(...)
```
```91:124:app/templates/data_export.html
<a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
<a class="csv" href="{{ dataset.csv_url }}">CSV</a>
```

- `app/core/config.py` & `app/db/mongo.py`：集中管理環境變數（`MONGODB_URI`, `DATABASE_NAME`, `EXPORT_TOKEN`, `CORS_ALLOW_ORIGINS`），並在啟動/關閉時建立 Motor client，提供 `get_collection()` 給路由使用。  

- `requirements.txt`、`render.yaml`、`env.example`、`README.md`：列出開發/部署依賴、Render 環境變數需求以及 README 指南（包含資料匯出頁面 URI、部署步驟、前端路徑與測試流程）。

總結：整套程式已是可部署的 EmoGo 後端，涵蓋 MongoDB 連線、資料寫入/查詢、資料匯出頁面、健康檢查及部署文件，只要設定環境變數即可在本地或 Render 上提供 TA 下載三類資料。

```python
app = FastAPI(title=settings.project_name)
...
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(export_routes.router)
```

```python
cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
...
result = await collection.insert_one(document)
```

```python
if format == "csv":
    ...
return JSONResponse(...)
```

```html
<a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
<a class="csv" href="{{ dataset.csv_url }}">CSV</a>
```

---

**User**

所以現在有資料可以讀嗎?

---

**Cursor**

目前 MongoDB 端預設還沒有資料；若要測試 `/api/v1/datasets/...` 或 `/data-export`，可以：

- 透過 MongoDB Compass 連上您剛建立的 `cluster0.lwlsy3w.mongodb.net`，在 `emogo` 資料庫底下新增 `vlogs`、`sentiments`、`gps` 三個 collection，手動插入幾筆測試 document。
- 或啟動後端、呼叫 `POST /api/v1/datasets/{dataset}`（例如 `sentiments`）送入資料。

只要其中任何一個 collection 有 document，`/data-export` 頁面與 JSON/CSV 下載就會顯示並讀取到那些資料。

---

**User**

The homework this week is totally independent of the last one—you can use fake data (or data from the outstanding submissions) to populate your backend DB/storage, even if you didn't get Emogo data last time.



Also, the export/download/dashboard page is an HTML page returned by FastAPI rather than a separate frontend.

Finally, the video download/export function is not fully functioning in some of the current submissions. Please check if you have a backend URI (rather than a frontend URI) or allow users to download those videos without requiring knowledge of the URIs.





根據上述要求，假設跟前端沒關係了，你的程式有做到上述要求嗎

---

**Cursor**

目前的後端狀態與三項補充要求對照如下：

- **可用假資料／不依賴前端**：已達成。`POST /api/v1/datasets/{vlogs|sentiments|gps}` 可直接塞入假資料；或用 MongoDB Compass/FastAPI 補資料，完全不需要前端參與。
- **資料匯出頁面由 FastAPI 回傳 HTML**：已達成。`/data-export` 由 Jinja2 模板產生，提供三類資料的 JSON/CSV 下載連結（也支援 token）。
- **影片下載必須由後端 URI 供應，而非只吐前端 URI**：目前**尚未做到**。`vlogs` 集合只儲存 `videoUri`（假設影片另存於前端或外部儲存），因此助教仍需知道檔案 URL 才能下載，不符合「後端就能直接提供影片」的條件。

若要完全符合第三點，需要再做以下調整（可接續開發）：
1. 讓後端接受影片上傳（`UploadFile`）並存到後端儲存（例如 Mongo GridFS、Render 磁碟或外部 S3），記錄檔案 id。
2. 新增 `/media/{file_id}` 之類的下載路由，在 `/data-export` 中改用這個後端 URI 供下載。
3. 匯出 CSV/JSON 時同樣提供後端可用的檔案網址。

只要完成上述第 3 點，整份作業就完全符合新的要求。需要我繼續實作影片上傳/下載與媒體路由嗎？

---

**User**

所以還是符合對嗎?現在的code

---

**Cursor**

目前這份後端程式 **尚未完全符合** 最新補充要求，主要差在：

- `vlogs` 集合仍只存放 `videoUri`（假設影片位於前端或其他儲存位置），`/data-export` 下載連結也只會指出該 URI，所以助教若不知道影片實際位置/權限，仍無法直接從後端取得影片。
- 未提供後端自有的影片下載端點（例如 `GET /media/{id}`）或直接由 FastAPI 匯出影片檔案。

若要符合「影片必須透過後端 URI 直接下載」的規範，還需要新增：後端上傳/儲存影片（幾種選擇：GridFS、Render 磁碟、S3…），並提供對應下載路由，在匯出頁面中使用這些後端 URI。若您希望我繼續補上這段功能，再告訴我即可動工。

---

**User**

好的請補上

---

**Cursor**

```pip-requirements
fastapi[all]
uvicorn[standard]
motor[srv]
pydantic-settings>=2.0.3
python-dotenv
jinja2
python-multipart
```

```python
from fastapi import HTTPException, status

from app.core.config import settings


def enforce_export_token(token: str | None) -> None:
    """Check whether the provided token matches settings.EXPORT_TOKEN (if set)."""

    if settings.export_token is None:
        return
    if token != settings.export_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token")

```

```python
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

    documents = await fetch_all_documents(get_collection(dataset))
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


```

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{ project_name }} · Data Export</title>
    <style>
      :root {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        color: #142748;
        background-color: #f6f7fb;
      }
      body {
        margin: 0;
        padding: 2rem;
      }
      h1 {
        margin-bottom: 0.25rem;
      }
      .subtitle {
        color: #506283;
        margin-bottom: 2rem;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(20, 39, 72, 0.08);
      }
      thead {
        background: #f0f3ff;
      }
      th,
      td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid #e4e8f3;
      }
      th {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c7ca4;
      }
      tr:last-child td {
        border-bottom: none;
      }
      .badge {
        display: inline-flex;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background: #eef3ff;
        color: #41509b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .actions a {
        display: inline-block;
        margin-right: 0.5rem;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: background 0.2s ease;
      }
      .actions a.json {
        background: #ebf8ff;
        color: #0a6fbf;
      }
      .actions a.csv {
        background: #fff3e6;
        color: #cc6a00;
      }
      .actions a:hover {
        filter: brightness(0.95);
      }
      .token-info {
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: #fff8d8;
        border: 1px solid #f7e5a0;
        border-radius: 10px;
        color: #8a6d1e;
      }
    </style>
  </head>
  <body>
    <h1>{{ project_name }} Export Portal</h1>
    <p class="subtitle">下載 EmoGo 前端蒐集的所有資料類型。</p>

    {% if token_enabled %}
    <div class="token-info">
      目前已啟用下載 Token 驗證。若看到「Unauthorized」請於網址列加上
      <code>?token=YOUR_TOKEN</code>，或在下方按鈕後自行補上 <code>&token=YOUR_TOKEN</code>。
    </div>
    {% endif %}

    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Total Records</th>
          <th>Last Entry</th>
          <th>Download</th>
        </tr>
      </thead>
      <tbody>
        {% for dataset in dataset_stats %}
        <tr>
          <td>
            <span class="badge">{{ dataset.name }}</span>
          </td>
          <td>{{ dataset.total }}</td>
          <td>{{ dataset.last_timestamp }}</td>
          <td class="actions">
            <a class="json" href="{{ dataset.json_url }}" target="_blank">JSON</a>
            <a class="csv" href="{{ dataset.csv_url }}">CSV</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    {% if recent_vlogs %}
    <h2 style="margin-top: 2.5rem">最新影片資料</h2>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Mood</th>
          <th>Label</th>
          <th>Download</th>
        </tr>
      </thead>
      <tbody>
        {% for vlog in recent_vlogs %}
        <tr>
          <td>{{ vlog.timestamp }}</td>
          <td>{{ vlog.moodScore }}</td>
          <td>{{ vlog.moodLabel }}</td>
          <td class="actions">
            <a class="csv" href="{{ vlog.download_url }}" download>下載 {{ vlog.filename }}</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </body>
</html>


```

```python
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

    grid_out = await open_file(file_id_obj)
    media_type = grid_out.metadata.get("content_type", "application/octet-stream") if grid_out.metadata else "application/octet-stream"
    filename = grid_out.metadata.get("original_filename") if grid_out.metadata else grid_out.filename

    return StreamingResponse(
        iter_gridfs_file(grid_out),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

```

```python
*** Begin Patch
*** Update File: app/services/storage.py
@@
-import csv
-import io
-from typing import Any, Iterable
-
-from fastapi.encoders import jsonable_encoder
-from motor.motor_asyncio import AsyncIOMotorCollection
-
-
-async def fetch_all_documents(collection: AsyncIOMotorCollection) -> list[dict[str, Any]]:
-    documents: list[dict[str, Any]] = []
-    cursor = collection.find().sort("timestamp", -1)
-    async for doc in cursor:
-        documents.append(jsonable_encoder(doc))
-    return documents
-
-
-def documents_to_csv(documents: Iterable[dict[str, Any]]) -> str:
-    """Convert a list of flat dictionaries to CSV format."""
-
-    rows = list(documents)
-    if not rows:
-        return ""
-
-    headers = sorted({key for row in rows for key in row.keys()})
-    buffer = io.StringIO()
-    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
-    writer.writeheader()
-    for row in rows:
-        writer.writerow(row)
-    return buffer.getvalue()
*** End Patch
```

```python
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

```

```python
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


```

```python
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import VlogEntry
from app.services.storage import save_upload_file

router = APIRouter(prefix="/vlogs", tags=["vlogs"])


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp format") from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_vlog(
    file: UploadFile = File(..., description="Vlog video file"),
    timestamp: str = Form(..., description="ISO timestamp captured on device"),
    moodScore: int = Form(..., ge=0, le=10),
    moodLabel: str | None = Form(default=None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: str | None = Form(default=None),
) -> dict:
    """Upload a vlog (video + metadata) and persist it together with a GridFS file."""

    metadata = {
        "content_type": file.content_type,
        "original_filename": file.filename,
    }
    file_id = await save_upload_file(file, metadata=metadata)
    video_uri = f"/media/{file_id}"

    entry_payload = {
        "timestamp": timestamp,
        "moodScore": moodScore,
        "moodLabel": moodLabel,
        "latitude": latitude,
        "longitude": longitude,
        "notes": notes,
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
    entry = VlogEntry(**entry_payload)

    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
    collection = get_collection("vlogs")
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document

```

```python
"""FastAPI routers registration helpers."""

from fastapi import APIRouter

from app.api.routes import datasets, health, vlogs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router, prefix="/datasets")
api_router.include_router(vlogs.router)


```

```python
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
```

```python
from . import datasets, export, health, media, vlogs

__all__ = ["datasets", "export", "health", "media", "vlogs"]


```

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str | None = Field(default=None, alias="_id")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO timestamp recorded when the entry was created on the device.",
    )
    notes: str | None = Field(default=None, description="Optional user-provided notes.")


class SentimentEntry(BaseEntry):
    mood_score: int = Field(
        ge=0,
        le=10,
        description="Mood score provided by the user (0-10).",
        serialization_alias="moodScore",
        validation_alias="moodScore",
    )
    mood_label: str | None = Field(
        default=None,
        description="Optional mood label (e.g., happy, sad).",
        serialization_alias="moodLabel",
        validation_alias="moodLabel",
    )


class VlogEntry(SentimentEntry):
    video_uri: str = Field(
        description="Publicly accessible URL or storage reference for the recorded vlog.",
        serialization_alias="videoUri",
        validation_alias="videoUri",
    )
    video_storage_id: str | None = Field(
        default=None,
        serialization_alias="videoStorageId",
        validation_alias="videoStorageId",
        description="GridFS file identifier for backend-hosted videos.",
    )
    video_filename: str | None = Field(
        default=None,
        serialization_alias="videoFilename",
        validation_alias="videoFilename",
        description="Original filename for download convenience.",
    )
    latitude: float
    longitude: float


class GPSEntry(BaseEntry):
    latitude: float
    longitude: float
    accuracy: float | None = Field(default=None, description="GPS accuracy in meters, if available.")


class DatasetResponse(BaseModel):
    dataset: Literal["vlogs", "sentiments", "gps"]
    total: int
    items: list[dict[str, Any]]


```

```python
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


```

```python
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


```

```markdown
# EmoGo Backend (FastAPI + MongoDB)

這是一個部署在 Render 的 FastAPI 服務，串接 MongoDB Atlas 以提供 EmoGo 前端（`emogo-frontend-BurningBright7214/`）上傳的 vlog、情緒問卷與 GPS 資料。專案滿足課程要求：提供線上資料下載頁面，支援 JSON/CSV 匯出，並以 FastAPI 端提供影片檔案下載連結（GridFS 儲存及 `/media/{file_id}` 下載）。

## 立即重點

- **資料匯出頁面 URI（請在部署後更新成實際網址）**
  - 本地開發：`http://127.0.0.1:8000/data-export`
  - Render 正式站：`https://<your-render-app>.onrender.com/data-export`
  - 若設定 `EXPORT_TOKEN`，請在網址後加上 `?token=YOUR_TOKEN`
- **三種資料型別**：`/api/v1/datasets/vlogs`、`/sentiments`、`/gps`
- **影片上傳/下載**：`POST /api/v1/vlogs`（multipart 上傳影片 + metadata），`GET /media/{file_id}` 下載 GridFS 檔案
- **健康檢查**：`/api/v1/health`

> ✅ 請務必在部署完成後，把正式站的 `https://.../data-export` URI 更新回本段，並於 NTU COOL 提交 GitHub 連結與此 URI。

## 專案結構

```
.
├── app/
│   ├── api/                 # FastAPI routers
│   ├── core/                # 設定檔與環境變數
│   ├── db/                  # Mongo 連線管理
│   ├── schemas/             # Pydantic schema
│   ├── services/            # 資料匯出 + GridFS 儲存工具
│   └── templates/           # 匯出頁面 (Jinja2)
├── emogo-frontend-BurningBright7214/   # 期中前端程式
├── main.py
├── render.yaml
├── requirements.txt
└── env.example
```

## 環境變數

| 變數名稱          | 說明                                               |
| ----------------- | -------------------------------------------------- |
| `MONGODB_URI`     | MongoDB Atlas 連線字串（白板指定：允許 IP = 0.0.0.0） |
| `DATABASE_NAME`   | Mongo 資料庫名稱，預設 `emogo`                     |
| `EXPORT_TOKEN`    | (可選) 匯出頁面密鑰。設定後需用 `?token=...` 存取   |
| `CORS_ALLOW_ORIGINS` | (可選) 允許的前端網域，逗號分隔                   |

> 建議：將 `env.example` 複製為 `.env` 後填上以上值，再啟動本地服務。

## 本地開發

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
setx MONGODB_URI "<your-connection-uri>"    # 或使用 .env
uvicorn main:app --reload
```

啟動後即可：
- `http://127.0.0.1:8000/docs`：互動式 API 說明
- `http://127.0.0.1:8000/data-export`：資料下載頁面（有 token 就加上 query）

## 影片上傳 / 匯出 / 測試流程

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料（可造假）。
2. 測試影片上傳（儲存於 Mongo GridFS，下載 URI 為 `/media/{file_id}`）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/vlogs \
  -F "file=@sample.mp4" \
  -F "timestamp=2025-12-03T10:00:00" \
  -F "moodScore=7" \
  -F "moodLabel=happy" \
  -F "latitude=25.0418" \
  -F "longitude=121.5431" \
  -F "notes=first fake vlog"
```

呼叫成功後會回傳 `videoStorageId` 與 `videoUri=/media/{id}`，所有下載都透過 FastAPI 端點處理，不需知道前端 URI。

3. 確認後端啟動，造訪 `/data-export` 頁面。此頁面：
   - 顯示三種資料的筆數與最後一筆時間，並提供 JSON/CSV 匯出按鈕（`/export/{dataset}?format=json|csv`）。
   - 額外列出最新 10 筆 vlog，提供直接的影片下載按鈕（指向 `/media/{file_id}`）。
4. 點選 JSON/CSV 下載或影片下載按鈕即可驗證整體流程；若有設定 token，請在網址後加上 `?token=YOUR_TOKEN`。

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料。
2. 確認後端啟動，造訪 `/data-export` 頁面，應可看到三種資料的筆數與最後一筆時間。
3. 點選 JSON/CSV 即可下載所有資料；header 會顯示 `X-Total-Count`。

## 部署到 Render

1. 到 Render 建立 **Web Service**，連結此 GitHub repo。
2. Build / Start 指令已寫在 `render.yaml`：
   - `pip install -r requirements.txt`
   - `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. 於 Render 的 Environment variables 區填入：
   - `MONGODB_URI`（請勿直接硬編碼到 repo）
   - `DATABASE_NAME=emogo`
   - `EXPORT_TOKEN=<自訂安全 token>`
4. 部署完成後，瀏覽 `https://<render-app>.onrender.com/data-export`，確認可以下載三大資料集。

## 前端整合（可選）

`emogo-frontend-BurningBright7214/` 內含 Expo App 原始碼。若要完整串接本後端，可：
- 在 APP 內新增設定畫面，填寫後端 base URL。
- 於拍攝 vlog 後直接呼叫 `POST /api/v1/vlogs`（multipart）上傳影片；後端會自動存到 GridFS 並回傳可下載的 backend URI。
- 用 `expo-location` 取得 GPS，送至 `/api/v1/datasets/gps`；心情量表可傳到 `/api/v1/datasets/sentiments`。

## 待辦 / 交付清單

- [ ] 在 README 中更新正式部署的 `data-export` URI。
- [ ] Render 後端可正常啟動並連線 MongoDB。
- [ ] MongoDB 中預先放入測試資料供 TA 驗證下載。
- [ ] 於 NTU COOL 提交 GitHub repo 連結與資料匯出 URI（deadline：12/4 Thu 20:00）。
```

```markdown
# EmoGo Backend (FastAPI + MongoDB)

這是一個部署在 Render 的 FastAPI 服務，串接 MongoDB Atlas 以提供 EmoGo 前端（`emogo-frontend-BurningBright7214/`）上傳的 vlog、情緒問卷與 GPS 資料。專案滿足課程要求：提供線上資料下載頁面，支援 JSON/CSV 匯出，並以 FastAPI 端提供影片檔案下載連結（GridFS 儲存及 `/media/{file_id}` 下載）。

## 立即重點

- **資料匯出頁面 URI（請在部署後更新成實際網址）**
  - 本地開發：`http://127.0.0.1:8000/data-export`
  - Render 正式站：`https://<your-render-app>.onrender.com/data-export`
  - 若設定 `EXPORT_TOKEN`，請在網址後加上 `?token=YOUR_TOKEN`
- **三種資料型別**：`/api/v1/datasets/vlogs`、`/sentiments`、`/gps`
- **影片上傳/下載**：`POST /api/v1/vlogs`（multipart 上傳影片 + metadata），`GET /media/{file_id}` 下載 GridFS 檔案
- **健康檢查**：`/api/v1/health`

> ✅ 請務必在部署完成後，把正式站的 `https://.../data-export` URI 更新回本段，並於 NTU COOL 提交 GitHub 連結與此 URI。

## 專案結構

```
.
├── app/
│   ├── api/                 # FastAPI routers
│   ├── core/                # 設定檔與環境變數
│   ├── db/                  # Mongo 連線管理
│   ├── schemas/             # Pydantic schema
│   ├── services/            # 資料匯出 + GridFS 儲存工具
│   └── templates/           # 匯出頁面 (Jinja2)
├── emogo-frontend-BurningBright7214/   # 期中前端程式
├── main.py
├── render.yaml
├── requirements.txt
└── env.example
```

## 環境變數

| 變數名稱          | 說明                                               |
| ----------------- | -------------------------------------------------- |
| `MONGODB_URI`     | MongoDB Atlas 連線字串（白板指定：允許 IP = 0.0.0.0） |
| `DATABASE_NAME`   | Mongo 資料庫名稱，預設 `emogo`                     |
| `EXPORT_TOKEN`    | (可選) 匯出頁面密鑰。設定後需用 `?token=...` 存取   |
| `CORS_ALLOW_ORIGINS` | (可選) 允許的前端網域，逗號分隔                   |

> 建議：將 `env.example` 複製為 `.env` 後填上以上值，再啟動本地服務。

## 本地開發

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
setx MONGODB_URI "<your-connection-uri>"    # 或使用 .env
uvicorn main:app --reload
```

啟動後即可：
- `http://127.0.0.1:8000/docs`：互動式 API 說明
- `http://127.0.0.1:8000/data-export`：資料下載頁面（有 token 就加上 query）

## 影片上傳 / 匯出 / 測試流程

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料（可造假）。
2. 測試影片上傳（儲存於 Mongo GridFS，下載 URI 為 `/media/{file_id}`）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/vlogs \
  -F "file=@sample.mp4" \
  -F "timestamp=2025-12-03T10:00:00" \
  -F "moodScore=7" \
  -F "moodLabel=happy" \
  -F "latitude=25.0418" \
  -F "longitude=121.5431" \
  -F "notes=first fake vlog"
```

呼叫成功後會回傳 `videoStorageId` 與 `videoUri=/media/{id}`，所有下載都透過 FastAPI 端點處理，不需知道前端 URI。

3. 確認後端啟動，造訪 `/data-export` 頁面。此頁面：
   - 顯示三種資料的筆數與最後一筆時間，並提供 JSON/CSV 匯出按鈕（`/export/{dataset}?format=json|csv`）。
   - 額外列出最新 10 筆 vlog，提供直接的影片下載按鈕（指向 `/media/{file_id}`）。
4. 點選 JSON/CSV 下載或影片下載按鈕即可驗證整體流程；若有設定 token，請在所有匯出/下載網址後加上 `?token=YOUR_TOKEN`（`/media/{file_id}` 也支援）。

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料。
2. 確認後端啟動，造訪 `/data-export` 頁面，應可看到三種資料的筆數與最後一筆時間。
3. 點選 JSON/CSV 即可下載所有資料；header 會顯示 `X-Total-Count`。

## 部署到 Render

1. 到 Render 建立 **Web Service**，連結此 GitHub repo。
2. Build / Start 指令已寫在 `render.yaml`：
   - `pip install -r requirements.txt`
   - `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. 於 Render 的 Environment variables 區填入：
   - `MONGODB_URI`（請勿直接硬編碼到 repo）
   - `DATABASE_NAME=emogo`
   - `EXPORT_TOKEN=<自訂安全 token>`
4. 部署完成後，瀏覽 `https://<render-app>.onrender.com/data-export`，確認可以下載三大資料集。

## 前端整合（可選）

`emogo-frontend-BurningBright7214/` 內含 Expo App 原始碼。若要完整串接本後端，可：
- 在 APP 內新增設定畫面，填寫後端 base URL。
- 於拍攝 vlog 後直接呼叫 `POST /api/v1/vlogs`（multipart）上傳影片；後端會自動存到 GridFS 並回傳可下載的 backend URI。
- 用 `expo-location` 取得 GPS，送至 `/api/v1/datasets/gps`；心情量表可傳到 `/api/v1/datasets/sentiments`。

## 待辦 / 交付清單

- [ ] 在 README 中更新正式部署的 `data-export` URI。
- [ ] Render 後端可正常啟動並連線 MongoDB。
- [ ] MongoDB 中預先放入測試資料供 TA 驗證下載。
- [ ] 於 NTU COOL 提交 GitHub repo 連結與資料匯出 URI（deadline：12/4 Thu 20:00）。
```

```python
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import VlogEntry
from app.services.storage import save_upload_file

router = APIRouter(prefix="/vlogs", tags=["vlogs"])


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp format") from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_vlog(
    file: UploadFile = File(..., description="Vlog video file"),
    timestamp: str = Form(..., description="ISO timestamp captured on device"),
    moodScore: int = Form(..., ge=0, le=10),
    moodLabel: str | None = Form(default=None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: str | None = Form(default=None),
) -> dict:
    """Upload a vlog (video + metadata) and persist it together with a GridFS file."""

    parsed_timestamp = parse_timestamp(timestamp)
    metadata = {
        "content_type": file.content_type,
        "original_filename": file.filename,
    }
    file_id = await save_upload_file(file, metadata=metadata)
    video_uri = f"/media/{file_id}"

    entry_payload = {
        "timestamp": parsed_timestamp,
        "moodScore": moodScore,
        "moodLabel": moodLabel,
        "latitude": latitude,
        "longitude": longitude,
        "notes": notes,
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
    entry = VlogEntry(**entry_payload)

    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
    collection = get_collection("vlogs")
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document


```

```python
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_collection
from app.schemas.datasets import VlogEntry
from app.services.storage import save_upload_file

router = APIRouter(prefix="/vlogs", tags=["vlogs"])


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp format") from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_vlog(
    file: UploadFile = File(..., description="Vlog video file"),
    timestamp: str = Form(..., description="ISO timestamp captured on device"),
    moodScore: int = Form(..., ge=0, le=10),
    moodLabel: str | None = Form(default=None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: str | None = Form(default=None),
) -> dict:
    """Upload a vlog (video + metadata) and persist it together with a GridFS file."""

    parsed_timestamp = parse_timestamp(timestamp)
    metadata = {
        "content_type": file.content_type,
        "original_filename": file.filename,
    }
    file_id = await save_upload_file(file, metadata=metadata)
    video_uri = f"/media/{file_id}"

    entry_payload = {
        "timestamp": parsed_timestamp,
        "moodScore": moodScore,
        "moodLabel": moodLabel,
        "latitude": latitude,
        "longitude": longitude,
        "notes": notes,
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
    entry = VlogEntry(**entry_payload)

    document = jsonable_encoder(entry, exclude_none=True, by_alias=True)
    collection = get_collection("vlogs")
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document


```

```markdown
# EmoGo Backend (FastAPI + MongoDB)

這是一個部署在 Render 的 FastAPI 服務，串接 MongoDB Atlas 以提供 EmoGo 前端（`emogo-frontend-BurningBright7214/`）上傳的 vlog、情緒問卷與 GPS 資料。專案滿足課程要求：提供線上資料下載頁面，支援 JSON/CSV 匯出，並以 FastAPI 端提供影片檔案下載連結（GridFS 儲存及 `/media/{file_id}` 下載）。

## 立即重點

- **資料匯出頁面 URI（請在部署後更新成實際網址）**
  - 本地開發：`http://127.0.0.1:8000/data-export`
  - Render 正式站：`https://<your-render-app>.onrender.com/data-export`
  - 若設定 `EXPORT_TOKEN`，請在網址後加上 `?token=YOUR_TOKEN`
- **三種資料型別**：`/api/v1/datasets/vlogs`、`/sentiments`、`/gps`
- **影片上傳/下載**：`POST /api/v1/vlogs`（multipart 上傳影片 + metadata），`GET /media/{file_id}` 下載 GridFS 檔案
- **健康檢查**：`/api/v1/health`

> ✅ 請務必在部署完成後，把正式站的 `https://.../data-export` URI 更新回本段，並於 NTU COOL 提交 GitHub 連結與此 URI。

## 專案結構

```
.
├── app/
│   ├── api/                 # FastAPI routers
│   ├── core/                # 設定檔與環境變數
│   ├── db/                  # Mongo 連線管理
│   ├── schemas/             # Pydantic schema
│   ├── services/            # 資料匯出 + GridFS 儲存工具
│   └── templates/           # 匯出頁面 (Jinja2)
├── emogo-frontend-BurningBright7214/   # 期中前端程式
├── main.py
├── render.yaml
├── requirements.txt
└── env.example
```

## 環境變數

| 變數名稱          | 說明                                               |
| ----------------- | -------------------------------------------------- |
| `MONGODB_URI`     | MongoDB Atlas 連線字串（白板指定：允許 IP = 0.0.0.0） |
| `DATABASE_NAME`   | Mongo 資料庫名稱，預設 `emogo`                     |
| `EXPORT_TOKEN`    | (可選) 匯出頁面密鑰。設定後需用 `?token=...` 存取   |
| `CORS_ALLOW_ORIGINS` | (可選) 允許的前端網域，逗號分隔                   |

> 建議：將 `env.example` 複製為 `.env` 後填上以上值，再啟動本地服務。

## 本地開發

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
setx MONGODB_URI "<your-connection-uri>"    # 或使用 .env
uvicorn main:app --reload
```

啟動後即可：
- `http://127.0.0.1:8000/docs`：互動式 API 說明
- `http://127.0.0.1:8000/data-export`：資料下載頁面（有 token 就加上 query）

## 影片上傳 / 匯出 / 測試流程

1. 依白板教學建立 MongoDB Atlas，使用 `MongoDB Compass` 建立 `vlogs`、`sentiments`、`gps` 集合並插入測試資料（可造假）。
2. 測試影片上傳（儲存於 Mongo GridFS，下載 URI 為 `/media/{file_id}`）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/vlogs \
  -F "file=@sample.mp4" \
  -F "timestamp=2025-12-03T10:00:00" \
  -F "moodScore=7" \
  -F "moodLabel=happy" \
  -F "latitude=25.0418" \
  -F "longitude=121.5431" \
  -F "notes=first fake vlog"
```

呼叫成功後會回傳 `videoStorageId` 與 `videoUri=/media/{id}`，所有下載都透過 FastAPI 端點處理，不需知道前端 URI。

3. 確認後端啟動，造訪 `/data-export` 頁面。此頁面：
   - 顯示三種資料的筆數與最後一筆時間，並提供 JSON/CSV 匯出按鈕（`/export/{dataset}?format=json|csv`）。
   - 額外列出最新 10 筆 vlog，提供直接的影片下載按鈕（指向 `/media/{file_id}`）。
4. 點選 JSON/CSV 下載或影片下載按鈕即可驗證整體流程；若有設定 token，請在所有匯出/下載網址後加上 `?token=YOUR_TOKEN`（`/media/{file_id}` 也支援）。

## 部署到 Render

1. 到 Render 建立 **Web Service**，連結此 GitHub repo。
2. Build / Start 指令已寫在 `render.yaml`：
   - `pip install -r requirements.txt`
   - `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. 於 Render 的 Environment variables 區填入：
   - `MONGODB_URI`（請勿直接硬編碼到 repo）
   - `DATABASE_NAME=emogo`
   - `EXPORT_TOKEN=<自訂安全 token>`
4. 部署完成後，瀏覽 `https://<render-app>.onrender.com/data-export`，確認可以下載三大資料集。

## 前端整合（可選）

`emogo-frontend-BurningBright7214/` 內含 Expo App 原始碼。若要完整串接本後端，可：
- 在 APP 內新增設定畫面，填寫後端 base URL。
- 於拍攝 vlog 後直接呼叫 `POST /api/v1/vlogs`（multipart）上傳影片；後端會自動存到 GridFS 並回傳可下載的 backend URI。
- 用 `expo-location` 取得 GPS，送至 `/api/v1/datasets/gps`；心情量表可傳到 `/api/v1/datasets/sentiments`。

## 待辦 / 交付清單

- [ ] 在 README 中更新正式部署的 `data-export` URI。
- [ ] Render 後端可正常啟動並連線 MongoDB。
- [ ] MongoDB 中預先放入測試資料供 TA 驗證下載。
- [ ] 於 NTU COOL 提交 GitHub repo 連結與資料匯出 URI（deadline：12/4 Thu 20:00）。
```

## 更新內容
- 影片現在透過後端 `POST /api/v1/vlogs` 上傳並存入 Mongo GridFS，metadata 會自動寫入 `vlogs` 集合，並提供 `/media/{file_id}` 的後端 URI（含 `videoStorageId`、`videoFilename`）。  
```1:50:app/api/routes/vlogs.py
    entry_payload = {
        "timestamp": parsed_timestamp,
        "moodScore": moodScore,
        ...
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
```
```1:35:app/api/routes/media.py
@router.get("/media/{file_id}", name="download_media")
async def download_media(...):
    enforce_export_token(token)
    ...
    return StreamingResponse(
        iter_gridfs_file(grid_out),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
```
- `/data-export` 頁面與 `/export/{dataset}` 匯出 API 會自動將 vlog 的 `videoUri` 轉成可下載的後端連結，並在頁面上列出最新 10 筆 vlog 供直接下載，完全符合「由後端提供影片下載」的要求。  
```30:105:app/api/routes/export.py
    if dataset == "vlogs":
        for document in documents:
            storage_id = document.get("videoStorageId") ...
            document["videoUri"] = download_url
    ...
    recent_vlogs = await get_collection("vlogs").find().sort("timestamp", -1).limit(10).to_list(10)
```
```92:124:app/templates/data_export.html
    {% if recent_vlogs %}
    <h2 style="margin-top: 2.5rem">最新影片資料</h2>
    ...
            <a class="csv" href="{{ vlog.download_url }}" download>下載 {{ vlog.filename }}</a>
```
- README 重新說明影片上傳／下載流程、curl 範例、部署注意事項與 NTU COOL 交付重點，並提醒所有匯出/下載端點（包含 `/media/{file_id}`）如設定 `EXPORT_TOKEN` 需附加 `?token=`。  
```1:82:README.md
呼叫成功後會回傳 `videoStorageId` 與 `videoUri=/media/{id}`...
4. 點選 JSON/CSV 下載或影片下載按鈕... `?token=YOUR_TOKEN`（`/media/{file_id}` 也支援）。
```

## 測試
- `python -m compileall main.py app`

## 下一步建議
- 依 `env.example` 建立 `.env`，填入您提供的 `MONGODB_URI`、`DATABASE_NAME=emogo` 以及自訂 `EXPORT_TOKEN`，然後 `uvicorn main:app --reload` 本地驗證。
- 用 MongoDB Compass 或上述 curl 指令造幾筆假 vlog/情緒/GPS 資料，確認 `/data-export?token=...` 能下載 JSON、CSV 與影片。
- 部署至 Render 時把三個環境變數寫入服務設定，部署完成後更新 README 的正式 `https://.../data-export` URI，再將 repo 連結與該 URI 回報 NTU COOL。

```python
    entry_payload = {
        "timestamp": parsed_timestamp,
        "moodScore": moodScore,
        ...
        "videoUri": video_uri,
        "videoStorageId": file_id,
        "videoFilename": file.filename or "vlog.mp4",
    }
```

```python
@router.get("/media/{file_id}", name="download_media")
async def download_media(...):
    enforce_export_token(token)
    ...
    return StreamingResponse(
        iter_gridfs_file(grid_out),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
```

```python
    if dataset == "vlogs":
        for document in documents:
            storage_id = document.get("videoStorageId") ...
            document["videoUri"] = download_url
    ...
    recent_vlogs = await get_collection("vlogs").find().sort("timestamp", -1).limit(10).to_list(10)
```

```html
    {% if recent_vlogs %}
    <h2 style="margin-top: 2.5rem">最新影片資料</h2>
    ...
            <a class="csv" href="{{ vlog.download_url }}" download>下載 {{ vlog.filename }}</a>
```

```plaintext
呼叫成功後會回傳 `videoStorageId` 與 `videoUri=/media/{id}`...
4. 點選 JSON/CSV 下載或影片下載按鈕... `?token=YOUR_TOKEN`（`/media/{file_id}` 也支援）。
```

---

**User**

詳細告訴我接著要幹嘛

---

**Cursor**

後續建議照這個順序進行：

1. **建立本地環境變數 / .env**
   - 依 `env.example` 建一份 `.env`，填入：
     - `MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/`
     - `DATABASE_NAME=emogo`
     - `EXPORT_TOKEN=<自訂安全字串>`（給 TA 的下載密鑰）
   - 若要限制 CORS，可再加 `CORS_ALLOW_ORIGINS=https://...`。

2. **本地啟動與測試**
   - 建議指令：
     ```
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     uvicorn main:app --reload
     ```
   - 用 curl 或 Postman 測試：
     - `POST /api/v1/vlogs`（multipart，上傳小 mp4，確認回傳 `videoStorageId`）。
     - `POST /api/v1/datasets/sentiments`、`/gps`（送假資料）。
     - `GET /api/v1/datasets/vlogs?token=...`（JSON 內的 `videoUri` 應是 `/media/{id}?token=...`）。
     - `GET /media/{file_id}?token=...`（瀏覽器會下載影片）。
   - 造訪 `http://127.0.0.1:8000/data-export?token=...`，核查三種資料筆數、最近 vlog 表格與下載按鈕是否正常。

3. **MongoDB 側驗證**
   - 用 MongoDB Compass 連線 Atlas，檢查 `emogo` DB：
     - `vlogs.files`/`vlogs.chunks`（GridFS bucket `vlog_files`）確認有檔案。
     - `vlogs` 集合是否有 `videoStorageId`、`videoUri=/media/...`。
     - `sentiments`、`gps` 是否有測試資料。
   - 可另外插入 TA 方便檢查的假資料（例如三種資料各 5 筆）。

4. **部署到 Render**
   - 登入 Render → New Web Service → 連 GitHub repo。
   - 確認 `render.yaml` 生效或在面板設定：
     - Build: `pip install -r requirements.txt`
     - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - 在 Render 環境變數區填：
     - `MONGODB_URI`（同本地）
     - `DATABASE_NAME=emogo`
     - `EXPORT_TOKEN=<與本地一致或另設>`
   - 部署完成後打開 `https://<app>.onrender.com/data-export?token=...` 進行 Smoke Test。

5. **更新 README 與繳交**
   - 把 README「資料匯出頁面 URI」改成實際 Render 網址。
   - 若 `EXPORT_TOKEN` 需提供給 TA，請在繳交時一併附上使用方式（例如「請以 `?token=xxx` 存取」）。
   - 確認 repo push 完成 → 在 NTU COOL 提交 GitHub 連結與 `https://.../data-export` URI。

6. **Optional：前端更新（如需）**
   - 如果要讓 `emogo-frontend` 串接：
     - 新增設定：可輸入後端 base URL、token。
     - 改寫上傳流程：拍完 vlog 直接呼叫 `POST /api/v1/vlogs`，把影片檔 (`FormData` file) + metadata 一起送出。
     - Sentiment/GPS 上傳對應 API。
   - 測試 Expo App → 影片/情緒/GPS 都能出現在後端匯出頁面。

若執行過程遇到任何錯誤（特別是 GridFS 權限、Render 記憶體限制、影片檔案過大等），請把錯誤訊息貼給我，我可以協助除錯。

```plaintext
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     uvicorn main:app --reload
```

---

**User**

我們一步步來，先告訴我第一步要做什麼

---

**Cursor**

第一步：建立本地環境設定。請把 `env.example` 複製成 `.env`（或直接新增 `.env`），並填入以下值：

```
MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/
DATABASE_NAME=emogo
EXPORT_TOKEN=<自訂安全字串，例如 emogo-test-token>
```

若未來有特定前端網域要允許，再加 `CORS_ALLOW_ORIGINS=`。完成後告訴我，我們就進到下一步。

```plaintext
MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/
DATABASE_NAME=emogo
EXPORT_TOKEN=<自訂安全字串，例如 emogo-test-token>
```

---

**User**

你直接幫我新增即可

---

**Cursor**

我嘗試直接建立 `.env` 檔，但此專案環境啟用了安全限制（globalignore），無法由我這邊新增 `.env`。請您在專案根目錄手動建立 `.env`，內容如下：

```
MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/
DATABASE_NAME=emogo
EXPORT_TOKEN=emogo-test-token
# CORS_ALLOW_ORIGINS=https://your-frontend.example.com
```

存檔後告訴我，我們再進行下一步。

```plaintext
MONGODB_URI=mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/
DATABASE_NAME=emogo
EXPORT_TOKEN=emogo-test-token
# CORS_ALLOW_ORIGINS=https://your-frontend.example.com
```

---

**User**

好的新增完畢

---

**Cursor**

太好了！下一步：

### 步驟 2：建立與啟動本地環境
1. 在專案根目錄開 PowerShell 依序執行：
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
2. 啟動後在瀏覽器開啟 `http://127.0.0.1:8000/docs` 或 `http://127.0.0.1:8000/data-export?token=emogo-test-token`。

完成或遇到問題隨時告訴我，我再協助下一步（例如上傳假資料、測試下載等）。

```plaintext
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload
```

---

**User**

<html>{"service":"EmoGo Backend","docs":"/docs","export_page":"/data-export"} html</html> 我要怎麼檢查是否正確

---

**Cursor**

看到首頁（`GET /`）回傳 `{"service":"EmoGo Backend","docs":"/docs","export_page":"/data-export"}`，表示後端已成功啟動。接下來可以這樣檢查更完整：

1. **打開 Swagger**  
   在瀏覽器進入 `http://127.0.0.1:8000/docs`，確認所有 API 都列在裡面（health、datasets、vlogs、export、media 等），可以直接在頁面上試呼叫。

2. **健康檢查**  
   用瀏覽器或 curl：  
   ```
   curl http://127.0.0.1:8000/api/v1/health
   ```
   會看到 Mongo `status: ok`、`database: reachable` 之類的回應。

3. **資料匯出頁面**  
   造訪 `http://127.0.0.1:8000/data-export?token=emogo-test-token`（如果沒設 token 可省略 query）。這裡應顯示三類資料的筆數、最後時間，還會有 JSON/CSV 下載連結。

4. **上傳測試 vlog**（若想確定影音功能）  
   用 curl 上傳一個小 MP4：
   ```
   curl -X POST http://127.0.0.1:8000/api/v1/vlogs \
     -F "file=@sample.mp4" \
     -F "timestamp=2025-12-03T10:00:00" \
     -F "moodScore=7" \
     -F "latitude=25.0418" \
     -F "longitude=121.5431"
   ```
   回傳 JSON 會包含 `videoStorageId` 與 `/media/{id}` URI。

5. **下載影片驗證**  
   直接在瀏覽器開 `http://127.0.0.1:8000/media/<videoStorageId>?token=emogo-test-token`，應該會觸發檔案下載。

這些測試都通過就代表本地環境沒問題，接下來就可以進行資料造假/部署等後續步驟。需要我陪你跑其中某個測試可以再說！

```plaintext
   curl http://127.0.0.1:8000/api/v1/health
```

```plaintext
   curl -X POST http://127.0.0.1:8000/api/v1/vlogs \
     -F "file=@sample.mp4" \
     -F "timestamp=2025-12-03T10:00:00" \
     -F "moodScore=7" \
     -F "latitude=25.0418" \
     -F "longitude=121.5431"
```

---

**User**

好的下一步

---

**Cursor**

下一步：先放幾筆測試資料進 Mongo，之後 `/data-export` 才看得到內容。建議用 API 直接塞假資料（省得在 Compass 手動輸入）：

1. **情緒資料 (sentiments)**  
   ```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
     -H "Content-Type: application/json" ^
     -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
   ```

2. **GPS 資料 (gps)**  
   ```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/datasets/gps ^
     -H "Content-Type: application/json" ^
     -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
   ```

3. **Vlog + 影片**（準備一個小的 MP4 或任何影片檔）：  
   ```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/vlogs ^
     -F "file=@sample.mp4" ^
     -F "timestamp=2025-12-03T10:10:00" ^
     -F "moodScore=8" ^
     -F "moodLabel=excited" ^
     -F "latitude=25.033" ^
     -F "longitude=121.565" ^
     -F "notes=fake vlog upload"
   ```

每跑一次就多一筆資料；若想多幾筆可調整時間、分數再重複送。完成後告訴我，我們接著檢查 `/data-export` 與影片下載是否正常。

```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
     -H "Content-Type: application/json" ^
     -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/datasets/gps ^
     -H "Content-Type: application/json" ^
     -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
```

```powershell
   curl -X POST http://127.0.0.1:8000/api/v1/vlogs ^
     -F "file=@sample.mp4" ^
     -F "timestamp=2025-12-03T10:10:00" ^
     -F "moodScore=8" ^
     -F "moodLabel=excited" ^
     -F "latitude=25.033" ^
     -F "longitude=121.565" ^
     -F "notes=fake vlog upload"
```

---

**User**

請問你講的要怎麼做，是要在mongodb comprass 裡面做嗎

---

**Cursor**

不用一定在 MongoDB Compass 裡面手動新增，我建議直接在你剛啟動的 FastAPI 後端用 API 方式插入假資料，比較快速，也能順便驗證 API。方法有兩種：

---

### 方法 A：用命令列 `curl`（推薦）
1. 開啟 PowerShell，確認後端還在 `uvicorn` 執行。
2. 複製以下指令（逐條貼上回車）：

**情緒資料 (sentiments)**
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

**GPS 資料 (gps)**
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/datasets/gps ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
```

**Vlog + 影片**
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/vlogs ^
  -F "file=@sample.mp4" ^
  -F "timestamp=2025-12-03T10:10:00" ^
  -F "moodScore=8" ^
  -F "moodLabel=excited" ^
  -F "latitude=25.033" ^
  -F "longitude=121.565" ^
  -F "notes=fake vlog upload"
```
把 `sample.mp4` 換成你電腦上任意小影片的路徑（同資料夾可直接寫檔名）。這樣就會透過後端 API 寫入 MongoDB，GridFS 也會自動儲存影片並產生 `/media/{file_id}`。

---

### 方法 B：在 MongoDB Compass 手動新增
如果你更習慣用 Compass，也可以這樣做：
1. Compass 連線 `mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/`.
2. 選擇 `emogo` DB，建三個 collection：`vlogs`、`sentiments`、`gps`。
3. 每個 collection 點 `Insert Document`，填入 JSON（例如 sentiments 的 JSON 內容同上）。
4. vlogs 的 document 要自行填 `videoUri`。但這樣影片檔本身不會在 GridFS 裡，後端也無法提供下載，會違反作業要求，所以這個方式不建議用於 vlog。

---

總結：**情緒、GPS 可以用 Compass 或 API 隨意建立；vlog 一定要透過 `POST /api/v1/vlogs` 才能讓影片檔進後端 GridFS。** 建議照方法 A 一次解決三種資料。完成後告訴我，我們就去 `/data-export` 驗證。

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/datasets/gps ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
```

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/vlogs ^
  -F "file=@sample.mp4" ^
  -F "timestamp=2025-12-03T10:10:00" ^
  -F "moodScore=8" ^
  -F "moodLabel=excited" ^
  -F "latitude=25.033" ^
  -F "longitude=121.565" ^
  -F "notes=fake vlog upload"
```

---

**User**

@powershell (4-41) 

---

**Cursor**

在 PowerShell 裡輸入 `curl` 其實是對 `Invoke-WebRequest` 的別名，所以 `-X/-H/-d` 這些參數會被當成未知參數。可以改用下列方式之一：

### 方式 1：使用 `curl.exe`
在指令前面寫 `curl.exe`（或 `Invoke-WebRequest`），讓 PowerShell 不要把它轉成別名：
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```
其他指令（GPS、vlogs）同理，把最前面的 `curl` 改成 `curl.exe` 即可。

### 方式 2：用 `Invoke-WebRequest` 的參數寫法
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -Method Post `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"timestamp":"2025-12-03T10:00:00","moodScore":7,"moodLabel":"happy","notes":"fake sentiment"}'
```

建議簡單起見直接用方式 1（`curl.exe` 或 `curl.exe -k ...`）。試重新送一次，成功時會看到 API 回傳 JSON。若還有錯誤再告訴我。

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -Method Post `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"timestamp":"2025-12-03T10:00:00","moodScore":7,"moodLabel":"happy","notes":"fake sentiment"}'
```

---

**User**

@powershell (42-72) 

---

**Cursor**

問題有兩個：

1. **FastAPI 可能沒在跑**  
   `curl: (7) Failed to connect to 127.0.0.1 port 8000` 表示 `uvicorn` 伺服器不是正在跑（或跑在別的 port）。請確認另一個終端有執行 `uvicorn main:app --reload` 且沒有結束。

2. **PowerShell 的換行符號**  
   PowerShell 裡換行要用反引號 `` ` ``，不是 `^`。`^` 是 cmd 的寫法，所以 `-H/-d` 會被 PS 當成新指令。

請照以下方式測試（把敘述貼進 PowerShell）：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -H "Content-Type: application/json" `
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

如果你不想換行，可以整串放在同一行：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

先確認 `uvicorn` 有在跑，再執行上述命令，應該就能成功寫入假資料。

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -H "Content-Type: application/json" `
  -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

---

**User**

先確認 uvicorn 有在跑怎麼確認

---

**Cursor**

確認 `uvicorn` 是否在執行，可以用以下方式：

1. **查看啟動終端畫面**  
   - 若你之前在某個 PowerShell 視窗執行 `uvicorn main:app --reload`，那個視窗中應持續顯示類似：
     ```
     INFO:     Started server process [xxxx]
     INFO:     Application startup complete.
     INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
     ```
   - 如果該視窗已經沒有這樣的訊息，或已回到指令提示符號，代表伺服器已停止，需再次執行 `uvicorn main:app --reload`。

2. **瀏覽器測試**  
   - 在瀏覽器輸入 `http://127.0.0.1:8000/`，若能看到 `{"service":"EmoGo Backend",...}` 的 JSON，表示伺服器正在運行。

3. **PowerShell 測試**  
   - 在任一 PowerShell 視窗輸入 `Invoke-WebRequest http://127.0.0.1:8000/`。
   - 成功會顯示 JSON；如果顯示「無法連線」，代表 `uvicorn` 未啟動。

請先確認上述其中一項，如果伺服器沒跑，再在專案根目錄執行：
```
.venv\Scripts\activate
uvicorn main:app --reload
```
啟動後再回到插入假資料的步驟。

```plaintext
     INFO:     Started server process [xxxx]
     INFO:     Application startup complete.
     INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

```plaintext
.venv\Scripts\activate
uvicorn main:app --reload
```

---

**User**

@uvicorn (4-20) 然後

---

**Cursor**

伺服器已經成功啟動了，接下來回到另一個 PowerShell 視窗（不要中斷跑 uvicorn 的視窗），開始塞測試資料。請依序執行：

### 1. Sentiment 假資料
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

### 2. GPS 假資料
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/gps -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
```

### 3. Vlog（需準備影片檔，例如 `sample.mp4`）
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs -F "file=@sample.mp4" -F "timestamp=2025-12-03T10:10:00" -F "moodScore=8" -F "moodLabel=excited" -F "latitude=25.033" -F "longitude=121.565" -F "notes=fake vlog upload"
```
把最後一行的 `sample.mp4` 換成電腦上的實際檔名。

每條指令都要在新視窗/分頁執行（原本跑 `uvicorn` 的視窗保持開著）。成功後終端會回傳 JSON。完成後再回報，我們去 `/data-export` 確認資料與影片下載。

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:00:00\",\"moodScore\":7,\"moodLabel\":\"happy\",\"notes\":\"fake sentiment\"}"
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/gps -H "Content-Type: application/json" -d "{\"timestamp\":\"2025-12-03T10:05:00\",\"latitude\":25.0418,\"longitude\":121.5431,\"accuracy\":5.2}"
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs -F "file=@sample.mp4" -F "timestamp=2025-12-03T10:10:00" -F "moodScore=8" -F "moodLabel=excited" -F "latitude=25.033" -F "longitude=121.565" -F "notes=fake vlog upload"
```

---

**User**

@powershell (5-16) 

---

**Cursor**

你的 `-d "{\"timestamp\":...}` 雖然有跳脫，但在 PowerShell 中還是容易被重新解讀，導致 JSON 內容壞掉。建議改用單引號包住整個 JSON 字串（單引號在 PowerShell 內是純文字，不會再處理跳脫），或乾脆用 PowerShell 內建的 `Invoke-RestMethod`。

下面直接給可複製的範例，這次改用單引號，並保持所有內容在同一行：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d '{"timestamp":"2025-12-03T10:00:00","moodScore":7,"moodLabel":"happy","notes":"fake sentiment"}'
```

同理，GPS 那條也改成：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/gps -H "Content-Type: application/json" -d '{"timestamp":"2025-12-03T10:05:00","latitude":25.0418,"longitude":121.5431,"accuracy":5.2}'
```

重點：整個 JSON 用單引號 `'...'` 包住即可，不用再加跳脫符號。Vlog 那條因為是 multipart，就保持原寫法但確認影片路徑正確。再次嘗試後若還有問題再告訴我。

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/sentiments -H "Content-Type: application/json" -d '{"timestamp":"2025-12-03T10:00:00","moodScore":7,"moodLabel":"happy","notes":"fake sentiment"}'
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/gps -H "Content-Type: application/json" -d '{"timestamp":"2025-12-03T10:05:00","latitude":25.0418,"longitude":121.5431,"accuracy":5.2}'
```

---

**User**

@powershell (18-26) 

---

**Cursor**

看來 PowerShell 還是把 `-d ' {... }'` 裡的 JSON 讀壞了；改用 PowerShell 原生的 `Invoke-RestMethod` 最穩。請在 **另一個** PowerShell 視窗（uvicorn 那個保持運行）貼上以下指令：

### Sentiments
```powershell
$sentiment = @{
    timestamp = "2025-12-03T10:00:00"
    moodScore = 7
    moodLabel = "happy"
    notes     = "fake sentiment"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -Method Post `
  -ContentType "application/json" `
  -Body $sentiment
```

### GPS
```powershell
$gps = @{
    timestamp = "2025-12-03T10:05:00"
    latitude  = 25.0418
    longitude = 121.5431
    accuracy  = 5.2
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/datasets/gps `
  -Method Post `
  -ContentType "application/json" `
  -Body $gps
```

指令執行後會直接回傳新增的 JSON，這樣就完成假資料插入。成功後我們再進行 vlog 上傳與資料匯出檢查。

```powershell
$sentiment = @{
    timestamp = "2025-12-03T10:00:00"
    moodScore = 7
    moodLabel = "happy"
    notes     = "fake sentiment"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/datasets/sentiments `
  -Method Post `
  -ContentType "application/json" `
  -Body $sentiment
```

```powershell
$gps = @{
    timestamp = "2025-12-03T10:05:00"
    latitude  = 25.0418
    longitude = 121.5431
    accuracy  = 5.2
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/datasets/gps `
  -Method Post `
  -ContentType "application/json" `
  -Body $gps
```

---

**User**

影片呢

---

**Cursor**

接下來補 vlog 影片那筆資料。步驟如下（記得保持 `uvicorn` 在另一個視窗持續執行）：

1. 準備一個小型影片檔（MP4、MOV 皆可），例如 `C:\Users\b8759\Downloads\sample.mp4`。  
2. 在新的 PowerShell 視窗執行（請把檔案路徑換成你的實際檔案）：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs `
  -F "file=@C:\Users\b8759\Downloads\sample.mp4" `
  -F "timestamp=2025-12-03T10:10:00" `
  -F "moodScore=8" `
  -F "moodLabel=excited" `
  -F "latitude=25.033" `
  -F "longitude=121.565" `
  -F "notes=fake vlog upload"
```

- `@` 後面一定要是檔案完整路徑。  
- 若影片名稱有空白，可用引號包住整個 `file=@"C:\path with spaces\video.mp4"`。

成功後會得到新增的 vlog JSON，包含 `videoStorageId`（Mongo GridFS 的檔案 ID）與 `videoUri`（後端 `/media/{id}` 下載連結）。

完成後回報我，再帶你去 `/data-export?token=...` 驗證影片下載與 JSON/CSV 匯出。

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs `
  -F "file=@C:\Users\b8759\Downloads\sample.mp4" `
  -F "timestamp=2025-12-03T10:10:00" `
  -F "moodScore=8" `
  -F "moodLabel=excited" `
  -F "latitude=25.033" `
  -F "longitude=121.565" `
  -F "notes=fake vlog upload"
```

---

**User**

你可以隨便幫我生成嗎

---

**Cursor**

我沒辦法在你電腦上直接生成實體影片檔，但你可以利用專案裡現成的檔案來測試：`emogo-frontend-BurningBright7214\data\videos\` 目錄下已經有多個 `.mp4`。例如選其中一個：

```
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs `
  -F "file=@.\emogo-frontend-BurningBright7214\data\videos\0b9d73e3-25a9-413e-bde9-15c0b291bfe5.mp4" `
  -F "timestamp=2025-12-03T10:10:00" `
  -F "moodScore=8" `
  -F "moodLabel=excited" `
  -F "latitude=25.033" `
  -F "longitude=121.565" `
  -F "notes=fake vlog upload"
```

這樣就會把那個現有影片送進後端 GridFS，滿足測試需求。若想換別的檔案，只要替換 `file=@...` 路徑即可。需要我幫你調整其他欄位或再新增多筆，也可以再說。

```plaintext
curl.exe -X POST http://127.0.0.1:8000/api/v1/vlogs `
  -F "file=@.\emogo-frontend-BurningBright7214\data\videos\0b9d73e3-25a9-413e-bde9-15c0b291bfe5.mp4" `
  -F "timestamp=2025-12-03T10:10:00" `
  -F "moodScore=8" `
  -F "moodLabel=excited" `
  -F "latitude=25.033" `
  -F "longitude=121.565" `
  -F "notes=fake vlog upload"
```

---

**User**

好的接著要做什麼

---

**Cursor**

接下來做資料檢查＆驗證下載：

1. **開資料匯出頁面**  
   在瀏覽器輸入 `http://127.0.0.1:8000/data-export?token=emogo-test-token`。  
   - 確認三個 dataset（vlogs/sentiments/gps）都有筆數、最後時間。
   - 下面的「最新影片資料」表格應列出你剛上傳的 vlog，並有下載按鈕。

2. **測試 JSON/CSV 匯出**  
   - 點 `vlogs` / `sentiments` / `gps` 每個的 JSON/CSV，確認檔案能下載且內容正確。  
   - 若瀏覽器會直接顯示 JSON，記錄下 URL（驗證 Render 時要提供給 TA）。

3. **測試影片下載**  
   - 在「最新影片資料」表格點擊下載鈕或直接開 `http://127.0.0.1:8000/media/<videoStorageId>?token=emogo-test-token`。  
   - 瀏覽器應下載 mp4，證明後端 URI 可直接取得影片。

完成這些後，我們就可以進行部署步驟（設定 Render 環境、推到 GitHub 等）。若檢視途中有任何異常，隨時截圖或貼錯誤給我，我會幫忙排查。

---

**User**

點json出現Internal Server Error

---

**Cursor**

```python
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


```

我已補上 `ObjectId` 的轉換（`app/services/exporter.py` 會把 Mongo 的 `_id` 直接轉成字串），`uvicorn --reload` 會自動重新載入，所以不用手動重啟。請重新整理或再次點擊 `/export/vlogs?format=json&token=...`（或其他 dataset 的 JSON/CSV 連結），應該就不會再出現 Internal Server Error；若還有問題再告訴我。

---

**User**

好的有了，接著要幹嘛

---

**Cursor**

太好了，現在資料匯出與下載都 OK，接下來可以進入部署與交付階段，建議依序進行：

1. **推送最新版程式到 GitHub**
   - 在專案根目錄檢查 git 狀態：`git status`
   - 若未初始化 git 或尚未綁 GitHub，先建立遠端 repo。
   - `git add .` → `git commit -m "Implement EmoGo backend with Mongo/GridFS export"` → `git push origin main`
   - 確認 GitHub 上能看到最新程式與 README。

2. **部署到 Render**
   - 登入 Render → New Web Service → 選 GitHub repo。
   - 建議使用 `render.yaml` 或直接設定：
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - 設定環境變數：
     - `MONGODB_URI`（同 .env）
     - `DATABASE_NAME=emogo`
     - `EXPORT_TOKEN=emogo-test-token`（或部署版專用字串）
   - 部署完成後測試 `https://<render-app>.onrender.com/data-export?token=...`、`/media/...`，確認線上行為與本地一致。

3. **在 README 更新正式 URL**
   - 把 README「資料匯出頁面 URI」替換成 Render 實際網址。
   - 若 `EXPORT_TOKEN` 需要提供 TA，記得說明使用方式（例如：`https://xxx.onrender.com/data-export?token=xxxx`）。

4. **最終檢查與繳交**
   - 再次確認 MongoDB Atlas 有足夠的假資料供助教下載檢驗。
   - 在 NTU COOL 繳交 GitHub repo 連結＋資料匯出頁面 URI（deadline 12/4 20:00）。
   - 若還要做 Optional 前端整合，可再行討論。

若上述任何一個步驟需要協助（git push、Render 設定、README 修改等），隨時告訴我。

---

**User**

我現在的程式跟後端與之前傳給你的mongodb有什麼關係

---

**Cursor**

你的 MongoDB Atlas（`mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/`、DB `emogo`）就是現在這個 FastAPI 後端的資料庫來源，整個流程關係如下：

1. **FastAPI 啟動時讀取 .env**  
   - `main.py` 透過 `Settings` 把 `MONGODB_URI`、`DATABASE_NAME`、`EXPORT_TOKEN` 載入，並建立 Motor (`AsyncIOMotorClient`) 連線；GridFS bucket (`vlog_files`) 也在這一步初始化。

2. **所有 API 都操作同一個 MongoDB**  
   - `/api/v1/datasets/sentiments`、`/gps`：直接將 JSON 寫入 Atlas 的 `sentiments`、`gps` collection。
   - `/api/v1/vlogs`：把你指定的影片寫進 Atlas 的 GridFS，同時在 `vlogs` collection 留下 metadata（含 `videoStorageId`、`videoUri=/media/{id}`）。
   - 查詢和匯出 (`/export/{dataset}`, `/data-export`) 也都是從這個 Atlas 取資料。

3. **資料匯出/下載頁面反映 Atlas 內容**  
   - 你在 Atlas（或透過 API）新增的假資料，會被匯出頁面 `/data-export?token=...` 讀出，並提供 JSON/CSV 下載連結、影片下載按鈕 `/media/{file_id}`。  
   - 因為影片存在 GridFS，助教下載時不需要知道原始檔案 URI，全由後端提供。

4. **部署後也一樣**  
   - 不論在本地或 Render，`.env` 或 Render 的環境變數都指向同一組 MongoDB URI，因此你已有的假資料會被直接沿用；只要 Atlas 資料不刪，線上匯出頁面就能看到同樣內容。

總結：FastAPI 是 API/匯出/下載入口，MongoDB Atlas 則是唯一的資料儲存（包含 GridFS 影片）。你之前提供的連線資訊就是這套系統的後端資料庫。

---

**User**

在compress看的到嗎

---

**Cursor**

可以的。只要你用 API 成功寫入資料並確保 `MONGODB_URI` 指向 Atlas，MongoDB Compass 連到 `mongodb+srv://b8759758_db_user:emo@cluster0.lwlsy3w.mongodb.net/`，選擇 `emogo` 資料庫，就能看到：

- `sentiments`、`gps` 集合中的 JSON documents
- `vlogs` 集合中的 metadata（含 `videoStorageId`, `videoUri`）
- `vlog_files.files` 與 `vlog_files.chunks` 這兩個 collection（GridFS 會自動建立）裡面存放的影片檔案內容

需要我帶你在 Compass 裡定位這些 collection，也可以再告訴我。

---

**User**

很好我看到了。



1.檢查你有達成以下作業要求嗎

The homework this week is totally independent of the last one—you can use fake data (or data from the outstanding submissions) to populate your backend DB/storage, even if you didn't get Emogo data last time.



Also, the export/download/dashboard page is an HTML page returned by FastAPI rather than a separate frontend.

Finally, the video download/export function is not fully functioning in some of the current submissions. Please check if you have a backend URI (rather than a frontend URI) or allow users to download those videos without requiring knowledge of the URIs.

[Goal] Making an EmoGo backend on a public server using FastAPI+MongoDB.



[Required] List the URI of the data-exporting/downloading page of your backend in the README.md of your GitHub repo. In there, TAs & Tren should be able to see/download all three types of data collected by the EmoGo frontend (i.e., vlogs, sentiments, & GPS coordinates).





2.告訴我下一步要幹嘛

---

**Cursor**

1. 目前程式已符合各項要求：  
   - **可用假資料填充**：我們透過 API 直接寫入假的情緒、GPS 與 vlog 影片（放在 Atlas GridFS），不依賴期中前端，符合「作業與前次獨立」的指示。  
   - **FastAPI 回傳的資料匯出頁面**：`/data-export` 使用 Jinja2 由 FastAPI 生成 HTML，提供筆數、最後時間、JSON/CSV 下載、最新 vlog 下載按鈕，沒有額外前端。  
   - **影片下載走後端 URI**：影片上傳時存進 GridFS，`videoUri` 會指向 `GET /media/{file_id}`，助教只要點按鈕即可下載，不需知道原始 URI。  
   - **FastAPI + MongoDB**：主流程即為 FastAPI + Motor（MongoDB Atlas / GridFS），達成 Goal。  
   - **README 有列資料匯出 URI**：README 的「立即重點」段落已標示 `http://127.0.0.1:8000/data-export`（待部署後替換成 Render URL）並說明 Token；同頁描述 TA 可下載三種資料與影片。  

2. 下一步建議進行部署與繳交流程：  
   - **推送到 GitHub**：`git status` → `git add .` → `git commit ...` → `git push origin main`。  
   - **Render 部署**：用 repo 建 Web Service，設定 `MONGODB_URI/DATABASE_NAME/EXPORT_TOKEN`（與本地相同），完成後測試 `https://<app>.onrender.com/data-export?token=...` 與 `/media/...`。  
   - **README 更新正式網址**：把 `http://127.0.0.1:8000` 替換成 Render 版，並提醒 TA 使用 Token。  
   - **NTU COOL 繳交**：提供 GitHub repo 連結 + Render 的 `/data-export` URI（含 token 說明）。  
需要我協助任一步（例如 git、Render 或 README 更新）再告訴我就行。

