# EmoGo Backend (FastAPI + MongoDB)

這是一個部署在 Render 的 FastAPI 服務，串接 MongoDB Atlas 以提供 EmoGo 前端（`emogo-frontend-BurningBright7214/`）上傳的 vlog、情緒問卷與 GPS 資料。專案滿足課程要求：提供線上資料下載頁面，支援 JSON/CSV 匯出，並以 FastAPI 端提供影片檔案下載連結（GridFS 儲存及 `/media/{file_id}` 下載）。

## 立即重點

- **資料匯出頁面 URI**
  - 本地開發：`http://127.0.0.1:8000/data-export?token=emogo-test-token`
  - Render 正式站：[`https://emogo-backend-burningbright7214.onrender.com/data-export?token=emogo-test-token`](https://emogo-backend-burningbright7214.onrender.com/data-export?token=emogo-test-token)
- **三種資料匯出 API**：`/export/vlogs`、`/export/sentiments`、`/export/gps`（支援 `?format=json|csv`）
- **影片上傳/下載**：`POST /api/v1/vlogs`（multipart 上傳影片 + metadata），`GET /media/{file_id}` 下載 GridFS 檔案
- **健康檢查**：`/api/v1/health`
- **公共首頁**：[`https://emogo-backend-burningbright7214.onrender.com`](https://emogo-backend-burningbright7214.onrender.com) 會回傳 `{ "service": "EmoGo Backend", ... }`

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
- `http://127.0.0.1:8000/data-export?token=emogo-test-token`：資料下載頁面（若更換 token 記得更新 query）

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
4. 部署完成後，瀏覽 `https://emogo-backend-burningbright7214.onrender.com/data-export?token=emogo-test-token`，確認可以下載三大資料集與影片。

## 線上驗證 (Render)

1. **根路由/健檢**
   - `https://emogo-backend-burningbright7214.onrender.com/`：應回傳服務資訊與文件入口。
   - `https://emogo-backend-burningbright7214.onrender.com/api/v1/health`：應顯示 `status: ok`、`database: reachable`。
2. **下載測試**
   - `https://emogo-backend-burningbright7214.onrender.com/data-export?token=emogo-test-token`：檢視三類資料筆數、最新 vlog，以及 JSON/CSV 下載按鈕。
   - `https://emogo-backend-burningbright7214.onrender.com/export/vlogs?format=json&token=emogo-test-token`（或 `sentiments` / `gps`）應直接回傳 JSON。
   - `https://emogo-backend-burningbright7214.onrender.com/media/<videoStorageId>?token=emogo-test-token`：應下載 MP4 檔案（可從資料匯出頁的「最新影片」表格取得 `videoStorageId`）。
3. **Swagger 文件**
   - `https://emogo-backend-burningbright7214.onrender.com/docs`：可在瀏覽器內直接呼叫 API 測試。

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

## Cursor 對話紀錄

- `cursor_deploying_emogo_backend_with_fas.md`：此檔紀錄在 Cursor 內的完整對話與操作歷程，助教若需驗證過程可直接檢視。