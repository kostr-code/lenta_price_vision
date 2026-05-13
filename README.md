# lenta_price_vision

Retail price-tag recognition stack (backend + ML + mock store API).

## What runs today

- `backend` (FastAPI): `http://localhost:8001`
- `ml` (FastAPI): `http://localhost:8000`
- `mock-store-api` (FastAPI): `http://localhost:8002`
- `postgres`: `localhost:5433`
- `frontend` (PySide6 desktop app): local desktop window

## Prerequisites

- Docker + Docker Compose
- `uv` (for local non-Docker run)
- Python `3.11.x` (for local non-Docker run)

## Quick Launch (Docker, recommended)

1. Copy env file:

```powershell
Copy-Item .env.example .env
```

2. Start dev stack:

```powershell
docker compose `
  -f docker-compose.yml `
  -f infra/compose/docker-compose.dev.yml `
  up --build
```

3. Check health:

```powershell
curl http://localhost:8001/health
curl http://localhost:8000/health
curl http://localhost:8002/health
```

4. Open API docs:

- Backend docs: `http://localhost:8001/docs`
- ML docs: `http://localhost:8000/docs`

## Launch via Make (if `make` is installed)

```bash
make dev
```

Smoke test:

```bash
make smoke
```

Stop:

```bash
make down
```

## Local Launch (without Docker)

1. Install dependencies:

```powershell
uv sync
cd packages/ml
uv sync
cd ../..
```

2. Start services in separate terminals:

Terminal 1 (ML):

```powershell
cd packages/ml
uv run ml-service
```

Terminal 2 (Backend):

```powershell
cd packages/backend
$env:ML_URL="http://localhost:8000"
uv run backend
```

Terminal 3 (Mock API):

```powershell
cd packages/mock_store_api
uv run mock-store-api
```

3. Start frontend in a 4th terminal:

```powershell
cd packages/frontend
uv run frontend
```

4. Verify:

```powershell
curl http://localhost:8001/health
```

In the frontend window:

1. Set backend URL (`http://localhost:8001` by default).
2. Select a video file.
3. Click `Run Recognition`.
4. Save CSV/debug artifacts using `Save CSV` and `Save Debug JSON`.

## Main backend endpoints for frontend integration

- `POST /api/v1/predict/video` (upload video and run recognition)
- `POST /api/v1/predict/path` (run by local file path on ML host)
- `GET /api/v1/schema`
- `GET /api/v1/datasets`
- `GET /api/v1/download/{run_id}/{filename}`

The backend proxies ML responses and adds:

- `backend_download`
- `backend_debug_download`

so a UI can download artifacts through backend directly.
