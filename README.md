# lenta_price_vision

Retail price-tag recognition stack:

- `backend` (FastAPI gateway)
- `ml` (FastAPI inference service, run outside Docker by default)
- `frontend` (React + Vite web UI)

## Services and URLs

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8001`
- ML: `http://localhost:8000` (local service used by backend through `ML_URL`)
- Postgres: `localhost:5433`

## Prerequisites

- Docker + Docker Compose
- `uv` (for Python local runs)
- Node.js `20+` and npm (for frontend local run)

## Quick Launch (Docker, recommended)

1. Copy env file:

```powershell
Copy-Item .env.example .env
```

2. Start development stack:

```powershell
docker compose `
  -f docker-compose.yml `
  -f infra/compose/docker-compose.dev.yml `
  up --build
```

3. Check health:

```powershell
curl http://localhost:8001/health
```

4. Open apps:

- Frontend UI: `http://localhost:5173`
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

1. Install Python dependencies:

```powershell
uv sync
cd packages/ml
uv sync
cd ../..
```

2. Start backend services in separate terminals:

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

3. Start frontend:

```powershell
cd packages/frontend
npm install
npm run dev
```

4. Open frontend at `http://localhost:5173`.

## Frontend to backend integration

Frontend uses backend endpoints:

- `POST /api/v1/predict/video`
- `POST /api/v1/predict/image`
- `GET /api/v1/schema`
- `GET /api/v1/datasets`
- `GET /api/v1/download/{run_id}/{filename}`

Backend response includes:

- `backend_download`
- `backend_debug_download`

Frontend uses these links to download CSV/debug files.

Frontend file upload supports both video and image files. Video uploads are routed to
`/api/v1/predict/video`, image uploads to `/api/v1/predict/image`.
