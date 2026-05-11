FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

ENV UV_PROJECT_ENVIRONMENT=/app/.venv

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

COPY packages/mock_store_api/pyproject.toml ./packages/mock_store_api/pyproject.toml

RUN uv sync --frozen --package mock-store-api --no-dev

COPY packages/mock_store_api ./packages/mock_store_api

WORKDIR /app/packages/mock_store_api

EXPOSE 8002

CMD ["uv", "run", "uvicorn", "mock_store_api.main:app", "--host", "localhost", "--port", "8002"]