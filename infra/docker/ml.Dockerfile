FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

ENV UV_PROJECT_ENVIRONMENT=/app/packages/ml/.venv

ENV PATH="/app/packages/ml/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY packages/ml/pyproject.toml packages/ml/uv.lock ./packages/ml/

WORKDIR /app/packages/ml

RUN uv sync --frozen --no-dev

WORKDIR /app

COPY packages/ml ./packages/ml

WORKDIR /app/packages/ml

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ml.main:app", "--host", "0.0.0.0", "--port", "8000"]
