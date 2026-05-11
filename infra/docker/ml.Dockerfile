FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY packages/ml/pyproject.toml packages/ml/uv.lock ./packages/ml/
COPY packages/ml ./packages/ml
COPY models ./models

WORKDIR /app/packages/ml

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ml.main:app", "--host", "localhost", "--port", "8000"]