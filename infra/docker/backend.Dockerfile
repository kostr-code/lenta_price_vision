FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY packages/backend ./packages/backend

RUN uv sync --frozen --package backend

WORKDIR /app/packages/backend

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "localhost", "--port", "8001"]