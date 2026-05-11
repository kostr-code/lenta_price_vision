FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY packages/mock_store_api ./packages/mock_store_api
COPY packages/mock_store_api ./packages/mock_store_api

WORKDIR /app/packages/mock_store_api

RUN uv sync

EXPOSE 8002

CMD ["uv", "run", "uvicorn", "mock_store_api.main:app", "--host", "localhost", "--port", "8002"]