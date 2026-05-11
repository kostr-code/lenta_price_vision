from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Price Tag Audit ML Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml"}


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ml.main:app", host=host, port=port)


if __name__ == "__main__":
    run()
