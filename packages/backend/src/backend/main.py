from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Price Tag Audit Backend")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "backend"}


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("backend.main:app", host=host, port=port)


if __name__ == "__main__":
    run()
