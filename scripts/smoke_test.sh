#!/usr/bin/env bash

set -e

echo "Checking backend..."

curl -f http://localhost:8001/health

echo "Checking ML..."

curl -f http://localhost:8000/health

echo "Checking mock store API..."

curl -f http://localhost:8002/health

echo "Smoke test passed"