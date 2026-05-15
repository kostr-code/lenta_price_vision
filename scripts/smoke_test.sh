#!/usr/bin/env bash

set -e

echo "Checking backend..."

curl -f http://localhost:8001/health

echo "Checking ML..."

curl -f http://localhost:8000/health

echo "Smoke test passed"
