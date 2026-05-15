#!/usr/bin/env bash

set -e

echo "Checking backend..."

curl -f http://localhost:8001/health

echo "Checking frontend..."

curl -f http://localhost:5173

echo "Smoke test passed"
