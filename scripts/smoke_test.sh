#!/usr/bin/env bash

set -e

echo "Checking backend..."

curl -f http://localhost:8001/health

echo "Smoke test passed"
