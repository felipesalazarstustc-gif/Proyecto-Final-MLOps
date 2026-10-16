#!/bin/sh
cp -n .env.example .env 2>/dev/null || true
docker compose up -d --build
docker compose ps
echo "MLflow: http://localhost:5000"
echo "API: http://localhost:8000/health"
