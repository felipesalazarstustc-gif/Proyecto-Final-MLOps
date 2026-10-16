@echo off
echo Copiando variables...
if not exist .env copy .env.example .env
echo Levantando proyecto...
docker compose up -d --build
echo.
echo MLflow: http://localhost:5000
echo API: http://localhost:8000/health
echo.
docker compose ps
pause
