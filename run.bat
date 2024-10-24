@echo off
cd 08\ligand_search

:: Start Docker Compose
docker-compose up -d

:: Poll to check if the application is up
echo Waiting for the server to be ready...

:CHECK_SERVER
timeout /t 1 >nul
curl -s http://localhost:8000 >nul
if %errorlevel% neq 0 (
    echo Server not ready yet...
    goto CHECK_SERVER
)

:: When the server is ready, open the browser
echo Server is up, opening browser...
start http://localhost:8000/

:: Keep Docker Compose logs in the foreground
docker-compose logs -f
