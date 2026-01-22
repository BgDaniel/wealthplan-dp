@echo off
REM -----------------------------
REM Build DEV Docker Image
REM -----------------------------

REM Name of the Docker image
SET IMAGE_NAME=wealthplan-dp-dev

REM Project root folder (assumes this batch file is in docker/dev/)
SET PROJECT_ROOT=C:\Projects\wealthplan-dp

REM Path to Dockerfile relative to the project root
SET DOCKERFILE=%PROJECT_ROOT%\docker\dev\Dockerfile

echo Building Docker image %IMAGE_NAME% using %DOCKERFILE% with context %PROJECT_ROOT%

REM Build the Docker image with context set to project root
docker build -f %DOCKERFILE% -t %IMAGE_NAME%:latest %PROJECT_ROOT%

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to build Docker image!
    exit /b %ERRORLEVEL%
)

echo Docker image %IMAGE_NAME%:latest built successfully.
