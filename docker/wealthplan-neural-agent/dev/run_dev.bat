@echo off
REM -----------------------------
REM Run DEV Docker Image with custom params file
REM -----------------------------

REM Name of the Docker image
SET IMAGE_NAME=wealthplan-neural-agent-dev

REM Set env variable for S3 bucket
SET S3_BUCKET=wealthplan-neural-agent-dev

REM Set run ID
SET RUN_ID=a04280a1

REM AWS credentials from your environment
SET AWS_KEY=%AWS_ACCESS_KEY_ID%
SET AWS_SECRET=%AWS_SECRET_ACCESS_KEY%
SET AWS_REGION=%AWS_DEFAULT_REGION%

echo Running Docker container %IMAGE_NAME% with params file %PARAMS_FILE%...

docker run --rm ^
    -e AWS_ACCESS_KEY_ID=%AWS_KEY% ^
    -e AWS_SECRET_ACCESS_KEY=%AWS_SECRET% ^
    -e AWS_DEFAULT_REGION=%AWS_REGION% ^
    %IMAGE_NAME% --params-file %PARAMS_FILE%

IF %ERRORLEVEL% NEQ 0 (
    echo Docker run failed!
    exit /b %ERRORLEVEL%
)

echo Docker container finished successfully.
