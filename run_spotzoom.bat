@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=E:\Anaconda\envs\cp11_torch22\python.exe"
set "SCRIPT_PATH=%SCRIPT_DIR%SpotZoom.py"

if not exist "%PYTHON_EXE%" (
    echo [SpotZoom] Python runtime not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%SCRIPT_PATH%" (
    echo [SpotZoom] Script not found: %SCRIPT_PATH%
    exit /b 1
)

call "%PYTHON_EXE%" "%SCRIPT_PATH%" %*
exit /b %ERRORLEVEL%
