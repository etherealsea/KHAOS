@echo off
setlocal enabledelayedexpansion

REM Set VeighNa Studio Installation Directory
set "VNSTUDIO_DIR=D:\veighna_studio"

REM Set Path to Python Executable
set "PYTHON_EXE=%VNSTUDIO_DIR%\python.exe"

REM Configure PATH environment variable to include VeighNa's DLLs and Scripts
set "PATH=%VNSTUDIO_DIR%;%VNSTUDIO_DIR%\Scripts;%VNSTUDIO_DIR%\Library\bin;%PATH%"

REM Verify Python Executable exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] python.exe not found at: %PYTHON_EXE%
    echo Please confirm that VeighNa Studio is installed in D:\veighna_studio
    pause
    exit /b 1
)

echo Starting VeighNa Station with Custom Modules...
"%PYTHON_EXE%" "%~dp0run_custom.py"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] VeighNa exited with error code %errorlevel%
    pause
)
endlocal
