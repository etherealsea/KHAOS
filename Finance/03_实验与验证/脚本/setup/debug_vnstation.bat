@echo off
echo ========================================================
echo VeighNa Station Launch Diagnostic Tool
echo ========================================================
echo.
echo Attempting to launch from: D:\veighna_studio
echo.

if not exist "D:\veighna_studio\python.exe" (
    echo [ERROR] Could not find D:\veighna_studio\python.exe
    echo Please confirm your installation path.
    pause
    exit /b
)

"D:\veighna_studio\python.exe" -m vnstation

echo.
echo ========================================================
echo CRASH REPORT
echo ========================================================
echo If the window closed immediately before, the error should be visible above.
echo Please copy the error message (lines starting with Traceback or Error).
echo.
pause
