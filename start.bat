@echo off
REM Double-click this file to start the project - no terminal typing needed.
REM It automatically runs from wherever THIS file is located, so the
REM "wrong folder" / "cd" confusion that trips up nearly everyone the
REM first time can't happen with this launcher.

cd /d "%~dp0"

echo Starting Delivery Time Predictor...
echo.

python3 control.py start 2> nul
if errorlevel 1 (
    python control.py start
)

pause
