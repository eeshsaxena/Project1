@echo off
title TruthfulRAG v5 — Launcher
color 0B
echo.
echo  ████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗███████╗██╗   ██╗██╗
echo  ╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔════╝██║   ██║██║
echo     ██║   ██████╔╝██║   ██║   ██║   ███████║█████╗  ██║   ██║██║
echo     ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║██╔══╝  ██║   ██║██║
echo     ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║██║     ╚██████╔╝███████╗
echo     ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚══════╝
echo.
echo  TruthfulRAG v5 ^| Eesh Saxena 230101032 ^| IIIT Manipur
echo  ─────────────────────────────────────────────────────────
echo.

set PY=C:\Users\eeshs\AppData\Local\Programs\Python\Python312\python.exe

:: Check Python exists
if not exist "%PY%" (
    echo  ERROR: Python not found at %PY%
    pause & exit /b 1
)

:: Install flask-cors if not present (needed by launcher.py)
"%PY%" -c "import flask_cors" 2>nul || (
    echo  Installing flask-cors...
    "%PY%" -m pip install flask-cors -q
)

echo  [1/2] Starting Launcher Backend on port 5001...
start "TruthfulRAG-Launcher" /MIN "%PY%" "D:\Project-1\launcher.py"

:: Wait for launcher backend to come up
echo  Waiting for launcher to start...
:wait_launcher
timeout /t 1 /nobreak >nul
powershell -Command "try { Invoke-WebRequest http://127.0.0.1:5001/health -TimeoutSec 1 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto wait_launcher

echo  [2/2] Opening Launcher UI in browser...
start "" "D:\Project-1\launcher.html"

echo.
echo  ─────────────────────────────────────────────────────────
echo   Launcher is running!
echo   Browser window should open automatically.
echo   Click "Launch All Services" in the browser.
echo  ─────────────────────────────────────────────────────────
echo.
echo  This window can be minimized. Do NOT close it.
echo.
pause
