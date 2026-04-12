@echo off
echo ============================================================
echo   TruthfulRAG v5 - Live API Server
echo ============================================================
echo.

REM Use the Python that has all ML packages installed
set PY=C:\Users\eeshs\AppData\Local\Programs\Python\Python312\python.exe

REM Check Python exists
if not exist "%PY%" (
    echo ERROR: Python not found at %PY%
    echo Edit this file and set PY= to your Python 3.12 path
    pause
    exit /b 1
)

echo Using: %PY%
echo.
echo Starting server at http://localhost:5000
echo Open web_demo\chatbot_live.html in your browser
echo.
echo NOTE: Neo4j must be running for TruthfulRAG v5 to work.
echo       LangChain RAG works without Neo4j.
echo.

"%PY%" web_demo\server.py

pause
