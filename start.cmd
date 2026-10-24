@echo off
setlocal
cd /d "%~dp0"
chcp 65001 > nul
title STEMForge - Local Server
if not exist "runtime\venv\Scripts\python.exe" (
    echo STEMForge is not installed. Starting one-click installer...
    call "%~dp0install.cmd"
    exit /b %errorlevel%
)
set "PATH=%CD%\runtime\ffmpeg;%PATH%"
set "UV_CACHE_DIR=%CD%\runtime\cache"
set "HF_HOME=%CD%\data\hf_cache"
set "TORCH_HOME=%CD%\data\torch_cache"
set "AUDIO_SEPARATOR_MODEL_DIR=%CD%\data\models"
echo ---------------------------------------------------------
echo STEMForge Local Audio Lab
 echo Open in browser: http://127.0.0.1:8765
 echo To stop service close this window or press Ctrl+C
 echo ---------------------------------------------------------
"%~dp0runtime\venv\Scripts\python.exe" "%~dp0app\server.py"
if errorlevel 1 (
  echo.
  echo App exited with an error. If port 8765 is in use, close the other instance.
  pause
)
