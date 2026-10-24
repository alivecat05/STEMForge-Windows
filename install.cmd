@echo off
setlocal
cd /d "%~dp0"
chcp 65001 > nul
title STEMForge - One Click Installer
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed. Please read the error above.
  echo [ERROR] Log: runtime\install.log
  pause
  exit /b 1
)
echo.
echo [OK] Installation completed. Use start.cmd to open the app.
choice /m "Open STEMForge now"
if errorlevel 2 exit /b 0
call "%~dp0start.cmd"
