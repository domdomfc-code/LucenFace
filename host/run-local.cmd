@echo off
REM Chay LucenFace local ma khong can doi ExecutionPolicy (goi PowerShell Bypass).
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run-local.ps1"
if errorlevel 1 pause
