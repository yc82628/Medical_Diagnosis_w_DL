@echo off
REM Start Retinal Screening for the hospital network (HTTPS, staff networks only).
REM Run deploy\setup.py once first.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Follow SETUP.md to create it, then run deploy\setup.py.
  exit /b 1
)
".venv\Scripts\python.exe" deploy\start_server.py
