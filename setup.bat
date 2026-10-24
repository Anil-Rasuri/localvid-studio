@echo off
rem One-time setup for Windows: creates .venv, installs PyTorch (CUDA 12.8) and all packages.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
pause
