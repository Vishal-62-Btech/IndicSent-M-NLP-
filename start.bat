@echo off
title IndicSent-M Launcher
cd /d "%~dp0"
python run.py
if errorlevel 1 (
    echo.
    echo An error occurred while running IndicSent-M.
    pause
)
