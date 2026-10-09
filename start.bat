@echo off
REM Double-click to install, set up (first time only) and start the paper trading bot.
cd /d "%~dp0"
python install.py || (echo Setup failed. If Python is missing, install it from https://www.python.org/downloads/ and tick "Add to PATH". & pause & exit /b 1)
if not exist .env python main.py setup
if exist .env python main.py run
pause
