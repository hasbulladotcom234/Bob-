#!/bin/bash
# Double-click (Mac) or run `bash start.command` to install, set up (first
# time only) and start the paper trading bot.
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "Python not found. Install it from https://www.python.org/downloads/"; exit 1; fi
"$PY" -m pip install -q -r requirements.txt
[ -f .env ] || "$PY" main.py setup
[ -f .env ] && "$PY" main.py run
