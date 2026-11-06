#!/usr/bin/env bash
# One-time setup: creates .venv and installs requirements. Works in Git Bash, Linux, macOS.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY=$(command -v python3 || command -v python)
"$PY" -m venv .venv
if [ -x ".venv/Scripts/python.exe" ]; then VPY=".venv/Scripts/python.exe"; else VPY=".venv/bin/python"; fi
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt
echo "Setup finished. Launch with: bash run_app.sh"
