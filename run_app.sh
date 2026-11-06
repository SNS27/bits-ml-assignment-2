#!/usr/bin/env bash
# Everyday launch: no activation needed, safe for paths with spaces.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$DIR/.venv/Scripts/python.exe" ]; then VPY="$DIR/.venv/Scripts/python.exe"; else VPY="$DIR/.venv/bin/python"; fi
exec "$VPY" -m streamlit run "$DIR/app.py"
