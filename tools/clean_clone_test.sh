#!/usr/bin/env bash
# Acceptance test: fresh clone + fresh venv + install requirements + verify + workflow tests.
set -e
T="$(mktemp -d)"; echo "temp folder: $T"
git clone https://github.com/SNS27/bits-ml-assignment-2.git "$T/repo"
cd "$T/repo"
PY=$(command -v python3 || command -v python); "$PY" -m venv .venv
if [ -x ".venv/Scripts/python.exe" ]; then VPY=".venv/Scripts/python.exe"; else VPY=".venv/bin/python"; fi
"$VPY" -m pip install -q -r requirements.txt
ls model test_data.csv app.py requirements.txt README.md
"$VPY" tools/verify_models.py
"$VPY" tools/test_app_workflow.py
echo "CLEAN CLONE ACCEPTANCE: PASSED"
