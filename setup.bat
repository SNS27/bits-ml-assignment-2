@echo off
cd /d "%~dp0"
python -m venv .venv || py -3 -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo Setup finished. Launch with run_app.bat
