@echo off
setlocal
cd /d "%~dp0..\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Starting TreeAIBox Standalone with CUDA acceleration...
call .venv\Scripts\activate.bat
python TreeAIBox.py
pause
