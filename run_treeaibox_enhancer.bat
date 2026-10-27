@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo Starting TreeAIBox Enhancer / Forestry AI Suite...
call .venv\Scripts\activate.bat
python scripts\visualization\treeAIbox_Enhancer.py %*
pause
