@echo off
setlocal
cd /d "%~dp0..\.."
echo Starting Forestry AI Suite (TreeAIBox / TreeLearn / ForestFormer)...
call .venv\Scripts\activate.bat
python scripts\visualization\treeAIbox_Enhancer.py
pause
