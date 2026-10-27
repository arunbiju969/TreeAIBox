# Launch Forestry AI Suite GUI
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

Write-Host "Starting Forestry AI Suite (TreeAIBox / TreeLearn / ForestFormer) in virtual environment (.venv)..." -ForegroundColor Green
& "$repoRoot\.venv\Scripts\python.exe" "$repoRoot\scripts\visualization\treeAIbox_Enhancer.py"
