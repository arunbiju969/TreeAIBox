# Launch Modern Tree Visualizer GUI
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Cyan
& "$repoRoot\.venv\Scripts\python.exe" "$repoRoot\scripts\visualization\tree_visu.py"
