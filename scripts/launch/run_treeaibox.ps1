# Launch TreeAIBox Standalone with CUDA support
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Cyan
& "$repoRoot\.venv\Scripts\python.exe" "$repoRoot\TreeAIBox.py"
