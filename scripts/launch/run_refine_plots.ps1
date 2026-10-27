# Run Automated Tree Segmentation Refiner
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

Write-Host "Running Automated Tree Segmentation Refiner (.venv)..." -ForegroundColor Cyan
& "$repoRoot\.venv\Scripts\python.exe" "$repoRoot\scripts\pointcloud_processing\refine_segmented_plots.py"
