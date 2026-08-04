# DTV-REA offline demo and test run - Windows (PowerShell).
#
# The macOS/Linux twin is scripts/demo.sh and runs the identical three
# commands. Nothing here needs an API key or an internet connection.
#
#     powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
#
# Activate your virtual environment first, or this will use whichever Python
# is on your PATH.

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== Running the test suite =="
python -m pytest

Write-Host ""
Write-Host "== Running the offline interview demo =="
python -m dtv_rea.cli --stub --session demo

Write-Host ""
Write-Host "== Scoring every persona =="
python -m eval.run_eval --stub

Write-Host ""
Write-Host "Done. Look at runs/demo/requirements.md and eval/report.md"
