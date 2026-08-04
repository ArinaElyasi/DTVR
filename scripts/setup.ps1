# DTV-REA setup - Windows (PowerShell).
#
# The macOS/Linux twin of this file is scripts/setup.sh and does exactly the
# same three things. Neither script does anything you cannot do by hand; they
# exist only to save typing. The README spells out every step for both
# platforms.
#
# Run it from the project folder:
#     powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "1/3  Creating a virtual environment in .venv"
py -3 -m venv .venv

Write-Host "2/3  Installing dependencies"
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "3/3  Setting up your .env file"
if (Test-Path .env) {
  Write-Host "     .env already exists - leaving it alone."
} else {
  Copy-Item .env.example .env
  Write-Host "     Created .env. Open it and paste in your Groq API key."
}

Write-Host ""
Write-Host "Done. Next, activate the environment:"
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then try the offline demo (no API key needed):"
Write-Host "    python -m dtv_rea.cli --stub"
