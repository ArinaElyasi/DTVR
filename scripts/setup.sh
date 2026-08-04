#!/usr/bin/env bash
# DTV-REA setup - macOS and Linux.
#
# The Windows twin of this file is scripts/setup.ps1 and does exactly the same
# three things. Neither script does anything you cannot do by hand; they exist
# only to save typing. The README spells out every step for both platforms.
#
# Run it from the project folder:
#     bash scripts/setup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "1/3  Creating a virtual environment in .venv"
python3 -m venv .venv

echo "2/3  Installing dependencies"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "3/3  Setting up your .env file"
if [ -f .env ]; then
  echo "     .env already exists - leaving it alone."
else
  cp .env.example .env
  echo "     Created .env. Open it and paste in your Groq API key."
fi

echo
echo "Done. Next, activate the environment:"
echo "    source .venv/bin/activate"
echo
echo "Then try the offline demo (no API key needed):"
echo "    python -m dtv_rea.cli --stub"
