#!/usr/bin/env bash
# DTV-REA offline demo and test run - macOS and Linux.
#
# The Windows twin is scripts/demo.ps1 and runs the identical three commands.
# Nothing here needs an API key or an internet connection.
#
#     bash scripts/demo.sh
#
# Activate your virtual environment first, or this will use whichever Python
# is on your PATH.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Running the test suite =="
python -m pytest

echo
echo "== Running the offline interview demo =="
python -m dtv_rea.cli --stub --session demo

echo
echo "== Scoring every persona =="
python -m eval.run_eval --stub

echo
echo "Done. Look at runs/demo/requirements.md and eval/report.md"
