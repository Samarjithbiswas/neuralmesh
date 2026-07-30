#!/usr/bin/env bash
# Run exactly what CI runs, in the same order. Use this before every push.
#
# This file exists because a commit was pushed whose only failure was
# "ruff format --check" on a file that had been linted but never formatted.
# Running the real gate locally is cheaper than finding out from CI.
set -e
python -m ruff check neuralmesh tests examples
python -m ruff format --check neuralmesh tests examples
python -m pytest -q
python -m neuralmesh.cli verify
echo "all CI gates pass locally"
