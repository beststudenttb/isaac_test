#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

./IsaacLab/isaaclab.sh -p train/teacher.py

./IsaacLab/isaaclab.sh -p val/teacher.py
