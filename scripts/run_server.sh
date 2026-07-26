#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the existing LeRobot conda env (has pyserial + torch already).
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lerobot
fi

pip install -q -r requirements.txt
export PYTHONPATH="."
exec python -m robot.server
