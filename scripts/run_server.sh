#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
  # The existing LeRobot conda env already has pyserial and torch. Nice if it is
  # there, not worth insisting on if it isn't.
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lerobot 2>/dev/null || true
fi

if [ "${CORGI_SKIP_INSTALL:-0}" != "1" ]; then
  pip install -q -r requirements.txt
fi

export PYTHONPATH="."
exec python -m robot.server
