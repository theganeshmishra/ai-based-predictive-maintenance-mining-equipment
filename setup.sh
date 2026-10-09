#!/usr/bin/env bash
# One-step setup on macOS or Linux. From this folder:
#
#     bash setup.sh          create .venv, install packages, register the "Capstone (.venv)" notebook kernel
#     bash setup.sh --data   the same, then download the data for the 3 machines (about 1.1 GB)
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n== %s\n' "$1"; }

step "1/4  Find Python 3.10 or newer"
BASE=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    BASE="$candidate"; break
  fi
done
if [ -z "$BASE" ]; then
  echo "Python 3.10 or newer was not found. Install Python 3.11 from https://www.python.org/downloads/ and run this again." >&2
  exit 1
fi
echo "Using $BASE ($("$BASE" --version))"

step "2/4  Create the project environment (.venv)"
PY=.venv/bin/python
if [ -x "$PY" ]; then echo ".venv already exists"; else "$BASE" -m venv .venv; fi

step "3/4  Install packages (a few minutes the first time)"
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt --quiet

step "4/4  Register the notebook kernel 'Capstone (.venv)'"
"$PY" -m ipykernel install --user --name capstone-week4 --display-name "Capstone (.venv)"
"$PY" -c "import cv2, numpy, pandas, sklearn; print('Packages OK: OpenCV', cv2.__version__, '| scikit-learn', sklearn.__version__)"

if [ "${1:-}" = "--data" ]; then
  step "Download data (Pump vibration, Haul Truck vibration, Belt Conveyor CORA)"
  "$PY" Models/pump.py intake
  "$PY" Models/haul_truck.py intake --sensor-only
  "$PY" Models/belt_conveyor.py intake
fi

step "Done"
cat <<'EOF'
Next:
  1. Open this folder in VS Code (File > Open Folder). Install the Python and Jupyter extensions if VS Code asks.
  2. Open Notebook/Pump.ipynb, Belt_Conveyor.ipynb or Haul_Truck.ipynb.
     Check the top right says 'Capstone (.venv)' (click it to choose it if not), then click Run All.
  3. Or press Cmd+Shift+B (Ctrl+Shift+B on Linux) to run all 3 machines.

Without the data the notebooks still show the saved results.
To download the data and rerun the models: bash setup.sh --data
EOF
