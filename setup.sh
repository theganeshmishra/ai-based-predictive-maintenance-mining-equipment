#!/usr/bin/env bash
# One-step setup on macOS or Linux. From this folder:
#
#     bash setup.sh    create .venv, install packages, register the "Capstone (.venv)" notebook kernel,
#                      then list any raw data files that are still missing
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n== %s\n' "$1"; }

step "1/4  Find Python 3.10 or newer"
BASE=""
PY=.venv/bin/python
if [ -x "$PY" ]; then
  if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "Existing .venv uses an unsupported Python. Move it aside, install Python 3.11 or newer, then rerun setup." >&2
    exit 1
  fi
  BASE="$PY"
else
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      BASE="$candidate"; break
    fi
  done
fi
if [ -z "$BASE" ]; then
  echo "Python 3.10 or newer was not found. Install Python 3.11 from https://www.python.org/downloads/ and run this again." >&2
  exit 1
fi
echo "Using $BASE ($("$BASE" --version))"

step "2/4  Create the project environment (.venv)"
if [ -x "$PY" ]; then echo ".venv already exists"; else "$BASE" -m venv .venv; fi

step "3/4  Install packages (a few minutes the first time)"
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt --quiet

step "4/4  Register the notebook kernel 'Capstone (.venv)'"
KERNEL_ENV=()
if [ "$(uname -s)" = "Darwin" ]; then
  TORCH_LIB=$("$PY" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')
  if [ -f "$TORCH_LIB/libomp.dylib" ]; then
    KERNEL_ENV=(--env DYLD_LIBRARY_PATH "$TORCH_LIB")
  fi
fi
"$PY" -m ipykernel install --user --name capstone-week4 --display-name "Capstone (.venv)" "${KERNEL_ENV[@]}"
"$PY" -c "import cv2, numpy, pandas, sklearn; print('Packages OK: OpenCV', cv2.__version__, '| scikit-learn', sklearn.__version__)"

step "Check which raw data files are present"
"$PY" Notebook/pdm.py data-check

step "Done"
cat <<'EOF'
Next:
  1. Open this folder in VS Code (File > Open Folder). Install the Python and Jupyter extensions if VS Code asks.
  2. Open Notebook/Mining_Equipment_Condition_Monitoring.ipynb.
     Check the top right says 'Capstone (.venv)' (click it to choose it if not), then click Run All.

The results section works without the raw data. Sections 2, 3 and 5 need the files in Raw_Data/;
download them by hand from the sources in Dataset_Manifest/manifest.md.
To rerun a model: .venv/bin/python Notebook/pdm.py <machine> sensor   (see README.md)
EOF
