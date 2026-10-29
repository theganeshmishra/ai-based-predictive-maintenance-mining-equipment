#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$project_root/.venv/bin/python"
model="$1"
shift

# The project environment includes libomp with PyTorch. Make it visible to
# XGBoost and LightGBM when VS Code starts a model task on macOS.
torch_lib="$("$python_bin" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
if [ -f "$torch_lib/libomp.dylib" ]; then
  export DYLD_LIBRARY_PATH="$torch_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

exec "$python_bin" "$model" "$@"
