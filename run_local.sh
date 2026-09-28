#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
fi
.venv/bin/python -m pip install -r requirements.txt

sample_path="data/raw/tina_sample_2017-09-08_to_10.parquet"
if [[ ! -f "$sample_path" ]]; then
  .venv/bin/python src/build_week2_subset.py
fi

exec .venv/bin/jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port="${JUPYTER_PORT:-8888}" \
  --ServerApp.root_dir="$project_dir"
