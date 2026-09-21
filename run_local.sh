#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

sample_path="data/raw/tina_sample_2017-09-06.parquet"
if [[ ! -f "$sample_path" ]]; then
  mkdir -p data/raw
  curl -fL \
    "https://raw.githubusercontent.com/ari-dasci/OD-TINA/main/data/day%3D2017-09-06/data.parquet" \
    -o "$sample_path"
fi

export TINA_DATA_PATH="${TINA_DATA_PATH:-$sample_path}"
export TINA_TIME_COLUMN="${TINA_TIME_COLUMN:-unix}"
export TINA_MAX_SENSOR_FEATURES="${TINA_MAX_SENSOR_FEATURES:-5}"

exec .venv/bin/jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port="${JUPYTER_PORT:-8888}" \
  --ServerApp.root_dir="$project_dir"
