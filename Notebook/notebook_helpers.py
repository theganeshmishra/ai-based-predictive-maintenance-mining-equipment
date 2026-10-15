"""Shared set-up for the machine notebooks in this folder.

The experiment code lives in Models/ (one .py file per equipment). The notebooks only call
those files and show their results, using the small helpers below, so each
notebook stays short.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import runpy
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from IPython.display import display

_here = Path.cwd().resolve()
ROOT = next(p for p in (_here, *_here.parents) if (p / "Models").is_dir() and (p / "Results").is_dir())
RAW, PROCESSED, RESULTS = ROOT / "Raw_Data", ROOT / "Processed_Data", ROOT / "Results"
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")


def _package(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def setup(data: list[tuple[str, str, str, int]]) -> dict[str, bool]:
    """Show the Python environment and which datasets are present.

    `data` lists (label, folder or file, file pattern, expected file count).
    When everything is local, network access is blocked so the run is provably offline.
    Returns {label: True if present}.
    """
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    in_venv = (ROOT / ".venv").resolve() in Path(sys.executable).resolve().parents
    environment = {"git commit": commit, "python": platform.python_version(),
                   "kernel": "project .venv" if in_venv else "NOT the project .venv"}
    environment.update({name: _package(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "opencv-python", "matplotlib")})
    display(pd.Series(environment, name="environment").to_frame())
    if not in_venv:
        print("WARNING: this notebook is not using the project's .venv, so packages such as cv2 may be missing.\n"
              "In VS Code: click the kernel name at the top right -> Select Another Kernel -> Jupyter Kernel -> Capstone (.venv).")

    rows = []
    for label, location, pattern, expected in data:
        path = ROOT / location
        files = [f for f in path.rglob(pattern) if f.is_file()] if path.is_dir() else ([path] if path.is_file() else [])
        rows.append({"data": label, "location": location, "files": f"{len(files)}/{expected}",
                     "MB": round(sum(f.stat().st_size for f in files) / 1e6, 1), "local": len(files) >= expected})
    table = pd.DataFrame(rows).set_index("data")
    display(table)
    local = table["local"].to_dict()
    if all(local.values()):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ[name] = "http://127.0.0.1:9"  # any network call now fails loudly
        os.environ["NO_PROXY"] = ""
        print("All data is local: running fully offline.")
    else:
        print("Some data is missing, so those steps are skipped and the saved results are shown instead.\n"
              "To download the data, run  .\\setup.ps1 -Data  (Windows) or  bash setup.sh --data  (Mac/Linux) in the project folder.")
    return local


def use_local_pump_zips() -> None:
    """Make the Pump event-camera script read the two local XJTU zips instead of downloading byte ranges."""
    import remotezip

    def local_zip(url, *args, **kwargs):
        part = "part1" if "part1" in url else "part2" if "part2" in url else None
        if part is None:
            raise ValueError(f"Unexpected archive URL: {url}")
        return zipfile.ZipFile(RAW / "xjtu_dv_pump" / f"XJTU-DV-Pump_{part}.zip")

    remotezip.RemoteZip = local_zip


def run_script(relative_path: str, *args: str, tail: int = 3) -> None:
    """Run a project script exactly as `python <script>` would and print its last output lines."""
    path, buffer, saved = ROOT / relative_path, io.StringIO(), sys.argv
    sys.argv = [path.name, *args]
    try:
        with contextlib.redirect_stdout(buffer):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved
    text = buffer.getvalue().replace(str(ROOT) + os.sep, "").replace(str(ROOT), ".")
    print(f"[{' '.join((relative_path, *args))}]")
    for line in [line for line in text.splitlines() if line.strip()][-tail:]:
        print("  " + line)


def load(name: str):
    """Read a saved result file from Results/ (None if it does not exist yet)."""
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def metrics(block: dict) -> dict:
    return {k: round(block[k], 3) for k in METRICS if isinstance(block.get(k), (int, float))}


def show(rows: dict) -> None:
    """Display a table of scores: {row name: result block}."""
    display(pd.DataFrame({name: metrics(block) for name, block in rows.items()}).T)


def check(before, after, paths, name: str) -> None:
    """Say whether this run reproduced the previously saved scores."""
    def pick(doc, path):
        for key in path:
            doc = doc[key]
        return metrics(doc)
    same = before is not None and all(pick(before, p) == pick(after, p) for p in paths)
    print(f"{name}: {'matches' if same else 'DIFFERS from'} the saved result")


def show_baselines(result: dict) -> None:
    """Display the Week 5 baseline comparison: one row per branch and model type."""
    show({f"{branch} | {name}": score
          for branch, block in result["branches"].items()
          for name, score in block["models"].items()})


def show_week6(machine: str) -> None:
    """Display the Week 6 results: 4-class CSV sensor models and the CNN on images."""
    csv_result, cnn_result = load(f"{machine}_csv_4class.json"), load(f"{machine}_cnn_4class.json")
    rows = {}
    if csv_result:
        rows.update({f"CSV sensor | {name}": score for name, score in csv_result["models"].items()})
    if cnn_result:
        rows["Camera images | CNN (per recording)"] = cnn_result["recording_level"]
        rows["Camera images | CNN (per image)"] = cnn_result["image_level"]
    show(rows)
