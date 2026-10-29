"""Shared set-up for the single combined equipment notebook.

The experiment code lives in Models/ (one .py file per equipment). The combined
notebook uses these helpers to run a selected model and display saved results.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import runpy
import subprocess
import sys
import warnings
import zipfile
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

# Harmless scikit-learn/joblib notice that can repeat thousands of times inside a notebook kernel
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")

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


class _RecentOutput:
    """Keep only the last non-empty output lines from a model run."""

    def __init__(self, limit: int):
        self.lines: deque[str] = deque(maxlen=limit)
        self.pending = ""

    def write(self, chunk: str) -> int:
        text = self.pending + chunk
        self.pending = ""
        for line in text.splitlines(keepends=True):
            if line.endswith(("\n", "\r")):
                if line.strip():
                    self.lines.append(line.rstrip("\r\n"))
            else:
                self.pending = line
        return len(chunk)

    def flush(self) -> None:
        pass

    def recent_lines(self) -> list[str]:
        if self.pending.strip():
            self.lines.append(self.pending)
            self.pending = ""
        return list(self.lines)


def run_script(relative_path: str, *args: str, tail: int = 3) -> None:
    """Run a project script and show only its last lines without buffering all output."""
    if tail < 1:
        raise ValueError("tail must be positive")
    path, buffer, saved = ROOT / relative_path, _RecentOutput(tail), sys.argv
    sys.argv = [path.name, *args]
    try:
        with contextlib.redirect_stdout(buffer):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved
    print(f"[{' '.join((relative_path, *args))}]")
    for line in buffer.recent_lines():
        print("  " + line.replace(str(ROOT) + os.sep, "").replace(str(ROOT), "."))


def select_equipment(active: str | None, equipment: str):
    """Return model and score-check functions for one notebook section.

    The returned functions call the module-level helpers directly, so they do
    not accidentally wrap a previous equipment section's functions.
    """
    def run_selected(relative_path: str, *args: str, **kwargs):
        if active == equipment:
            return run_script(relative_path, *args, **kwargs)

    def check_selected(before, after, paths, name: str):
        if active == equipment:
            return check(before, after, paths, name)
        print(f"{name}: showing saved results (not rerun).")

    return run_selected, check_selected


def load(name: str):
    """Read a saved result file from Results/<machine>/ (None if it does not exist yet)."""
    path = next(RESULTS.glob(f"*/{name}"), RESULTS / name)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def metrics(block: dict) -> dict:
    return {k: round(block[k], 3) for k in METRICS if isinstance(block.get(k), (int, float))}


def show(rows: dict) -> None:
    """Display a table of scores: {row name: result block}."""
    display(pd.DataFrame({name: metrics(block) for name, block in rows.items()}).T)


def show_overview() -> None:
    """Summarize the eight planned mappings from their saved result files."""
    equipment = (
        ("Pump", "pump"), ("Haul Truck", "haul_truck"),
        ("Belt Conveyor", "belt_conveyor"), ("Excavator", "excavator"),
        ("Dozer", "dozer"), ("Wheel Loader", "wheel_loader"),
        ("Drill Rig", "drill_rig"), ("Grader", "grader"),
    )

    def result(machine: str, suffix: str) -> dict:
        path = RESULTS / machine / f"{machine}_{suffix}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def percent(score):
        return round(score * 100, 2) if score is not None else None

    rows = []
    for name, machine in equipment:
        sensor = result(machine, "csv_4class").get("validation_selected", {}).get("score", {})
        visual = result(machine, "cnn_pretrained_4class").get("recording_level", {})
        sensor_time = result(machine, "csv_4class_timesplit").get("models", {}).get("Random forest", {})
        visual_time = result(machine, "cnn_pretrained_4class_timesplit").get("recording_level", {})
        s, v = sensor.get("accuracy"), visual.get("accuracy")
        rows.append({
            "Equipment": name,
            "Strict sensor %": percent(s),
            "Strict visual %": percent(v),
            "Time sensor %": percent(sensor_time.get("accuracy")),
            "Time visual %": percent(visual_time.get("accuracy")),
            "Status": "Both >80%" if s is not None and v is not None and s > .8 and v > .8
                      else "Evaluated" if s is not None and v is not None else "Pending",
        })
    display(pd.DataFrame(rows).set_index("Equipment"))


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
    csv_result = load(f"{machine}_csv_4class.json")
    rows = {}
    if csv_result:
        if "validation_selected" in csv_result:
            rows["CSV sensor | validation-selected (main)"] = csv_result["validation_selected"]["score"]
        rows.update({f"CSV sensor | {name}": score for name, score in csv_result["models"].items()})
    dense = load(f"{machine}_cnn_pretrained_4class_denseframes.json")
    pretrained_name = f"{machine}_cnn_pretrained_4class{'_denseframes' if dense else ''}.json"
    pretrained_label = "Camera images | expanded ResNet18" if dense else "Camera images | original ResNet18"
    for name, label in ((pretrained_name, pretrained_label),
                        (f"{machine}_cnn_4class.json", "Camera images | historical small CNN")):
        result = load(name)
        if result:
            rows[f"{label}, per recording"] = result["recording_level"]
            if "segment_level" in result:
                rows[f"{label}, per 1/10 recording segment"] = result["segment_level"]
    time_csv, time_cnn = load(f"{machine}_csv_4class_timesplit.json"), load(f"{machine}_cnn_pretrained_4class_timesplit.json")
    if time_csv:
        rows["TIME SPLIT (secondary) | CSV sensor | Random forest"] = time_csv["models"]["Random forest"]
    if time_cnn:
        rows["TIME SPLIT (secondary) | Camera images | pretrained ResNet18"] = time_cnn["recording_level"]
    show(rows)


def show_ensembles(machine: str) -> None:
    """Display the proposal Week 6 ensemble comparison (XGBoost / LightGBM vs the other models)."""
    rows = {}
    for suffix, label in (("", "MAIN"), ("_timesplit", "TIME SPLIT (secondary)")):
        result = load(f"{machine}_ensembles{suffix}.json")
        for name, block in (result or {}).get("models", {}).items():
            rows[f"{label} | {name}"] = {**metrics(block), "train_seconds": block["train_seconds"]}
    display(pd.DataFrame(rows).T)


def show_anomaly(machine: str) -> None:
    """Display the Week 7 anomaly detection results (trained on healthy data only)."""
    rows = {}
    for suffix, label in (("", "MAIN"), ("_timesplit", "TIME SPLIT (secondary)")):
        result = load(f"{machine}_anomaly{suffix}.json")
        for name, block in (result or {}).get("detectors", {}).items():
            rows[f"{label} | {name}"] = {k: block[k] for k in ("roc_auc", "fault_detection_rate", "false_alarm_rate",
                                                                "accuracy", "test_recordings")}
    display(pd.DataFrame(rows).T)


def show_fulldata(machine: str) -> None:
    """Display the Week 7 full-data sensor experiment (every CORA sensor, window and start-up recording)."""
    result = load(f"{machine}_fulldata.json")
    if not result:
        print(f"No full-data result yet: run  python Models/{machine}.py fulldata")
        return
    def row(block):
        return {"accuracy (all recordings)": block["all"]["accuracy"], "test recordings": block["all"]["test_recordings"],
                "stationary runs": block.get("stationary", {}).get("accuracy"),
                "start-up tests": block.get("transient", {}).get("accuracy"), "macro F1": block["all"]["macro_f1"]}
    rows = {"MAIN (unseen speed) | validation-selected": row(result["validation_selected"]["score"])}
    rows.update({f"MAIN | {name}": row(block) for name, block in result["all_combinations"].items()})
    rows.update({f"TIME SPLIT (stationary) | {name}": row(block)
                 for name, block in result["time_split_stationary"].items()})
    display(pd.DataFrame(rows).T)
    print("Validation choices per held-out speed:",
          "; ".join(f"{c['held_out_speed']}: {c['feature_set']} / {c['model']}" for c in result["validation_selected"]["choices"]))


def show_fulldata_extras(machine: str) -> None:
    """Display the Week 7 full-data ensemble comparison and anomaly detection (fair test: unseen speed)."""
    ens, ano = load(f"{machine}_fulldata_ensembles.json"), load(f"{machine}_fulldata_anomaly.json")
    if not ens and not ano:
        print(f"No result yet: run  python Models/{machine}.py fulldata --part all")
        return
    if ens:
        display(pd.DataFrame({name: {"MAIN accuracy (unseen speed)": m["main"]["all"]["accuracy"],
                                     "MAIN macro F1": m["main"]["all"]["macro_f1"],
                                     "MAIN stationary": m["main"]["stationary"]["accuracy"],
                                     "MAIN start-up": m["main"]["transient"]["accuracy"],
                                     "TIME SPLIT (secondary)": m["time_split_stationary"]["all"]["accuracy"],
                                     "seconds": m["seconds"]} for name, m in ens["models"].items()}).T)
    if ano:
        display(pd.DataFrame({f"{label} | {name}": {k: b[part][k] for k in ("roc_auc", "fault_detection_rate",
                                                                            "false_alarm_rate", "test_recordings")}
                              for name, b in ano["detectors"].items()
                              for part, label in (("main", "MAIN"), ("time_split_stationary", "TIME SPLIT"))}).T)


def show_allframes(machine: str) -> None:
    """Compare the Week 6 image CNN with the Week 7 run on every frame (separate result files)."""
    rows = {}
    for suffix, label in (("", "Week 6 frames"), ("_allframes", "Week 7 all frames")):
        for split, name in (("", "MAIN"), ("_timesplit", "TIME SPLIT")):
            result = load(f"{machine}_cnn_pretrained_4class{suffix}{split}.json")
            if result:
                rows[f"{name} | {label} ({result['images']} images)"] = result["recording_level"]
                if "segment_level" in result:
                    rows[f"{name} | {label}, per segment"] = result["segment_level"]
    show(rows)


def show_per_recording(machine: str, suffix: str = "") -> None:
    """Every test image, every 1/10 segment and every recording of the pretrained CNN, with the counts tested."""
    result = load(f"{machine}_cnn_pretrained_4class{suffix}.json")
    if not result:
        print(f"No result yet: run  python Models/{machine}.py cnn-pretrained")
        return
    levels = {"every test image": result["image_level"]}
    if "segment_level" in result:
        levels["per 1/10 recording segment"] = result["segment_level"]
    levels["per whole recording"] = result["recording_level"]
    _levels_figure(levels, result["classes"],
                   f"{machine}: pretrained ResNet18 on {result['images']:,} images in total, "
                   f"{result['recording_level']['test_units']} test recordings"
                   + (" (time split, secondary)" if "timesplit" in suffix else ", each never seen in training"))
    if "per_recording" in result:
        display(pd.DataFrame({unit: {"class": r["class"], **r["images_predicted"], "final answer": r["recording_prediction"]}
                              for unit, r in result["per_recording"].items()}).T)
    else:
        print(f"Images per class for each recording appear after rerunning  python Models/{machine}.py cnn-pretrained")


_TEST_IMAGE_MACHINES = (("Pump", "pump"), ("Haul Truck", "haul_truck"), ("Belt Conveyor", "belt_conveyor"),
                        ("Excavator", "excavator"), ("Dozer", "dozer"), ("Wheel Loader", "wheel_loader"))


def _preferred_cnn_result(machine: str):
    """Prefer the expanded held-out CNN evaluation, then all-frame and original results."""
    for suffix in ("_denseframes", "_allframes", ""):
        result = load(f"{machine}_cnn_pretrained_4class{suffix}.json")
        if result and "image_level" in result:
            return result
    return None


def _levels_figure(levels: dict, labels: list, title: str) -> None:
    """One image: the score table on top and one confusion matrix per scoring level below."""
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(4.4 * len(levels), 6.2), dpi=90)
    grid = figure.add_gridspec(2, len(levels), height_ratios=[1, 4.2])
    top = figure.add_subplot(grid[0, :])
    top.axis("off")
    columns = ["tested", *METRICS]
    cells = [[f"{block['test_units']:,}", *(f"{block[k]:.3f}" for k in METRICS)] for block in levels.values()]
    table = top.table(cellText=cells, rowLabels=list(levels), colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (name, block), column in zip(levels.items(), range(len(levels))):
        _draw_image_matrix(figure.add_subplot(grid[1, column]), block, labels, name, level=None)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    plt.show()


def _draw_image_matrix(axis, block, labels, title, level="every test image"):
    import numpy as np
    array = np.asarray(block["confusion_matrix"])
    axis.imshow(array, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    prefix = f"{level}: " if level else ""
    axis.set_title(f"{title}\n{prefix}{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                      color="white" if array[i, j] > array.max() / 2 else "black")


def show_test_images(machine: str | None = None) -> None:
    """Confusion matrix over every test image of the fair-test CNN: one machine, or each of the six in turn
    (a separate image per machine) when machine is None."""
    import matplotlib.pyplot as plt
    machines = [(n, m) for n, m in _TEST_IMAGE_MACHINES if machine in (None, m)]
    found = [(n, m, r) for n, m in machines if (r := _preferred_cnn_result(m))]
    if not found:
        print(f"No CNN result yet: run  python Models/{machine}.py cnn-pretrained")
        return
    for name, m, result in found:
        figure, axis = plt.subplots(figsize=(4.8, 4.6), dpi=100)
        _draw_image_matrix(axis, result["image_level"], result["classes"],
                           f"{name}: {result['images']:,} images, "
                           f"{result['recording_level']['test_units']} test recordings")
        figure.tight_layout()
        plt.show()


def show_thermal_motor() -> None:
    """Excavator, separate image data: Kulkarni motor thermal images, healthy vs corroded bearing, unseen load."""
    result = load("excavator_thermal_motor_2class.json")
    if not result:
        print("No result yet: run  python Models/excavator.py thermal-motor")
        return
    levels = {"every test image": result["image_level"], "per 1/10 session segment": result["segment_level"],
              "per whole session": result["session_level"],
              "control: brightness only (per image)": result["brightness_control_image_level"]}
    if "border_only_control_image_level" in result:
        levels["control: border only, motor hidden (per image)"] = result["border_only_control_image_level"]
    table = pd.DataFrame({name: metrics(block) for name, block in levels.items()}).T
    table.insert(0, "tested", [f"{block['test_units']:,}" for block in levels.values()])
    display(table)
    display(pd.DataFrame(result["folds"]).set_index("test_load"))
    figure = ROOT / "Figures" / "excavator" / "excavator_thermal_motor_2class.png"
    if figure.exists():
        import io
        from PIL import Image as PILImage
        buffer = io.BytesIO()
        picture = PILImage.open(figure).convert("RGB")
        picture.thumbnail((1000, 1000))
        picture.save(buffer, "JPEG", quality=60)
        display(Image(data=buffer.getvalue(), format="jpeg", width=1000))
    print("Caveat:", result["caveat"])


def save_test_image_counts() -> Path:
    """Recreate the notebook's held-out image-count chart from saved evaluations."""
    import matplotlib.pyplot as plt
    found = [(name, result) for name, machine in _TEST_IMAGE_MACHINES
             if (result := _preferred_cnn_result(machine))]
    if not found:
        raise ValueError("No saved held-out image evaluations are available")
    figure, axis = plt.subplots(figsize=(10, 5))
    counts = [result["image_level"]["test_units"] for _, result in found]
    bars = axis.barh([name for name, _ in found], counts, color="#2878a5")
    axis.bar_label(bars, labels=[f"{n:,} images tested" for n in counts], padding=6)
    axis.set_xlim(0, max(counts) * 1.35)
    axis.invert_yaxis()
    axis.set_xlabel("Number of held-out test images")
    axis.set_title("Images tested for each equipment mapping")
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(.5, .015, "Frames from the same recording are related. Dozer reuses the Belt Conveyor benchmark.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, 1))
    path = ROOT / "Figures" / "test_image_counts.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
