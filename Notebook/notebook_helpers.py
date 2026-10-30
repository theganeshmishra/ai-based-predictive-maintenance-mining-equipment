"""Shared set-up for the single combined equipment notebook.

All equipment model implementations, shared data loading and display helpers
live in this file. The combined notebook calls a selected model or shows saved results.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from types import SimpleNamespace
import csv
import json
import os
import platform
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

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, RESULTS = ROOT / "Raw_Data", ROOT / "Processed_Data", ROOT / "Results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Shared data readers. Files are loaded only when requested.
RAW_DATA, PROCESSED_DATA, IMAGE_DATA = RAW, PROCESSED, ROOT / "Images"


def load_mat(path, *args, **kwargs):
    from scipy.io import loadmat
    return loadmat(path, *args, **kwargs)


def load_csv(path, *args, **kwargs):
    import pandas as pd
    return pd.read_csv(path, *args, **kwargs)


def load_excel(path, *args, **kwargs):
    import pandas as pd
    return pd.read_excel(path, *args, **kwargs)


def load_numpy(path, *args, **kwargs):
    import numpy as np
    return np.load(path, *args, **kwargs)


def load_image(path, flags=None):
    import cv2
    return cv2.imread(str(path), cv2.IMREAD_COLOR if flags is None else flags)



def load_pump_sensor(filename):
    """Load one Ottawa Pump bearing recording as a flat vibration signal."""
    path = RAW_DATA / "uottawa_bearing" / filename
    data = load_mat(path, variable_names=["Channel_1"])
    return data["Channel_1"].ravel()


def load_haul_truck_sensor(filename):
    """Load the accelerometer column from one UODS component recording."""
    import numpy as np

    path = RAW_DATA / "uods_vafdc" / filename
    data = load_mat(path)
    recording_name = Path(filename).stem
    return data[recording_name][:, 0].astype(np.float64)


def load_cora_sensor(filename):
    """Load the first vibration column for the four CORA equipment mappings."""
    import numpy as np

    path = RAW_DATA / "cora_vibration" / filename
    table = load_csv(path, header=None, dtype=np.float64)
    return table.iloc[:, 0].to_numpy()


def load_model_images(image_folder, records):
    """Read RGB images and prepare a normalized NCHW array for pretrained CNNs.

    Records come from index.csv and retain their existing labels and run groups.
    This function only loads pixels; splitting remains in the equipment model.
    """
    import cv2
    import numpy as np

    images = []
    for record in records:
        path = Path(image_folder) / record["file"]
        image = load_image(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing image {record['file']}: run export-images first")

        # OpenCV reads BGR; the pretrained CNN expects RGB, channels first.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32).transpose(2, 0, 1)

        # Normalize each image independently, without using other recordings.
        image = (image - image.mean()) / (image.std() + 1e-6)
        images.append(image)

    return np.stack(images)


def open_archive(path, *args, **kwargs):
    """Return a ZIP reader; use a with statement to close it after reading."""
    return zipfile.ZipFile(path, *args, **kwargs)


def load_image_index(machine, variant=''):
    """Read filenames, labels and recording groups without loading image pixels."""
    if variant not in ('', 'allframes', 'denseframes'):
        raise ValueError('variant must be empty, allframes or denseframes')
    if machine not in ('pump', 'haul_truck', 'belt_conveyor', 'excavator', 'dozer', 'wheel_loader'):
        raise ValueError('Unknown equipment mapping')
    folder = machine + (f'_{variant}' if variant else '')
    with (IMAGE_DATA / folder / 'index.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


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
            equipment = Path(relative_path).stem
            get_model(equipment).cli()
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


def show_drill_rig() -> None:
    """Drill Rig, separate data: MCC5-THU gearbox vibration, 4 classes, each speed held out in turn."""
    result = load("drill_rig_csv_4class.json")
    if not result:
        print("No result yet: run  python Notebook/notebook_helpers.py drill_rig all")
        return
    rows = {"validation-selected model (main result)": result["validation_selected"]["score"], **result["models"]}
    table = pd.DataFrame({name: metrics(block) for name, block in rows.items()}).T
    table.insert(0, "tested", [f"{block['test_units']} recordings" for block in rows.values()])
    display(table)
    display(pd.DataFrame(result["folds"]).drop(columns="inner_macro_f1").set_index("held_out_rpm"))
    figure = ROOT / "Figures" / "drill_rig" / "drill_rig_csv_4class.png"
    if figure.exists():
        display(Image(filename=str(figure), width=480))
    print(f"{result['recordings']} recordings, {result['windows']:,} windows. Caveat: {result['caveat']}")


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


def save_test_image_result_graphs() -> None:
    """Recreate image counts and performance graphs from the saved held-out evaluations."""
    import matplotlib.pyplot as plt
    root=ROOT;machines=('pump','haul_truck','belt_conveyor','excavator','dozer','wheel_loader')
    for m in machines:
     folder=root/'Results'/m
     p=next(folder/f'{m}_cnn_pretrained_4class{s}.json' for s in ('_denseframes','_allframes','') if (folder/f'{m}_cnn_pretrained_4class{s}.json').exists())
     r=json.loads(p.read_text());b=r['image_level'];matrix=b['confusion_matrix'];correct=sum(row[i] for i,row in enumerate(matrix));total=b['test_units'];wrong=total-correct
     fig,axes=plt.subplots(1,2,figsize=(10,4.5))
     bars=axes[0].bar(['Tested','Correct','Incorrect'],[total,correct,wrong],color=['#2878a5','#33865a','#bf5656'])
     axes[0].bar_label(bars,labels=[f'{n:,}' for n in (total,correct,wrong)],padding=5)
     axes[0].set_ylim(0,total*1.18);axes[0].set_ylabel('Number of held-out test images')
     keys=('accuracy','macro_precision','macro_recall','macro_f1');vals=[b[k]*100 for k in keys]
     bars=axes[1].bar(['Accuracy','Precision','Recall','F1'],vals,color='#2878a5')
     axes[1].bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=5)
     axes[1].set_ylim(0,115);axes[1].set_ylabel('Image-level performance (%)')
     for ax in axes:ax.spines[['top','right']].set_visible(False)
     fig.suptitle(f"{m.replace('_',' ').title()}: {total:,} held-out test images from {r['recording_level']['test_units']} recordings")
     fig.text(.5,.015,'Frames from the same recording are related. Precision, recall and F1 are macro averages.',ha='center',fontsize=9)
     fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(root/'Figures'/m/f'{m}_test_image_results.png',dpi=150);plt.close(fig)


# PUMP MODEL IMPLEMENTATION
def _make_pump():
    __doc__ = "Pump: support bearing, vibration sensor vs XJTU event camera.\n\nAll of this machine's code in one file. The combined notebook\nNotebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.\n\nUsage:\n    python Models/pump.py <command> [options]\n    python Models/pump.py <command> --help\n\nCommands:\n    intake          download and verify the 48 Ottawa vibration recordings\n    sensor          vibration sensor model\n    camera          XJTU event-camera model (OpenCV HOG)\n    compare         sensor vs camera report and figure\n    original-xlsx   extra: original centrifugal-pump XLSX diagnostic (sensor only)\n    uci-hydraulic   extra: UCI hydraulic pump leakage (sensor only)\n    espset          extra: ESPset V3 pump vibration (sensor only)\n    baselines       Week 5: compare majority class, logistic regression, decision tree and random forest\n    export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)\n    csv             Week 6: 4-class sensor model from a CSV feature table\n    cnn             Week 6: CNN with data augmentation on the named images (run export-images first)\n    cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)\n    all             run sensor, then camera, then compare\n\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Pump: support bearing, vibration sensor vs XJTU event camera.

    All of this machine's code in one file. The combined notebook
    Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

    Usage:
        python Models/pump.py <command> [options]
        python Models/pump.py <command> --help

    Commands:
        intake          download and verify the 48 Ottawa vibration recordings
        sensor          vibration sensor model
        camera          XJTU event-camera model (OpenCV HOG)
        compare         sensor vs camera report and figure
        original-xlsx   extra: original centrifugal-pump XLSX diagnostic (sensor only)
        uci-hydraulic   extra: UCI hydraulic pump leakage (sensor only)
        espset          extra: ESPset V3 pump vibration (sensor only)
        baselines       Week 5: compare majority class, logistic regression, decision tree and random forest
        export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)
        csv             Week 6: 4-class sensor model from a CSV feature table
        cnn             Week 6: CNN with data augmentation on the named images (run export-images first)
        cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)
        all             run sensor, then camera, then compare

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import sys
    from pathlib import Path



    # ==================================================================================================
    # intake: download and verify the 48 Ottawa vibration recordings
    # (formerly Preprocessing/pump_bearing_independent_intake.py)
    # ==================================================================================================

    def intake(run: bool = True) -> dict:
        """Acquire the four-class University of Ottawa bearing-vibration proxy.

        Source: Huang and Baddour, Mendeley Data V2, DOI 10.17632/v43hmbwxpm.2.
        Only H/I/O/B MAT files are selected; the compound-fault class is excluded.
        The files describe an independent bearing test rig, not the XJTU pump.
        """


        import hashlib
        import json
        import os
        import re
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from pathlib import Path
        from urllib.request import Request, urlopen


        CAPSTONE = ROOT
        RAW = CAPSTONE / "Raw_Data" / "uottawa_bearing"
        MANIFEST = CAPSTONE / "Results" / "pump" / "pump_uottawa_intake.json"
        BASE = "https://data.mendeley.com/public-api/datasets/v43hmbwxpm"
        FOLDERS = {
            "H": "1 Data collected from a healthy bearing",
            "I": "2 Data collected from a bearing with inner race fault",
            "O": "3 Data collected from a bearing with outer race fault",
            "B": "4 Data collected from a bearing with ball fault",
        }
        NAME = re.compile(r"^([HIOB])-([ABCD])-([123])\.mat$")


        def get_json(url: str):
            with urlopen(Request(url, headers={"User-Agent": "PROJ613-capstone/1.0"}), timeout=45) as response:
                return json.load(response)


        def inventory() -> list[dict]:
            folders = get_json(f"{BASE}/folders/2")
            by_name = {item["name"]: item["id"] for item in folders}
            if not set(FOLDERS.values()).issubset(by_name):
                raise ValueError("Expected publisher fault folders are missing")
            selected = []
            for prefix, folder_name in FOLDERS.items():
                folder_id = by_name[folder_name]
                url = f"{BASE}/files?folder_id={folder_id}&version=2&%24start=0&%24limit=1000"
                members = get_json(url)
                if len(members) != 12:
                    raise ValueError(f"Expected 12 separate files in {folder_name}; found {len(members)}")
                for item in members:
                    name = item["filename"]
                    match = NAME.fullmatch(name)
                    if not match or match.group(1) != prefix:
                        raise ValueError(f"Unexpected source file in {folder_name}: {name}")
                    details = item["content_details"]
                    digest = details["sha256_hash"]
                    if not re.fullmatch(r"[a-f0-9]{64}", digest):
                        raise ValueError(f"Missing publisher SHA-256 for {name}")
                    selected.append({
                        "name": name,
                        "class_code": prefix,
                        "speed_profile": match.group(2),
                        "trial": int(match.group(3)),
                        "source_file_id": item["id"],
                        "source_sha256": digest,
                        "source_size_bytes": int(details["size"]),
                        "url": details["download_url"] + "?version=2",
                    })
            if len({item["name"] for item in selected}) != 48:
                raise ValueError("Expected 48 unique H/I/O/B recordings")
            return sorted(selected, key=lambda item: item["name"])


        def sha256(path: Path) -> str:
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            return checksum.hexdigest()


        def acquire(item: dict) -> tuple[str, str]:
            destination = RAW / item["name"]
            if destination.is_file():
                if destination.stat().st_size != item["source_size_bytes"] or sha256(destination) != item["source_sha256"]:
                    raise ValueError(f"Existing source file failed publisher hash check: {destination}")
                return item["name"], "cached-and-verified"
            temporary = destination.with_suffix(".mat.part")
            for attempt in range(3):
                try:
                    request = Request(item["url"], headers={"User-Agent": "PROJ613-capstone/1.0"})
                    with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                    if temporary.stat().st_size != item["source_size_bytes"] or sha256(temporary) != item["source_sha256"]:
                        raise ValueError(f"Publisher hash/size mismatch for {item['name']}")
                    os.replace(temporary, destination)
                    return item["name"], "downloaded-and-verified"
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            raise AssertionError("Unreachable")


        def main() -> None:
            items = inventory()
            RAW.mkdir(parents=True, exist_ok=True)
            status = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(acquire, item): item["name"] for item in items}
                for completed, future in enumerate(as_completed(futures), 1):
                    name, outcome = future.result()
                    status[name] = outcome
                    print(f"{completed}/48 {name}: {outcome}", flush=True)
            for item in items:
                item.pop("url")
                item["local_status"] = status[item["name"]]
            result = {
                "source": "https://data.mendeley.com/datasets/v43hmbwxpm/2",
                "doi": "10.17632/v43hmbwxpm.2",
                "license": "CC BY 4.0",
                "source_domain": "University of Ottawa laboratory bearing rig, not a pump or mining machine",
                "selected": "H/I/O/B only; four fault classes × four speed profiles × three trials",
                "selected_recordings": len(items),
                "selected_bytes": sum(item["source_size_bytes"] for item in items),
                "files": items,
            }
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved verified file manifest: {MANIFEST}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # sensor: vibration sensor model
    # (formerly Models/pump_bearing_independent_baseline.py)
    # ==================================================================================================

    def sensor(run: bool = True) -> dict:
        """Recording-held-out vibration benchmark for the Pump bearing-fault taxonomy.

        Source recordings are from a University of Ottawa bearing rig, not a pump.
        The professor allows comparable components on different physical machines.
        This model uses whole-run evaluation and never joins its rows to Pump images.
        """


        import argparse
        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


        CAPSTONE = ROOT
        INTAKE = CAPSTONE / "Results" / "pump" / "pump_uottawa_intake.json"
        RAW = CAPSTONE / "Raw_Data" / "uottawa_bearing"
        CACHE = CAPSTONE / "Processed_Data" / "pump_bearing_independent_features.npz"
        RESULT = CAPSTONE / "Results" / "pump" / "pump_bearing_independent.json"
        FIGURE = CAPSTONE / "Figures" / "pump" / "pump_bearing_independent_confusions.png"
        CLASSES = ("healthy", "inner", "outer", "ball")
        CODE_TO_CLASS = {"H": "healthy", "I": "inner", "O": "outer", "B": "ball"}
        SAMPLE_RATE_HZ = 200_000
        WINDOW_SAMPLES = 20_000  # 100 ms; 100 non-overlapping windows per 10 s source recording
        BANDS_HZ = ((0, 2_000), (2_000, 5_000), (5_000, 10_000),
                    (10_000, 20_000), (20_000, 50_000), (50_000, 100_000))
        FEATURE_NAMES = (
            "mean", "std", "rms", "peak_to_peak", "crest_factor", "skew", "excess_kurtosis",
            "spectral_centroid_hz", "dominant_frequency_hz", "spectral_entropy",
            *(f"power_{low}_{high}_hz" for low, high in BANDS_HZ),
        )


        def extract(signal: np.ndarray) -> np.ndarray:
            count = len(signal) // WINDOW_SAMPLES
            if count < 80:
                raise ValueError(f"Recording is unexpectedly short: {len(signal)} samples")
            windows = signal[:count * WINDOW_SAMPLES].reshape(count, WINDOW_SAMPLES)
            mean = windows.mean(axis=1)
            centered = windows - mean[:, None]
            std = centered.std(axis=1)
            safe_std = np.maximum(std, 1e-12)
            rms = np.sqrt(np.mean(windows**2, axis=1))
            crest = np.max(np.abs(centered), axis=1) / safe_std
            skew = np.mean(centered**3, axis=1) / safe_std**3
            kurt = np.mean(centered**4, axis=1) / safe_std**4 - 3
            frequency = np.fft.rfftfreq(WINDOW_SAMPLES, d=1 / SAMPLE_RATE_HZ)
            power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
            total = np.maximum(power.sum(axis=1), 1e-12)
            relative = power / total[:, None]
            centroid = (relative * frequency[None, :]).sum(axis=1)
            dominant = frequency[np.argmax(power[:, 1:], axis=1) + 1]
            entropy = -(relative * np.log(np.maximum(relative, 1e-12))).sum(axis=1)
            band_features = [relative[:, (frequency >= low) & (frequency < high)].sum(axis=1)
                             for low, high in BANDS_HZ]
            columns = [mean, std, rms, np.ptp(windows, axis=1), crest, skew, kurt,
                       centroid, dominant, entropy, *band_features]
            result = np.column_stack(columns).astype(np.float32)
            if not np.isfinite(result).all():
                raise ValueError("Non-finite vibration feature encountered")
            return result


        def preprocess() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
            intake = json.loads(INTAKE.read_text(encoding="utf-8"))
            files = intake["files"]
            if len(files) != 48 or len({item["name"] for item in files}) != 48:
                raise ValueError("Intake manifest must contain 48 unique source recordings")
            all_x, all_y, all_profile, all_run, provenance = [], [], [], [], []
            for number, item in enumerate(files, 1):
                name = item["name"]
                signal = load_pump_sensor(name)
                if signal.dtype.kind not in "fi" or not np.isfinite(signal).all():
                    raise ValueError(f"Invalid accelerometer signal in {name}")
                run_x = extract(np.asarray(signal, dtype=np.float64))
                label = CODE_TO_CLASS[item["class_code"]]
                all_x.append(run_x)
                all_y.extend([label] * len(run_x))
                all_profile.extend([item["speed_profile"]] * len(run_x))
                all_run.extend([name] * len(run_x))
                provenance.append({"file": name, "label": label, "speed_profile": item["speed_profile"],
                                   "trial": item["trial"], "source_sha256": item["source_sha256"],
                                   "samples": int(len(signal)), "windows": len(run_x)})
                print(f"{number}/48 {name}: {len(run_x)} non-overlapping windows", flush=True)
            x = np.vstack(all_x)
            y, profile, run = np.asarray(all_y), np.asarray(all_profile), np.asarray(all_run)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE, x=x, y=y, profile=profile, run=run,
                                provenance=np.asarray(json.dumps(provenance)))
            return x, y, profile, run, provenance


        def build_model() -> RandomForestClassifier:
            return RandomForestClassifier(n_estimators=250, max_features="sqrt",
                                          class_weight="balanced_subsample", min_samples_leaf=2,
                                          random_state=42, n_jobs=-1)


        def run_predictions(model, x: np.ndarray, y: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            actual, predicted = [], []
            for name in sorted(set(run)):
                mask = run == name
                if len(set(y[mask])) != 1:
                    raise ValueError(f"Mixed fault labels within {name}")
                actual.append(y[mask][0])
                probability = model.predict_proba(x[mask]).mean(axis=0)
                predicted.append(model.classes_[int(np.argmax(probability))])
            return np.asarray(actual), np.asarray(predicted)


        def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
            precision, recall, f1, support = precision_recall_fscore_support(
                actual, predicted, labels=CLASSES, zero_division=0
            )
            return {
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
                "per_class": {label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                      "f1": round(float(f), 4), "support": int(n)}
                              for label, p, r, f, n in zip(CLASSES, precision, recall, f1, support)},
            }


        def plot(primary: list[list[int]], secondary: list[list[int]]) -> None:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for axis, matrix, title in zip(axes, (primary, secondary),
                                           ("A/B train → C/D test (24 runs)",
                                            "Held-out speed-profile folds (48 runs)")):
                array = np.asarray(matrix)
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(4), CLASSES, rotation=30)
                axis.set_yticks(range(4), CLASSES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(title)
                for i in range(4):
                    for j in range(4):
                        axis.text(j, i, str(array[i, j]), ha="center", va="center",
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=160)
            plt.close(figure)


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--refresh", action="store_true", help="Re-extract features from verified raw MAT files")
            args = parser.parse_args()
            if CACHE.is_file() and not args.refresh:
                with load_numpy(CACHE, allow_pickle=False) as saved:
                    x, y, profile, run = (saved[key] for key in ("x", "y", "profile", "run"))
                    provenance = json.loads(str(saved["provenance"]))
            else:
                x, y, profile, run, provenance = preprocess()
            train, test = np.isin(profile, ["A", "B"]), np.isin(profile, ["C", "D"])
            if len(set(run[train])) != 24 or len(set(run[test])) != 24 or set(run[train]) & set(run[test]):
                raise ValueError("Expected disjoint groups of 24 full recordings")
            forest = build_model()
            forest.fit(x[train], y[train])
            actual, predicted = run_predictions(forest, x[test], y[test], run[test])
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(x[train], y[train])
            dummy_actual, dummy_pred = run_predictions(dummy, x[test], y[test], run[test])

            # Sensitivity check: every run is tested once with its entire speed profile
            # absent from training. Reused source rig/bearing means this is not a new site.
            fold_actual, fold_pred = [], []
            for held_out in "ABCD":
                probe = build_model()
                fit_mask, test_mask = profile != held_out, profile == held_out
                probe.fit(x[fit_mask], y[fit_mask])
                a, p = run_predictions(probe, x[test_mask], y[test_mask], run[test_mask])
                fold_actual.extend(a)
                fold_pred.extend(p)

            # Permute labels by whole training run, never independently by window.
            training_runs = sorted(set(run[train]))
            original = {name: y[run == name][0] for name in training_runs}
            rng = np.random.default_rng(42)
            shuffled = dict(zip(training_runs, rng.permutation([original[name] for name in training_runs])))
            shuffled_y = np.asarray([shuffled[name] for name in run[train]])
            negative = build_model()
            negative.fit(x[train], shuffled_y)
            _, negative_pred = run_predictions(negative, x[test], y[test], run[test])

            result = {
                "status": "independent_recording_cross_speed_profile_evaluation",
                "source": "Huang and Baddour, https://data.mendeley.com/datasets/v43hmbwxpm/2, CC BY 4.0",
                "source_domain": "University of Ottawa laboratory bearing rig, not a pump or mining machine",
                "target": "current bearing condition: healthy/inner/outer/ball",
                "class_order": list(CLASSES),
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "window_samples": WINDOW_SAMPLES,
                "feature_names": list(FEATURE_NAMES),
                "selection": "48 source runs: four classes × four speed profiles × three trials",
                "split": "A and B speed-profile recordings train; disjoint C and D recordings test; score one averaged prediction per complete recording",
                "train_runs": 24,
                "test_runs": 24,
                "source_recordings": provenance,
                "random_forest_run_level": metrics(actual, predicted),
                "dummy_run_level": metrics(dummy_actual, dummy_pred),
                "held_out_speed_profile_run_level": metrics(np.asarray(fold_actual), np.asarray(fold_pred)),
                "shuffled_training_run_labels_control": metrics(actual, negative_pred),
                "limitations": [
                    "Separate whole recordings are held out, but the dataset does not document independent physical bearing specimens for every trial/profile.",
                    "The rig is not a pump; transfer to pump support bearings is a mechanical-component analogy, not demonstrated pump accuracy.",
                    "No sensor recording is paired with an XJTU event-camera run; only the four fault labels are comparable.",
                    "The target is present fault type, not future repair timing or remaining useful life.",
                ],
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(result["random_forest_run_level"]["confusion_matrix"],
                 result["held_out_speed_profile_run_level"]["confusion_matrix"])
            print(f"Saved {RESULT} and {FIGURE}")
            print(f"Held-out run accuracy {result['random_forest_run_level']['accuracy']:.4f}; macro F1 {result['random_forest_run_level']['macro_f1']:.4f}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # camera: XJTU event-camera model (OpenCV HOG)
    # (formerly Models/pump_event_visual_baseline.py)
    # ==================================================================================================

    def camera(run: bool = True) -> dict:
        """Pump event-camera fault classifier using independent XJTU-DV-Pump runs.

        Read selected members of the public Zenodo ZIPs by HTTP range. Convert the
        first two seconds of each event stream into 250 ms polarity-count images,
        extract OpenCV HOG features, train on 20 Hz runs, and test on 30 Hz runs.
        Each source recording is one evaluation unit; no sensor row is paired to it.
        """


        import argparse
        import hashlib
        import json
        import os
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        from remotezip import RemoteZip
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


        CAPSTONE = ROOT
        CACHE = CAPSTONE / "Processed_Data" / "pump_event_features.npz"
        RUN_CACHE = CAPSTONE / "Processed_Data" / "pump_event_runs"
        RESULT = CAPSTONE / "Results" / "pump" / "pump_event_visual.json"
        FIGURE = CAPSTONE / "Figures" / "pump" / "pump_event_visual_confusion.png"
        ZENODO = "https://zenodo.org/api/records/17447083/files/"
        CLASSES = ("healthy", "inner", "outer", "ball")
        SOURCE_FOLDERS = {"healthy": "Healthy", "inner": "Inner", "outer": "Outer", "ball": "Ball"}
        START_US = 0
        END_US = 2_000_000
        BIN_US = 250_000
        FRAME_COUNT = (END_US - START_US) // BIN_US
        HEIGHT, WIDTH = 480, 640
        GRID_H, GRID_W = 48, 64
        EVENT_DTYPE = np.dtype([("t", "<u4"), ("xy", "<u4")])
        HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


        def selected_members(archive: RemoteZip, hz: int) -> list[tuple[str, str, int, int]]:
            selected = []
            for info in archive.infolist():
                for label, folder in SOURCE_FOLDERS.items():
                    prefix = f"XJTU-DV-Pump/Data/{folder}/{hz}"
                    if info.filename.startswith(prefix) and info.filename.endswith(".dat"):
                        selected.append((info.filename, label, hz, info.CRC))
            counts = {label: sum(item[1] == label for item in selected) for label in CLASSES}
            if counts != {label: 6 for label in CLASSES}:
                raise ValueError(f"Expected six runs per class at {hz} Hz; found {counts}")
            return sorted(selected)


        def parse_header(stream) -> tuple[int, int]:
            while True:
                first = stream.read(1)
                if not first:
                    raise ValueError("Event file ended before binary header")
                if first != b"%":
                    event_type = first[0]
                    break
                line = first + stream.readline()
                if line.startswith(b"% Height ") and int(line.split()[2]) != HEIGHT:
                    raise ValueError("Unexpected event camera height")
                if line.startswith(b"% Width ") and int(line.split()[2]) != WIDTH:
                    raise ValueError("Unexpected event camera width")
            event_size = stream.read(1)[0]
            if event_size != 8:
                raise ValueError(f"Expected 8-byte events, found {event_size}")
            return event_type, event_size


        def event_images(stream) -> tuple[np.ndarray, int, int]:
            event_type, _ = parse_header(stream)
            counts = np.zeros((FRAME_COUNT, 2, GRID_H, GRID_W), dtype=np.uint32)
            retained = 0
            last_timestamp = 0
            while True:
                payload = stream.read(8 * 250_000)
                if not payload:
                    break
                if len(payload) % 8:
                    raise ValueError("Truncated event record")
                events = np.frombuffer(payload, dtype=EVENT_DTYPE)
                last_timestamp = int(events["t"][-1])
                mask = (events["t"] >= START_US) & (events["t"] < END_US)
                if mask.any():
                    t = events["t"][mask]
                    packed = events["xy"][mask]
                    x = packed & 0x3FFF
                    y = (packed >> 14) & 0x3FFF
                    if (x >= WIDTH).any() or (y >= HEIGHT).any():
                        raise ValueError("Event coordinates exceed the stated sensor size")
                    p = (packed >> 28) & 1
                    frame = (t - START_US) // BIN_US
                    coarse_x = x // (WIDTH // GRID_W)
                    coarse_y = y // (HEIGHT // GRID_H)
                    index = (((frame * 2 + p) * GRID_H + coarse_y) * GRID_W + coarse_x).astype(np.int64)
                    counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                    retained += len(t)
                if last_timestamp >= END_US:
                    break
            if last_timestamp < END_US or retained < 1000:
                raise ValueError(f"Recording did not cover the requested interval (last={last_timestamp}, kept={retained})")
            return counts, event_type, retained


        def frame_features(counts: np.ndarray) -> np.ndarray:
            rows = []
            for frame in counts:
                # Signed event contrast; normalization suppresses absolute event-count
                # differences caused by illumination or exposure, but not all rig bias.
                signed = np.log1p(frame[1].astype(np.float32)) - np.log1p(frame[0].astype(np.float32))
                image = cv2.resize(signed, (64, 64), interpolation=cv2.INTER_LINEAR)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                rows.append(HOG.compute(image).ravel())
            return np.asarray(rows, dtype=np.float32)


        def process_member(job: tuple[str, str, str, int, int]) -> tuple[np.ndarray, dict]:
            url, member, label, hz, crc = job
            protocol_key = f"{member}|{START_US}|{END_US}|{BIN_US}"
            cache_name = hashlib.sha256(protocol_key.encode("utf-8")).hexdigest()[:20] + ".npz"
            cached_run = RUN_CACHE / cache_name
            if cached_run.is_file():
                with load_numpy(cached_run, allow_pickle=False) as saved:
                    x = saved["x"]
                    record = json.loads(str(saved["record"]))
                if (record["member"] != member or record["zip_member_crc32"] != f"{crc:08x}"
                        or record["analysis_interval_us"] != [START_US, END_US]):
                    raise ValueError(f"Cached event run does not match current protocol: {member}")
                return x, record

            for attempt in range(3):
                try:
                    archive = RemoteZip(url)
                    with archive.open(member) as stream:
                        counts, event_type, retained = event_images(stream)
                    archive.close()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            x = frame_features(counts)
            record = {"member": member, "label": label, "speed_hz": hz,
                      "frames": len(x), "events_in_interval": retained,
                      "event_type": event_type, "zip_member_crc32": f"{crc:08x}",
                      "analysis_interval_us": [START_US, END_US]}
            RUN_CACHE.mkdir(parents=True, exist_ok=True)
            temporary = cached_run.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, x=x, record=np.asarray(json.dumps(record)))
            os.replace(temporary, cached_run)
            return x, record


        def preprocess() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
            features, labels, speeds, run_ids, provenance = [], [], [], [], []
            for hz, part in ((20, "part1"), (30, "part2")):
                url = f"{ZENODO}XJTU-DV-Pump_{part}.zip/content"
                archive = RemoteZip(url)
                members = selected_members(archive, hz)
                archive.close()
                jobs = [(url, member, label, hz, crc) for member, label, _, crc in members]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for number, (x, record) in enumerate(pool.map(process_member, jobs), 1):
                        member, label = record["member"], record["label"]
                        features.extend(x)
                        labels.extend([label] * len(x))
                        speeds.extend([hz] * len(x))
                        run_ids.extend([member] * len(x))
                        provenance.append(record)
                        print(f"{hz} Hz {number}/{len(members)}: {label} {record['events_in_interval']:,} events", flush=True)
            x = np.asarray(features, dtype=np.float32)
            y = np.asarray(labels)
            speed = np.asarray(speeds)
            run = np.asarray(run_ids)
            if not np.isfinite(x).all() or len(set(run)) != 48:
                raise ValueError("Non-finite features or incorrect released-run count")
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE, x=x, y=y, speed=speed, run=run,
                                provenance=np.asarray(json.dumps(provenance)))
            return x, y, speed, run, provenance


        def run_predictions(model, x: np.ndarray, y: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            true_labels, predicted = [], []
            for run_id in sorted(set(run)):
                mask = run == run_id
                if len(set(y[mask])) != 1:
                    raise ValueError(f"Mixed class labels within {run_id}")
                true_labels.append(y[mask][0])
                probability = model.predict_proba(x[mask]).mean(axis=0)
                predicted.append(model.classes_[int(np.argmax(probability))])
            return np.asarray(true_labels), np.asarray(predicted)


        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASSES, zero_division=0
            )
            return {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "macro_precision": round(float(np.mean(precision)), 4),
                "macro_recall": round(float(np.mean(recall)), 4),
                "macro_f1": round(float(np.mean(f1)), 4),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
                "per_class": {label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                       "f1": round(float(f), 4), "support": int(n)}
                              for label, p, r, f, n in zip(CLASSES, precision, recall, f1, support)},
            }


        def camera_configuration(run_id: str) -> str:
            parts = Path(run_id).stem.split("_")
            if len(parts) < 4 or parts[-2] not in {"15", "18"} or parts[-1] not in {"-30", "0", "30"}:
                raise ValueError(f"Cannot identify camera configuration in {run_id}")
            return "_".join(parts[-2:])


        def plot_confusion(matrix: list[list[int]]) -> None:
            figure, axis = plt.subplots(figsize=(6, 5))
            axis.imshow(matrix, cmap="Blues", vmin=0)
            axis.set_xticks(range(4), CLASSES, rotation=30)
            axis.set_yticks(range(4), CLASSES)
            axis.set_xlabel("Predicted run condition")
            axis.set_ylabel("Actual run condition")
            axis.set_title("Pump event-camera, held-out 30 Hz runs")
            for i in range(4):
                for j in range(4):
                    axis.text(j, i, str(matrix[i][j]), ha="center", va="center",
                              color="white" if matrix[i][j] > np.max(matrix) / 2 else "black")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=160)
            plt.close(figure)


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--refresh", action="store_true", help="Reprocess the remote event streams")
            args = parser.parse_args()
            if CACHE.is_file() and not args.refresh:
                with load_numpy(CACHE, allow_pickle=False) as saved:
                    x, y, speed, run = (saved[key] for key in ("x", "y", "speed", "run"))
                    provenance = json.loads(str(saved["provenance"]))
            else:
                x, y, speed, run, provenance = preprocess()
            train, test = speed == 20, speed == 30
            if len(set(run[train])) != 24 or len(set(run[test])) != 24 or set(run[train]) & set(run[test]):
                raise ValueError("Expected 24 independent source recordings per speed")
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(x[train], y[train])
            forest = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                            min_samples_leaf=2, random_state=42, n_jobs=-1)
            forest.fit(x[train], y[train])
            yt_dummy, yp_dummy = run_predictions(dummy, x[test], y[test], run[test])
            yt, yp = run_predictions(forest, x[test], y[test], run[test])
            # Secondary sensitivity check: hold out each camera configuration across
            # both speeds, including every frame from its eight recordings.
            config = np.asarray([camera_configuration(name) for name in run])
            config_true, config_pred = [], []
            for held_out in sorted(set(config)):
                train_config = config != held_out
                test_config = config == held_out
                probe = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                               min_samples_leaf=2, random_state=42, n_jobs=-1)
                probe.fit(x[train_config], y[train_config])
                actual, predicted = run_predictions(probe, x[test_config], y[test_config], run[test_config])
                config_true.extend(actual)
                config_pred.extend(predicted)

            # A shuffled *run-label* control detects an accidental pipeline shortcut.
            training_runs = sorted(set(run[train]))
            rng = np.random.default_rng(42)
            original_labels = {name: y[run == name][0] for name in training_runs}
            shuffled_labels = dict(zip(training_runs, rng.permutation([original_labels[name] for name in training_runs])))
            shuffled_y = np.asarray([shuffled_labels[name] for name in run[train]])
            shuffled_model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
            shuffled_model.fit(x[train], shuffled_y)
            _, shuffled_pred = run_predictions(shuffled_model, x[test], y[test], run[test])
            result = {
                "status": "cross_speed_independent_recording_evaluation",
                "source": "https://zenodo.org/records/17447083; XJTU-DV-Pump part1/part2 (CC BY 4.0)",
                "class_order": list(CLASSES),
                "target": "current pump support-bearing condition",
                "selection": "48 released recordings: four classes × two speeds × six camera configurations",
                "analysis_interval_seconds": [START_US / 1e6, END_US / 1e6],
                "frame_bin_ms": BIN_US / 1000,
                "preprocessing": "event polarity counts at 64×48, signed log contrast, OpenCV 64×64 HOG",
                "split": "all 20 Hz recordings train; all 30 Hz recordings test; evaluate by recording, averaging frame probabilities",
                "train_runs": 24, "test_runs": 24, "frames_per_run": FRAME_COUNT,
                "source_recordings": provenance,
                "dummy_run_level": metrics(yt_dummy, yp_dummy),
                "random_forest_run_level": metrics(yt, yp),
                "held_out_camera_configuration_run_level": metrics(np.asarray(config_true), np.asarray(config_pred)),
                "shuffled_training_run_labels_control": metrics(yt, shuffled_pred),
                "limitations": [
                    "Event frames show bearing-housing motion; they are not RGB photographs of visible damage.",
                    "Both speeds come from one laboratory pump rig, not independent mining pumps.",
                    "Only the first two seconds of each recording were processed; ZIP-member CRC was not verified from the full member.",
                    "Sensor and event recordings are from different rigs and are compared by fault taxonomy, not paired sample or fused prediction.",
                    "The four-class target describes current condition, not time-to-repair or remaining useful life.",
                ],
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot_confusion(result["random_forest_run_level"]["confusion_matrix"])
            print(f"Saved {RESULT} and {FIGURE}")
            print(f"Run-level test macro F1: {result['random_forest_run_level']['macro_f1']:.4f}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # compare: sensor vs camera report and figure
    # (formerly Models/pump_compare_modalities.py)
    # ==================================================================================================

    def compare(run: bool = True) -> dict:
        """Report aligned bearing-fault targets without pairing unrelated source rows.

        Both branches have recording-held-out tests; different rigs and shift tests mean
        their accuracy values are descriptive and cannot establish modality superiority.
        """


        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import binomtest


        CAPSTONE = ROOT
        SENSOR = CAPSTONE / "Results" / "pump" / "pump_bearing_independent.json"
        SENSOR_DIAGNOSTIC = CAPSTONE / "Results" / "pump" / "pump_bearing_exploratory.json"
        VISUAL = CAPSTONE / "Results" / "pump" / "pump_event_visual.json"
        REPORT = CAPSTONE / "Results" / "pump" / "pump_sensor_visual_comparison.md"
        FIGURE = CAPSTONE / "Figures" / "pump" / "pump_sensor_visual_confusions.png"
        CLASSES = ("healthy", "inner", "outer", "ball")


        def fmt(value: float) -> str:
            return f"{value:.3f}"


        def metric_line(name: str, unit: str, record: dict) -> str:
            return (f"| {name} | {unit} | {fmt(record['accuracy'])} | "
                    f"{fmt(record['macro_precision'])} | {fmt(record['macro_recall'])} | "
                    f"{fmt(record['macro_f1'])} |")


        def per_class(lines: list[str], record: dict, unit: str) -> None:
            lines.extend([f"| Condition | Precision | Recall | F1 | {unit} |",
                          "|---|---:|---:|---:|---:|"])
            for label in CLASSES:
                row = record["per_class"][label]
                lines.append(f"| {label} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
                             f"{fmt(row['f1'])} | {row['support']} |")


        def matrix_table(lines: list[str], title: str, record: dict) -> None:
            lines.extend([f"### {title}", "",
                          "| Actual \\ predicted | Healthy | Inner | Outer | Ball |",
                          "|---|---:|---:|---:|---:|"])
            for label, row in zip(CLASSES, record["confusion_matrix"]):
                lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
            lines.append("")


        def draw_matrices(records: list[dict], titles: list[str]) -> None:
            figure, axes = plt.subplots(1, 4, figsize=(19, 5.7))
            for axis, record, title in zip(axes, records, titles):
                array = np.asarray(record["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(4), CLASSES, rotation=30)
                axis.set_yticks(range(4), CLASSES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(title)
                for i in range(4):
                    for j in range(4):
                        axis.text(j, i, str(array[i, j]), ha="center", va="center",
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle("Separate recording-level evaluations — not a paired or controlled modality trial", fontsize=13)
            figure.subplots_adjust(left=0.05, right=0.99, bottom=0.20, top=0.84, wspace=0.40)
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=160)
            plt.close(figure)


        def main() -> None:
            sensor = json.loads(SENSOR.read_text(encoding="utf-8"))
            diagnostic = json.loads(SENSOR_DIAGNOSTIC.read_text(encoding="utf-8"))
            visual = json.loads(VISUAL.read_text(encoding="utf-8"))
            if tuple(sensor["class_order"]) != CLASSES or tuple(visual["class_order"]) != CLASSES:
                raise ValueError("Sensor and visual results must use the same four fault classes")
            if sensor["status"] != "independent_recording_cross_speed_profile_evaluation":
                raise ValueError("Sensor result lacks a full-recording holdout")
            if visual["status"] != "cross_speed_independent_recording_evaluation":
                raise ValueError("Visual result lacks a full-recording holdout")
            if diagnostic["independent_run_accuracy"] is not None:
                raise ValueError("Original Pump sensor score must remain a same-file diagnostic")
            sm = sensor["random_forest_run_level"]
            vm = visual["random_forest_run_level"]
            sp = sensor["held_out_speed_profile_run_level"]
            vc = visual["held_out_camera_configuration_run_level"]
            for name, record, expected in (("sensor primary", sm, 24), ("visual primary", vm, 24),
                                           ("sensor profile folds", sp, 48), ("visual camera folds", vc, 48)):
                if sum(map(sum, record["confusion_matrix"])) != expected:
                    raise ValueError(f"Incorrect recording-level confusion support: {name}")
            sensor_correct = sum(sm["confusion_matrix"][i][i] for i in range(4))
            visual_correct = sum(vm["confusion_matrix"][i][i] for i in range(4))
            sensor_ci = binomtest(sensor_correct, 24).proportion_ci(method="exact")
            visual_ci = binomtest(visual_correct, 24).proportion_ci(method="exact")

            lines = [
                "# Pump support-bearing sensor–visual fault comparison",
                "",
                "## What is—and is not—being compared",
                "",
                "The models predict the same four **present-condition** labels: healthy, inner-race, outer-race, and rolling-element/ball fault. The sensor input is real vibration from a University of Ottawa **bearing test rig**, used as a mechanically comparable *pump support-bearing proxy*. The visual input is real event-camera motion from the XJTU **pump-bearing housing**. The rigs, physical bearings, samples and capture processes differ. No sensor recording is matched to, joined with, or fused with an image recording.",
                "",
                "Both primary tests now hold out **entire source recordings**, with 24 train and 24 test recordings and six test recordings per class. That closes the earlier same-file sensor-window leakage gap. It does **not** make this a controlled modality-superiority trial: the sensor split transfers across speed *patterns* (A/B to C/D), while the visual split transfers across fixed speeds (20 to 30 Hz). Scores are descriptive evidence for each branch; **do not subtract them, rank modalities, or call the difference a fusion gain**.",
                "",
                "| Branch | Source reality | Training/test recordings | Test unit | Physical-machine holdout? |",
                "|---|---|---:|---|---|",
                "| Vibration | University of Ottawa laboratory rolling-bearing rig, **not a pump** | 24 / 24 | One 10 s recording | No; separate runs, shared rig and possibly same fault specimen |",
                "| Event camera / OpenCV | XJTU laboratory pump support-bearing housing, **not a mining pump** | 24 / 24 | One event recording | No; separate runs, shared rig |",
                "",
                "## Sensor branch: recording-held-out test",
                "",
                "| Evaluation | Test unit | Accuracy | Macro precision | Macro recall | Macro F1 |",
                "|---|---|---:|---:|---:|---:|",
                metric_line("A/B speed profiles train → C/D test", "24 distinct recordings", sm),
                metric_line("Four-fold held-out speed profile", "48 fold-held-out run predictions", sp),
                metric_line("Most-frequent-class control", "Same 24 primary test runs", sensor["dummy_run_level"]),
                metric_line("Shuffled training-run labels", "Same 24 primary test runs", sensor["shuffled_training_run_labels_control"]),
                "",
                f"Primary test: **{sensor_correct}/24** correctly classified; an exploratory exact 95% binomial interval is **{sensor_ci.low:.3f}–{sensor_ci.high:.3f}**. It treats run outcomes as independent, which may be optimistic on one rig/possibly shared fault specimens, and does not include uncertainty from changing pumps or sites.",
                "",
                "### Sensor per-class results on held-out C/D recordings",
                "",
            ]
            per_class(lines, sm, "Test runs")
            lines.extend([
                "",
                "## Visual branch: recording-held-out test",
                "",
                "| Evaluation | Test unit | Accuracy | Macro precision | Macro recall | Macro F1 |",
                "|---|---|---:|---:|---:|---:|",
                metric_line("20 Hz train → 30 Hz test", "24 distinct recordings", vm),
                metric_line("Six-fold held-out camera configuration", "48 fold-held-out run predictions", vc),
                metric_line("Most-frequent-class control", "Same 24 primary test runs", visual["dummy_run_level"]),
                metric_line("Shuffled training-run labels", "Same 24 primary test runs", visual["shuffled_training_run_labels_control"]),
                "",
                f"Primary test: **{visual_correct}/24** correctly classified; an exploratory exact 95% binomial interval is **{visual_ci.low:.3f}–{visual_ci.high:.3f}**. It treats run outcomes as independent despite one shared rig; the harder camera-configuration check is not a second independent dataset.",
                "",
                "### Visual per-class results on held-out 30 Hz recordings",
                "",
            ])
            per_class(lines, vm, "Test runs")
            lines.extend(["", "### Visual per-class sensitivity to held-out camera configurations", ""])
            per_class(lines, vc, "Fold-held-out runs")
            lines.extend([
                "",
                "## Confusion matrices",
                "",
                "Rows are actual classes and columns predicted classes, ordered healthy / inner / outer / ball. The secondary fold matrices reuse the same 48 source recordings across different training fits; they are sensitivity checks, not extra independent tests.",
                "",
            ])
            matrix_table(lines, "Sensor: held-out C/D speed-profile recordings", sm)
            matrix_table(lines, "Visual: held-out 30 Hz event recordings", vm)
            matrix_table(lines, "Sensor: held-out speed-profile folds", sp)
            matrix_table(lines, "Visual: held-out camera-configuration folds", vc)
            lines.extend([
                "## Method and provenance",
                "",
                "- **Sensor source:** [Huang and Baddour, University of Ottawa bearing vibration V2](https://data.mendeley.com/datasets/v43hmbwxpm/2), DOI 10.17632/v43hmbwxpm.2, CC BY 4.0. Four single-condition classes, four speed profiles and three trials per profile yield 48 source MAT files. The [verified intake record](pump_uottawa_intake.json) contains publisher SHA-256 values for every file. The compound-fault fifth class is excluded.",
                "- **Sensor preprocessing:** only accelerometer `Channel_1` at 200 kHz is used; encoder `Channel_2` and file names are not predictors. Each 10 s run becomes 100 non-overlapping 0.1 s windows, with 16 time/spectral features. A fixed random forest is fit only on A/B runs. Window probabilities are averaged before **one prediction per held-out C/D recording**; no window from a test recording appears in training. Hyperparameters were not selected on the test runs.",
                "- **Visual source and unchanged experiment:** [XJTU-DV-Pump](https://zenodo.org/records/17447083), CC BY 4.0. The first two seconds of each of 48 relevant released recordings become eight 250 ms polarity-count event images. OpenCV HOG feeds a fixed random forest trained on 20 Hz runs; average frame probabilities produce one prediction per separate 30 Hz recording. These are event-camera motion images, **not RGB photographs of visible surface damage**. Full-file CRCs were not checked because only bounded intervals were read remotely.",
                "- **Original pump-vibration source:** [Jiang centrifugal-pump multi-fault V1](https://data.mendeley.com/datasets/63gjxpn7s6/1) was retained as a pump-domain pipeline diagnostic, but it has one file per selected class. Its within-file window score is **excluded from the independent comparison**. The Ottawa rig supplies repeat-run evidence at the same bearing-fault taxonomy, not direct pump-domain validation.",
                "- **Limits:** none of these evaluations holds out a new physical pump or mine. The Ottawa data do not establish independent fault-bearing specimens across all trials. Different source rigs and operating shifts preclude a causal claim that one modality is better. No matching timestamps, sample-level pairing, multimodal fusion, maintenance dates, failure horizon, or remaining-useful-life target exist.",
                "",
                "## Professor-requirement decision",
                "",
                "**Yes, for the professor's explicitly allowed component/fault-level comparison using sufficiently similar machines:** real vibration and real event-camera images both classify healthy/inner/outer/ball bearing conditions, and both now have recording-held-out Accuracy, Precision, Recall, F1, per-class results and confusion matrices. **No**, if the requirement means paired sensor-plus-camera observations from the *same physical mining pump*, a controlled modality-superiority experiment, image–sensor fusion, or prediction of a future repair date. The sensor proxy is a bearing rig, so this is a defensible component-level Pump proof of concept, **not validated mining-pump deployment**. Confirm that scope with the professor before describing the project as fully predictive maintenance.",
                "",
                f"![Separate recording-level confusion matrices](../../Figures/pump/{FIGURE.name})",
                "",
            ])
            REPORT.write_text("\n".join(lines), encoding="utf-8")
            draw_matrices([sm, vm, sp, vc],
                          ["Sensor: held-out C/D runs", "Event camera: held-out 30 Hz",
                           "Sensor: held-out profiles", "Event camera: held-out configs"])
            print(f"Saved {REPORT} and {FIGURE}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # original-xlsx: extra: original centrifugal-pump XLSX diagnostic (sensor only)
    # (formerly Models/pump_bearing_baseline.py)
    # ==================================================================================================

    def original_xlsx(run: bool = True) -> dict:
        """Exploratory pump-bearing vibration baseline for the locked P1 comparison.

        Four official spreadsheets contain one continuous recording per condition. A
        within-recording split is useful for a code/feature smoke test, but its score
        is *not* independent-run, independent-pump, or image-versus-sensor accuracy.
        """


        import hashlib
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from scipy.stats import kurtosis, skew
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


        CAPSTONE = ROOT
        RAW = CAPSTONE / "Raw_Data" / "pump_bearing"
        OUTPUT = CAPSTONE / "Results" / "pump" / "pump_bearing_exploratory.json"
        SAMPLE_RATE_HZ = 2_000
        WINDOW_SAMPLES = 1_024  # 0.512 seconds; non-overlapping
        CLASSES = ("healthy", "inner", "outer", "ball")

        # Source: https://data.mendeley.com/datasets/63gjxpn7s6/1 (CC BY 4.0).
        # These are the four single-fault/healthy files, not compound impeller faults.
        SOURCES = {
            "healthy": ("0_正常_00000", "a16aa7cedbd9c471bbb29ae152821177d8bb94d2956a6d25604ff211d2a3d6bf"),
            "inner": ("1_内圈_10000", "b8f63a3925186c78c3100cacc15ce4a12755333f6fa3064da0948cc646843596"),
            "outer": ("2_外圈_20000", "1ef1fbdc195d7c7a66d34b93804c2604c529c80bb3d0e287cb5daed4f4e29307"),
            "ball": ("3_滚动体_30000", "49d7ef08b515d092931a8d7d212d686391a855470a713b974a8de8f4337af279"),
        }


        def read_signal(label: str) -> np.ndarray:
            sheet, expected_hash = SOURCES[label]
            path = RAW / f"{label}.xlsx"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; see the capstone README for the four official download links")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Unexpected source checksum for {path.name}: {actual_hash}")

            # The first worksheet row is a numeric observation, NOT a column heading.
            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                if workbook.sheet_names != [sheet]:
                    raise ValueError(f"Unexpected worksheet in {path.name}: {workbook.sheet_names}")
                frame = load_excel(workbook, sheet_name=sheet, header=None)
            if frame.shape[1] != 1:
                raise ValueError(f"Expected one vibration channel in {path.name}, found {frame.shape[1]}")
            signal = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(signal).all() or len(signal) < 10 * WINDOW_SAMPLES:
                raise ValueError(f"Non-finite or insufficient vibration samples in {path.name}")
            return signal


        def window_features(window: np.ndarray) -> list[float]:
            centered = window - window.mean()
            spectrum = np.abs(np.fft.rfft(centered)) ** 2
            frequencies = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE_HZ)
            total_power = float(spectrum.sum())
            rms = float(np.sqrt(np.mean(centered**2)))
            bands = ((0, 250), (250, 500), (500, 1_000))
            return [
                float(window.mean()),
                float(window.std()),
                rms,
                float(np.ptp(window)),
                float(np.max(np.abs(centered)) / rms) if rms else 0.0,
                float(skew(window)),
                float(kurtosis(window)),
                float(np.sum(frequencies * spectrum) / total_power) if total_power else 0.0,
                float(frequencies[int(np.argmax(spectrum[1:])) + 1]),
                *[float(spectrum[(frequencies >= low) & (frequencies < high)].sum() / total_power)
                  if total_power else 0.0 for low, high in bands],
            ]


        def summarize(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
            prediction = model.predict(x_test)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, prediction, labels=CLASSES, zero_division=0
            )
            return {
                "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
                "macro_precision": round(float(np.mean(precision)), 4),
                "macro_recall": round(float(np.mean(recall)), 4),
                "macro_f1": round(float(np.mean(f1)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, prediction)), 4),
                "confusion_matrix": confusion_matrix(y_test, prediction, labels=CLASSES).tolist(),
                "per_class": {
                    label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                            "f1": round(float(f), 4), "support": int(n)}
                    for label, p, r, f, n in zip(CLASSES, precision, recall, f1, support)
                },
            }


        def main() -> None:
            features = []
            labels = []
            partitions = []
            source_summary = {}
            for label in CLASSES:
                signal = read_signal(label)
                count = len(signal) // WINDOW_SAMPLES
                windows = signal[:count * WINDOW_SAMPLES].reshape(count, WINDOW_SAMPLES)
                # Chronological blocks with a 20% unused middle gap prevent adjacent
                # windows from straddling the train/test boundary. They do NOT remove
                # the shared-run confound: every class still has just one source file.
                train_end = int(count * 0.6)
                test_start = int(count * 0.8)
                if train_end < 10 or count - test_start < 5:
                    raise ValueError(f"Too few windows for the declared split: {label}")
                features.extend(window_features(window) for window in windows)
                labels.extend([label] * count)
                partitions.extend(["train"] * train_end + ["unused_gap"] * (test_start - train_end) + ["test"] * (count - test_start))
                source_summary[label] = {
                    "samples": int(len(signal)),
                    "seconds": round(len(signal) / SAMPLE_RATE_HZ, 3),
                    "nonoverlapping_windows": count,
                    "train_windows": train_end,
                    "unused_gap_windows": test_start - train_end,
                    "test_windows": count - test_start,
                    "sha256": SOURCES[label][1],
                }

            x = np.asarray(features, dtype=np.float64)
            y = np.asarray(labels)
            partition = np.asarray(partitions)
            if not np.isfinite(x).all():
                raise ValueError("Extracted features contain non-finite values")
            x_train, y_train = x[partition == "train"], y[partition == "train"]
            x_test, y_test = x[partition == "test"], y[partition == "test"]

            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(x_train, y_train)
            forest = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            forest.fit(x_train, y_train)
            feature_names = ["mean", "std", "rms", "peak_to_peak", "crest_factor", "skew", "kurtosis", "spectral_centroid_hz", "dominant_frequency_hz", "power_0_250", "power_250_500", "power_500_1000"]
            result = {
                "status": "exploratory_same_file_temporal_split_only",
                "independent_run_accuracy": None,
                "independent_run_accuracy_reason": "One sensor recording per class; no independent fault run can be held out.",
                "dataset": "Mendeley 10.17632/63gjxpn7s6.1; four selected single-condition vibration files",
                "source_license": "CC BY 4.0",
                "target": "current pump-bearing condition; healthy/inner/outer/ball",
                "class_order": list(CLASSES),
                "source_summary": source_summary,
                "sampling_rate_hz": SAMPLE_RATE_HZ,
                "window_samples": WINDOW_SAMPLES,
                "feature_names": feature_names,
                "split": "first 60% train, middle 20% unused gap, final 20% test within each source file",
                "dummy": summarize(dummy, x_test, y_test),
                "random_forest": summarize(forest, x_test, y_test),
                "limitations": [
                    "Only one source recording per class: test windows share the same physical run as training windows.",
                    "These metrics are a pipeline smoke test, not independent-pump/fault generalization or deployment accuracy.",
                    "No visual data are trained here; XJTU-DV-Pump event recordings remain remote and are not paired with these samples.",
                    "This classifies an existing condition; it does not predict a future repair time or remaining useful life.",
                ],
            }
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved exploratory result: {OUTPUT}")
            print(f"RF macro F1 {result['random_forest']['macro_f1']:.4f}; scientifically valid independent-run score: unavailable")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # uci-hydraulic: extra: UCI hydraulic pump leakage (sensor only)
    # (formerly Preprocessing/hydraulic_baseline.py)
    # ==================================================================================================

    def uci_hydraulic(run: bool = True) -> dict:
        """Reproduce a pump-leakage baseline on the UCI hydraulic test rig.

        This is a subsystem benchmark, not data recorded on a mining vehicle. The
        original ZIP stays under ignored Raw_Data. Features and results are derived
        locally; no raw observations are committed to Git.

        Run from the repository root:
            .venv/bin/python Models/pump.py uci-hydraulic
        """


        import hashlib
        import json
        import zipfile
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        PROJECT = ROOT
        ARCHIVE = PROJECT / "Raw_Data/uci_hydraulic/condition_monitoring_of_hydraulic_systems.zip"
        FEATURES = PROJECT / "Processed_Data/uci_hydraulic_cycle_features.npz"
        RESULTS = PROJECT / "Results/pump/uci_hydraulic_pump_leakage.json"
        FIGURE = PROJECT / "Figures/pump/uci_hydraulic_pump_leakage.png"
        EXPECTED_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
        SENSORS = (
            "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
            "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
        )
        LABELS = (0, 1, 2)


        def checked_archive() -> zipfile.ZipFile:
            if not ARCHIVE.is_file():
                raise FileNotFoundError(
                    f"Missing {ARCHIVE}. Download the UCI dataset ZIP as described in the project README."
                )
            digest = hashlib.sha256()
            with ARCHIVE.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != EXPECTED_SHA256:
                raise ValueError("UCI ZIP checksum differs from the inspected source; verify the new archive.")
            return open_archive(ARCHIVE)


        def extract_cycle_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
            with checked_archive() as source:
                profile = np.loadtxt(source.open("profile.txt"), delimiter="\t", dtype=np.int64)
                if profile.shape != (2205, 5):
                    raise ValueError(f"Unexpected profile shape: {profile.shape}")
                feature_arrays = []
                names = []
                sample_widths = {}
                for sensor in SENSORS:
                    values = np.loadtxt(source.open(f"{sensor}.txt"), delimiter="\t", dtype=np.float64)
                    if values.ndim != 2 or values.shape[0] != profile.shape[0]:
                        raise ValueError(f"Unexpected shape for {sensor}: {values.shape}")
                    sample_widths[sensor] = int(values.shape[1])
                    feature_arrays.extend((
                        values.mean(axis=1),
                        values.std(axis=1),
                        np.sqrt(np.mean(np.square(values), axis=1)),
                        np.ptp(values, axis=1),
                    ))
                    names.extend(f"{sensor}_{stat}" for stat in ("mean", "std", "rms", "range"))
                    del values
            x_all = np.column_stack(feature_arrays)
            if not np.isfinite(x_all).all():
                raise ValueError("Non-finite feature values require inspection before modelling.")
            # UCI profile column 5: 0 = stable, 1 = transient/not yet at steady state.
            stable = profile[:, 4] == 0
            x = x_all[stable]
            y = profile[stable, 2]  # Pump leakage: 0 none, 1 weak, 2 severe.
            states, groups = np.unique(profile[stable, :4], axis=0, return_inverse=True)
            if set(np.unique(y)) != set(LABELS):
                raise ValueError("The stable-cycle subset does not contain all three pump classes.")
            details = {
                "source_cycle_count": int(profile.shape[0]),
                "stable_cycle_count": int(stable.sum()),
                "transient_cycle_count": int((~stable).sum()),
                "sensor_channel_count": len(SENSORS),
                "samples_per_cycle_by_channel": sample_widths,
                "feature_count": int(x.shape[1]),
                "missing_or_nonfinite_feature_count": int((~np.isfinite(x)).sum()),
                "full_condition_state_group_count": int(states.shape[0]),
                "class_counts": {str(label): int((y == label).sum()) for label in LABELS},
            }
            FEATURES.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(FEATURES, x=x.astype(np.float32), y=y, groups=groups, names=np.array(names))
            return x, y, groups, names, details


        def split_by_condition(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            indices = np.arange(len(y))
            first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            train_val, test = next(first.split(indices, y, groups))
            second = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
            train_rel, val_rel = next(second.split(train_val, y[train_val], groups[train_val]))
            train, val = train_val[train_rel], train_val[val_rel]
            for left, right in ((train, val), (train, test), (val, test)):
                if set(groups[left]) & set(groups[right]):
                    raise AssertionError("A full condition state appears in two splits.")
            for part in (train, val, test):
                if set(np.unique(y[part])) != set(LABELS):
                    raise ValueError("A split lacks a pump-leakage class; revise the group split.")
            return train, val, test


        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=LABELS, zero_division=0
            )
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "per_class": {
                    str(label): {
                        "precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i]), "support": int(support[i]),
                    }
                    for i, label in enumerate(LABELS)
                },
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            }


        def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, details: dict) -> dict:
            train, val, test = split_by_condition(y, groups)
            candidates = {
                "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
                "logistic_regression": make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
                ),
                "random_forest": make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(
                        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                        random_state=42, n_jobs=-1,
                    ),
                ),
            }
            validation = {}
            for name, model in candidates.items():
                model.fit(x[train], y[train])
                validation[name] = metrics(y[val], model.predict(x[val]))
            selected = max(
                (name for name in candidates if name != "dummy_most_frequent"),
                key=lambda name: (validation[name]["macro_f1"], validation[name]["balanced_accuracy"]),
            )
            train_val = np.concatenate((train, val))
            selected_model = candidates[selected].fit(x[train_val], y[train_val])
            test_metrics = metrics(y[test], selected_model.predict(x[test]))
            dummy = DummyClassifier(strategy="most_frequent").fit(x[train_val], y[train_val])
            result = {
                "dataset": "UCI Condition monitoring of hydraulic systems",
                "doi": "10.24432/C5CW21",
                "provenance": "Experimental hydraulic test rig; not mining-machine telemetry",
                "task": "Three-class current pump-leakage condition (none, weak, severe); not future repair prediction",
                "feature_method": "Per-cycle mean, standard deviation, RMS and range for each of 17 channels",
                "quality_filter": "Retain stable_flag = 0 only",
                "split_method": "StratifiedGroupKFold; hold out complete combinations of cooler, valve, pump and accumulator conditions",
                "selection_rule": "Maximum validation macro F1 among logistic regression and random forest",
                "data": details,
                "splits": {
                    part_name: {
                        "cycles": int(len(part)),
                        "full_condition_state_groups": int(len(np.unique(groups[part]))),
                        "class_counts": {str(label): int((y[part] == label).sum()) for label in LABELS},
                    }
                    for part_name, part in (("train", train), ("validation", val), ("test", test))
                },
                "validation": validation,
                "selected_model": selected,
                "test_selected": test_metrics,
                "test_dummy": metrics(y[test], dummy.predict(x[test])),
                "limitations": [
                    "One controlled test rig; no independent mining-equipment validation.",
                    "Rows are operating cycles, not separate physical machines.",
                    "Condition labels are contemporaneous; there is no future maintenance outcome or RUL target.",
                ],
            }
            return result


        def save_figure(result: dict) -> None:
            classes = ["No leakage", "Weak", "Severe"]
            counts = [result["data"]["class_counts"][str(label)] for label in LABELS]
            matrix = np.array(result["test_selected"]["confusion_matrix"])
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(classes, counts, color="#386A8C")
            axes[0].set_title("Stable-cycle class counts")
            axes[0].set_ylabel("Cycles")
            axes[1].imshow(matrix, cmap="Blues")
            axes[1].set_title(f"Test confusion matrix: {result['selected_model']}")
            axes[1].set_xticks(range(3), classes, rotation=25, ha="right")
            axes[1].set_yticks(range(3), classes)
            axes[1].set_xlabel("Predicted condition")
            axes[1].set_ylabel("Actual condition")
            for row in range(3):
                for col in range(3):
                    axes[1].text(
                        col, row, str(matrix[row, col]), ha="center", va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                    )
            fig.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE, dpi=160)
            plt.close(fig)


        def main() -> None:
            x, y, groups, names, details = extract_cycle_features()
            result = evaluate(x, y, groups, details)
            result["feature_names"] = names
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            save_figure(result)
            print(f"Stable cycles: {details['stable_cycle_count']} / {details['source_cycle_count']}")
            print(f"Selected: {result['selected_model']}")
            print(f"Test macro F1: {result['test_selected']['macro_f1']:.3f}")
            print(f"Test balanced accuracy: {result['test_selected']['balanced_accuracy']:.3f}")
            print(f"Wrote {RESULTS} and {FIGURE}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # espset: extra: ESPset V3 pump vibration (sensor only)
    # (formerly Models/espset_baseline.py)
    # ==================================================================================================

    def espset(run: bool = True) -> dict:
        """Group-held-out fault baseline on ESPset V3's published feature file.

        ESPset records offshore oil-production pumps, not mining pumps. No images are
        paired with these sensor rows. Run from the repository root with the project's
        Python environment after placing the official features.zip under Raw_Data.
        """


        import hashlib
        import json
        import zipfile
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
            precision_score, recall_score,
        )
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.tree import DecisionTreeClassifier


        PROJECT = ROOT
        ARCHIVE = PROJECT / "Raw_Data/espset_v3/features.zip"
        PROCESSED = PROJECT / "Processed_Data/espset_v3_features.npz"
        RESULTS = PROJECT / "Results/pump/espset_v3_binary_fault.json"
        FIGURE = PROJECT / "Figures/pump/espset_v3_binary_fault.png"
        EXPECTED_SHA256 = "b8214c10ba5b5b211c00a058f2fffa3ed7ccc46f67219b4769c907baec9a724d"
        FEATURE_COLUMNS = (
            "median(8,13)", "rms(98,102)", "median(98,102)",
            "peak1x", "peak2x", "a", "b",
        )


        def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
            if not ARCHIVE.is_file():
                raise FileNotFoundError(f"Missing {ARCHIVE}; see the project README for the official link.")
            digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256:
                raise ValueError("ESPset features.zip checksum differs from the inspected V3 file.")
            with open_archive(ARCHIVE) as source:
                frame = load_csv(source.open("features.csv"), sep=";")
            required = {"id", "esp_id", "label", *FEATURE_COLUMNS}
            if frame.shape != (6032, 10) or set(frame.columns) != required:
                raise ValueError(f"Unexpected ESPset shape/columns: {frame.shape}, {frame.columns.tolist()}")
            if frame["id"].duplicated().any() or frame["esp_id"].isna().any():
                raise ValueError("ESPset IDs are missing or duplicated; inspect the source file.")
            x = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("ESPset contains missing or non-finite feature values.")
            y = (frame["label"] != "Normal").to_numpy(dtype=np.int64)
            groups = frame["esp_id"].to_numpy(dtype=np.int64)
            details = {
                "rows": int(len(frame)),
                "columns": frame.columns.tolist(),
                "distinct_pumps": int(frame["esp_id"].nunique()),
                "missing_values_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
                "original_condition_counts": {str(key): int(value) for key, value in frame["label"].value_counts().items()},
                "binary_counts": {"normal": int((y == 0).sum()), "abnormal": int((y == 1).sum())},
                "duplicate_feature_rows": int(frame[list(FEATURE_COLUMNS)].duplicated().sum()),
            }
            PROCESSED.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PROCESSED, x=x.astype(np.float32), y=y, esp_id=groups,
                                names=np.array(FEATURE_COLUMNS))
            return x, y, groups, details


        def split_by_pump(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            indexes = np.arange(len(y))
            outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_val, test = next(outer.split(indexes, y, groups))
            inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
            train_rel, val_rel = next(inner.split(train_val, y[train_val], groups[train_val]))
            train, val = train_val[train_rel], train_val[val_rel]
            for left, right in ((train, val), (train, test), (val, test)):
                if set(groups[left]) & set(groups[right]):
                    raise AssertionError("A physical pump occurs in more than one split.")
            for part in (train, val, test):
                if set(np.unique(y[part])) != {0, 1}:
                    raise ValueError("A split lacks normal or abnormal rows; revise the group split.")
            return train, val, test


        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_abnormal": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall_abnormal": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_abnormal": float(f1_score(y_true, y_pred, zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            }


        def main() -> None:
            x, y, groups, details = load_data()
            train, val, test = split_by_pump(y, groups)
            candidates = {
                "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
                "random_forest": RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                    random_state=42, n_jobs=-1,
                ),
            }
            validation = {}
            for name, model in candidates.items():
                model.fit(x[train], y[train])
                validation[name] = metrics(y[val], model.predict(x[val]))
            chosen = max(candidates, key=lambda name: (
                validation[name]["macro_f1"], validation[name]["recall_abnormal"]
            ))
            train_val = np.concatenate((train, val))
            model = candidates[chosen].fit(x[train_val], y[train_val])
            selected_test = metrics(y[test], model.predict(x[test]))
            dummy = DummyClassifier(strategy="most_frequent").fit(x[train_val], y[train_val])
            result = {
                "dataset": "ESPset V3 features.zip",
                "doi": "10.17632/m268jsw339.3",
                "provenance": "Real offshore oil-production pumps; not mining pumps",
                "task": "Current abnormal-vs-normal pump vibration condition; not future failure or RUL",
                "predictor_columns": list(FEATURE_COLUMNS),
                "excluded_from_predictors": ["id", "esp_id", "label"],
                "split_method": "Two-stage GroupShuffleSplit by esp_id; no pump appears in more than one split",
                "selection_rule": "Maximum validation macro F1, then abnormal recall",
                "data": details,
                "splits": {
                    name: {
                        "rows": int(len(part)),
                        "esp_ids": [int(item) for item in np.unique(groups[part])],
                        "normal": int((y[part] == 0).sum()),
                        "abnormal": int((y[part] == 1).sum()),
                    }
                    for name, part in (("train", train), ("validation", val), ("test", test))
                },
                "validation": validation,
                "selected_model": chosen,
                "test_selected": selected_test,
                "test_dummy": metrics(y[test], dummy.predict(x[test])),
                "limitations": [
                    "Only 11 source pumps; holdout performance is sensitive to which pumps are chosen.",
                    "The source is offshore oil production, not mining; transfer must be tested on mining pumps.",
                    "These seven features were supplied by the publisher; spectrum.zip was not required for this first baseline.",
                    "No timestamps, images, future repairs or remaining-useful-life labels are present in features.csv.",
                ],
            }
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            counts = [details["binary_counts"]["normal"], details["binary_counts"]["abnormal"]]
            matrix = np.array(selected_test["confusion_matrix"])
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].bar(["Normal", "Abnormal"], counts, color="#386A8C")
            axes[0].set_ylabel("Vibration tests")
            axes[0].set_title("ESPset V3 class counts")
            axes[1].imshow(matrix, cmap="Blues")
            axes[1].set_title(f"Held-out pumps: {chosen}")
            axes[1].set_xticks([0, 1], ["Normal", "Abnormal"])
            axes[1].set_yticks([0, 1], ["Normal", "Abnormal"])
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Actual")
            for row in range(2):
                for col in range(2):
                    axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center",
                                 color="white" if matrix[row, col] > matrix.max() / 2 else "black")
            fig.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE, dpi=160)
            plt.close(fig)
            print(f"Inspected {details['rows']} tests from {details['distinct_pumps']} pumps")
            print(f"Selected: {chosen}; test macro F1: {selected_test['macro_f1']:.3f}")
            print(f"Test abnormal recall: {selected_test['recall_abnormal']:.3f}")
            print(f"Wrote {RESULTS} and {FIGURE}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # baselines: Week 5 baseline model comparison (majority class, logistic regression, decision tree, random forest)
    # ====================================================================================================

    def baselines(run: bool = True) -> dict:
        """Week 5: compare baseline model families on the same held-out tests as the main experiment.

        A majority-class control, logistic regression, a decision tree and a random forest are trained
        with settings fixed in advance and scored on exactly the same held-out test units. No model or
        setting is chosen from these test results; the main model of the experiment stays the same.
        """
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        RESULT = ROOT / "Results" / "pump" / "pump_baselines.json"
        FIGURE = ROOT / "Figures" / "pump" / "pump_baselines.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Decision tree": lambda: DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=42),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }
        SETTINGS = {
            "Majority class (control)": "always predicts the most common training class",
            "Logistic regression": "standardised features, C=1, class-balanced, lbfgs (multi-class)",
            "Decision tree": "class-balanced, min_samples_leaf=2",
            "Random forest": "250 trees, sqrt features, class-balanced subsample, min_samples_leaf=2",
        }

        def unit_predictions(model, x, y, units):
            """One prediction per test unit (recording or run): average the model's probabilities over its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(model.classes_[int(np.argmax(model.predict_proba(x[rows]).mean(axis=0)))])
            return actual, predicted

        def evaluate(x, y, units, folds, classes):
            """`folds` are boolean test masks; every test unit is predicted by a model that never saw it."""
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds:
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model, x[test], y[test], units[test])
                    actual += list(a)
                    predicted += list(p)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predicted, labels=classes, zero_division=0)
                scores[name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_precision": round(float(precision.mean()), 4),
                    "macro_recall": round(float(recall.mean()), 4),
                    "macro_f1": round(float(f1.mean()), 4),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
                }
            return scores

        def plot(result):
            branches, names = list(result["branches"]), list(MODELS)
            width = 0.8 / len(names)
            figure, axis = plt.subplots(figsize=(2.2 + 2.8 * len(branches), 4.0))
            for i, name in enumerate(names):
                values = [result["branches"][b]["models"][name]["macro_f1"] for b in branches]
                bars = axis.bar(np.arange(len(branches)) + (i - (len(names) - 1) / 2) * width, values, width, label=name)
                axis.bar_label(bars, fmt="%.2f", fontsize=7)
            axis.set_xticks(range(len(branches)), branches)
            axis.set_ylim(0, 1.15)
            axis.set_ylabel("Macro F1 on held-out test")
            axis.set_title(result["title"], fontsize=10)
            axis.legend(fontsize=8, ncol=2, loc="upper right")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=140)
            plt.close(figure)

        def main():
            branches = {}
            s = sensor(run=False)
            if s["CACHE"].is_file():
                with load_numpy(s["CACHE"], allow_pickle=False) as saved:
                    x, y, profile, units = saved["x"], saved["y"], saved["profile"], saved["run"]
            else:
                x, y, profile, units, _ = s["preprocess"]()
            branches["Sensor: vibration"] = {
                "test": "train on speed profiles A/B; test on 24 separate C/D recordings",
                "classes": list(s["CLASSES"]),
                "models": evaluate(x, y, units, [np.isin(profile, ["C", "D"])], list(s["CLASSES"]))}
            c = camera(run=False)
            if c["CACHE"].is_file():
                with load_numpy(c["CACHE"], allow_pickle=False) as saved:
                    x, y, speed, units = saved["x"], saved["y"], saved["speed"], saved["run"]
            else:
                x, y, speed, units, _ = c["preprocess"]()
            branches["Camera: event camera"] = {
                "test": "train on 24 recordings at 20 Hz; test on 24 separate recordings at 30 Hz",
                "classes": list(c["CLASSES"]),
                "models": evaluate(x, y, units, [speed == 30], list(c["CLASSES"]))}
            result = {
                "title": "Pump: baseline models on held-out recordings",
                "purpose": "Week 5 baseline model comparison on the same held-out tests as the main experiment",
                "model_settings": SETTINGS,
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "branches": branches,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(result)
            for branch, block in branches.items():
                for name, score in block["models"].items():
                    print(f"{branch:<24} {name:<26} accuracy {score['accuracy']:.3f}  macro F1 {score['macro_f1']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # export-images: Week 6, save the camera images as named files (professor remark)
    # ====================================================================================================

    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/pump/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "pump"
        OUT = ROOT / "Images" / MACHINE
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5
        NAME = "t{k:02d}"
        FRAMES = SECONDS * 1_000_000 // BIN_US

        def event_frames(stream):
            """Bin events using the configured duration and frame interval; channels preserve polarity."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})

        def export():
            rows = []
            folders = {"healthy": "Healthy", "inner": "Inner", "outer": "Outer", "ball": "Ball"}
            for hz, part in (("20Hz", "part1"), ("30Hz", "part2")):
                with open_archive(ROOT / "Raw_Data" / "xjtu_dv_pump" / f"XJTU-DV-Pump_{part}.zip") as archive:
                    for label, folder in folders.items():
                        members = sorted(i.filename for i in archive.infolist()
                                         if i.filename.startswith(f"XJTU-DV-Pump/Data/{folder}/{hz[:2]}") and i.filename.endswith(".dat"))
                        for member in members:
                            setup = "_".join(Path(member).stem.split("_")[-2:])
                            unit = f"{hz}_{label}_{setup}"
                            with archive.open(member) as stream:
                                counts = event_frames(stream)
                            for k, frame in enumerate(counts):
                                if frame.sum() > 0:
                                    write(rows, label, unit, hz, f"{MACHINE}_{label}_{hz}_{setup}_" + NAME.format(k=k) + ".png", event_png(frame))
                            print(f"{unit}: done", flush=True)
            return rows


        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.add_argument("--dense-frames", action="store_true", help="125 ms frames over up to 30 seconds; separate denseframes output")
            args = parser.parse_args()
            nonlocal OUT, SECONDS, FRAMES, BIN_US, NAME
            if args.dense_frames:
                OUT = ROOT / "Images" / f"{MACHINE}_denseframes"
                SECONDS, BIN_US, NAME = 30, 125_000, "t{k:03d}"
                FRAMES = SECONDS * 1_000_000 // BIN_US
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # csv: Week 6, 4-class sensor model from a CSV feature table (professor remark: "csv accuracy 90 plus")
    # ====================================================================================================

    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "pump"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "pump" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "pump" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def build_table():
            s = sensor(run=False)
            if s["CACHE"].is_file():
                with load_numpy(s["CACHE"], allow_pickle=False) as saved:
                    x, y, profile, units = saved["x"], saved["y"], saved["profile"], saved["run"]
            else:
                x, y, profile, units, _ = s["preprocess"]()
            names = list(s["FEATURE_NAMES"])
            table = pd.DataFrame(x, columns=names)
            table.insert(0, "class", y)
            table.insert(0, "speed_profile", profile)
            table.insert(0, "unit", units)
            folds = lambda t: [t["speed_profile"].isin(["C", "D"]).to_numpy()]
            return (table, names, ["healthy", "inner", "outer", "ball"], folds,
                    "train on speed profiles A/B; test on 24 separate C/D recordings",
                    ["Ottawa bearing rig vibration; one row per 0.1 s window, 100 windows per recording"])


        INNER_GROUP = "unit"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.parse_args()
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn: Week 6, CNN with data augmentation on the named images (professor remark)
    # ====================================================================================================

    def _augment_event_batch(batch, generator):
        """Apply the fixed training augmentation without constructing another CNN pipeline."""
        import torch

        n = batch.shape[0]
        flip = torch.rand(n, generator=generator) < 0.5
        batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
        shifted = []
        for image in batch:
            dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
            shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
        batch = torch.stack(shifted)
        scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
        return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)



    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "pump"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "pump" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "pump" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "inner", "outer", "ball"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [2, 1]  # red = brighter events, green = darker events
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        augment = _augment_event_batch

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == "30Hz"], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "train on the 24 recordings at 20 Hz; test on the 24 separate recordings at 30 Hz",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn-pretrained: Week 6, transfer learning with a pretrained ResNet18 on the named images
    # ====================================================================================================

    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "pump"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "pump" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "pump" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "inner", "outer", "ball"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        augment = _augment_event_batch

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--dense-frames", action="store_true", help="use separately exported 125 ms full-duration frames")
            args = parser.parse_args()
            nonlocal IMAGES, RESULT, FIGURE
            if args.dense_frames:
                IMAGES = ROOT / "Images" / f"{MACHINE}_denseframes"
                RESULT = RESULT.with_name(RESULT.stem + "_denseframes.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_denseframes.png")
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                rows = list(csv_module.DictReader(handle))
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            for number, test in enumerate([groups == "30Hz"], 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "train on the 24 recordings at 20 Hz; test on the 24 separate recordings at 30 Hz",
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # train-final / predict: use the trained sensor model on any CSV
    # ====================================================================================================

    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "base" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "pump_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. B-C-1.mat")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["base"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "pump",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/pump_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of accelerometer values sampled at 200 kHz (at least 1.6 million values).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "pump_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            return sensor(run=False)["extract"](values)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, inner, outer, ball")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")

            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Proposal Week 6: advanced ensemble models (XGBoost / LightGBM comparison table)
    # ====================================================================================================

    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "pump"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: anomaly detection (proposal Week 7: Isolation Forest / One-Class SVM)
    # ====================================================================================================

    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "pump"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    COMMANDS = {
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
        "intake": intake,
        "sensor": sensor,
        "camera": camera,
        "compare": compare,
        "original-xlsx": original_xlsx,
        "uci-hydraulic": uci_hydraulic,
        "espset": espset,
        "baselines": baselines,
        "export-images": export_images,
        "csv": csv,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
    }
    RUN_ALL = ['sensor', 'camera', 'compare']


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"pump.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# HAUL_TRUCK MODEL IMPLEMENTATION
def _make_haul_truck():
    __doc__ = "Haul/Dump Truck: wheel-motor bearing, UODS vibration vs XJTU-Rotor event camera.\n\nAll of this machine's code in one file. The combined notebook\nNotebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.\n\nUsage:\n    python Models/haul_truck.py <command> [options]\n    python Models/haul_truck.py <command> --help\n\nCommands:\n    intake          download UODS vibration and the XJTU-Rotor zip\n    experiment      vibration and event-camera models, report and figure\n    baselines       Week 5: compare majority class, logistic regression, decision tree and random forest\n    export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)\n    csv             Week 6: 4-class sensor model from a CSV feature table\n    cnn             Week 6: CNN with data augmentation on the named images (run export-images first)\n    cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)\n    all             run experiment\n\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Haul/Dump Truck: wheel-motor bearing, UODS vibration vs XJTU-Rotor event camera.

    All of this machine's code in one file. The combined notebook
    Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

    Usage:
        python Models/haul_truck.py <command> [options]
        python Models/haul_truck.py <command> --help

    Commands:
        intake          download UODS vibration and the XJTU-Rotor zip
        experiment      vibration and event-camera models, report and figure
        baselines       Week 5: compare majority class, logistic regression, decision tree and random forest
        export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)
        csv             Week 6: 4-class sensor model from a CSV feature table
        cnn             Week 6: CNN with data augmentation on the named images (run export-images first)
        cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)
        all             run experiment

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import sys
    from pathlib import Path



    # ==================================================================================================
    # intake: download UODS vibration and the XJTU-Rotor zip
    # (formerly Preprocessing/haul_truck_intake.py)
    # ==================================================================================================

    def intake(run: bool = True) -> dict:
        """Acquire the Haul/Dump Truck wheel-motor bearing sources (matrix row B1).

        Sensor: University of Ottawa UODS-VAFDC (Mendeley Data, DOI 10.17632/y2px5tg92h.1),
        CC BY 4.0. Only bearings 1-10 are selected: their healthy stage-0 recordings and
        their final stage-2 inner-race (bearings 1-5) or outer-race (bearings 6-10)
        recordings. Ball faults are excluded because the publisher states they were
        recorded under no load, while healthy/inner/outer used 400 N. Stage 1
        (fault developing) has no XJTU counterpart and is also excluded.

        Visual: XJTU-DV-Rotor event-camera ZIP (Zenodo record 17447083), CC BY 4.0,
        downloaded in full and checked against Zenodo's published MD5.

        Both are laboratory rigs, not haul-truck data. Run with --sensor-only or
        --visual-only to fetch one source.
        """


        import argparse
        import hashlib
        import json
        import os
        import re
        import time
        from pathlib import Path
        from urllib.request import Request, urlopen


        CAPSTONE = ROOT
        SENSOR_RAW = CAPSTONE / "Raw_Data" / "uods_vafdc"
        VISUAL_RAW = CAPSTONE / "Raw_Data" / "xjtu_dv_rotor"
        MANIFEST = CAPSTONE / "Results" / "haul_truck" / "haul_truck_intake.json"
        MENDELEY = "https://data.mendeley.com/public-api/datasets/y2px5tg92h"
        FOLDERS = {"H": "1_Healthy", "I": "2_Inner_Race_Faults", "O": "3_Outer_Race_Faults"}
        NAME = re.compile(r"^([HIOB])_(\d+)_([012])\.mat$")
        BALL_FOLDERS = {"H": "1_Healthy", "B": "4_Ball_Faults"}
        ROTOR_URL = "https://zenodo.org/api/records/17447083/files/XJTU-DV-Rotor.zip/content"
        ROTOR_NAME = "XJTU-DV-Rotor.zip"
        ROTOR_SIZE = 2_812_106_658
        ROTOR_MD5 = "2dfed028950f3769ae62effa47faaccb"
        HEADERS = {"User-Agent": "PROJ613-capstone/1.0"}


        def get_json(url: str):
            with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
                return json.load(response)


        def selected(name: str) -> bool:
            match = NAME.fullmatch(name)
            if not match:
                return False
            code, bearing, stage = match.group(1), int(match.group(2)), int(match.group(3))
            if code == "H":
                return 1 <= bearing <= 10 and stage == 0
            if code == "I":
                return 1 <= bearing <= 5 and stage == 2
            if code == "O":
                return 6 <= bearing <= 10 and stage == 2
            return False


        def selected_ball(name: str) -> bool:
            """Optional 4th class: ball-fault bearings 11-15 (final stage 2) and their healthy stage-0 recordings.
            The publisher recorded ball faults with no load, while the other classes used 400 N."""
            match = NAME.fullmatch(name)
            if not match:
                return False
            code, bearing, stage = match.group(1), int(match.group(2)), int(match.group(3))
            return 11 <= bearing <= 15 and ((code == "H" and stage == 0) or (code == "B" and stage == 2))


        def sensor_inventory(ball: bool = False) -> list[dict]:
            folders, pick = (BALL_FOLDERS, selected_ball) if ball else (FOLDERS, selected)
            by_name = {item["name"]: item["id"] for item in get_json(f"{MENDELEY}/folders/1")}
            if not set(folders.values()).issubset(by_name):
                raise ValueError("Expected publisher fault folders are missing")
            items = []
            for code, folder in folders.items():
                members = get_json(f"{MENDELEY}/files?folder_id={by_name[folder]}&version=1&%24start=0&%24limit=1000")
                for member in members:
                    name = member["filename"]
                    if not pick(name):
                        continue
                    details = member["content_details"]
                    if not re.fullmatch(r"[a-f0-9]{64}", details.get("sha256_hash", "")):
                        raise ValueError(f"Missing publisher SHA-256 for {name}")
                    match = NAME.fullmatch(name)
                    items.append({
                        "name": name,
                        "class_code": code,
                        "bearing": int(match.group(2)),
                        "stage": int(match.group(3)),
                        "source_file_id": member["id"],
                        "source_sha256": details["sha256_hash"],
                        "source_size_bytes": int(details["size"]),
                        "url": details["download_url"] + "?version=1",
                    })
            counts = {code: sum(item["class_code"] == code for item in items) for code in folders}
            if counts != ({"H": 5, "B": 5} if ball else {"H": 10, "I": 5, "O": 5}):
                raise ValueError(f"Unexpected UODS selection: {counts}")
            return sorted(items, key=lambda item: (item["bearing"], item["stage"]))


        def digest(path: Path, algorithm: str) -> str:
            checksum = hashlib.new(algorithm)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    checksum.update(block)
            return checksum.hexdigest()


        def fetch_sensor(item: dict) -> str:
            destination = SENSOR_RAW / item["name"]
            if destination.is_file():
                if destination.stat().st_size != item["source_size_bytes"] or digest(destination, "sha256") != item["source_sha256"]:
                    raise ValueError(f"Existing file failed publisher hash check: {destination}")
                return "cached-and-verified"
            temporary = destination.with_suffix(".mat.part")
            for attempt in range(3):
                try:
                    with urlopen(Request(item["url"], headers=HEADERS), timeout=120) as response, temporary.open("wb") as out:
                        for block in iter(lambda: response.read(1 << 20), b""):
                            out.write(block)
                    if temporary.stat().st_size != item["source_size_bytes"] or digest(temporary, "sha256") != item["source_sha256"]:
                        raise ValueError(f"Publisher hash/size mismatch for {item['name']}")
                    os.replace(temporary, destination)
                    return "downloaded-and-verified"
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            raise AssertionError("Unreachable")


        def fetch_rotor() -> str:
            """Download the full Rotor ZIP, resuming a partial file, then check Zenodo's MD5."""
            VISUAL_RAW.mkdir(parents=True, exist_ok=True)
            destination = VISUAL_RAW / ROTOR_NAME
            if destination.is_file():
                if destination.stat().st_size != ROTOR_SIZE or digest(destination, "md5") != ROTOR_MD5:
                    raise ValueError(f"Existing Rotor ZIP failed the Zenodo MD5 check: {destination}")
                return "cached-and-verified"
            partial = destination.with_suffix(".zip.part")
            for attempt in range(20):
                done = partial.stat().st_size if partial.is_file() else 0
                if done >= ROTOR_SIZE:
                    break
                try:
                    request = Request(ROTOR_URL, headers={**HEADERS, "Range": f"bytes={done}-"})
                    with urlopen(request, timeout=120) as response, partial.open("ab") as out:
                        if done and response.status != 206:
                            raise ValueError("Server ignored the resume request")
                        last = time.time()
                        for block in iter(lambda: response.read(1 << 22), b""):
                            out.write(block)
                            done += len(block)
                            if time.time() - last > 30:
                                print(f"Rotor ZIP: {done / ROTOR_SIZE:.1%}", flush=True)
                                last = time.time()
                except Exception as error:
                    print(f"Rotor ZIP interrupted at {done:,} bytes ({error}); resuming", flush=True)
                    time.sleep(min(30, 3 * (attempt + 1)))
            if partial.stat().st_size != ROTOR_SIZE or digest(partial, "md5") != ROTOR_MD5:
                raise ValueError("Rotor ZIP failed the Zenodo size/MD5 check; delete the .part file and retry")
            os.replace(partial, destination)
            return "downloaded-and-verified"


        def main() -> None:
            parser = argparse.ArgumentParser()
            group = parser.add_mutually_exclusive_group()
            group.add_argument("--sensor-only", action="store_true")
            group.add_argument("--visual-only", action="store_true")
            parser.add_argument("--with-ball", action="store_true",
                                help="Also fetch the ball-fault bearings 11-15 for the 4-class experiment")
            args = parser.parse_args()
            record = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}

            if not args.visual_only:
                items = sensor_inventory()
                SENSOR_RAW.mkdir(parents=True, exist_ok=True)
                for number, item in enumerate(items, 1):
                    item["local_status"] = fetch_sensor(item)
                    item.pop("url")
                    print(f"{number}/{len(items)} {item['name']}: {item['local_status']}", flush=True)
                record["sensor"] = {
                    "source": "https://data.mendeley.com/datasets/y2px5tg92h/1",
                    "doi": "10.17632/y2px5tg92h.1",
                    "license": "CC BY 4.0",
                    "source_domain": "University of Ottawa laboratory bearing rig, not a haul truck",
                    "selected": "bearings 1-10: healthy stage 0 for all ten, final stage 2 inner (1-5) or outer (6-10); ball, cage and fault-developing stage excluded",
                    "selected_recordings": len(items),
                    "selected_bytes": sum(item["source_size_bytes"] for item in items),
                    "files": items,
                }

            if args.with_ball and not args.visual_only:
                items = sensor_inventory(ball=True)
                for number, item in enumerate(items, 1):
                    item["local_status"] = fetch_sensor(item)
                    item.pop("url")
                    print(f"ball {number}/{len(items)} {item['name']}: {item['local_status']}", flush=True)
                record["sensor_ball"] = {
                    "source": "https://data.mendeley.com/datasets/y2px5tg92h/1",
                    "doi": "10.17632/y2px5tg92h.1",
                    "license": "CC BY 4.0",
                    "selected": "bearings 11-15: healthy stage 0 and final stage 2 ball fault; for the 4-class experiment only",
                    "caveat": "the publisher recorded ball-fault data with no load; healthy, inner and outer used 400 N",
                    "selected_recordings": len(items),
                    "selected_bytes": sum(item["source_size_bytes"] for item in items),
                    "files": items,
                }

            if not args.sensor_only:
                status = fetch_rotor()
                print(f"{ROTOR_NAME}: {status}", flush=True)
                record["visual"] = {
                    "source": "https://zenodo.org/records/17447083",
                    "doi": "10.5281/zenodo.17447083",
                    "license": "CC BY 4.0",
                    "source_domain": "XJTU laboratory rotor-bearing rig, not a haul truck",
                    "file": ROTOR_NAME,
                    "size_bytes": ROTOR_SIZE,
                    "zenodo_md5": ROTOR_MD5,
                    "local_status": status,
                }

            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"Saved intake record: {MANIFEST.relative_to(CAPSTONE)}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # experiment: vibration and event-camera models, report and figure
    # (formerly Models/haul_truck_wheel_motor_bearing.py)
    # ==================================================================================================

    def experiment(run: bool = True) -> dict:
        """Haul/Dump Truck electric wheel-motor bearing: vibration sensor vs event camera.

        Matrix row B1. Both sources are laboratory bearing rigs, not haul-truck data;
        the comparison is at fault-taxonomy level (healthy, inner race, outer race).

        Sensor: UODS-VAFDC accelerometer, bearings 1-10. Each bearing contributes its
        healthy stage-0 recording and its final stage-2 fault recording, so every
        fold holds out whole physical bearings (both their recordings).
        Visual: XJTU-DV-Rotor event camera, 18 recordings (3 classes x 3 camera angles
        x 2 speeds). Train on 1000 rpm, test on 2000 rpm; a second check holds out
        each camera angle. Event images reuse the Pump event-camera pipeline.
        No sensor recording is paired with a camera recording.
        """


        import argparse
        import hashlib
        import json
        import os
        import re
        import sys
        import zipfile
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        from remotezip import RemoteZip
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

        _pump_file = get_model("pump")
        _pump = _pump_file.camera(run=False)
        BIN_US, END_US, START_US = _pump["BIN_US"], _pump["END_US"], _pump["START_US"]
        event_images, frame_features = _pump["event_images"], _pump["frame_features"]


        CAPSTONE = ROOT
        INTAKE = CAPSTONE / "Results" / "haul_truck" / "haul_truck_intake.json"
        SENSOR_RAW = CAPSTONE / "Raw_Data" / "uods_vafdc"
        ROTOR_ZIP = CAPSTONE / "Raw_Data" / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip"
        ROTOR_URL = "https://zenodo.org/api/records/17447083/files/XJTU-DV-Rotor.zip/content"
        SENSOR_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_sensor_features.npz"
        RUN_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_event_runs"
        RESULT = CAPSTONE / "Results" / "haul_truck" / "haul_truck_wheel_motor_bearing.json"
        REPORT = CAPSTONE / "Results" / "haul_truck" / "haul_truck_sensor_visual_comparison.md"
        FIGURE = CAPSTONE / "Figures" / "haul_truck" / "haul_truck_confusions.png"
        CLASSES = ("healthy", "inner", "outer")
        CODE_TO_CLASS = {"H": "healthy", "I": "inner", "O": "outer"}
        SAMPLE_RATE_HZ = 42_000
        WINDOW_SAMPLES = 4_200  # 100 ms; 100 non-overlapping windows per 10 s recording
        BANDS_HZ = ((0, 1_000), (1_000, 2_000), (2_000, 5_000), (5_000, 10_000), (10_000, 21_000))
        # The raw mean is excluded on purpose: every inner-race recording carries a DC
        # offset near +50 (and healthy H_2_0 one near +164), a recording artefact that
        # would otherwise let the model classify by sensor offset instead of vibration.
        FEATURE_NAMES = (
            "std", "peak_to_peak", "crest_factor", "skew", "excess_kurtosis",
            "spectral_centroid_hz", "dominant_frequency_hz", "spectral_entropy",
            *(f"power_{low}_{high}_hz" for low, high in BANDS_HZ),
        )
        ROTOR_MEMBER = re.compile(r"^Data/(Healthy|Inner|Outer)/angle([123])_(Healthy|Inner|Outer)_(1000|2000)\.dat$")


        def forest() -> RandomForestClassifier:
            return RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                          min_samples_leaf=2, random_state=42, n_jobs=-1)


        # ----------------------------------------------------------------------------- sensor branch
        def vibration_features(signal: np.ndarray) -> np.ndarray:
            count = len(signal) // WINDOW_SAMPLES
            if count < 90:
                raise ValueError(f"Recording is unexpectedly short: {len(signal)} samples")
            windows = signal[:count * WINDOW_SAMPLES].reshape(count, WINDOW_SAMPLES)
            centered = windows - windows.mean(axis=1, keepdims=True)
            std = centered.std(axis=1)
            safe_std = np.maximum(std, 1e-12)
            crest = np.max(np.abs(centered), axis=1) / safe_std
            skew = np.mean(centered**3, axis=1) / safe_std**3
            kurt = np.mean(centered**4, axis=1) / safe_std**4 - 3
            frequency = np.fft.rfftfreq(WINDOW_SAMPLES, d=1 / SAMPLE_RATE_HZ)
            power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
            relative = power / np.maximum(power.sum(axis=1), 1e-12)[:, None]
            centroid = (relative * frequency[None, :]).sum(axis=1)
            dominant = frequency[np.argmax(power[:, 1:], axis=1) + 1]
            entropy = -(relative * np.log(np.maximum(relative, 1e-12))).sum(axis=1)
            bands = [relative[:, (frequency >= low) & (frequency < high)].sum(axis=1) for low, high in BANDS_HZ]
            result = np.column_stack([std, np.ptp(centered, axis=1), crest, skew, kurt,
                                      centroid, dominant, entropy, *bands]).astype(np.float32)
            if not np.isfinite(result).all():
                raise ValueError("Non-finite vibration feature")
            return result


        def sensor_data(refresh: bool):
            if SENSOR_CACHE.is_file() and not refresh:
                with load_numpy(SENSOR_CACHE, allow_pickle=False) as saved:
                    return saved["x"], saved["y"], saved["bearing"], saved["run"], json.loads(str(saved["provenance"]))
            files = json.loads(INTAKE.read_text(encoding="utf-8"))["sensor"]["files"]
            xs, ys, bearings, runs, provenance = [], [], [], [], []
            for item in files:
                name = item["name"]
                signal = load_haul_truck_sensor(name)
                x = vibration_features(signal)
                label = CODE_TO_CLASS[item["class_code"]]
                xs.append(x)
                ys += [label] * len(x)
                bearings += [item["bearing"]] * len(x)
                runs += [name] * len(x)
                provenance.append({"file": name, "label": label, "bearing": item["bearing"], "stage": item["stage"],
                                   "source_sha256": item["source_sha256"], "samples": len(signal), "windows": len(x)})
            x, y, bearing, run = np.vstack(xs), np.asarray(ys), np.asarray(bearings), np.asarray(runs)
            SENSOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(SENSOR_CACHE, x=x, y=y, bearing=bearing, run=run, provenance=np.asarray(json.dumps(provenance)))
            return x, y, bearing, run, provenance


        # ----------------------------------------------------------------------------- visual branch
        def open_rotor():
            return open_archive(ROTOR_ZIP) if ROTOR_ZIP.is_file() else RemoteZip(ROTOR_URL)


        def rotor_members() -> list[tuple[str, str, int, int, int]]:
            with open_rotor() as archive:
                members = []
                for info in archive.infolist():
                    match = ROTOR_MEMBER.match(info.filename)
                    if match and match.group(1) == match.group(3):
                        members.append((info.filename, match.group(1).lower(), int(match.group(2)),
                                        int(match.group(4)), info.CRC))
            counts = {label: sum(m[1] == label for m in members) for label in CLASSES}
            if counts != {label: 6 for label in CLASSES}:
                raise ValueError(f"Expected six Rotor recordings per class; found {counts}")
            return sorted(members)


        REFRESH_EVENTS = False


        def event_run(job: tuple[str, str, int, int, int]) -> tuple[np.ndarray, dict]:
            member, label, angle, rpm, crc = job
            key = f"rotor|{member}|{START_US}|{END_US}|{BIN_US}"
            cached = RUN_CACHE / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".npz")
            if cached.is_file() and not REFRESH_EVENTS:
                with load_numpy(cached, allow_pickle=False) as saved:
                    x, record = saved["x"], json.loads(str(saved["record"]))
                if record["member"] != member or record["zip_member_crc32"] != f"{crc:08x}":
                    raise ValueError(f"Cached event run does not match the source: {member}")
                return x, record
            with open_rotor() as archive, archive.open(member) as stream:
                counts, event_type, retained = event_images(stream)
            x = frame_features(counts)
            record = {"member": member, "label": label, "camera_angle": angle, "speed_rpm": rpm,
                      "frames": len(x), "events_in_interval": retained, "event_type": event_type,
                      "zip_member_crc32": f"{crc:08x}", "analysis_interval_us": [START_US, END_US]}
            RUN_CACHE.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, x=x, record=np.asarray(json.dumps(record)))
            os.replace(temporary, cached)
            return x, record


        def visual_data():
            xs, ys, angles, speeds, runs, provenance = [], [], [], [], [], []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for number, (x, record) in enumerate(pool.map(event_run, rotor_members()), 1):
                    xs.append(x)
                    n = len(x)
                    ys += [record["label"]] * n
                    angles += [record["camera_angle"]] * n
                    speeds += [record["speed_rpm"]] * n
                    runs += [record["member"]] * n
                    provenance.append(record)
                    print(f"Rotor {number}/18: {record['label']} angle {record['camera_angle']} "
                          f"{record['speed_rpm']} rpm, {record['events_in_interval']:,} events", flush=True)
            return (np.vstack(xs), np.asarray(ys), np.asarray(angles), np.asarray(speeds), np.asarray(runs), provenance)


        # ----------------------------------------------------------------------------- evaluation
        def run_predictions(model, x, y, run):
            actual, predicted = [], []
            for name in sorted(set(run)):
                mask = run == name
                if len(set(y[mask])) != 1:
                    raise ValueError(f"Mixed labels within {name}")
                actual.append(y[mask][0])
                predicted.append(model.classes_[int(np.argmax(model.predict_proba(x[mask]).mean(axis=0)))])
            return actual, predicted


        def shuffled_labels(y, run, mask, seed):
            """Permute labels between whole training recordings, never between windows."""
            names = sorted(set(run[mask]))
            original = [y[run == name][0] for name in names]
            mapping = dict(zip(names, np.random.default_rng(seed).permutation(original)))
            return np.asarray([mapping[name] for name in run[mask]])


        def grouped_evaluation(x, y, run, folds):
            """folds: list of boolean test masks. Returns model, dummy and shuffled-label run predictions."""
            out = {"model": ([], []), "most_frequent_control": ([], []), "shuffled_training_labels_control": ([], [])}
            for number, test in enumerate(folds):
                train = ~test
                if set(run[train]) & set(run[test]):
                    raise ValueError("A recording appears in both train and test")
                for key, model, labels in (("model", forest(), y[train]),
                                           ("most_frequent_control", DummyClassifier(strategy="most_frequent"), y[train]),
                                           ("shuffled_training_labels_control", forest(), shuffled_labels(y, run, train, 42 + number))):
                    model.fit(x[train], labels)
                    a, p = run_predictions(model, x[test], y[test], run[test])
                    out[key][0].extend(a)
                    out[key][1].extend(p)
            return {key: metrics(np.asarray(a), np.asarray(p)) for key, (a, p) in out.items()}


        def metrics(actual, predicted) -> dict:
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=CLASSES, zero_division=0)
            return {
                "test_recordings": int(len(actual)),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
                "per_class": {label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                      "f1": round(float(f), 4), "support": int(n)}
                              for label, p, r, f, n in zip(CLASSES, precision, recall, f1, support)},
            }


        def plot(panels) -> None:
            figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.4))
            for axis, (title, matrix) in zip(axes, panels):
                array = np.asarray(matrix)
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(3), CLASSES, rotation=30)
                axis.set_yticks(range(3), CLASSES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(title, fontsize=10)
                for i in range(3):
                    for j in range(3):
                        axis.text(j, i, str(array[i, j]), ha="center", va="center",
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=140)
            plt.close(figure)


        def row(name, block):
            return (f"| {name} | {block['test_recordings']} | {block['accuracy']:.3f} | {block['macro_precision']:.3f} | "
                    f"{block['macro_recall']:.3f} | {block['macro_f1']:.3f} |")


        def interpretation(result) -> list[str]:
            lines = []
            for name, block in (("Sensor (held-out bearings)", result["sensor"]["held_out_bearings"]),
                                ("Event camera (2000 rpm test)", result["visual"]["train_1000_test_2000_rpm"]),
                                ("Event camera (held-out camera angle)", result["visual"]["held_out_camera_angle"])):
                model, control = block["model"], block["most_frequent_control"]
                missed = [label for label, stats in model["per_class"].items() if stats["recall"] == 0]
                extra = round((model["accuracy"] - control["accuracy"]) * model["test_recordings"])
                if extra <= 0:
                    verdict = "not above"
                elif extra * 6 <= model["test_recordings"]:
                    verdict = f"only {extra} more correct recording{'s' if extra > 1 else ''} than"
                else:
                    verdict = "well above"
                text = (f"{name}: accuracy {model['accuracy']:.3f} on {model['test_recordings']} recordings, "
                        f"{verdict} the {control['accuracy']:.3f} most-frequent control.")
                if missed:
                    text += f" Never correctly predicted: {', '.join(missed)}."
                lines.append(text)
            lines.append("The two branches use different rigs and very small test sets, so the gap between them is not "
                         "evidence that one modality is better in general; it describes these two fixed pipelines on these data.")
            return lines


        def write_report(result) -> None:
            s, v = result["sensor"], result["visual"]
            lines = [
                "# Haul/Dump Truck wheel-motor bearing: sensor vs event camera",
                "",
                "Generated by `Models/haul_truck.py experiment`. Target: present bearing condition "
                "(healthy, inner race, outer race). Both sources are **laboratory bearing rigs**, used as an analogue for an "
                "electric-drive haul truck's wheel-motor bearing; they are not truck data, are not paired, and are not fused.",
                "",
                "| Branch and test | Test recordings | Accuracy | Macro precision | Macro recall | Macro F1 |",
                "|---|---:|---:|---:|---:|---:|",
                row("Sensor: UODS vibration, held-out bearings (5 folds)", s["held_out_bearings"]["model"]),
                row("Sensor: most-frequent control", s["held_out_bearings"]["most_frequent_control"]),
                row("Sensor: shuffled training labels", s["held_out_bearings"]["shuffled_training_labels_control"]),
                row("CV: XJTU-Rotor event camera, train 1000 rpm, test 2000 rpm", v["train_1000_test_2000_rpm"]["model"]),
                row("CV: most-frequent control", v["train_1000_test_2000_rpm"]["most_frequent_control"]),
                row("CV: shuffled training labels", v["train_1000_test_2000_rpm"]["shuffled_training_labels_control"]),
                row("CV: held-out camera angle (3 folds)", v["held_out_camera_angle"]["model"]),
                "",
                "![Confusion matrices](../../Figures/haul_truck/haul_truck_confusions.png)",
                "",
                "## Interpretation",
                "",
                *[f"- {line}" for line in interpretation(result)],
                "",
                "## Limits",
                "",
                *[f"- {item}" for item in result["limitations"]],
                "",
                "Data: UODS-VAFDC, DOI 10.17632/y2px5tg92h.1; XJTU-DV, DOI 10.5281/zenodo.17447083; both CC BY 4.0.",
            ]
            REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--refresh", action="store_true", help="Rebuild all features from the raw source files")
            args = parser.parse_args()
            nonlocal REFRESH_EVENTS
            REFRESH_EVENTS = args.refresh

            # Sensor: fold k holds out inner-race bearing k and outer-race bearing k+5, with both of their recordings.
            sx, sy, sbearing, srun, sprov = sensor_data(args.refresh)
            if len(set(srun)) != 20:
                raise ValueError("Expected 20 UODS recordings")
            sensor_folds = [np.isin(sbearing, [k, k + 5]) for k in range(1, 6)]
            sensor = {"held_out_bearings": grouped_evaluation(sx, sy, srun, sensor_folds)}

            vx, vy, vangle, vspeed, vrun, vprov = visual_data()
            if len(set(vrun)) != 18:
                raise ValueError("Expected 18 Rotor recordings")
            visual = {
                "train_1000_test_2000_rpm": grouped_evaluation(vx, vy, vrun, [vspeed == 2000]),
                "held_out_camera_angle": grouped_evaluation(vx, vy, vrun, [vangle == a for a in (1, 2, 3)]),
            }

            result = {
                "status": "haul_truck_wheel_motor_bearing_sensor_vs_event_camera",
                "target": "present bearing condition: healthy / inner race / outer race",
                "class_order": list(CLASSES),
                "sensor": {
                    "source": "UODS-VAFDC, https://data.mendeley.com/datasets/y2px5tg92h/1, CC BY 4.0",
                    "selection": "bearings 1-10: healthy stage 0 (10 recordings), final stage 2 inner (bearings 1-5) and outer (bearings 6-10)",
                    "sample_rate_hz": SAMPLE_RATE_HZ, "window_samples": WINDOW_SAMPLES, "feature_names": list(FEATURE_NAMES),
                    "preprocessing_note": "Each window is mean-removed and the raw mean is not a feature, because inner-race recordings carry a DC offset near +50 that is a recording artefact.",
                    "split": "5 folds; fold k holds out bearings k (inner) and k+5 (outer) with both their healthy and fault recordings; one averaged prediction per recording",
                    "source_recordings": sprov,
                    **sensor,
                },
                "visual": {
                    "source": "XJTU-DV-Rotor, https://zenodo.org/records/17447083, CC BY 4.0",
                    "selection": "18 recordings: healthy/inner/outer x camera angles 1-3 x 1000/2000 rpm; ball excluded to match the sensor classes",
                    "preprocessing": "first 2 s, eight 250 ms event-count images, signed log contrast, OpenCV 64x64 HOG (same code as the Pump event-camera branch)",
                    "split": "primary: train 1000 rpm, test 2000 rpm; check: hold out each camera angle across both speeds",
                    "source_recordings": vprov,
                    **visual,
                },
                "limitations": [
                    "Both rigs are laboratory bearing rigs; transfer to a haul truck's electric wheel-motor bearing is a component analogy, not demonstrated truck accuracy.",
                    "Sensor and camera data come from different rigs and are compared by fault label only; there are no paired samples and no fusion.",
                    "Small test sets: 20 sensor recordings from 10 bearings, and 9 camera recordings in the primary test, so one error changes accuracy by 5 to 11 points.",
                    "Each UODS bearing has one healthy and one fault recording, so healthy and fault recordings also differ in test stage and date.",
                    "The camera sees rotor-housing motion, not visible bearing damage; only the first 2 s of each recording are used.",
                    "The models classify present condition, not remaining useful life or a repair date.",
                ],
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot([("Sensor: held-out bearings (20 recordings)", sensor["held_out_bearings"]["model"]["confusion_matrix"]),
                  ("CV: train 1000, test 2000 rpm (9)", visual["train_1000_test_2000_rpm"]["model"]["confusion_matrix"]),
                  ("CV: held-out camera angle (18)", visual["held_out_camera_angle"]["model"]["confusion_matrix"])])
            write_report(result)
            print(f"Saved {RESULT.relative_to(CAPSTONE)}, {REPORT.relative_to(CAPSTONE)} and {FIGURE.relative_to(CAPSTONE)}")
            s, v = sensor["held_out_bearings"]["model"], visual["train_1000_test_2000_rpm"]["model"]
            print(f"Sensor accuracy {s['accuracy']:.3f}, macro F1 {s['macro_f1']:.3f}; "
                  f"camera accuracy {v['accuracy']:.3f}, macro F1 {v['macro_f1']:.3f}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # baselines: Week 5 baseline model comparison (majority class, logistic regression, decision tree, random forest)
    # ====================================================================================================

    def baselines(run: bool = True) -> dict:
        """Week 5: compare baseline model families on the same held-out tests as the main experiment.

        A majority-class control, logistic regression, a decision tree and a random forest are trained
        with settings fixed in advance and scored on exactly the same held-out test units. No model or
        setting is chosen from these test results; the main model of the experiment stays the same.
        """
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        RESULT = ROOT / "Results" / "haul_truck" / "haul_truck_baselines.json"
        FIGURE = ROOT / "Figures" / "haul_truck" / "haul_truck_baselines.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Decision tree": lambda: DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=42),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }
        SETTINGS = {
            "Majority class (control)": "always predicts the most common training class",
            "Logistic regression": "standardised features, C=1, class-balanced, lbfgs (multi-class)",
            "Decision tree": "class-balanced, min_samples_leaf=2",
            "Random forest": "250 trees, sqrt features, class-balanced subsample, min_samples_leaf=2",
        }

        def unit_predictions(model, x, y, units):
            """One prediction per test unit (recording or run): average the model's probabilities over its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(model.classes_[int(np.argmax(model.predict_proba(x[rows]).mean(axis=0)))])
            return actual, predicted

        def evaluate(x, y, units, folds, classes):
            """`folds` are boolean test masks; every test unit is predicted by a model that never saw it."""
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds:
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model, x[test], y[test], units[test])
                    actual += list(a)
                    predicted += list(p)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predicted, labels=classes, zero_division=0)
                scores[name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_precision": round(float(precision.mean()), 4),
                    "macro_recall": round(float(recall.mean()), 4),
                    "macro_f1": round(float(f1.mean()), 4),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
                }
            return scores

        def plot(result):
            branches, names = list(result["branches"]), list(MODELS)
            width = 0.8 / len(names)
            figure, axis = plt.subplots(figsize=(2.2 + 2.8 * len(branches), 4.0))
            for i, name in enumerate(names):
                values = [result["branches"][b]["models"][name]["macro_f1"] for b in branches]
                bars = axis.bar(np.arange(len(branches)) + (i - (len(names) - 1) / 2) * width, values, width, label=name)
                axis.bar_label(bars, fmt="%.2f", fontsize=7)
            axis.set_xticks(range(len(branches)), branches)
            axis.set_ylim(0, 1.15)
            axis.set_ylabel("Macro F1 on held-out test")
            axis.set_title(result["title"], fontsize=10)
            axis.legend(fontsize=8, ncol=2, loc="upper right")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=140)
            plt.close(figure)

        def main():
            branches = {}
            e = experiment(run=False)
            classes = list(e["CLASSES"])
            x, y, bearing, units, _ = e["sensor_data"](False)
            branches["Sensor: vibration"] = {
                "test": "five rounds, each holding out two whole bearings; 20 recording-level predictions",
                "classes": classes,
                "models": evaluate(x, y, units, [np.isin(bearing, [k, k + 5]) for k in range(1, 6)], classes)}
            vx, vy, angle, speed, vunits, _ = e["visual_data"]()
            branches["Camera: event camera"] = {
                "test": "train on 1000 rpm; test on 9 separate recordings at 2000 rpm",
                "classes": classes,
                "models": evaluate(vx, vy, vunits, [speed == 2000], classes)}
            result = {
                "title": "Haul Truck: baseline models on held-out bearings / speed",
                "purpose": "Week 5 baseline model comparison on the same held-out tests as the main experiment",
                "model_settings": SETTINGS,
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "branches": branches,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(result)
            for branch, block in branches.items():
                for name, score in block["models"].items():
                    print(f"{branch:<24} {name:<26} accuracy {score['accuracy']:.3f}  macro F1 {score['macro_f1']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # export-images: Week 6, save the camera images as named files (professor remark)
    # ====================================================================================================

    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/haul_truck/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "haul_truck"
        OUT = ROOT / "Images" / MACHINE
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5   # --all-frames: the whole recording (up to 30 s; recordings are 16-26 s)
        FRAMES = SECONDS * 1_000_000 // BIN_US
        NAME = "t{k:02d}"

        def event_frames(stream):
            """Bin events using the configured duration and frame interval; channels preserve polarity."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})

        def export():
            rows = []
            folders = {"healthy": "Healthy", "inner": "Inner", "outer": "Outer", "ball": "Ball"}
            with open_archive(ROOT / "Raw_Data" / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip") as archive:
                for label, folder in folders.items():
                    for angle in (1, 2, 3):
                        for rpm in (1000, 2000):
                            unit = f"{rpm}rpm_{label}_angle{angle}"
                            with archive.open(f"Data/{folder}/angle{angle}_{folder}_{rpm}.dat") as stream:
                                counts = event_frames(stream)
                            for k, frame in enumerate(counts):
                                if frame.sum() > 0:
                                    write(rows, label, unit, f"{rpm}rpm",
                                          f"{MACHINE}_{label}_{rpm}rpm_angle{angle}_" + NAME.format(k=k) + ".png", event_png(frame))
                            print(f"{unit}: done", flush=True)
            return rows


        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.add_argument("--all-frames", action="store_true",
                                help="whole recordings (all 250 ms frames) to Images/haul_truck_allframes/ (Week 7)")
            parser.add_argument("--dense-frames", action="store_true", help="125 ms frames over the whole recording; separate output")
            args = parser.parse_args()
            nonlocal OUT, SECONDS, FRAMES, NAME, BIN_US
            if args.all_frames:
                OUT, SECONDS, NAME = ROOT / "Images" / f"{MACHINE}_allframes", 30, "t{k:03d}"
                FRAMES = SECONDS * 1_000_000 // BIN_US
            if args.dense_frames:
                OUT = ROOT / "Images" / f"{MACHINE}_denseframes"
                SECONDS, BIN_US, NAME = 30, 125_000, "t{k:03d}"
                FRAMES = SECONDS * 1_000_000 // BIN_US
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # csv: Week 6, 4-class sensor model from a CSV feature table (professor remark: "csv accuracy 90 plus")
    # ====================================================================================================

    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "haul_truck"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "haul_truck" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "haul_truck" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        ENVELOPE_BANDS = ((5, 50), (50, 90), (90, 130), (130, 180), (180, 250), (250, 400), (400, 800))
        ENVELOPE_NAMES = ["envelope_kurtosis", "envelope_peak_hz",
                          *(f"envelope_power_{lo}_{hi}_hz" for lo, hi in ENVELOPE_BANDS)]

        def envelope_features(signal, n_windows, rate=42_000, window=4_200):
            """Envelope analysis, the standard bearing-fault method: band-pass the vibration (1-15 kHz), take
            the Hilbert envelope and measure where its energy sits (fault 'knock' rates are 50-400 Hz)."""
            from scipy.signal import butter, hilbert, sosfiltfilt
            sos = butter(4, (1_000, 15_000), btype="bandpass", fs=rate, output="sos")
            band = sosfiltfilt(sos, signal - signal.mean())
            env = np.abs(hilbert(band))[: n_windows * window].reshape(n_windows, window)
            env = env - env.mean(axis=1, keepdims=True)
            std = np.maximum(env.std(axis=1), 1e-12)
            freq = np.fft.rfftfreq(window, d=1 / rate)
            power = np.abs(np.fft.rfft(env, axis=1)) ** 2
            low = (freq >= 5) & (freq < 800)
            share = power[:, low] / np.maximum(power[:, low].sum(axis=1, keepdims=True), 1e-12)
            f = freq[low]
            columns = [(env ** 4).mean(axis=1) / std ** 4, f[np.argmax(share, axis=1)],
                       *[share[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in ENVELOPE_BANDS]]
            return np.column_stack(columns)

        def build_table():
            e = experiment(run=False)
            intake = json.loads((ROOT / "Results" / "haul_truck" / "haul_truck_intake.json").read_text(encoding="utf-8"))
            if "sensor_ball" not in intake:
                raise SystemExit("Ball-fault files missing: run  python Models/haul_truck.py intake --sensor-only --with-ball")
            files = intake["sensor"]["files"] + intake["sensor_ball"]["files"]
            codes = {"H": "healthy", "I": "inner", "O": "outer", "B": "ball"}
            frames = []
            for item in files:
                name = item["name"]
                data = loadmat(ROOT / "Raw_Data" / "uods_vafdc" / name)[name[:-4]]
                signal = data[:, 0].astype(np.float64)       # column 0 = accelerometer
                microphone = data[:, 1].astype(np.float64)   # column 1 = microphone; speed and load columns not used
                features = e["vibration_features"](signal)
                frame = pd.DataFrame(features, columns=list(e["FEATURE_NAMES"]))
                envelope = envelope_features(signal, len(frame))
                for k, column in enumerate(ENVELOPE_NAMES):
                    frame[column] = envelope[:, k]
                # 0.5 s envelope windows (2 Hz resolution) repeated for their five 0.1 s rows
                long_envelope = np.repeat(envelope_features(signal, len(frame) // 5, window=21_000), 5, axis=0)
                for k, column in enumerate(ENVELOPE_NAMES):
                    frame[f"long_{column}"] = long_envelope[: len(frame), k]
                sound = e["vibration_features"](microphone)
                for k, column in enumerate(e["FEATURE_NAMES"]):
                    frame[f"mic_{column}"] = sound[: len(frame), k]
                frame.insert(0, "class", codes[item["class_code"]])
                frame.insert(0, "bearing", item["bearing"])
                frame.insert(0, "unit", name[:-4])
                frames.append(frame)
            table = pd.concat(frames, ignore_index=True)
            folds = lambda t: [t["bearing"].isin([k, k + 5, k + 10]).to_numpy() for k in range(1, 6)]
            base = list(e["FEATURE_NAMES"])
            long_env = [f"long_{c}" for c in ENVELOPE_NAMES]
            mic = [f"mic_{c}" for c in e["FEATURE_NAMES"]]
            return (table, {"base": base, "base+envelope": base + ENVELOPE_NAMES,
                            "base+long envelope": base + long_env,
                            "base+microphone": base + mic,
                            "base+long envelope+microphone": base + long_env + mic},
                    ["healthy", "inner", "outer", "ball"], folds,
                    "five rounds, each holding out three whole bearings (inner, outer, ball) with their healthy recordings; 30 recording-level predictions",
                    ["UODS vibration, bearings 1-15; one row per 0.1 s window; window mean removed (inner-race files carry a DC offset)",
                     "ball faults were recorded with no load (load sensor reads zero) while all other files ran at about 400 N; the load column is not used, but no-load running also changes the vibration, so ball results may partly reflect load"])


        INNER_GROUP = "bearing"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test, as described in the result) or time (secondary, easier): train on "
                                     "the first 60%% of every recording, test on the last 30%% (10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier "
                                    "than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            if args.split == "time":
                result["split"] = "time (secondary)"
                RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                main_score = scores["Random forest"]
                plot_matrix(main_score["confusion_matrix"], labels,
                            f"{MACHINE}: CSV sensor, TIME SPLIT (random forest), accuracy {main_score['accuracy']:.3f}", FIGURE)
                for name, s in scores.items():
                    print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units, time split)")
                print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
                return
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn: Week 6, CNN with data augmentation on the named images (professor remark)
    # ====================================================================================================

    def _augment_event_batch(batch, generator):
        """Apply the fixed training augmentation without constructing another CNN pipeline."""
        import torch

        n = batch.shape[0]
        flip = torch.rand(n, generator=generator) < 0.5
        batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
        shifted = []
        for image in batch:
            dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
            shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
        batch = torch.stack(shifted)
        scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
        return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)



    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "haul_truck"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "haul_truck" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "haul_truck" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "inner", "outer", "ball"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [2, 1]  # red = brighter events, green = darker events
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        augment = _augment_event_batch

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == "2000rpm"], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "train on the 12 recordings at 1000 rpm; test on the 12 separate recordings at 2000 rpm",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn-pretrained: Week 6, transfer learning with a pretrained ResNet18 on the named images
    # ====================================================================================================

    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "haul_truck"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "haul_truck" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "haul_truck" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "inner", "outer", "ball"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        augment = _augment_event_batch

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary, easier: first 60%% of each recording's frames "
                                     "train, last 30%% test, 10%% gap)")
            parser.add_argument("--all-frames", action="store_true",
                                help="use Images/haul_truck_allframes/ (whole recordings); results saved as *_allframes")
            parser.add_argument("--dense-frames", action="store_true", help="use separately exported 125 ms full-duration frames")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE, IMAGES
            if args.all_frames:
                IMAGES = ROOT / "Images" / f"{MACHINE}_allframes"
                RESULT = RESULT.with_name(RESULT.stem + "_allframes.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_allframes.png")
            if args.dense_frames:
                if args.all_frames or args.split != "main":
                    raise ValueError("dense-frames uses the main held-out test only")
                IMAGES = ROOT / "Images" / f"{MACHINE}_denseframes"
                RESULT = RESULT.with_name(RESULT.stem + "_denseframes.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_denseframes.png")
            rows = list(csv_module.DictReader((IMAGES / "index.csv").open(encoding="utf-8")))
            test_description = "train on the 12 recordings at 1000 rpm; test on the 12 separate recordings at 2000 rpm"
            if args.split == "time":
                import re
                frame = {row["file"]: int(re.search(r"_t(\d+)\.png$", row["file"]).group(1)) for row in rows}
                length = {}
                for row in rows:
                    length[row["unit"]] = max(length.get(row["unit"], 0), frame[row["file"]] + 1)
                frames_per = 20 if not args.all_frames else None
                position = {row["file"]: frame[row["file"]] / (frames_per or length[row["unit"]]) for row in rows}
                rows = [dict(row, unit=row["unit"] + ("_late" if position[row["file"]] > 0.7 else "_early"))
                        for row in rows if position[row["file"]] <= 0.6 or position[row["file"]] > 0.7]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording's frames train, last 30% test, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            folds = ([np.char.endswith(units.astype(str), "_late")] if args.split == "time"
                     else [groups == "2000rpm"])
            for number, test in enumerate(folds, 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": test_description,
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # train-final / predict: use the trained sensor model on any CSV
    # ====================================================================================================

    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "base" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "haul_truck_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. O_6_2")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["base"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "haul_truck",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/haul_truck_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of accelerometer values sampled at 42 kHz (at least 9 seconds).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "haul_truck_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            return experiment(run=False)["vibration_features"](values)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, inner, outer, ball")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")

            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Proposal Week 6: advanced ensemble models (XGBoost / LightGBM comparison table)
    # ====================================================================================================

    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "haul_truck"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: anomaly detection (proposal Week 7: Isolation Forest / One-Class SVM)
    # ====================================================================================================

    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "haul_truck"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    COMMANDS = {
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
        "intake": intake,
        "experiment": experiment,
        "baselines": baselines,
        "export-images": export_images,
        "csv": csv,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
    }
    RUN_ALL = ['experiment']


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"haul_truck.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# BELT_CONVEYOR MODEL IMPLEMENTATION
def _make_belt_conveyor():
    __doc__ = "Belt Conveyor: geared drive, CORA T1 temperature vs infrared camera.\n\nAll of this machine's code in one file. The combined notebook\nNotebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.\n\nUsage:\n    python Models/belt_conveyor.py <command> [options]\n    python Models/belt_conveyor.py <command> --help\n\nCommands:\n    intake          download CORA infrared stacks and bounded T1 samples\n    inspect-file    helper: inspect a local CORA publisher file\n    preprocess      helper: CORA sensor/thermal preprocessing smoke test\n    experiment      T1 temperature and infrared models, report and figure\n    baselines       Week 5: compare majority class, logistic regression, decision tree and random forest\n    export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)\n    csv             Week 6: 4-class sensor model from a CSV feature table\n    cnn             Week 6: CNN with data augmentation on the named images (run export-images first)\n    cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images\n    intake-vibration  check selectively downloaded CORA vibration files (MD5) and file them\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)\n    fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)\n                    (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)\n    all             run experiment\n\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Belt Conveyor: geared drive, CORA T1 temperature vs infrared camera.

    All of this machine's code in one file. The combined notebook
    Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

    Usage:
        python Models/belt_conveyor.py <command> [options]
        python Models/belt_conveyor.py <command> --help

    Commands:
        intake          download CORA infrared stacks and bounded T1 samples
        inspect-file    helper: inspect a local CORA publisher file
        preprocess      helper: CORA sensor/thermal preprocessing smoke test
        experiment      T1 temperature and infrared models, report and figure
        baselines       Week 5: compare majority class, logistic regression, decision tree and random forest
        export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)
        csv             Week 6: 4-class sensor model from a CSV feature table
        cnn             Week 6: CNN with data augmentation on the named images (run export-images first)
        cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images
        intake-vibration  check selectively downloaded CORA vibration files (MD5) and file them
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)
        fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)
                        (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)
        all             run experiment

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import sys
    from pathlib import Path



    # ==================================================================================================
    # intake: download CORA infrared stacks and bounded T1 samples
    # (formerly Preprocessing/cora_gear_intake.py)
    # ==================================================================================================

    def intake(run: bool = True) -> dict:
        """Acquire a bounded CORA gear-wear sensor/infrared experiment set.

        The 16 stationary experiments are H/W25/W50/W75 at F5/F15/F50/F60.
        Thermal MATLAB stacks are downloaded in full; only eight deterministic,
        equally spaced byte ranges of original T1 CSV per run are retained locally.
        The byte-range sample is a resource-bounded proxy, not the full sensor stream.
        All raw/processed data are ignored by Git. No cross-run or frame-level pairing
        is inferred from matching filenames.
        """


        import argparse
        import hashlib
        import json
        import os
        import re
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from pathlib import Path
        from urllib.request import Request, urlopen

        import numpy as np
        import truststore
        from scipy.io import whosmat

        # The CORA server sends an incomplete certificate chain, which Python's own certificate list rejects.
        # Use the computer's certificate store instead; certificates are still fully verified.
        truststore.inject_into_ssl()


        CAPSTONE = ROOT
        RAW = CAPSTONE / "Raw_Data" / "cora_gear_wear"
        PROCESSED = CAPSTONE / "Processed_Data" / "cora_gear_wear"
        INTAKE = CAPSTONE / "Results" / "belt_conveyor" / "belt_conveyor_cora_intake.json"
        BASE = "https://dataverse.csuc.cat/api/access/datafile"
        CHUNK_BYTES = 256 * 1024
        N_RANGES = 8
        SOURCE = "https://doi.org/10.34810/DATA2500"
        FILE_IDS = {
            "H_F5_S": (334968, 335123),
            "H_F15_S": (334964, 335059),
            "H_F50_S": (334886, 335054),
            "H_F60_S": (334901, 335165),
            "W25_F5_S": (334980, 335370),
            "W25_F15_S": (334958, 335494),
            "W25_F50_S": (334909, 335479),
            "W25_F60_S": (334991, 335458),
            "W50_F5_S": (334971, 335495),
            "W50_F15_S": (334982, 335438),
            "W50_F50_S": (334950, 335415),
            "W50_F60_S": (334937, 335423),
            "W75_F5_S": (334949, 335449),
            "W75_F15_S": (334977, 335391),
            "W75_F50_S": (334993, 335488),
            "W75_F60_S": (334918, 335400),
        }
        PUBLISHER_MD5 = {"H_F5_S": "18b4bf748275343a220cca7d4f854d60"}


        def digest(path: Path, algorithm: str) -> str:
            result = hashlib.new(algorithm)
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    result.update(block)
            return result.hexdigest()


        def request(file_id: int, byte_range: tuple[int, int] | None = None) -> Request:
            headers = {"User-Agent": "PROJ613-capstone/1.0"}
            if byte_range is not None:
                headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
            return Request(f"{BASE}/{file_id}?format=original", headers=headers)


        def download_thermal(run: str, file_id: int) -> dict:
            destination = RAW / f"{run}.mat"
            temporary = RAW / f"{run}.mat.part"
            if not destination.is_file():
                for attempt in range(3):
                    try:
                        with urlopen(request(file_id), timeout=180) as response, temporary.open("wb") as output:
                            content_name = response.headers.get("Content-Disposition", "")
                            if f'filename="{run}.mat"' not in content_name:
                                raise ValueError(f"Unexpected thermal response for {run}: {content_name}")
                            for block in iter(lambda: response.read(1024 * 1024), b""):
                                output.write(block)
                        if temporary.stat().st_size < 1_000_000:
                            raise ValueError(f"Thermal response too small for {run}")
                        os.replace(temporary, destination)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(3 * (attempt + 1))
            if run in PUBLISHER_MD5 and digest(destination, "md5") != PUBLISHER_MD5[run]:
                raise ValueError(f"Publisher MD5 mismatch: {run}")
            variables = whosmat(destination)
            if ("imagenes_celda", (1, 450), "cell") not in variables:
                raise ValueError(f"Unexpected MATLAB image stack: {run}: {variables}")
            return {
                "file_id": file_id,
                "filename": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": digest(destination, "sha256"),
                "publisher_md5_checked": run in PUBLISHER_MD5,
                "image_frames": 450,
            }


        def sensor_length(file_id: int, run: str) -> int:
            with urlopen(request(file_id, (0, 0)), timeout=90) as response:
                if response.status != 206:
                    raise ValueError(f"Range request not honored for {run}")
                disposition = response.headers.get("Content-Disposition", "")
                if f'filename="{run}_T1.csv"' not in disposition:
                    raise ValueError(f"Unexpected sensor response for {run}: {disposition}")
                match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise ValueError(f"Missing source size for {run}")
                response.read()
            return int(match.group(1))


        def sample_sensor(run: str, file_id: int) -> tuple[np.ndarray, dict]:
            total = sensor_length(file_id, run)
            if total < CHUNK_BYTES * N_RANGES:
                raise ValueError(f"Unusually short T1 source for {run}: {total}")
            starts = np.linspace(0, total - CHUNK_BYTES, N_RANGES, dtype=np.int64)
            values: list[np.ndarray] = []
            ranges = []
            for start_value in starts:
                start = int(start_value)
                end = start + CHUNK_BYTES - 1
                with urlopen(request(file_id, (start, end)), timeout=90) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or content_range != f"bytes {start}-{end}/{total}":
                        raise ValueError(f"Unexpected range response for {run}: {content_range}")
                    block = response.read()
                if len(block) != CHUNK_BYTES:
                    raise ValueError(f"Truncated range for {run}: {start}-{end}")
                lines = block.splitlines()
                complete = lines[1:-1] if start else lines[:-1]
                chunk = np.fromstring(b"\n".join(complete).decode("ascii"), sep="\n")
                if len(chunk) < 5_000 or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid sensor readings for {run} at byte {start}")
                values.append(chunk.astype(np.float32))
                ranges.append({"first_byte": start, "last_byte": end, "readings": len(chunk),
                               "sha256": hashlib.sha256(block).hexdigest()})
            # Preserve one fixed six-second reading window at every selected position.
            fixed = np.stack([chunk[:6_000] for chunk in values])
            return fixed, {
                "file_id": file_id,
                "source_filename": f"{run}_T1.csv",
                "source_size_bytes": total,
                "sampled_bytes": N_RANGES * CHUNK_BYTES,
                "sampled_readings": int(fixed.size),
                "ranges": ranges,
            }


        def main() -> None:
            parser = argparse.ArgumentParser(description=intake.__doc__)
            parser.add_argument("--sensor-only", action="store_true", help="Skip full infrared downloads")
            parser.add_argument("--thermal-only", action="store_true", help="Skip sensor range sampling")
            args = parser.parse_args()
            if args.sensor_only and args.thermal_only:
                parser.error("Choose at most one of --sensor-only and --thermal-only")
            RAW.mkdir(parents=True, exist_ok=True)
            PROCESSED.mkdir(parents=True, exist_ok=True)
            records = {}
            existing = json.loads(INTAKE.read_text(encoding="utf-8")) if INTAKE.exists() else {}
            if not args.sensor_only:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(download_thermal, run, ids[0]): run
                               for run, ids in FILE_IDS.items()}
                    for completed, future in enumerate(as_completed(futures), 1):
                        run = futures[future]
                        records.setdefault(run, {})["infrared"] = future.result()
                        print(f"Infrared {completed}/16: {run}", flush=True)
            if not args.thermal_only:
                sensor_data = {}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(sample_sensor, run, ids[1]): run
                               for run, ids in FILE_IDS.items()}
                    for completed, future in enumerate(as_completed(futures), 1):
                        run = futures[future]
                        values, metadata = future.result()
                        sensor_data[run] = values
                        records.setdefault(run, {})["temperature_sensor"] = metadata
                        print(f"Sensor {completed}/16: {run}", flush=True)
                np.savez_compressed(PROCESSED / "t1_bounded_samples.npz", **sensor_data)
            previous = existing.get("runs", {})
            for run, values in previous.items():
                for modality, metadata in values.items():
                    records.setdefault(run, {}).setdefault(modality, metadata)
            result = {
                "source": SOURCE,
                "version": "2.1",
                "license": "CC BY 4.0",
                "source_domain": "One controlled motor/gearbox laboratory rig, not a mining conveyor",
                "target": "healthy versus imposed gear-tooth wear (W25/W50/W75)",
                "selection": "16 stationary runs; four classes by four speeds; one run per class/speed",
                "sensor_sampling": "Eight equally spaced 256-KiB ranges per original T1 CSV; first 6,000 complete readings in each range",
                "sensor_sample_fraction_approx": "about 4.3% of bytes per run; not the full signal",
                "frame_alignment": "No claimed one-to-one frame/window alignment",
                "runs": dict(sorted(records.items())),
            }
            INTAKE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {INTAKE}")

        if run:
            main()
        return locals()


    # ==================================================================================================
    # inspect-file: helper: inspect a local CORA publisher file
    # (formerly Preprocessing/cora_manifest.py)
    # ==================================================================================================

    def inspect_file(run: bool = True) -> dict:
        """Audit CORA sensor and thermal files without loading the large measurements.

        The filename provides an experiment-level key, not a one-to-one pairing of
        individual sensor samples and thermal frames. Run from the repository root:

            python Models/belt_conveyor.py inspect-file /path/to/CORA/files

        Pass either a directory of downloaded files or the repository's
        ``Information_about_database_files.txt`` guide. No files are downloaded,
        modified, or committed.
        """


        import argparse
        from collections import defaultdict
        from pathlib import Path

        import numpy as np


        FAULT_NAMES = {
            "H": "Healthy",
            "BD": "Bearing defect",
            "HB": "Half broken rotor bar",
            "OB": "One broken rotor bar",
            "U": "Unbalance",
            "M": "Misalignment",
            "W25": "Gear wear 25%",
            "W50": "Gear wear 50%",
            "W75": "Gear wear 75%",
        }
        FREQUENCIES = {"F5", "F15", "F50", "F60"}
        SENSOR_CHANNELS = {
            "C1", "C2", "C3", "VX", "VY", "VZ", "T1", "T2", "T3",
            "T4", "T5", "T6", "RPM", "AT", "TIMECU", "TIMEVI",
            "TIMETEMP", "TIMERPM",
        }
        MEASUREMENT_SUFFIXES = {".csv", ".tab"}
        THERMAL_SUFFIXES = {".mat", ".jpg", ".jpeg"}


        def parse_file(path: Path) -> dict[str, str] | None:
            """Extract the published fault, frequency, regime, and modality tokens."""
            suffix = path.suffix.lower()
            if suffix not in MEASUREMENT_SUFFIXES | THERMAL_SUFFIXES:
                return None
            parts = path.stem.upper().split("_")
            if len(parts) < 3:
                return None
            fault, frequency, regime = parts[:3]
            if fault not in FAULT_NAMES or frequency not in FREQUENCIES or regime not in {"S", "T"}:
                return None
            channel = parts[3] if len(parts) > 3 else ""
            if suffix in MEASUREMENT_SUFFIXES and channel in SENSOR_CHANNELS:
                modality = "sensor"
            elif suffix == ".mat" and regime == "S" and channel in {"", "IR"}:
                modality = "thermal_stack"
            elif suffix in {".jpg", ".jpeg"} and regime == "S" and channel == "IR":
                modality = "thermal_example"
            else:
                return None
            return {
                "experiment": f"{fault}_{frequency}_{regime}",
                "fault": fault,
                "frequency": frequency,
                "regime": regime,
                "channel": channel,
                "modality": modality,
                "filename": path.name,
            }


        def listed_paths(source: Path) -> list[Path]:
            """Read filenames from a folder or the official small text file guide."""
            if source.is_dir():
                return [path for path in source.rglob("*") if path.is_file()]
            if source.is_file() and source.suffix.lower() == ".txt":
                with source.open(encoding="utf-8-sig") as guide:
                    names = [line.split(";", 1)[0].split(",", 1)[0].strip()
                             for line in guide]
                return [Path(name) for name in names if name]
            raise FileNotFoundError(f"Expected a CORA data folder or file guide: {source}")


        def load_thermal_stack(path: Path) -> list[np.ndarray]:
            """Load the MATLAB-v5 ``imagenes_celda`` three-channel frame collection."""
            parsed = parse_file(path)
            if parsed is None or parsed["modality"] != "thermal_stack":
                raise ValueError(f"Not a stationary CORA thermal stack: {path.name}")
            data = loadmat(path, variable_names=["imagenes_celda"])
            if "imagenes_celda" not in data:
                raise ValueError(f"Missing imagenes_celda in {path.name}")
            cells = data["imagenes_celda"]
            frames = list(cells.flat)
            if not frames or any(
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
                or frame.shape != frames[0].shape
                for frame in frames
            ):
                raise ValueError(f"Unexpected thermal RGB frame structure in {path.name}")
            return frames


        def audit(source: Path, frequency: str = "F15", regime: str = "S") -> list[dict[str, str]]:
            """Print a condition-level inventory; never infer sample-level alignment."""
            records = [parsed for path in listed_paths(source)
                       if (parsed := parse_file(path)) is not None]
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for record in records:
                if record["frequency"] == frequency and record["regime"] == regime:
                    groups[record["fault"]].append(record)

            print(f"Recognized CORA files: {len(records)}")
            run_note = ("one published run per class/speed" if regime == "S"
                        else "transient measurements; thermal images were not recorded")
            print(f"Inventory: {frequency} / {regime} ({run_note})")
            print("Class | sensor channels | thermal stacks | JPG examples | status")
            print("------|-----------------|----------------|--------------|-------")
            for fault in FAULT_NAMES:
                files = groups[fault]
                channels = {item["channel"] for item in files if item["modality"] == "sensor"}
                thermal = sum(item["modality"] == "thermal_stack" for item in files)
                examples = sum(item["modality"] == "thermal_example" for item in files)
                status = "condition-level pair" if channels and thermal else "incomplete"
                print(f"{fault:5} | {len(channels):15} | {thermal:14} | {examples:12} | {status}")

            print("\nA shared experiment key links modalities by condition, not by individual sample.")
            print("Do not randomly split windows/images from one run into train and test.")
            print("Verify measurement time vectors and thermal timestamps before sample-level fusion.")
            return records


        def main() -> None:
            parser = argparse.ArgumentParser(description=inspect_file.__doc__)
            parser.add_argument("source", type=Path,
                                help="Local/cloud-mounted CORA folder or official file guide .txt")
            parser.add_argument("--frequency", choices=sorted(FREQUENCIES), default="F15")
            parser.add_argument("--regime", choices=("S", "T"), default="S")
            args = parser.parse_args()
            audit(args.source, args.frequency, args.regime)

        if run:
            main()
        return locals()


    # ==================================================================================================
    # preprocess: helper: CORA sensor/thermal preprocessing smoke test
    # (formerly Preprocessing/cora_preprocessing.py)
    # ==================================================================================================

    def preprocess(run: bool = True) -> dict:
        """Small-memory CORA preprocessing; no model training or cross-modal join.

        Use original, headerless CSV downloads for sensors. Dataverse's generated
        ``.tab`` view may treat the first numeric reading as a column heading.
        Thermal stacks are MATLAB-v5 ``.mat`` files with RGB frames. These functions
        retain experiment and within-file indices but do not claim that a frame index
        matches a sensor-window index.
        """


        import argparse
        from pathlib import Path

        import cv2
        import numpy as np
        import pandas as pd

        _manifest = inspect_file(run=False)
        load_thermal_stack, parse_file = _manifest["load_thermal_stack"], _manifest["parse_file"]


        SAMPLE_RATES_HZ = {
            **{name: 4000 for name in ("C1", "C2", "C3")},
            **{name: 3000 for name in ("VX", "VY", "VZ")},
            **{name: 1000 for name in ("T1", "T2", "T3", "T4", "T5", "T6")},
            "RPM": 1 / 30,
        }


        def sensor_window_features(
            path: Path, *, window_seconds: int = 6, max_windows: int | None = None,
        ) -> pd.DataFrame:
            """Stream one sensor channel into non-overlapping, fixed-duration features.

            A window index is relative to this file only. It is not a verified thermal
            timestamp or a guarantee that other channels start at the same instant.
            """
            record = parse_file(path)
            if record is None or record["modality"] != "sensor" or path.suffix.lower() != ".csv":
                raise ValueError("Use an original CORA sensor .csv file, not an ingested .tab file.")
            channel = record["channel"]
            if channel not in SAMPLE_RATES_HZ:
                raise ValueError(f"No verified sample rate for channel {channel}.")
            if window_seconds <= 0 or max_windows is not None and max_windows <= 0:
                raise ValueError("Window duration and maximum window count must be positive.")
            sample_rate = SAMPLE_RATES_HZ[channel]
            expected_samples = sample_rate * window_seconds
            samples_per_window = round(expected_samples)
            if samples_per_window < 1 or not np.isclose(expected_samples, samples_per_window):
                raise ValueError("Window duration is not an integer number of sensor samples.")

            rows: list[dict[str, float | int | str]] = []
            reader = load_csv(path, header=None, chunksize=samples_per_window)
            for index, chunk in enumerate(reader):
                if max_windows is not None and index >= max_windows:
                    break
                if chunk.shape != (samples_per_window, 1):
                    raise ValueError(f"Incomplete or multi-column window {index} in {path.name}.")
                values = pd.to_numeric(chunk.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite reading in window {index} of {path.name}.")
                mean = float(values.mean())
                centered = values - mean
                standard_deviation = float(values.std())
                rms = float(np.sqrt(np.mean(values ** 2)))
                peak = float(np.max(np.abs(values)))
                spectrum = np.abs(np.fft.rfft(centered)) ** 2
                if len(spectrum) > 1:
                    dominant_bin = int(np.argmax(spectrum[1:]) + 1)
                    dominant_hz = float(np.fft.rfftfreq(len(values), d=1 / sample_rate)[dominant_bin])
                    spectral_energy = float(spectrum[1:].sum() / len(values) ** 2)
                else:
                    dominant_hz = float("nan")
                    spectral_energy = 0.0
                rows.append({
                    "experiment": record["experiment"],
                    "channel": channel,
                    "window_index": index,
                    "mean": mean,
                    "std": standard_deviation,
                    "rms": rms,
                    "peak_to_peak": float(np.ptp(values)),
                    "crest_factor": peak / rms if rms else 0.0,
                    "dominant_hz": dominant_hz,
                    "spectral_energy": spectral_energy,
                })
            return pd.DataFrame(rows)


        def thermal_frame_features(path: Path, *, max_frames: int | None = None) -> pd.DataFrame:
            """Extract simple OpenCV image descriptors from one stationary .mat stack.

            RGB-shaped values may be repeated grayscale or false-colour renderings;
            do not interpret them as calibrated temperature readings. This is a
            preprocessing smoke test,
            not a nine-class image model.
            """
            record = parse_file(path)
            if record is None or record["modality"] != "thermal_stack":
                raise ValueError(f"Not a CORA thermal .mat stack: {path.name}")
            if max_frames is not None and max_frames <= 0:
                raise ValueError("Maximum frame count must be positive.")
            frames = load_thermal_stack(path)
            rows: list[dict[str, float | int | str]] = []
            for index, frame in enumerate(frames, start=1):
                if max_frames is not None and index > max_frames:
                    break
                resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 80, 160)
                rows.append({
                    "experiment": record["experiment"],
                    "frame_index": index,
                    "gray_mean": float(gray.mean()),
                    "gray_std": float(gray.std()),
                    "gray_p95": float(np.percentile(gray, 95)),
                    "edge_fraction": float(np.mean(edges > 0)),
                })
            return pd.DataFrame(rows)


        def main() -> None:
            parser = argparse.ArgumentParser(description=preprocess.__doc__)
            parser.add_argument("--sensor", type=Path, help="Original CORA sensor .csv")
            parser.add_argument("--thermal", type=Path, help="Stationary CORA .mat stack")
            parser.add_argument("--window-seconds", type=int, default=6)
            parser.add_argument("--max-windows", type=int, default=3)
            parser.add_argument("--max-frames", type=int, default=3)
            args = parser.parse_args()
            if not args.sensor and not args.thermal:
                parser.error("Provide --sensor and/or --thermal.")
            if args.sensor:
                features = sensor_window_features(
                    args.sensor, window_seconds=args.window_seconds,
                    max_windows=args.max_windows,
                )
                print("Sensor window features (not paired to thermal frames):")
                print(features.to_string(index=False))
            if args.thermal:
                features = thermal_frame_features(args.thermal, max_frames=args.max_frames)
                print("Thermal image features (not paired to sensor windows):")
                print(features.to_string(index=False))

        if run:
            main()
        return locals()


    # ==================================================================================================
    # experiment: T1 temperature and infrared models, report and figure
    # (formerly Models/belt_conveyor_cora_baseline.py)
    # ==================================================================================================

    def experiment(run: bool = True) -> dict:
        """Run-level CORA gear-wear sensor/infrared comparison for Equipment #2.

        The same 16 stationary motor/gearbox experiments feed separate T1 temperature
        and infrared-image classifiers. Leave-one-speed-out folds hold out all four
        condition runs at each speed. Windows/frames are never split across folds.
        This is a laboratory geared-drive analogue, not mining-conveyor validation.
        """


        import json
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import binomtest
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        CAPSTONE = ROOT
        RAW = CAPSTONE / "Raw_Data" / "cora_gear_wear"
        PROCESSED = CAPSTONE / "Processed_Data" / "cora_gear_wear"
        INTAKE = CAPSTONE / "Results" / "belt_conveyor" / "belt_conveyor_cora_intake.json"
        RESULT = CAPSTONE / "Results" / "belt_conveyor" / "belt_conveyor_cora.json"
        REPORT = CAPSTONE / "Results" / "belt_conveyor" / "belt_conveyor_sensor_visual_comparison.md"
        FIGURE = CAPSTONE / "Figures" / "belt_conveyor" / "belt_conveyor_cora_confusions.png"
        CLASSES = ("healthy", "gear_worn")
        CONDITIONS = ("H", "W25", "W50", "W75")
        SPEEDS = ("F5", "F15", "F50", "F60")
        RUNS = tuple(f"{condition}_{speed}_S" for condition in CONDITIONS for speed in SPEEDS)


        def sensor_features(values: np.ndarray) -> list[float]:
            if values.shape != (8, 6_000) or not np.isfinite(values).all():
                raise ValueError(f"Unexpected bounded T1 sample shape: {values.shape}")
            means = values.mean(axis=1)
            within_std = values.std(axis=1)
            low = np.percentile(values, 10, axis=1)
            high = np.percentile(values, 90, axis=1)
            trend = float(np.polyfit(np.arange(8), means, 1)[0])
            return [
                float(means.mean()), float(means.std()), float(np.median(means)),
                float(means.min()), float(means.max()), trend,
                float(within_std.mean()), float(np.mean(high - low)),
            ]


        def infrared_features(path: Path) -> list[float]:
            import cv2

            cells = loadmat(path, variable_names=["imagenes_celda"])["imagenes_celda"]
            if cells.size != 450:
                raise ValueError(f"Expected 450 infrared frames in {path.name}")
            rows = []
            for frame in cells.flat:
                if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"Invalid infrared frame in {path.name}")
                gray = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_RGB2GRAY)
                center = gray[32:96, 32:96].astype(np.float32)
                full = gray.astype(np.float32)
                rows.append([
                    float(full.mean()), float(full.std()),
                    float(np.percentile(full, 90)), float(np.percentile(full, 99)),
                    float(np.mean(full >= 220)),
                    float(center.mean() - full.mean()),
                    float(center.std()),
                    float(np.mean(np.abs(np.diff(full, axis=1)))),
                ])
            rows_array = np.asarray(rows)
            if not np.isfinite(rows_array).all():
                raise ValueError(f"Non-finite infrared descriptor in {path.name}")
            return np.concatenate([rows_array.mean(axis=0), rows_array.std(axis=0)]).tolist()


        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            correct = int(np.trace(matrix))
            interval = binomtest(correct, len(y_true)).proportion_ci(method="exact")
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_precision": float(np.mean(precision)),
                "macro_recall": float(np.mean(recall)),
                "macro_f1": float(np.mean(f1)),
                "correct_runs": correct,
                "test_runs": int(len(y_true)),
                "exploratory_accuracy_interval_95_percent": [float(interval.low), float(interval.high)],
                "confusion_matrix": matrix.tolist(),
                "per_class": {
                    label: {"precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(support[i])}
                    for i, label in enumerate(CLASSES)
                },
            }


        def evaluate(x: np.ndarray, y: np.ndarray, speeds: np.ndarray) -> dict:
            model_predictions = np.full(len(y), -1, dtype=int)
            dummy_predictions = np.full(len(y), -1, dtype=int)
            fold_rows = []
            for speed in SPEEDS:
                test = speeds == speed
                train = ~test
                if train.sum() != 12 or test.sum() != 4 or sorted(np.bincount(y[test])) != [1, 3]:
                    raise ValueError(f"Invalid whole-speed fold: {speed}")
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                       max_iter=1000, random_state=42),
                )
                model.fit(x[train], y[train])
                model_predictions[test] = model.predict(x[test])
                dummy = DummyClassifier(strategy="most_frequent")
                dummy.fit(x[train], y[train])
                dummy_predictions[test] = dummy.predict(x[test])
                fold_rows.append({
                    "held_out_speed": speed,
                    "train_runs": int(train.sum()),
                    "test_runs": int(test.sum()),
                    "actual": y[test].tolist(),
                    "predicted": model_predictions[test].tolist(),
                    "accuracy": float(accuracy_score(y[test], model_predictions[test])),
                })
            if np.any(model_predictions < 0) or np.any(dummy_predictions < 0):
                raise ValueError("Some runs lack an out-of-fold prediction")
            return {"model": metrics(y, model_predictions),
                    "most_frequent_control": metrics(y, dummy_predictions),
                    "folds": fold_rows,
                    "predictions": {run: int(prediction)
                                    for run, prediction in zip(RUNS, model_predictions)}}


        def draw_figures(sensor: dict, visual: dict) -> None:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
            for axis, record, title in zip(
                axes, (sensor, visual), ("T1 sensor (held-out speed)", "Infrared images (held-out speed)")
            ):
                matrix = np.asarray(record["model"]["confusion_matrix"])
                axis.imshow(matrix, cmap="Blues", vmin=0)
                axis.set_xticks([0, 1], CLASSES, rotation=20)
                axis.set_yticks([0, 1], CLASSES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(title)
                for row in range(2):
                    for col in range(2):
                        axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
            fig.suptitle("CORA gear-wear analogue: 16 whole-run out-of-fold predictions")
            fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.78, wspace=0.45)
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE, dpi=170)
            plt.close(fig)


        def metric_row(label: str, record: dict) -> str:
            return (f"| {label} | {record['accuracy']:.3f} | {record['balanced_accuracy']:.3f} | "
                    f"{record['macro_precision']:.3f} | {record['macro_recall']:.3f} | "
                    f"{record['macro_f1']:.3f} |")


        def render_report(result: dict) -> None:
            sensor = result["sensor"]
            visual = result["infrared"]
            lines = [
                "# Equipment #2 — Belt Conveyor geared-drive condition comparison",
                "",
                "## Scope and target",
                "",
                "The source is one controlled CORA motor/gearbox **laboratory rig**, not a mining conveyor. A geared conveyor drive reducer is mechanically comparable at the component level only; this result says nothing about belt cracks, idlers, or gearless drives. The binary target is **healthy (`H`) versus imposed gear-tooth wear (`W25`, `W50`, `W75`)**. It classifies present condition, not a future maintenance date.",
                "",
                "CORA supplies T1 temperature readings and infrared frames from the **same 16 stationary experiment runs**. We train separate models and compare their predictions at the run/fault-label level. We do **not** claim one-to-one frame/sensor-window synchronization, sample-level fusion, or two independent physical machines.",
                "",
                "## Leakage-safe evaluation",
                "",
                "Each of four folds holds out one complete speed setting (F5, F15, F50 or F60), containing one healthy and three worn runs. Training uses the other 12 whole runs. The run—not a sensor range or infrared frame—is the test unit. Both modalities use the **same four held-out-speed folds and same 16 test runs**. A fixed class-balanced logistic regression is fitted separately inside each fold, with feature scaling learned only on training runs; no test-selected features or hyperparameters. The majority-class control predicts worn in every fold.",
                "",
                "| Model / control | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |",
                "|---|---:|---:|---:|---:|---:|",
                metric_row("T1 sensor", sensor["model"]),
                metric_row("Infrared / OpenCV", visual["model"]),
                metric_row("Most-frequent control", sensor["most_frequent_control"]),
                "",
                "The scores above use **16 distinct experiment-level out-of-fold predictions per modality**, not thousands of independent frames/windows. Runs reuse one rig and may share a worn gear across speeds, so they are not 16 independent physical machines. The exploratory exact 95% intervals for accuracy are wide: sensor "
                + f"{sensor['model']['exploratory_accuracy_interval_95_percent'][0]:.3f}–{sensor['model']['exploratory_accuracy_interval_95_percent'][1]:.3f}; infrared "
                + f"{visual['model']['exploratory_accuracy_interval_95_percent'][0]:.3f}–{visual['model']['exploratory_accuracy_interval_95_percent'][1]:.3f}. These binomial intervals assume independent trials, an approximation here, and must not be interpreted as mine-fleet confidence bounds.",
                "",
                "Both learnt models are **below the 0.750 majority-class accuracy control**. The sensor model's macro F1 is above that control, while the infrared model's macro F1 is below it. Neither result supports reliable deployment or a claim that one modality is inherently superior.",
                "",
                "| Held-out speed | Sensor accuracy | Infrared accuracy | Test runs |",
                "|---|---:|---:|---:|",
            ]
            for sensor_fold, visual_fold in zip(sensor["folds"], visual["folds"]):
                if sensor_fold["held_out_speed"] != visual_fold["held_out_speed"]:
                    raise ValueError("Modality fold mismatch")
                lines.append(f"| {sensor_fold['held_out_speed']} | {sensor_fold['accuracy']:.3f} | {visual_fold['accuracy']:.3f} | {sensor_fold['test_runs']} |")
            lines.extend([
                "",
                "## Per-class and per-condition performance",
                "",
                "| Modality | Actual class | Precision | Recall | F1 | Runs |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for name, record in (("T1 sensor", sensor["model"]), ("Infrared", visual["model"])):
                for label in CLASSES:
                    item = record["per_class"][label]
                    lines.append(f"| {name} | {label} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |")
            lines.extend(["", "| Original CORA state | Runs | Sensor correctly classified | Infrared correctly classified |",
                          "|---|---:|---:|---:|"])
            for state in CONDITIONS:
                selected = [run for run in RUNS if run.startswith(state + "_")]
                actual = 0 if state == "H" else 1
                sensor_correct = sum(sensor["predictions"][run] == actual for run in selected)
                visual_correct = sum(visual["predictions"][run] == actual for run in selected)
                lines.append(f"| {state} | 4 | {sensor_correct}/4 | {visual_correct}/4 |")
            lines.extend([
                "",
                "## Run-level confusion matrices",
                "",
                "Rows are actual and columns predicted, in the order **healthy, gear-worn**.",
                "",
                "| Modality / actual | Predicted healthy | Predicted gear-worn |",
                "|---|---:|---:|",
            ])
            for name, record in (("Sensor healthy", sensor["model"]),
                                 ("Sensor worn", sensor["model"]),
                                 ("Infrared healthy", visual["model"]),
                                 ("Infrared worn", visual["model"])):
                row = 0 if name.endswith("healthy") else 1
                values = record["confusion_matrix"][row]
                lines.append(f"| {name} | {values[0]} | {values[1]} |")
            lines.extend([
                "",
                f"![Separate run-level confusion matrices](../../Figures/belt_conveyor/{FIGURE.name})",
                "",
                "## Data and reproducibility",
                "",
                "- [CORA version 2.1](https://doi.org/10.34810/DATA2500), CC BY 4.0. The [intake record](belt_conveyor_cora_intake.json) lists exact publisher file IDs, local SHA-256 checksums, source sizes and selected sensor byte ranges. All 16 MAT stacks passed filename/structure checks; H_F5_S also matched the publisher's MD5. The other 15 local SHA-256 values establish future local integrity but were not compared with publisher checksums. All source MAT stacks and sampled values remain in Git-ignored `Raw_Data/` and `Processed_Data/`.",
                "- **Sensor branch:** T1 is a source temperature channel sampled at 1 kHz. For each run we request eight deterministic, equally spaced 256-KiB ranges from the original CSV and retain the first 6,000 complete readings of each range. This is about 4.3% of source bytes and is **not full-signal modelling**. Eight predeclared time/variation features summarize each run.",
                "- **Visual branch:** all 450 published infrared frames per run are converted with OpenCV to grayscale intensity/spatial descriptors and summarized to one run-level vector. Pixel levels are not calibrated degrees Celsius. Publisher MATLAB files are used rather than equipment-identification photographs.",
                "- **No artificial pairing:** the two modalities share an experiment ID and fault condition, but their individual samples are not joined. Both branches have the same held-out speed/experiment units, making the run-level comparison more directly aligned than the earlier Pump cross-rig comparison.",
                "",
                "## Limitations and professor-requirement decision",
                "",
                "Only four speed settings and **one run per wear state/speed** are available. All 16 runs come from one laboratory rig; there is no unseen gearbox, mine, conveyor, load regime, or dated repair outcome. Reusing this CORA evidence for excavator, dozer, loader, drill, or grader later will **not** create additional independent equipment validations. Scores can compare how these fixed pipelines behave on the same held-out experiments, but are too small to establish sensor/infrared superiority or deployment performance.",
                "",
                "**Yes for completing the requested experimental comparison, conditional on the professor accepting a geared-drive laboratory reducer as a mechanically similar conveyor component.** Both real modalities address the same present gear-wear target using whole-run held-out-speed tests, and all requested metrics are reported. **No for an effective or validated predictive-maintenance model:** both accuracies are below the simple majority-class control, infrared healthy-class recall is 0.250, and there is no mining-conveyor test, sample-level fusion, or future-repair outcome. These weak results must be shown to the professor rather than described as a successful detector.",
                "",
            ])
            REPORT.write_text("\n".join(lines), encoding="utf-8")


        def main() -> None:
            intake = json.loads(INTAKE.read_text(encoding="utf-8"))
            if set(intake["runs"]) != set(RUNS):
                raise ValueError("Intake must contain all 16 CORA experiments")
            samples = load_numpy(PROCESSED / "t1_bounded_samples.npz")
            if set(samples.files) != set(RUNS):
                raise ValueError("Sensor cache must contain exactly 16 runs")
            sensor_rows = []
            visual_rows = []
            for index, run in enumerate(RUNS, 1):
                if set(intake["runs"][run]) != {"infrared", "temperature_sensor"}:
                    raise ValueError(f"Incomplete modality provenance: {run}")
                sensor_rows.append(sensor_features(samples[run]))
                visual_rows.append(infrared_features(RAW / f"{run}.mat"))
                print(f"Extracted run {index}/16: {run}", flush=True)
            sensor_x = np.asarray(sensor_rows)
            visual_x = np.asarray(visual_rows)
            y = np.asarray([0 if run.startswith("H_") else 1 for run in RUNS])
            speeds = np.asarray([run.split("_")[1] for run in RUNS])
            sensor = evaluate(sensor_x, y, speeds)
            visual = evaluate(visual_x, y, speeds)
            result = {
                "status": "complete_component_analogue_held_out_speed_evaluation",
                "equipment": "Belt Conveyor with a geared drive reducer",
                "source_domain": "Single controlled CORA motor/gearbox laboratory rig, not a mining conveyor",
                "source": "https://doi.org/10.34810/DATA2500",
                "target": "healthy versus imposed gear-tooth wear (W25/W50/W75)",
                "class_order": list(CLASSES),
                "original_conditions": list(CONDITIONS),
                "runs": list(RUNS),
                "split": "Four leave-one-speed-out folds; one held-out full run per original state in each fold",
                "independent_physical_machines": 1,
                "sensor": sensor,
                "infrared": visual,
                "same_experiment_level_ids": True,
                "sample_level_alignment_verified": False,
                "future_maintenance_prediction": False,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            draw_figures(sensor, visual)
            render_report(result)
            print(f"Saved {RESULT}, {REPORT}, and {FIGURE}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # baselines: Week 5 baseline model comparison (majority class, logistic regression, decision tree, random forest)
    # ====================================================================================================

    def baselines(run: bool = True) -> dict:
        """Week 5: compare baseline model families on the same held-out tests as the main experiment.

        A majority-class control, logistic regression, a decision tree and a random forest are trained
        with settings fixed in advance and scored on exactly the same held-out test units. No model or
        setting is chosen from these test results; the main model of the experiment stays the same.
        """
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        RESULT = ROOT / "Results" / "belt_conveyor" / "belt_conveyor_baselines.json"
        FIGURE = ROOT / "Figures" / "belt_conveyor" / "belt_conveyor_baselines.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)),
            "Decision tree": lambda: DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=42),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }
        SETTINGS = {
            "Majority class (control)": "always predicts the most common training class",
            "Logistic regression": "standardised features, C=1, class-balanced, liblinear",
            "Decision tree": "class-balanced, min_samples_leaf=2",
            "Random forest": "250 trees, sqrt features, class-balanced subsample, min_samples_leaf=2",
        }

        def unit_predictions(model, x, y, units):
            """One prediction per test unit (recording or run): average the model's probabilities over its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(model.classes_[int(np.argmax(model.predict_proba(x[rows]).mean(axis=0)))])
            return actual, predicted

        def evaluate(x, y, units, folds, classes):
            """`folds` are boolean test masks; every test unit is predicted by a model that never saw it."""
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds:
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model, x[test], y[test], units[test])
                    actual += list(a)
                    predicted += list(p)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predicted, labels=classes, zero_division=0)
                scores[name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_precision": round(float(precision.mean()), 4),
                    "macro_recall": round(float(recall.mean()), 4),
                    "macro_f1": round(float(f1.mean()), 4),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
                }
            return scores

        def plot(result):
            branches, names = list(result["branches"]), list(MODELS)
            width = 0.8 / len(names)
            figure, axis = plt.subplots(figsize=(2.2 + 2.8 * len(branches), 4.0))
            for i, name in enumerate(names):
                values = [result["branches"][b]["models"][name]["macro_f1"] for b in branches]
                bars = axis.bar(np.arange(len(branches)) + (i - (len(names) - 1) / 2) * width, values, width, label=name)
                axis.bar_label(bars, fmt="%.2f", fontsize=7)
            axis.set_xticks(range(len(branches)), branches)
            axis.set_ylim(0, 1.15)
            axis.set_ylabel("Macro F1 on held-out test")
            axis.set_title(result["title"], fontsize=10)
            axis.legend(fontsize=8, ncol=2, loc="upper right")
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=140)
            plt.close(figure)

        def main():
            branches = {}
            e = experiment(run=False)
            runs = list(e["RUNS"])
            samples = load_numpy(e["PROCESSED"] / "t1_bounded_samples.npz")
            sensor_x = np.asarray([e["sensor_features"](samples[r]) for r in runs])
            camera_x = np.asarray([e["infrared_features"](e["RAW"] / f"{r}.mat") for r in runs])
            y = np.asarray(["healthy" if r.startswith("H_") else "gear_worn" for r in runs])
            speeds = np.asarray([r.split("_")[1] for r in runs])
            units = np.asarray(runs)
            folds = [speeds == s for s in e["SPEEDS"]]
            classes = list(e["CLASSES"])
            test = "four rounds, each holding out one whole speed (4 runs); 16 run-level predictions"
            branches["Sensor: T1 temperature"] = {"test": test, "classes": classes,
                                                  "models": evaluate(sensor_x, y, units, folds, classes)}
            branches["Camera: infrared"] = {"test": test, "classes": classes,
                                            "models": evaluate(camera_x, y, units, folds, classes)}
            result = {
                "title": "Belt Conveyor: baseline models on held-out speeds",
                "purpose": "Week 5 baseline model comparison on the same held-out tests as the main experiment",
                "model_settings": SETTINGS,
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "branches": branches,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(result)
            for branch, block in branches.items():
                for name, score in block["models"].items():
                    print(f"{branch:<24} {name:<26} accuracy {score['accuracy']:.3f}  macro F1 {score['macro_f1']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # export-images: Week 6, save the camera images as named files (professor remark)
    # ====================================================================================================

    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/belt_conveyor/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "belt_conveyor"
        OUT = ROOT / "Images" / MACHINE
        STEP = 3  # every 3rd infrared frame (150 per run); --all-frames uses all 450
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5
        FRAMES = SECONDS * 1_000_000 // BIN_US

        def event_frames(stream):
            """First 5 s of an event recording as 20 images of 250 ms; channel 0 = brighter, 1 = darker events."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})

        def export():
            rows = []
            labels = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
            for condition, label in labels.items():
                for speed in ("F5", "F15", "F50", "F60"):
                    unit = f"{condition}_{speed}_S"
                    frames = loadmat(ROOT / "Raw_Data" / "cora_gear_wear" / f"{unit}.mat",
                                     variable_names=["imagenes_celda"])["imagenes_celda"]
                    for k, frame in enumerate(frames.flat):
                        if k % STEP:
                            continue  # every 3rd frame: 150 images per run
                        gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                        write(rows, label, unit, speed, f"{MACHINE}_{label}_{speed}_f{k:03d}.png", gray)
                    print(f"{unit}: done", flush=True)
            return rows


        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.add_argument("--all-frames", action="store_true",
                                help="save all 450 frames per run to Images/<machine>_allframes/ (Week 7 full-data run)")
            args = parser.parse_args()
            nonlocal OUT, STEP
            if args.all_frames:
                OUT, STEP = ROOT / "Images" / f"{MACHINE}_allframes", 1
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # csv: Week 6, 4-class sensor model from a CSV feature table (professor remark: "csv accuracy 90 plus")
    # ====================================================================================================

    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "belt_conveyor"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "belt_conveyor" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "belt_conveyor" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        VIB_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
        VIB_NAMES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                     *(f"order_power_{lo}_{hi}" for lo, hi in VIB_ORDERS)]

        def vibration_rows(run_name, label, speed):
            """CORA Vx vibration (3 kHz) in 1 s windows (every 10th window), measured in shaft orders.

            Orders = frequency / speed setting, so 1x, 2x and higher multiples line up across speeds:
            unbalance shows at 1x, misalignment at 2x, bearing and gear damage at higher orders and spikes.
            """
            cache = ROOT / "Processed_Data" / "cora_vibration" / f"{run_name}_Vx_features.csv"
            if cache.is_file():
                return load_csv(cache)
            source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
            if not source.is_file():
                raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
            values = load_cora_sensor(source.name)
            frame = vibration_frame(values, speed)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", label)
            frame.insert(0, "speed", speed)
            frame.insert(0, "unit", run_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame

        def vibration_frame(values, speed):
            """Order-domain vibration features for a raw 3 kHz Vx signal at speed setting F5/F15/F50/F60."""
            rate, shaft = 3000, float(speed[1:])
            n = len(values) // rate
            windows = values[: n * rate].reshape(n, rate)[::10]
            centered = windows - windows.mean(axis=1, keepdims=True)
            std = np.maximum(centered.std(axis=1), 1e-12)
            freq = np.fft.rfftfreq(rate, d=1 / rate)
            power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
            order = freq / shaft
            usable = (order >= 0.5) & (order < 20.5)
            share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
            o = order[usable]
            bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in VIB_ORDERS]
            frame = pd.DataFrame(np.column_stack([
                np.log(std), (centered ** 4).mean(axis=1) / std ** 4, (centered ** 3).mean(axis=1) / std ** 3,
                np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
                np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]), columns=VIB_NAMES)
            return frame

        def build_table():
            e = experiment(run=False)
            samples = load_numpy(e["PROCESSED"] / "t1_bounded_samples.npz")
            labels = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
            frames = []
            for run_name in e["RUNS"]:
                condition, speed = run_name.split("_")[:2]
                frame = vibration_rows(run_name, labels[condition], speed)
                t1 = samples[run_name].astype(np.float64)
                frame["t1_mean"], frame["t1_std"] = t1.mean(), t1.std()
                frame["t1_slope"] = np.polyfit(np.arange(8), t1.mean(axis=1), 1)[0]
                frames.append(frame)
            table = pd.concat(frames, ignore_index=True)
            folds = lambda t: [(t["speed"] == s).to_numpy() for s in e["SPEEDS"]]
            return (table, {"vibration": VIB_NAMES,
                            "vibration+temperature": VIB_NAMES + ["t1_mean", "t1_std", "t1_slope"]},
                    ["healthy", "wear25", "wear50", "wear75"], folds,
                    "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                    ["CORA Vx vibration (3 kHz, 1 s windows, every 10th window) in shaft orders, optionally with run-level T1 temperature",
                     "only one run per wear level and speed",
                     "per-speed normalisation is excluded because its group statistics would include held-out-speed samples"])
        INNER_GROUP = "speed"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test, as described in the result) or time (secondary, easier): train on "
                                     "the first 60%% of every recording, test on the last 30%% (10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier "
                                    "than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            if args.split == "time":
                result["split"] = "time (secondary)"
                RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                main_score = scores["Random forest"]
                plot_matrix(main_score["confusion_matrix"], labels,
                            f"{MACHINE}: CSV sensor, TIME SPLIT (random forest), accuracy {main_score['accuracy']:.3f}", FIGURE)
                for name, s in scores.items():
                    print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units, time split)")
                print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
                return
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn: Week 6, CNN with data augmentation on the named images (professor remark)
    # ====================================================================================================

    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "belt_conveyor"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "belt_conveyor" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "belt_conveyor" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "wear25", "wear50", "wear75"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [0]  # infrared images are grayscale
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        def augment(batch, generator):
            n = batch.shape[0]
            flip = torch.rand(n, generator=generator) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
            shifted = []
            for image in batch:
                dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
                shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
            batch = torch.stack(shifted)
            scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
            return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == s for s in ("F5", "F15", "F50", "F60")], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn-pretrained: Week 6, transfer learning with a pretrained ResNet18 on the named images
    # ====================================================================================================

    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "belt_conveyor"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "belt_conveyor" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "belt_conveyor" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "wear25", "wear50", "wear75"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        _small = cnn(run=False)
        read_index, augment = _small["read_index"], _small["augment"]

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary, easier: first 60%% of each recording's frames "
                                     "train, last 30%% test, 10%% gap)")
            parser.add_argument("--all-frames", action="store_true",
                                help="use Images/<machine>_allframes/ (all 450 frames per run); results saved as *_allframes")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE, IMAGES
            if args.all_frames:
                IMAGES = ROOT / "Images" / f"{MACHINE}_allframes"
                RESULT = RESULT.with_name(RESULT.stem + "_allframes.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_allframes.png")
            rows = list(csv_module.DictReader((IMAGES / "index.csv").open(encoding="utf-8")))
            test_description = "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions"
            if args.split == "time":
                import re
                position = {row["file"]: int(re.search(r"_f(\d{3})\.png$", row["file"]).group(1)) / 450 for row in rows}
                rows = [dict(row, unit=row["unit"] + ("_late" if position[row["file"]] > 0.7 else "_early"))
                        for row in rows if position[row["file"]] <= 0.6 or position[row["file"]] > 0.7]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording's frames train, last 30% test, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            folds = ([np.char.endswith(units.astype(str), "_late")] if args.split == "time"
                     else [groups == s for s in ("F5", "F15", "F50", "F60")])
            for number, test in enumerate(folds, 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": test_description,
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # intake-vibration: check and file selectively downloaded CORA files
    # ====================================================================================================

    def intake_vibration(run: bool = True) -> dict:
        """Check downloaded CORA files against the repository's MD5 and move them into Raw_Data/.

        The CORA repository (dataverse.csuc.cat, DOI 10.34810/DATA2500, CC BY 4.0) now shows a "not a bot"
        check, so these files cannot be fetched by code; download them from the dataset page in a browser
        (API: /api/access/datafile/<id>?format=original) and run this command. Default folder: ~/Downloads.
        """
        import argparse
        import hashlib
        import json
        import shutil
        from pathlib import Path

        MANIFEST = ROOT / "Results" / "belt_conveyor" / "belt_conveyor_vibration_intake.json"
        FILES = [  # (file name, destination folder in Raw_Data, size in bytes, publisher MD5)
            ("H_F5_S_Vx.csv", "cora_vibration", 111260279, "10e3d07a5647058f20d2f3565675baf7"),
            ("H_F15_S_Vx.csv", "cora_vibration", 107470262, "f75e4537077a2f2c9f8c7303aedc1d1e"),
            ("H_F50_S_Vx.csv", "cora_vibration", 110648041, "0247574694c415940832698ec0fd0bd3"),
            ("H_F60_S_Vx.csv", "cora_vibration", 110998372, "9d4549a7cd33d8cfdbf14afa320140ab"),
            ("W25_F5_S_Vx.csv", "cora_vibration", 108428177, "a47a92fb3172642047c908e828f1c679"),
            ("W25_F15_S_Vx.csv", "cora_vibration", 106848333, "e5efeff75c21ce8c694ea3a5e8bd30cc"),
            ("W25_F50_S_Vx.csv", "cora_vibration", 110528281, "fbf90f15b1882681779007964b414e2e"),
            ("W25_F60_S_Vx.csv", "cora_vibration", 112725506, "ebefd1cd62021596351dd092c8d79401"),
            ("W50_F5_S_Vx.csv", "cora_vibration", 107609236, "224b82f1b4f625b19b167842f322dba0"),
            ("W50_F15_S_Vx.csv", "cora_vibration", 106611962, "2cbcc2858cc9c5cd75d5b8649a53211c"),
            ("W50_F50_S_Vx.csv", "cora_vibration", 110525640, "a46a799430b5ced3953c05d679c5513e"),
            ("W50_F60_S_Vx.csv", "cora_vibration", 111405768, "b37c7b5ffb01c9e3110f6c19cbb5e1a3"),
            ("W75_F5_S_Vx.csv", "cora_vibration", 109445630, "70b4d708f21ea66177ad447ecf2b38f4"),
            ("W75_F15_S_Vx.csv", "cora_vibration", 105862619, "29c2239cb8af5419e8b5d3677ff52a0d"),
            ("W75_F50_S_Vx.csv", "cora_vibration", 111939715, "7c108c3eb71b7e2de1e8bf624e1e8418"),
            ("W75_F60_S_Vx.csv", "cora_vibration", 112056722, "ac87f5616ee8e935cf6cb81512c276a7"),
        ]

        def md5(path):
            checksum = hashlib.md5()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    checksum.update(chunk)
            return checksum.hexdigest()

        def main():
            parser = argparse.ArgumentParser(description=intake_vibration.__doc__)
            parser.add_argument("--from", dest="source", default=str(Path.home() / "Downloads"),
                                help="folder holding the browser downloads")
            args = parser.parse_args()
            source = Path(args.source)
            records, missing = [], []
            for name, folder, size, expected in FILES:
                destination = ROOT / "Raw_Data" / folder / name
                candidate = destination if destination.is_file() else source / name
                if not candidate.is_file():
                    missing.append(name)
                    continue
                if candidate.stat().st_size != size or md5(candidate) != expected:
                    raise SystemExit(f"{candidate}: size or MD5 does not match the repository; download it again")
                if candidate != destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidate), destination)
                records.append({"file": name, "folder": f"Raw_Data/{folder}", "size_bytes": size, "md5": expected,
                                "status": "verified against publisher MD5"})
                print(f"verified {name}", flush=True)
            MANIFEST.write_text(json.dumps({
                "source": "https://doi.org/10.34810/DATA2500", "license": "CC BY 4.0",
                "note": "selectively downloaded from CORA; every file checked against the publisher MD5",
                "files": records, "missing": missing}, indent=2) + "\n", encoding="utf-8")
            print(f"{len(records)} files verified; {len(missing)} still missing" + (f": {missing}" if missing else ""))

        if run:
            main()
        return locals()


    # ====================================================================================================
    # train-final / predict: use the trained sensor model on any CSV
    # ====================================================================================================

    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "vibration" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "belt_conveyor_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. W50_F15_S")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["vibration"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "belt_conveyor",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/belt_conveyor_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of CORA Vx vibration values sampled at 3 kHz (give --speed).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "belt_conveyor_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            if not args.speed:
                raise SystemExit("Raw recordings need --speed (F5, F15, F50 or F60)")
            ns = csv(run=False)
            return ns["vibration_frame"](values, args.speed)[ns["VIB_NAMES"]].to_numpy(dtype=np.float64)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, wear25, wear50, wear75")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")
            parser.add_argument("--speed", choices=("F5", "F15", "F50", "F60"), help="speed setting of a raw recording")
            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Proposal Week 6: advanced ensemble models (XGBoost / LightGBM comparison table)
    # ====================================================================================================

    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "belt_conveyor"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: anomaly detection (proposal Week 7: Isolation Forest / One-Class SVM)
    # ====================================================================================================

    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "belt_conveyor"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: full-data sensor experiment (every CORA sensor, window and recording)
    # ====================================================================================================

    def fulldata(run: bool = True) -> dict:
        """Week 7: every CORA sensor (vibration X/Y/Z, current C1-C3, temperatures T1-T6), every 1 s window, and the
        five start-up test repeats per condition and speed. See Models/cora_fulldata.py. Saves *_fulldata.json."""
        import argparse
        import importlib.util

        def main():
            parser = argparse.ArgumentParser(description=fulldata.__doc__)
            parser.add_argument("--part", choices=("classify", "ensembles", "anomaly", "all"), default="classify",
                                help="classify (default), ensembles (XGBoost/LightGBM comparison), anomaly, or all")
            part = parser.parse_args().part
            engine = get_model("cora_fulldata")
            conditions = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
            for name in ("classify", "ensembles", "anomaly"):
                if part in (name, "all"):
                    (engine.run if name == "classify" else getattr(engine, name))("belt_conveyor", conditions)

        if run:
            main()
        return locals()


    COMMANDS = {
        "fulldata": fulldata,
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
        "intake": intake,
        "inspect-file": inspect_file,
        "preprocess": preprocess,
        "experiment": experiment,
        "baselines": baselines,
        "export-images": export_images,
        "csv": csv,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
        "intake-vibration": intake_vibration,
    }
    RUN_ALL = ['experiment']


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"belt_conveyor.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# EXCAVATOR MODEL IMPLEMENTATION
def _make_excavator():
    __doc__ = "Excavator (machine 4): swing-drive motor and gearbox, CORA vibration vs infrared camera.\n\n4 classes from the CORA motor/gearbox rig: healthy, bearing defect, unbalance, misalignment. This is a\ndifferent task from the Belt Conveyor (gear wear) on the same laboratory rig. All of this machine's code\nis in this file; the combined notebook\nNotebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.\n\nUsage:\n    python Models/excavator.py <command> [options]\n    python Models/excavator.py <command> --help\n\nCommands:\n    intake          check selectively downloaded CORA files (MD5) and move them into Raw_Data/\n    csv             4-class vibration model from a CSV feature table (validation-selected model)\n    export-images   save the infrared images as named files in Images/excavator/ (local only)\n    cnn             CNN with data augmentation on the named images\n    cnn-pretrained  transfer learning, pretrained ResNet18 on the named images\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)\n    fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)\n                    (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)\n    thermal-motor   Week 8: separate thermal-image data (Kulkarni motor, healthy vs corroded bearing),\n                    leave one load out\n    all             run csv, then export-images, then cnn-pretrained\n\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Excavator (machine 4): swing-drive motor and gearbox, CORA vibration vs infrared camera.

    4 classes from the CORA motor/gearbox rig: healthy, bearing defect, unbalance, misalignment. This is a
    different task from the Belt Conveyor (gear wear) on the same laboratory rig. All of this machine's code
    is in this file; the combined notebook
    Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

    Usage:
        python Models/excavator.py <command> [options]
        python Models/excavator.py <command> --help

    Commands:
        intake          check selectively downloaded CORA files (MD5) and move them into Raw_Data/
        csv             4-class vibration model from a CSV feature table (validation-selected model)
        export-images   save the infrared images as named files in Images/excavator/ (local only)
        cnn             CNN with data augmentation on the named images
        cnn-pretrained  transfer learning, pretrained ResNet18 on the named images
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)
        fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)
                        (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)
        thermal-motor   Week 8: separate thermal-image data (Kulkarni motor, healthy vs corroded bearing),
                        leave one load out
        all             run csv, then export-images, then cnn-pretrained

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import sys
    from pathlib import Path




    # ====================================================================================================
    # intake: check and file selectively downloaded CORA files
    # ====================================================================================================

    def intake(run: bool = True) -> dict:
        """Check downloaded CORA files against the repository's MD5 and move them into Raw_Data/.

        The CORA repository (dataverse.csuc.cat, DOI 10.34810/DATA2500, CC BY 4.0) now shows a "not a bot"
        check, so these files cannot be fetched by code; download them from the dataset page in a browser
        (API: /api/access/datafile/<id>?format=original) and run this command. Default folder: ~/Downloads.
        """
        import argparse
        import hashlib
        import json
        import shutil
        from pathlib import Path

        MANIFEST = ROOT / "Results" / "excavator" / "excavator_intake.json"
        FILES = [  # (file name, destination folder in Raw_Data, size in bytes, publisher MD5)
            ("H_F5_S_Vx.csv", "cora_vibration", 111260279, "10e3d07a5647058f20d2f3565675baf7"),
            ("H_F15_S_Vx.csv", "cora_vibration", 107470262, "f75e4537077a2f2c9f8c7303aedc1d1e"),
            ("H_F50_S_Vx.csv", "cora_vibration", 110648041, "0247574694c415940832698ec0fd0bd3"),
            ("H_F60_S_Vx.csv", "cora_vibration", 110998372, "9d4549a7cd33d8cfdbf14afa320140ab"),
            ("BD_F5_S.mat", "cora_drive_faults", 14354542, "a67a54403673dcf7c271b637d3d8a736"),
            ("BD_F5_S_Vx.csv", "cora_vibration", 110475747, "1e3555d8576359997bb591f9b4a4b469"),
            ("BD_F15_S.mat", "cora_drive_faults", 20385070, "de197bf1881570c4ead8a7cf9c70fbd7"),
            ("BD_F15_S_Vx.csv", "cora_vibration", 108521895, "c54d048194cde04822bd6d515db5a12c"),
            ("BD_F50_S.mat", "cora_drive_faults", 23558647, "8c2b56f22e153a30e69bca324f93f19c"),
            ("BD_F50_S_Vx.csv", "cora_vibration", 110972006, "043b29da44cd0fa5e864a63e2d7adb72"),
            ("BD_F60_S.mat", "cora_drive_faults", 30278851, "23ace1e1df6c8f926bb81153615dd843"),
            ("BD_F60_S_Vx.csv", "cora_vibration", 111344017, "5fcbe8de8a247a4170cf4685dc34b6f1"),
            ("U_F5_S.mat", "cora_drive_faults", 16025524, "c008855036e5a9ebf217947975e6ff49"),
            ("U_F5_S_Vx.csv", "cora_vibration", 109911443, "a6f7900e31c33a6ee87a273d21535f33"),
            ("U_F15_S.mat", "cora_drive_faults", 20064343, "aac0ea78921239e400cae643f0504ccc"),
            ("U_F15_S_Vx.csv", "cora_vibration", 108320261, "5f9f1af5327aa1bbd5bb65eb361fd50c"),
            ("U_F50_S.mat", "cora_drive_faults", 29546944, "f40860b67574fb97d31a251e533d17d0"),
            ("U_F50_S_Vx.csv", "cora_vibration", 111114283, "f8840dbe57a8a0efed33f6141838f00e"),
            ("U_F60_S.mat", "cora_drive_faults", 25336431, "908b5cc3532a487f625a3dca9a3c7b2a"),
            ("U_F60_S_Vx.csv", "cora_vibration", 110897231, "935b73352713042265af7243ca79196b"),
            ("M_F5_S.mat", "cora_drive_faults", 12711576, "948aed478f80fbecf6efe82ceae1fd49"),
            ("M_F5_S_Vx.csv", "cora_vibration", 110661131, "184a132391cdeaa3353c3d98e906a57e"),
            ("M_F15_S.mat", "cora_drive_faults", 19258746, "a57280e7e3caf7fee5bb7ca8b7c3745e"),
            ("M_F15_S_Vx.csv", "cora_vibration", 108987453, "aea33473463e419d2f2906bef8b799fb"),
            ("M_F50_S.mat", "cora_drive_faults", 27878286, "0e29b3f286faea28645d13355fe3eed1"),
            ("M_F50_S_Vx.csv", "cora_vibration", 111444412, "39f0a36e3a70752ccee7717af7b361e0"),
            ("M_F60_S.mat", "cora_drive_faults", 30204185, "6d9c864a6e6cc9a426f75ba95d79c9e2"),
            ("M_F60_S_Vx.csv", "cora_vibration", 111054830, "673e7fb658880f129de97169074efc2d"),
        ]

        def md5(path):
            checksum = hashlib.md5()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    checksum.update(chunk)
            return checksum.hexdigest()

        def main():
            parser = argparse.ArgumentParser(description=intake.__doc__)
            parser.add_argument("--from", dest="source", default=str(Path.home() / "Downloads"),
                                help="folder holding the browser downloads")
            args = parser.parse_args()
            source = Path(args.source)
            records, missing = [], []
            for name, folder, size, expected in FILES:
                destination = ROOT / "Raw_Data" / folder / name
                candidate = destination if destination.is_file() else source / name
                if not candidate.is_file():
                    missing.append(name)
                    continue
                if candidate.stat().st_size != size or md5(candidate) != expected:
                    raise SystemExit(f"{candidate}: size or MD5 does not match the repository; download it again")
                if candidate != destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidate), destination)
                records.append({"file": name, "folder": f"Raw_Data/{folder}", "size_bytes": size, "md5": expected,
                                "status": "verified against publisher MD5"})
                print(f"verified {name}", flush=True)
            MANIFEST.write_text(json.dumps({
                "source": "https://doi.org/10.34810/DATA2500", "license": "CC BY 4.0",
                "note": "selectively downloaded from CORA; every file checked against the publisher MD5",
                "files": records, "missing": missing}, indent=2) + "\n", encoding="utf-8")
            print(f"{len(records)} files verified; {len(missing)} still missing" + (f": {missing}" if missing else ""))

        if run:
            main()
        return locals()


    # ====================================================================================================
    # export-images: Week 6, save the camera images as named files (professor remark)
    # ====================================================================================================

    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/excavator/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "excavator"
        OUT = ROOT / "Images" / MACHINE
        STEP = 3  # every 3rd infrared frame (150 per run); --all-frames uses all 450
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5
        FRAMES = SECONDS * 1_000_000 // BIN_US

        def event_frames(stream):
            """First 5 s of an event recording as 20 images of 250 ms; channel 0 = brighter, 1 = darker events."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})


        def export():
            rows = []
            labels = {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"}
            for condition, label in labels.items():
                folder = "cora_gear_wear" if condition == "H" else "cora_drive_faults"
                for speed in ("F5", "F15", "F50", "F60"):
                    unit = f"{condition}_{speed}_S"
                    frames = loadmat(ROOT / "Raw_Data" / folder / f"{unit}.mat",
                                     variable_names=["imagenes_celda"])["imagenes_celda"]
                    for k, frame in enumerate(frames.flat):
                        if k % STEP:
                            continue  # every 3rd frame: 150 images per run
                        gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                        write(rows, label, unit, speed, f"{MACHINE}_{label}_{speed}_f{k:03d}.png", gray)
                    print(f"{unit}: done", flush=True)
            return rows

        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.add_argument("--all-frames", action="store_true",
                                help="save all 450 frames per run to Images/<machine>_allframes/ (Week 7 full-data run)")
            args = parser.parse_args()
            nonlocal OUT, STEP
            if args.all_frames:
                OUT, STEP = ROOT / "Images" / f"{MACHINE}_allframes", 1
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # csv: Week 6, 4-class sensor model from a CSV feature table (professor remark: "csv accuracy 90 plus")
    # ====================================================================================================

    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "excavator"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "excavator" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "excavator" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        VIB_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
        VIB_NAMES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                     *(f"order_power_{lo}_{hi}" for lo, hi in VIB_ORDERS)]

        def vibration_rows(run_name, label, speed):
            """CORA Vx vibration (3 kHz) in 1 s windows (every 10th window), measured in shaft orders.

            Orders = frequency / speed setting, so 1x, 2x and higher multiples line up across speeds:
            unbalance shows at 1x, misalignment at 2x, bearing and gear damage at higher orders and spikes.
            """
            cache = ROOT / "Processed_Data" / "cora_vibration" / f"{run_name}_Vx_features.csv"
            if cache.is_file():
                return load_csv(cache)
            source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
            if not source.is_file():
                raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
            values = load_cora_sensor(source.name)
            frame = vibration_frame(values, speed)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", label)
            frame.insert(0, "speed", speed)
            frame.insert(0, "unit", run_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame

        def vibration_frame(values, speed):
            """Order-domain vibration features for a raw 3 kHz Vx signal at speed setting F5/F15/F50/F60."""
            rate, shaft = 3000, float(speed[1:])
            n = len(values) // rate
            windows = values[: n * rate].reshape(n, rate)[::10]
            centered = windows - windows.mean(axis=1, keepdims=True)
            std = np.maximum(centered.std(axis=1), 1e-12)
            freq = np.fft.rfftfreq(rate, d=1 / rate)
            power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
            order = freq / shaft
            usable = (order >= 0.5) & (order < 20.5)
            share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
            o = order[usable]
            bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in VIB_ORDERS]
            frame = pd.DataFrame(np.column_stack([
                np.log(std), (centered ** 4).mean(axis=1) / std ** 4, (centered ** 3).mean(axis=1) / std ** 3,
                np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
                np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]), columns=VIB_NAMES)
            return frame

        def per_speed_normalised(table, names):
            """Rescale each feature within each speed using all recordings at that speed together (no labels used).

            Removes what each speed 'sounds like', so the model compares machines running at the same speed.
            It assumes several machines are recorded at each speed, as in this dataset.
            """
            out = []
            for name in names:
                column = f"{name}_per_speed"
                grouped = table.groupby("speed")[name]
                table[column] = (table[name] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
                out.append(column)
            return out

        def build_table():
            labels = {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"}
            frames = [vibration_rows(f"{c}_{s}_S", label, s)
                      for c, label in labels.items() for s in ("F5", "F15", "F50", "F60")]
            table = pd.concat(frames, ignore_index=True)
            folds = lambda t: [(t["speed"] == s).to_numpy() for s in ("F5", "F15", "F50", "F60")]
            normalised = per_speed_normalised(table, VIB_NAMES)
            return (table, {"vibration": VIB_NAMES, "vibration (per-speed normalised)": normalised},
                    ["healthy", "bearing", "unbalance", "misalignment"], folds,
                    "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions",
                    ["CORA motor/gearbox rig, Vx vibration at 3 kHz; one row per 1 s window, every 10th window of each run",
                     "features in shaft orders (frequency / speed setting) so they line up across the held-out speed"])
        INNER_GROUP = "speed"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.add_argument("--split", choices=("speed", "time"), default="speed",
                                help="speed (main, fair test): hold out each whole speed. time (secondary): train on the "
                                     "first 60%% of every run, test on the last 30%% (10%% gap); the machine has been seen "
                                     "before, and each fault comes from one recording, so this is an easier test")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            if args.split == "time":
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every run trains, last 30% tests, 10% gap; "
                                    "16 run-level predictions; the same recordings appear in training, so this is easier "
                                    "than the main unseen-speed test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            if args.split == "time":
                result["split"] = "time (secondary)"
                RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                main_score = scores["Random forest"]
                plot_matrix(main_score["confusion_matrix"], labels,
                            f"{MACHINE}: CSV sensor, TIME SPLIT (random forest), accuracy {main_score['accuracy']:.3f}", FIGURE)
                for name, s in scores.items():
                    print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units, time split)")
                print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
                return
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn: Week 6, CNN with data augmentation on the named images (professor remark)
    # ====================================================================================================

    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "excavator"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "excavator" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "excavator" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "bearing", "unbalance", "misalignment"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [0]  # infrared images are grayscale
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        def augment(batch, generator):
            n = batch.shape[0]
            flip = torch.rand(n, generator=generator) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
            shifted = []
            for image in batch:
                dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
                shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
            batch = torch.stack(shifted)
            scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
            return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == s for s in ("F5", "F15", "F50", "F60")], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn-pretrained: Week 6, transfer learning with a pretrained ResNet18 on the named images
    # ====================================================================================================

    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "excavator"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "excavator" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "excavator" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "bearing", "unbalance", "misalignment"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        _small = cnn(run=False)
        read_index, augment = _small["read_index"], _small["augment"]

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--split", choices=("speed", "time"), default="speed",
                                help="speed (main, fair test) or time (secondary: first 60%% of each run's frames train, "
                                     "last 30%% test, 10%% gap; easier, the same recordings appear in training)")
            parser.add_argument("--all-frames", action="store_true",
                                help="use Images/<machine>_allframes/ (all 450 frames per run); results saved as *_allframes")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE, IMAGES
            if args.all_frames:
                IMAGES = ROOT / "Images" / f"{MACHINE}_allframes"
                RESULT = RESULT.with_name(RESULT.stem + "_allframes.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_allframes.png")
            rows = list(csv_module.DictReader((IMAGES / "index.csv").open(encoding="utf-8")))
            test_description = "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions"
            if args.split == "time":
                import re
                position = {row["file"]: int(re.search(r"_f(\d{3})\.png$", row["file"]).group(1)) / 450 for row in rows}
                rows = [dict(row, unit=row["unit"] + ("_late" if position[row["file"]] > 0.7 else "_early"))
                        for row in rows if position[row["file"]] <= 0.6 or position[row["file"]] > 0.7]
                test_description = ("TIME SPLIT (secondary): first 60% of every run's frames train, last 30% test, 10% gap; "
                                    "16 run-level predictions; the same recordings appear in training, so this is easier")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            folds = ([np.char.endswith(units.astype(str), "_late")] if args.split == "time"
                     else [groups == s for s in ("F5", "F15", "F50", "F60")])
            for number, test in enumerate(folds, 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": test_description,
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # train-final / predict: use the trained sensor model on any CSV
    # ====================================================================================================

    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "vibration" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "excavator_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. U_F15_S")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["vibration"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "excavator",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/excavator_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of CORA Vx vibration values sampled at 3 kHz (give --speed).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "excavator_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            if not args.speed:
                raise SystemExit("Raw recordings need --speed (F5, F15, F50 or F60)")
            ns = csv(run=False)
            return ns["vibration_frame"](values, args.speed)[ns["VIB_NAMES"]].to_numpy(dtype=np.float64)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, bearing, unbalance, misalignment")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")
            parser.add_argument("--speed", choices=("F5", "F15", "F50", "F60"), help="speed setting of a raw recording")
            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Proposal Week 6: advanced ensemble models (XGBoost / LightGBM comparison table)
    # ====================================================================================================

    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "excavator"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: anomaly detection (proposal Week 7: Isolation Forest / One-Class SVM)
    # ====================================================================================================

    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "excavator"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: full-data sensor experiment (every CORA sensor, window and recording)
    # ====================================================================================================

    def fulldata(run: bool = True) -> dict:
        """Week 7: every CORA sensor (vibration X/Y/Z, current C1-C3, temperatures T1-T6), every 1 s window, and the
        five start-up test repeats per condition and speed. See Models/cora_fulldata.py. Saves *_fulldata.json."""
        import argparse
        import importlib.util

        def main():
            parser = argparse.ArgumentParser(description=fulldata.__doc__)
            parser.add_argument("--part", choices=("classify", "ensembles", "anomaly", "all"), default="classify",
                                help="classify (default), ensembles (XGBoost/LightGBM comparison), anomaly, or all")
            part = parser.parse_args().part
            engine = get_model("cora_fulldata")
            conditions = {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"}
            for name in ("classify", "ensembles", "anomaly"):
                if part in (name, "all"):
                    (engine.run if name == "classify" else getattr(engine, name))("excavator", conditions)

        if run:
            main()
        return locals()


    def thermal_motor(run: bool = True) -> dict:
        """Week 8: separate image data for the Excavator. Induction-motor thermal images (Kulkarni et al.,
        Mendeley Data pymtdhfzbj v2, CC BY 4.0): healthy vs corroded bearing, each filmed at no, medium and full
        load (6 sessions, 488 images). Fair test: leave one load out (train on two loads, test on the unseen
        third), 3 rounds. The burned-in temperature reading and logo (top band) are masked. Inside each training
        set the other load is used to choose the number of epochs. A brightness-only model is the control: if it
        scores as well as the CNN, the CNN is only reading overall temperature.
        """
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        SOURCE = ROOT / "Raw_Data" / "kulkarni_motor_thermal" / "ext" / "Induction Motor-Thermal Images"
        RESULT = ROOT / "Results" / "excavator" / "excavator_thermal_motor_2class.json"
        FIGURE = ROOT / "Figures" / "excavator" / "excavator_thermal_motor_2class.png"
        LABELS = ["healthy", "bearing"]
        FOLDERS = {"H": "healthy", "C": "bearing"}           # H = healthy, C = corroded bearing
        LOADS = {"NL": "no load", "ML": "medium load", "FL": "full load"}
        SIZE, MASK_TOP, SEGMENTS = 224, 0.15, 10
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 8, 32, 3e-4, 613

        def load_images():
            x, y, load, session = [], [], [], []
            for folder in sorted(SOURCE.iterdir()):
                label, load_code = FOLDERS[folder.name[0]], folder.name[1:]
                for path in sorted(folder.glob("*.jpg")):
                    image = cv2.cvtColor(load_image(str(path)), cv2.COLOR_BGR2RGB)
                    image[: int(MASK_TOP * image.shape[0])] = 0        # hide the temperature reading and logo
                    x.append(cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
                    y.append(LABELS.index(label)), load.append(load_code), session.append(folder.name)
            return np.stack(x), np.array(y), np.array(load), np.array(session)

        def to_tensor(images):
            mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
            return (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255 - mean) / std

        def augment(batch, generator):
            flip = torch.rand(batch.shape[0], generator=generator) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
            scale = 0.8 + 0.4 * torch.rand(batch.shape[0], 1, 1, 1, generator=generator)
            return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            for name, parameter in model.named_parameters():      # fine-tune only the last block (CPU friendly)
                parameter.requires_grad = name.startswith(("layer4", "fc"))
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def predict(model, x):
            model.eval()
            with torch.no_grad():
                return torch.cat([model(x[i:i + 64]).softmax(1) for i in range(0, len(x), 64)]).numpy()

        def train(x, y, epochs, x_check=None, y_check=None):
            """Train for `epochs`; when check data is given, return the check accuracy after every epoch."""
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
            loss_fn = nn.CrossEntropyLoss()
            yt, history = torch.from_numpy(y).long(), []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(x), generator=generator)
                for start in range(0, len(x), BATCH):
                    rows = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss_fn(model(augment(x[rows], generator)), yt[rows]).backward()
                    optimiser.step()
                if x_check is not None:
                    history.append(accuracy_score(y_check, predict(model, x_check).argmax(1)))
            return model, history

        def score(actual, predicted):
            return {"test_units": int(len(actual)),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=range(len(LABELS))).tolist()}

        def plot(result, path):
            levels = [("every test image", result["image_level"]), ("per 1/10 session segment", result["segment_level"]),
                      ("per whole session", result["session_level"]),
                      ("control: brightness only", result["brightness_control_image_level"]),
                      ("control: border only, motor hidden", result["border_only_control_image_level"])]
            figure, axes = plt.subplots(1, len(levels), figsize=(3.4 * len(levels), 3.8))
            for axis, (name, block) in zip(axes, levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS)
                axis.set_yticks(range(len(LABELS)), LABELS)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=9,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(f"Excavator bearing, separate thermal-image data (Kulkarni): {result['images']} images, "
                            f"6 sessions; each load held out in turn", fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        def main():
            if not SOURCE.is_dir():
                raise SystemExit(f"Missing {SOURCE}: download Mendeley Data pymtdhfzbj v2 and extract it there")
            random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
            images, y, load, session = load_images()
            x = to_tensor(images)
            brightness = images.reshape(len(images), -1).mean(axis=1, keepdims=True)
            # second control: tiny grey images with the motor area hidden, so only framing and background remain
            border = images.mean(axis=3).copy()
            border[:, int(0.25 * SIZE):int(0.85 * SIZE), int(0.2 * SIZE):int(0.8 * SIZE)] = 0
            border = np.stack([cv2.resize(b, (16, 12), interpolation=cv2.INTER_AREA).ravel() for b in border]) / 255
            print(f"{len(y)} images, {len(set(session))} sessions: "
                  + ", ".join(f"{s} {int((session == s).sum())}" for s in sorted(set(session))))
            image_actual, image_predicted, control_predicted, border_predicted = [], [], [], []
            segment_actual, segment_predicted, session_actual, session_predicted = [], [], [], []
            per_session, folds = {}, []
            for test_load in LOADS:
                train_rows, test_rows = load != test_load, load == test_load
                # choose epochs: train on one training load, check on the other, both ways; average the curves
                curves = []
                for check_load in sorted(set(load[train_rows])):
                    inner_train, inner_check = train_rows & (load != check_load), load == check_load
                    curves.append(train(x[inner_train], y[inner_train], MAX_EPOCHS, x[inner_check], y[inner_check])[1])
                epochs = int(np.argmax(np.mean(curves, axis=0))) + 1
                model, _ = train(x[train_rows], y[train_rows], epochs)
                p = predict(model, x[test_rows])
                control = LogisticRegression().fit(brightness[train_rows], y[train_rows]).predict(brightness[test_rows])
                image_actual += list(y[test_rows])
                image_predicted += list(p.argmax(1))
                control_predicted += list(control)
                border_control = LogisticRegression(max_iter=3000).fit(border[train_rows], y[train_rows]).predict(border[test_rows])
                border_predicted += list(border_control)
                for s in sorted(set(session[test_rows])):
                    rows = session[test_rows] == s
                    ps, true = p[rows], int(y[test_rows][rows][0])
                    for part in np.array_split(np.arange(len(ps)), SEGMENTS):
                        segment_actual.append(true), segment_predicted.append(int(ps[part].mean(0).argmax()))
                    session_actual.append(true), session_predicted.append(int(ps.mean(0).argmax()))
                    per_session[s] = {"class": LABELS[true], "images": int(len(ps)),
                                      "images_predicted": {l: int((ps.argmax(1) == i).sum()) for i, l in enumerate(LABELS)},
                                      "session_prediction": LABELS[session_predicted[-1]]}
                folds.append({"test_load": LOADS[test_load], "epochs_chosen": epochs, "test_images": int(test_rows.sum()),
                              "image_accuracy": round(float(accuracy_score(y[test_rows], p.argmax(1))), 4),
                              "brightness_control_accuracy": round(float(accuracy_score(y[test_rows], control)), 4),
                              "border_only_control_accuracy": round(float(accuracy_score(y[test_rows], border_control)), 4)})
                print(f"held-out {LOADS[test_load]:11s}: {epochs} epochs; image accuracy {folds[-1]['image_accuracy']:.3f}"
                      f" (brightness control {folds[-1]['brightness_control_accuracy']:.3f}, "
                      f"border-only control {folds[-1]['border_only_control_accuracy']:.3f})", flush=True)
            result = {
                "purpose": "separate image data for the Excavator bearing task (not CORA)",
                "source": "Kulkarni et al., Induction Motor-Thermal Images, Mendeley Data pymtdhfzbj v2, CC BY 4.0",
                "images": int(len(y)), "classes": LABELS,
                "test": "leave one load out (no / medium / full load), 3 rounds; epochs chosen on the training loads only",
                "preprocessing": f"top {int(MASK_TOP * 100)}% of every image masked (temperature reading and logo), "
                                 f"resized to {SIZE}x{SIZE}; ImageNet ResNet18 with only layer4 and the head trained",
                "settings": {"max_epochs": MAX_EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED},
                "folds": folds,
                "image_level": score(image_actual, image_predicted),
                "segment_level": {**score(segment_actual, segment_predicted), "segments_per_session": SEGMENTS},
                "session_level": score(session_actual, session_predicted),
                "brightness_control_image_level": score(image_actual, control_predicted),
                "border_only_control_image_level": score(image_actual, border_predicted),
                "per_session": per_session,
                "caveat": "only 6 sessions (one per class and load) from one motor; images within a session are "
                          "consecutive and related, and only 2 classes match the Excavator's 4. The healthy and "
                          "corroded sessions were filmed with different framing: a model that sees only the image "
                          "border (motor hidden) also scores well, so a high CNN score does not prove it detects "
                          "bearing corrosion",
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(result, FIGURE)
            print(f"Image accuracy {result['image_level']['accuracy']:.3f} on {len(y)} images; session accuracy "
                  f"{result['session_level']['accuracy']:.3f} on 6 sessions; brightness control "
                  f"{result['brightness_control_image_level']['accuracy']:.3f}; border-only control "
                  f"{result['border_only_control_image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    COMMANDS = {
        "thermal-motor": thermal_motor,
        "fulldata": fulldata,
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
        "intake": intake,
        "csv": csv,
        "export-images": export_images,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
    }
    RUN_ALL = ['csv', 'export-images', 'cnn-pretrained']


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"excavator.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# DOZER MODEL IMPLEMENTATION
def _make_dozer():
    __doc__ = "Dozer (machine 5): track final-drive gear wear, CORA vibration/temperature vs infrared camera.\n\nAll of this machine's code in one file. The combined notebook\nNotebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.\n\nIMPORTANT: there is no public dozer dataset with matching sensor and camera data. As planned in the\ndataset manifest, the Dozer final drive uses the SAME 16 CORA gear-wear runs as the Belt Conveyor\n(not new data), with the same 4 classes: healthy, wear25, wear50, wear75. Its scores are\ntherefore expected to match the Belt Conveyor's; it maps that rig to the dozer track final drive.\nThe data loading is shared with Models/belt_conveyor.py.\n\nUsage:\n    python Models/dozer.py <command> [options]\n    python Models/dozer.py <command> --help\n\nCommands:\n    csv             4-class sensor model from a CSV feature table (--split time for the secondary test)\n    export-images   save the infrared images as named files in Images/dozer/<class>/ (local only)\n    cnn             small CNN with data augmentation on the named images (run export-images first)\n    cnn-pretrained  pretrained ResNet18 on the named images (--split time for the secondary test)\n    ensembles       XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         anomaly detection trained on healthy data only (--split time for the secondary test)\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)\n                    (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)\n    all             run csv, ensembles and anomaly\n\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Dozer (machine 5): track final-drive gear wear, CORA vibration/temperature vs infrared camera.

    All of this machine's code in one file. The combined notebook
    Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

    IMPORTANT: there is no public dozer dataset with matching sensor and camera data. As planned in the
    dataset manifest, the Dozer final drive uses the SAME 16 CORA gear-wear runs as the Belt Conveyor
    (not new data), with the same 4 classes: healthy, wear25, wear50, wear75. Its scores are
    therefore expected to match the Belt Conveyor's; it maps that rig to the dozer track final drive.
    The data loading is shared with Models/belt_conveyor.py.

    Usage:
        python Models/dozer.py <command> [options]
        python Models/dozer.py <command> --help

    Commands:
        csv             4-class sensor model from a CSV feature table (--split time for the secondary test)
        export-images   save the infrared images as named files in Images/dozer/<class>/ (local only)
        cnn             small CNN with data augmentation on the named images (run export-images first)
        cnn-pretrained  pretrained ResNet18 on the named images (--split time for the secondary test)
        ensembles       XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         anomaly detection trained on healthy data only (--split time for the secondary test)
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)
                        (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)
        all             run csv, ensembles and anomaly

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import importlib.util
    import sys
    from pathlib import Path



    # The Dozer reuses the Belt Conveyor's CORA data loading (same rig, same runs).
    BELT = get_model("belt_conveyor")


    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/dozer/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "dozer"
        OUT = ROOT / "Images" / MACHINE
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5
        FRAMES = SECONDS * 1_000_000 // BIN_US

        def event_frames(stream):
            """First 5 s of an event recording as 20 images of 250 ms; channel 0 = brighter, 1 = darker events."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})

        def export():
            rows = []
            labels = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
            for condition, label in labels.items():
                for speed in ("F5", "F15", "F50", "F60"):
                    unit = f"{condition}_{speed}_S"
                    frames = loadmat(ROOT / "Raw_Data" / "cora_gear_wear" / f"{unit}.mat",
                                     variable_names=["imagenes_celda"])["imagenes_celda"]
                    for k, frame in enumerate(frames.flat):
                        if k % 3:
                            continue  # every 3rd frame: 150 images per run
                        gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                        write(rows, label, unit, speed, f"{MACHINE}_{label}_{speed}_f{k:03d}.png", gray)
                    print(f"{unit}: done", flush=True)
            return rows


        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.parse_args()
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()



    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "dozer"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        VIB_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
        VIB_NAMES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                     *(f"order_power_{lo}_{hi}" for lo, hi in VIB_ORDERS)]

        def vibration_rows(run_name, label, speed):
            """CORA Vx vibration (3 kHz) in 1 s windows (every 10th window), measured in shaft orders.

            Orders = frequency / speed setting, so 1x, 2x and higher multiples line up across speeds:
            unbalance shows at 1x, misalignment at 2x, bearing and gear damage at higher orders and spikes.
            """
            cache = ROOT / "Processed_Data" / "cora_vibration" / f"{run_name}_Vx_features.csv"
            if cache.is_file():
                return load_csv(cache)
            source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
            if not source.is_file():
                raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
            values = load_cora_sensor(source.name)
            frame = vibration_frame(values, speed)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", label)
            frame.insert(0, "speed", speed)
            frame.insert(0, "unit", run_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame

        def vibration_frame(values, speed):
            """Order-domain vibration features for a raw 3 kHz Vx signal at speed setting F5/F15/F50/F60."""
            rate, shaft = 3000, float(speed[1:])
            n = len(values) // rate
            windows = values[: n * rate].reshape(n, rate)[::10]
            centered = windows - windows.mean(axis=1, keepdims=True)
            std = np.maximum(centered.std(axis=1), 1e-12)
            freq = np.fft.rfftfreq(rate, d=1 / rate)
            power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
            order = freq / shaft
            usable = (order >= 0.5) & (order < 20.5)
            share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
            o = order[usable]
            bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in VIB_ORDERS]
            frame = pd.DataFrame(np.column_stack([
                np.log(std), (centered ** 4).mean(axis=1) / std ** 4, (centered ** 3).mean(axis=1) / std ** 3,
                np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
                np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]), columns=VIB_NAMES)
            return frame

        def build_table():
            """Same CORA runs, features and 4 wear classes as the Belt Conveyor (no public dozer dataset exists)."""
            table, feature_sets, labels, folds, _, _ = BELT.csv(run=False)["build_table"]()
            return (table, feature_sets, labels, folds,
                    "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                    ["CORA motor/gearbox rig, the SAME 16 runs, features and 4 wear classes as the Belt Conveyor (not new "
                     "data), so the scores are expected to equal the Belt Conveyor's; mapped to the Dozer track final drive",
                     "per-speed normalisation is excluded because its group statistics would include held-out-speed samples"])
        INNER_GROUP = "speed"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test, as described in the result) or time (secondary, easier): train on "
                                     "the first 60%% of every recording, test on the last 30%% (10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier "
                                    "than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            if args.split == "time":
                result["split"] = "time (secondary)"
                RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                main_score = scores["Random forest"]
                plot_matrix(main_score["confusion_matrix"], labels,
                            f"{MACHINE}: CSV sensor, TIME SPLIT (random forest), accuracy {main_score['accuracy']:.3f}", FIGURE)
                for name, s in scores.items():
                    print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units, time split)")
                print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
                return
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()



    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "dozer"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "wear25", "wear50", "wear75"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [0]  # infrared images are grayscale
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        def augment(batch, generator):
            n = batch.shape[0]
            flip = torch.rand(n, generator=generator) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
            shifted = []
            for image in batch:
                dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
                shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
            batch = torch.stack(shifted)
            scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
            return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == s for s in ("F5", "F15", "F50", "F60")], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()



    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "dozer"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "wear25", "wear50", "wear75"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        _small = cnn(run=False)
        read_index, augment = _small["read_index"], _small["augment"]

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary, easier: first 60%% of each recording's frames "
                                     "train, last 30%% test, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            rows = read_index()
            test_description = "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions"
            if args.split == "time":
                import re
                position = {row["file"]: int(re.search(r"_f(\d{3})\.png$", row["file"]).group(1)) / 450 for row in rows}
                rows = [dict(row, unit=row["unit"] + ("_late" if position[row["file"]] > 0.7 else "_early"))
                        for row in rows if position[row["file"]] <= 0.6 or position[row["file"]] > 0.7]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording's frames train, last 30% test, 10% gap; "
                                    "one prediction per recording; the same recordings appear in training, so this is easier")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            folds = ([np.char.endswith(units.astype(str), "_late")] if args.split == "time"
                     else [groups == s for s in ("F5", "F15", "F50", "F60")])
            for number, test in enumerate(folds, 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": test_description,
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()



    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "vibration" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "dozer_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. W50_F15_S")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["vibration"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "dozer",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()



    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/dozer_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of CORA Vx vibration values sampled at 3 kHz (give --speed).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "dozer_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            if not args.speed:
                raise SystemExit("Raw recordings need --speed (F5, F15, F50 or F60)")
            ns = csv(run=False)
            return ns["vibration_frame"](values, args.speed)[ns["VIB_NAMES"]].to_numpy(dtype=np.float64)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, wear25, wear50, wear75")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")
            parser.add_argument("--speed", choices=("F5", "F15", "F50", "F60"), help="speed setting of a raw recording")
            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()



    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "dozer"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()



    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "dozer"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: full-data sensor experiment (every CORA sensor, window and recording)
    # ====================================================================================================

    def fulldata(run: bool = True) -> dict:
        """Week 7: every CORA sensor (vibration X/Y/Z, current C1-C3, temperatures T1-T6), every 1 s window, and the
        five start-up test repeats per condition and speed. See Models/cora_fulldata.py. Saves *_fulldata.json."""
        import argparse
        import importlib.util

        def main():
            parser = argparse.ArgumentParser(description=fulldata.__doc__)
            parser.add_argument("--part", choices=("classify", "ensembles", "anomaly", "all"), default="classify",
                                help="classify (default), ensembles (XGBoost/LightGBM comparison), anomaly, or all")
            part = parser.parse_args().part
            engine = get_model("cora_fulldata")
            conditions = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
            for name in ("classify", "ensembles", "anomaly"):
                if part in (name, "all"):
                    (engine.run if name == "classify" else getattr(engine, name))("dozer", conditions)

        if run:
            main()
        return locals()


    COMMANDS = {
        "fulldata": fulldata,
        "csv": csv,
        "export-images": export_images,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
    }
    RUN_ALL = ["csv", "ensembles", "anomaly"]


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"dozer.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# WHEEL_LOADER MODEL IMPLEMENTATION
def _make_wheel_loader():
    __doc__ = "Wheel Loader (machine 6): electric drive motor, CORA vibration vs infrared camera.\n\n4 classes from the CORA motor/gearbox rig: healthy, bearing defect, half-broken rotor bar and broken rotor\nbar. The two rotor-bar faults are CORA conditions no other machine in this project uses (new fault data);\nhealthy and bearing-defect runs are shared with the Excavator. Large wheel loaders can use electric drive,\nso the induction motor stands in for the loader's drive motor. Lab data, not a wheel loader.\n\nUsage:\n    python Models/wheel_loader.py <command> [options]\n    python Models/wheel_loader.py <command> --help\n\nCommands:\n    intake          check selectively downloaded CORA files (MD5) and move them into Raw_Data/\n    csv             4-class vibration model from a CSV feature table (validation-selected model)\n    export-images   save the infrared images as named files in Images/wheel_loader/ (local only)\n    cnn             CNN with data augmentation on the named images\n    cnn-pretrained  transfer learning, pretrained ResNet18 on the named images\n    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)\n    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known\n    ensembles       XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)\n    anomaly         anomaly detection trained on healthy data only (--split time for the secondary test)\n    fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)\n                    (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)\n    all             run csv, then export-images, then cnn-pretrained\n\nPreprocessing uses training data only (no per-speed normalisation across held-out runs).\nRaw data stays in Raw_Data/ and Processed_Data/ (not in Git)."
    """Wheel Loader (machine 6): electric drive motor, CORA vibration vs infrared camera.

    4 classes from the CORA motor/gearbox rig: healthy, bearing defect, half-broken rotor bar and broken rotor
    bar. The two rotor-bar faults are CORA conditions no other machine in this project uses (new fault data);
    healthy and bearing-defect runs are shared with the Excavator. Large wheel loaders can use electric drive,
    so the induction motor stands in for the loader's drive motor. Lab data, not a wheel loader.

    Usage:
        python Models/wheel_loader.py <command> [options]
        python Models/wheel_loader.py <command> --help

    Commands:
        intake          check selectively downloaded CORA files (MD5) and move them into Raw_Data/
        csv             4-class vibration model from a CSV feature table (validation-selected model)
        export-images   save the infrared images as named files in Images/wheel_loader/ (local only)
        cnn             CNN with data augmentation on the named images
        cnn-pretrained  transfer learning, pretrained ResNet18 on the named images
        train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
        predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
        ensembles       XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
        anomaly         anomaly detection trained on healthy data only (--split time for the secondary test)
        fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)
                        (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)
        all             run csv, then export-images, then cnn-pretrained

    Preprocessing uses training data only (no per-speed normalisation across held-out runs).
    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """


    import sys
    from pathlib import Path




    # ====================================================================================================
    # intake: check and file selectively downloaded CORA files
    # ====================================================================================================

    def intake(run: bool = True) -> dict:
        """Check downloaded CORA files against the repository's MD5 and move them into Raw_Data/.

        The CORA repository (dataverse.csuc.cat, DOI 10.34810/DATA2500, CC BY 4.0) now shows a "not a bot"
        check, so these files cannot be fetched by code; download them from the dataset page in a browser
        (API: /api/access/datafile/<id>?format=original) and run this command. Default folder: ~/Downloads.
        """
        import argparse
        import hashlib
        import json
        import shutil
        from pathlib import Path

        MANIFEST = ROOT / "Results" / "wheel_loader" / "wheel_loader_intake.json"
        FILES = [  # (file name, destination folder in Raw_Data, size in bytes, publisher MD5)
            ("H_F5_S_Vx.csv", "cora_vibration", 111260279, "10e3d07a5647058f20d2f3565675baf7"),
            ("H_F15_S_Vx.csv", "cora_vibration", 107470262, "f75e4537077a2f2c9f8c7303aedc1d1e"),
            ("H_F50_S_Vx.csv", "cora_vibration", 110648041, "0247574694c415940832698ec0fd0bd3"),
            ("H_F60_S_Vx.csv", "cora_vibration", 110998372, "9d4549a7cd33d8cfdbf14afa320140ab"),
            ("BD_F5_S.mat", "cora_drive_faults", 14354542, "a67a54403673dcf7c271b637d3d8a736"),
            ("BD_F5_S_Vx.csv", "cora_vibration", 110475747, "1e3555d8576359997bb591f9b4a4b469"),
            ("BD_F15_S.mat", "cora_drive_faults", 20385070, "de197bf1881570c4ead8a7cf9c70fbd7"),
            ("BD_F15_S_Vx.csv", "cora_vibration", 108521895, "c54d048194cde04822bd6d515db5a12c"),
            ("BD_F50_S.mat", "cora_drive_faults", 23558647, "8c2b56f22e153a30e69bca324f93f19c"),
            ("BD_F50_S_Vx.csv", "cora_vibration", 110972006, "043b29da44cd0fa5e864a63e2d7adb72"),
            ("BD_F60_S.mat", "cora_drive_faults", 30278851, "23ace1e1df6c8f926bb81153615dd843"),
            ("BD_F60_S_Vx.csv", "cora_vibration", 111344017, "5fcbe8de8a247a4170cf4685dc34b6f1"),
            ("HB_F5_S.mat", "cora_drive_faults", 15084827, "7b45c67f666a870aa563ec93ace213d7"),
            ("HB_F5_S_Vx.csv", "cora_vibration", 109501746, "cee8b92e732d37fae4b37407463df37d"),
            ("HB_F15_S.mat", "cora_drive_faults", 18621990, "0e2ff87d128f6da31b25c991831efe35"),
            ("HB_F15_S_Vx.csv", "cora_vibration", 105889293, "24bcc1bcb4b4e9cff57b207283e2d935"),
            ("HB_F50_S.mat", "cora_drive_faults", 28740479, "c842dc83ccee6a61811f398ec1362933"),
            ("HB_F50_S_Vx.csv", "cora_vibration", 110932121, "ffcf72671ff796984952ac140b8d0716"),
            ("HB_F60_S.mat", "cora_drive_faults", 27227964, "6294749ce6e1d244e47ee33544dcd63a"),
            ("HB_F60_S_Vx.csv", "cora_vibration", 111285344, "186d5006b2f34377ab3efad7f943a55f"),
            ("OB_F5_S.mat", "cora_drive_faults", 20840932, "a67776cf4261e3b64231e063d5402e83"),
            ("OB_F5_S_Vx.csv", "cora_vibration", 110381323, "a8f6cc78a18f170cf6325944a34726e4"),
            ("OB_F15_S.mat", "cora_drive_faults", 14529153, "0b753c46c9305c8d4fd3e493eaa4f951"),
            ("OB_F15_S_Vx.csv", "cora_vibration", 108549523, "e51708bc36451b96b2edc732b583afb7"),
            ("OB_F50_S.mat", "cora_drive_faults", 27532985, "5c48c30f51addec6377554fe9e746941"),
            ("OB_F50_S_Vx.csv", "cora_vibration", 110974447, "f9718fd790e9b28c9ee669156e2d0378"),
            ("OB_F60_S.mat", "cora_drive_faults", 29139968, "1504ac9b4b028b1da9476766856b119b"),
            ("OB_F60_S_Vx.csv", "cora_vibration", 110744307, "c55c63b37619b98d7b36f1a3896cd274"),
        ]

        def md5(path):
            checksum = hashlib.md5()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    checksum.update(chunk)
            return checksum.hexdigest()

        def main():
            parser = argparse.ArgumentParser(description=intake.__doc__)
            parser.add_argument("--from", dest="source", default=str(Path.home() / "Downloads"),
                                help="folder holding the browser downloads")
            args = parser.parse_args()
            source = Path(args.source)
            records, missing = [], []
            for name, folder, size, expected in FILES:
                destination = ROOT / "Raw_Data" / folder / name
                candidate = destination if destination.is_file() else source / name
                if not candidate.is_file():
                    missing.append(name)
                    continue
                if candidate.stat().st_size != size or md5(candidate) != expected:
                    raise SystemExit(f"{candidate}: size or MD5 does not match the repository; download it again")
                if candidate != destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidate), destination)
                records.append({"file": name, "folder": f"Raw_Data/{folder}", "size_bytes": size, "md5": expected,
                                "status": "verified against publisher MD5"})
                print(f"verified {name}", flush=True)
            MANIFEST.write_text(json.dumps({
                "source": "https://doi.org/10.34810/DATA2500", "license": "CC BY 4.0",
                "note": "selectively downloaded from CORA; every file checked against the publisher MD5",
                "files": records, "missing": missing}, indent=2) + "\n", encoding="utf-8")
            print(f"{len(records)} files verified; {len(missing)} still missing" + (f": {missing}" if missing else ""))

        if run:
            main()
        return locals()


    # ====================================================================================================
    # export-images: Week 6, save the camera images as named files (professor remark)
    # ====================================================================================================

    def export_images(run: bool = True) -> dict:
        """Week 6: save this machine's camera images as named PNG files for the CNN.

        Files go to Images/wheel_loader/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
        so every image carries its machine and class in its name. An index.csv lists each file with its
        class, its recording (test unit) and its group used for the held-out test. Images stay local
        (git-ignored); this command recreates them from Raw_Data/.
        """
        import argparse
        import csv as csv_module
        import shutil
        import zipfile

        import cv2
        import numpy as np

        MACHINE = "wheel_loader"
        OUT = ROOT / "Images" / MACHINE
        HEIGHT, WIDTH = 96, 128
        EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
        BIN_US, SECONDS = 250_000, 5
        FRAMES = SECONDS * 1_000_000 // BIN_US

        def event_frames(stream):
            """First 5 s of an event recording as 20 images of 250 ms; channel 0 = brighter, 1 = darker events."""
            while True:
                first = stream.read(1)
                if first != b"%":
                    break
                stream.readline()
            stream.read(1)
            counts = np.zeros((FRAMES, 2, HEIGHT, WIDTH), dtype=np.uint32)
            end = SECONDS * 1_000_000
            while True:
                chunk = stream.read(8 * 500_000)
                if not chunk:
                    break
                ev = np.frombuffer(chunk, dtype=EVENT)
                keep = ev["t"] < end
                t, xy = ev["t"][keep], ev["xy"][keep]
                x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
                index = ((((t // BIN_US) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                         + (x * WIDTH // 640)).astype(np.int64)
                counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
                if len(ev) and ev["t"][-1] >= end:
                    break
            return counts

        def event_png(frame):
            """Brighter events in red, darker events in green, log-scaled to 0-255."""
            scaled = np.log1p(frame.astype(np.float32))
            scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
            return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])  # BGR for OpenCV

        def write(rows, label, unit, group, name, image):
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), image)
            rows.append({"file": f"{label}/{name}", "machine": MACHINE, "class": label, "unit": unit, "group": group})


        def export():
            rows = []
            labels = {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}
            for condition, label in labels.items():
                folder = "cora_gear_wear" if condition == "H" else "cora_drive_faults"
                for speed in ("F5", "F15", "F50", "F60"):
                    unit = f"{condition}_{speed}_S"
                    frames = loadmat(ROOT / "Raw_Data" / folder / f"{unit}.mat",
                                     variable_names=["imagenes_celda"])["imagenes_celda"]
                    for k, frame in enumerate(frames.flat):
                        gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                        write(rows, label, unit, speed, f"{MACHINE}_{label}_{speed}_f{k:03d}.png", gray)
                    print(f"{unit}: done", flush=True)
            return rows

        def main():
            parser = argparse.ArgumentParser(description=export_images.__doc__)
            parser.parse_args()
            labels = {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}
            missing = []
            for condition in labels:
                folder = "cora_gear_wear" if condition == "H" else "cora_drive_faults"
                for speed in ("F5", "F15", "F50", "F60"):
                    source = ROOT / "Raw_Data" / folder / f"{condition}_{speed}_S.mat"
                    if not source.is_file():
                        missing.append(source)
            if missing:
                paths = ", ".join(str(path.relative_to(ROOT)) for path in missing)
                raise SystemExit(f"Missing infrared source files; no images were replaced: {paths}")
            if OUT.exists():
                shutil.rmtree(OUT)
            rows = export()
            with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["file", "machine", "class", "unit", "group"])
                writer.writeheader()
                writer.writerows(rows)
            counts = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            print(f"Saved {len(rows)} images to {OUT.relative_to(ROOT)} (per class: {counts})")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # csv: Week 6, 4-class sensor model from a CSV feature table (professor remark: "csv accuracy 90 plus")
    # ====================================================================================================

    def csv(run: bool = True) -> dict:
        """Week 6: write the sensor features to a CSV table and train 4-class models on it.

        The CSV has one row per window (or run) with its recording, class and feature values. A
        majority-class control, logistic regression and a random forest (the main model) are trained
        with settings fixed in advance and tested on whole held-out recordings/runs, as before.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        MACHINE = "wheel_loader"
        CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
        RESULT = ROOT / "Results" / "wheel_loader" / f"{MACHINE}_csv_4class.json"
        FIGURE = ROOT / "Figures" / "wheel_loader" / f"{MACHINE}_csv_4class.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
        }

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        VIB_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
        VIB_NAMES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                     *(f"order_power_{lo}_{hi}" for lo, hi in VIB_ORDERS)]

        def vibration_rows(run_name, label, speed):
            """CORA Vx vibration (3 kHz) in 1 s windows (every 10th window), measured in shaft orders.

            Orders = frequency / speed setting, so 1x, 2x and higher multiples line up across speeds:
            unbalance shows at 1x, misalignment at 2x, bearing and gear damage at higher orders and spikes.
            """
            cache = ROOT / "Processed_Data" / "cora_vibration" / f"{run_name}_Vx_features.csv"
            if cache.is_file():
                return load_csv(cache)
            source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
            if not source.is_file():
                raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
            values = load_cora_sensor(source.name)
            frame = vibration_frame(values, speed)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", label)
            frame.insert(0, "speed", speed)
            frame.insert(0, "unit", run_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame

        def vibration_frame(values, speed):
            """Order-domain vibration features for a raw 3 kHz Vx signal at speed setting F5/F15/F50/F60."""
            rate, shaft = 3000, float(speed[1:])
            n = len(values) // rate
            windows = values[: n * rate].reshape(n, rate)[::10]
            centered = windows - windows.mean(axis=1, keepdims=True)
            std = np.maximum(centered.std(axis=1), 1e-12)
            freq = np.fft.rfftfreq(rate, d=1 / rate)
            power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
            order = freq / shaft
            usable = (order >= 0.5) & (order < 20.5)
            share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
            o = order[usable]
            bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in VIB_ORDERS]
            frame = pd.DataFrame(np.column_stack([
                np.log(std), (centered ** 4).mean(axis=1) / std ** 4, (centered ** 3).mean(axis=1) / std ** 3,
                np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
                np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]), columns=VIB_NAMES)
            return frame

        def per_speed_normalised(table, names):
            """Rescale each feature within each speed using all recordings at that speed together (no labels used).

            Removes what each speed 'sounds like', so the model compares machines running at the same speed.
            It assumes several machines are recorded at each speed, as in this dataset.
            """
            out = []
            for name in names:
                column = f"{name}_per_speed"
                grouped = table.groupby("speed")[name]
                table[column] = (table[name] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
                out.append(column)
            return out

        def build_table():
            labels = {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}
            frames = [vibration_rows(f"{c}_{s}_S", label, s)
                      for c, label in labels.items() for s in ("F5", "F15", "F50", "F60")]
            table = pd.concat(frames, ignore_index=True)
            folds = lambda t: [(t["speed"] == s).to_numpy() for s in ("F5", "F15", "F50", "F60")]
            return (table, {"vibration": VIB_NAMES},
                    ["healthy", "bearing", "half_broken_bar", "broken_bar"], folds,
                    "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions",
                    ["CORA motor/gearbox rig, Vx vibration at 3 kHz; one row per 1 s window, every 10th window of each run",
                     "features in shaft orders (frequency / speed setting) so they line up across the held-out speed",
                     "no per-speed normalisation: preprocessing never looks at the held-out speed's runs"])
        INNER_GROUP = "speed"

        def candidate_models():
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
            from sklearn.svm import SVC
            return {
                "Random forest": lambda: RandomForestClassifier(
                    n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Extra trees": lambda: ExtraTreesClassifier(
                    n_estimators=300, max_features="sqrt", class_weight="balanced",
                    min_samples_leaf=2, random_state=42, n_jobs=-1),
                "Gradient boosting": lambda: HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
                "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(
                    C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)),
            }

        def nested_selection(table, feature_sets, labels, outer_folds, inner_group):
            """Pick a feature set and model on validation data inside each training set, then test once.

            Inner validation holds out whole groups (recordings, bearings or speeds) of the training set
            only; the outer test units are never used for the choice.
            """
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold

            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            groups = table[inner_group].to_numpy()
            models = candidate_models()
            actual, predicted, chosen = [], [], []
            for test in outer_folds:
                train_idx = np.flatnonzero(~test)
                n_splits = min(4, len(set(groups[train_idx])))
                inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
                best, best_score = None, -1.0
                for set_name, names in feature_sets.items():
                    x = table[names].to_numpy(dtype=np.float64)
                    for model_name, factory in models.items():
                        f1s = []
                        for fit_part, val_part in inner:
                            fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                            model = factory().fit(x[fit_rows], y[fit_rows])
                            a, p = unit_predictions(model.predict_proba(x[val_rows]), model.classes_, y[val_rows], units[val_rows])
                            f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
                        if float(np.mean(f1s)) > best_score:
                            best, best_score = (set_name, model_name), float(np.mean(f1s))
                x = table[feature_sets[best[0]]].to_numpy(dtype=np.float64)
                model = models[best[1]]().fit(x[train_idx], y[train_idx])
                a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                actual += a
                predicted += p
                chosen.append({"feature_set": best[0], "model": best[1], "validation_macro_f1": round(best_score, 4)})
            return score(actual, predicted, labels), chosen

        def main():
            parser = argparse.ArgumentParser(description=csv.__doc__)
            parser.add_argument("--split", choices=("speed", "time"), default="speed",
                                help="speed (main, fair test): hold out each whole speed. time (secondary): train on the "
                                     "first 60%% of every run, test on the last 30%% (10%% gap); the machine has been seen "
                                     "before, and each fault comes from one recording, so this is an easier test")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_names, labels, folds, test_description, notes = build_table()
            feature_sets = feature_names if isinstance(feature_names, dict) else {"base": feature_names}
            feature_names = next(iter(feature_sets.values()))
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CSV_PATH, index=False)
            if args.split == "time":
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every run trains, last 30% tests, 10% gap; "
                                    "16 run-level predictions; the same recordings appear in training, so this is easier "
                                    "than the main unseen-speed test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[feature_names].to_numpy(dtype=np.float64)
            y = table["class"].to_numpy()
            units = table["unit"].to_numpy()
            scores = {}
            for name, factory in MODELS.items():
                actual, predicted = [], []
                for test in folds(table):
                    train = ~test
                    if set(units[train]) & set(units[test]):
                        raise ValueError("A test unit also appears in training")
                    model = factory().fit(x[train], y[train])
                    a, p = unit_predictions(model.predict_proba(x[test]), model.classes_, y[test], units[test])
                    actual += a
                    predicted += p
                scores[name] = score(actual, predicted, labels)
            result = {
                "purpose": "Week 6 professor remark: 4-class sensor model from a CSV feature table",
                "csv_file": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(table)), "features": feature_names, "classes": labels,
                "test": test_description, "main_model": "Random forest",
                "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                "notes": notes, "models": scores,
            }
            if args.split == "time":
                result["split"] = "time (secondary)"
                RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                main_score = scores["Random forest"]
                plot_matrix(main_score["confusion_matrix"], labels,
                            f"{MACHINE}: CSV sensor, TIME SPLIT (random forest), accuracy {main_score['accuracy']:.3f}", FIGURE)
                for name, s in scores.items():
                    print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units, time split)")
                print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
                return
            selected, chosen = nested_selection(table, feature_sets, labels, folds(table), INNER_GROUP)
            result["validation_selected"] = {
                "description": "feature set and model chosen per round on validation groups inside the training set only "
                               "(candidates: " + ", ".join(candidate_models()) + "; feature sets: " + ", ".join(feature_sets) + ")",
                "feature_sets": feature_sets, "score": selected, "chosen_per_round": chosen}
            result["main_model"] = "validation-selected"
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            main_score = selected
            plot_matrix(main_score["confusion_matrix"], labels,
                        f"{MACHINE}: CSV sensor model (validation-selected), accuracy {main_score['accuracy']:.3f}", FIGURE)
            for name, s in scores.items():
                print(f"{name:<26} accuracy {s['accuracy']:.3f}  macro F1 {s['macro_f1']:.3f}  ({s['test_units']} test units)")
            print(f"Validation-selected model: accuracy {selected['accuracy']:.3f}  macro F1 {selected['macro_f1']:.3f}  "
                  f"(choices: {[c['feature_set'] + ' / ' + c['model'] for c in chosen]})")
            print(f"Saved {CSV_PATH.relative_to(ROOT)}, {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn: Week 6, CNN with data augmentation on the named images (professor remark)
    # ====================================================================================================

    def cnn(run: bool = True) -> dict:
        """Week 6: a small CNN with data augmentation, trained on the named images, 4 classes.

        Uses the same held-out test as the sensor model (whole recordings/runs never seen in training).
        Augmentation (training images only): random horizontal flip, shift of up to 8 pixels,
        brightness/contrast scaling 0.8-1.2 and Gaussian noise. Each image is standardised on its own,
        so overall brightness cannot be used as a shortcut. Settings are fixed in advance; the test set is
        never used to pick epochs or settings. Run `export-images` first.
        """

        import argparse
        import csv as csv_module
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)

        import random

        import cv2
        import torch
        from torch import nn

        MACHINE = "wheel_loader"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "wheel_loader" / f"{MACHINE}_cnn_4class.json"
        FIGURE = ROOT / "Figures" / "wheel_loader" / f"{MACHINE}_cnn_4class.png"
        LABELS = ["healthy", "bearing", "half_broken_bar", "broken_bar"]
        EPOCHS, BATCH, LEARNING_RATE, SEED = 20, 32, 1e-3, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)


        def read_index():
            with (IMAGES / "index.csv").open(encoding="utf-8") as handle:
                return list(csv_module.DictReader(handle))

        def load(rows):
            images = []
            for row in rows:
                image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
                channels = [0]  # infrared images are grayscale
                image = image[:, :, channels].astype(np.float32).transpose(2, 0, 1)
                image = (image - image.mean()) / (image.std() + 1e-6)  # per-image standardisation
                images.append(image)
            return np.stack(images)

        class Net(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                def block(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
                self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        def augment(batch, generator):
            n = batch.shape[0]
            flip = torch.rand(n, generator=generator) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
            shifted = []
            for image in batch:
                dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
                shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
            batch = torch.stack(shifted)
            scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
            return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)

        def train(x, y):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = Net(x.shape[1], len(LABELS))
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            for _ in range(EPOCHS):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    optimiser.zero_grad()
                    loss = loss_fn(model(augment(xs[batch], generator)), ys[batch])
                    loss.backward()
                    optimiser.step()
            return model

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 256])), dim=1).numpy()
                                       for i in range(0, len(x), 256)])

        def main():
            parser = argparse.ArgumentParser(description=cnn.__doc__)
            parser.parse_args()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            rows = read_index()
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
            for number, test in enumerate([groups == s for s in ("F5", "F15", "F50", "F60")], 1):
                train_mask = ~test
                if set(units[train_mask]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                model = train(x[train_mask], y[train_mask])
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                print(f"fold {number}: trained on {int(train_mask.sum())} images, tested on {int(test.sum())}", flush=True)
            result = {
                "purpose": "Week 6 professor remark: CNN with data augmentation on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions",
                "settings": {"epochs": EPOCHS, "batch": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
                             "architecture": "4 conv blocks (16, 32, 64, 64 channels) + global average pool + dropout 0.3",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05",
                             "normalisation": "each image standardised on its own"},
                "selection_rule": "settings fixed in advance; the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            r = result["recording_level"]
            plot_matrix(r["confusion_matrix"], LABELS,
                        f"{MACHINE}: CNN on images, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
            print(f"CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # cnn-pretrained: Week 6, transfer learning with a pretrained ResNet18 on the named images
    # ====================================================================================================

    def cnn_pretrained(run: bool = True) -> dict:
        """Week 6: transfer learning, an ImageNet-pretrained ResNet18 fine-tuned on the named images (4 classes).

        Same held-out test as before. Inside each training set, one recording per class is held out for
        validation to choose the number of epochs (1-15); the model is then retrained on the whole training
        set for that many epochs and tested once. Augmentation as in `cnn`. Run `export-images` first.
        """
        import argparse
        import csv as csv_module
        import json
        import random

        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        MACHINE = "wheel_loader"
        IMAGES = ROOT / "Images" / MACHINE
        RESULT = ROOT / "Results" / "wheel_loader" / f"{MACHINE}_cnn_pretrained_4class.json"
        FIGURE = ROOT / "Figures" / "wheel_loader" / f"{MACHINE}_cnn_pretrained_4class.png"
        LABELS = ["healthy", "bearing", "half_broken_bar", "broken_bar"]
        SEGMENTS = 10   # test segments per held-out recording (segment-level score)
        MAX_EPOCHS, BATCH, LEARNING_RATE, SEED = 15, 32, 3e-4, 42

        def unit_predictions(probabilities, classes, y, units):
            """One prediction per test unit (recording or run): average the class probabilities of its rows."""
            actual, predicted = [], []
            for unit in sorted(set(units)):
                rows = units == unit
                if len(set(y[rows])) != 1:
                    raise ValueError(f"Mixed labels within {unit}")
                actual.append(y[rows][0])
                predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
            return actual, predicted

        def score(actual, predicted, labels):
            precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
            return {
                "test_units": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                "macro_precision": round(float(precision.mean()), 4),
                "macro_recall": round(float(recall.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
            }

        def plot_matrix(matrix, labels, title, path):
            array = np.asarray(matrix)
            figure, axis = plt.subplots(figsize=(1.4 + 1.1 * len(labels), 1.2 + 1.0 * len(labels)))
            axis.imshow(array, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(labels)), labels, rotation=30)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_title(title, fontsize=9)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(array[i, j]), ha="center", va="center",
                              color="white" if array[i, j] > array.max() / 2 else "black")
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=140)
            plt.close(figure)

        def plot_levels(result, title, path):
            """One panel per scoring level, so the figure shows how many images were really tested."""
            levels = [("every test image", result["image_level"]), ("per 1/10 recording segment", result.get("segment_level")),
                      ("per whole recording", result["recording_level"])]
            levels = [(name, block) for name, block in levels if block]
            figure, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4))
            for axis, (name, block) in zip(np.atleast_1d(axes), levels):
                array = np.asarray(block["confusion_matrix"])
                axis.imshow(array, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
                axis.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                axis.set_title(f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9)
                for i in range(len(LABELS)):
                    for j in range(len(LABELS)):
                        axis.text(j, i, f"{array[i, j]:,}", ha="center", va="center", fontsize=8,
                                  color="white" if array[i, j] > array.max() / 2 else "black")
            figure.suptitle(title.replace("; ", "\n"), fontsize=10)
            figure.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=110)
            plt.close(figure)

        _small = cnn(run=False)
        read_index, augment = _small["read_index"], _small["augment"]

        def load(rows):
            return load_model_images(IMAGES, rows)

        def new_model():
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            return model

        def fit(x, y, epochs, val=None):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            model = new_model()
            counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))))
            optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.long)
            history = []
            for _ in range(epochs):
                model.train()
                order = torch.randperm(len(xs), generator=generator)
                for start in range(0, len(xs), BATCH):
                    batch = order[start:start + BATCH]
                    if len(batch) < 2:
                        continue
                    optimiser.zero_grad()
                    loss_fn(model(augment(xs[batch], generator)), ys[batch]).backward()
                    optimiser.step()
                if val is not None:
                    vx, vnames, vunits = val
                    a, p = unit_predictions(probabilities(model, vx), np.asarray(LABELS), vnames, vunits)
                    history.append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
            return model, history

        def probabilities(model, x):
            model.eval()
            with torch.no_grad():
                return np.concatenate([torch.softmax(model(torch.tensor(x[i:i + 128])), dim=1).numpy()
                                       for i in range(0, len(x), 128)])

        def main():
            parser = argparse.ArgumentParser(description=cnn_pretrained.__doc__)
            parser.add_argument("--split", choices=("speed", "time"), default="speed",
                                help="speed (main, fair test) or time (secondary: first 60%% of each run's frames train, "
                                     "last 30%% test, 10%% gap; easier, the same recordings appear in training)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            rows = read_index()
            test_description = "four rounds, each holding out one whole speed (4 runs, one per class); 16 run-level predictions"
            if args.split == "time":
                import re
                position = {row["file"]: int(re.search(r"_f(\d{3})\.png$", row["file"]).group(1)) / 450 for row in rows}
                rows = [dict(row, unit=row["unit"] + ("_late" if position[row["file"]] > 0.7 else "_early"))
                        for row in rows if position[row["file"]] <= 0.6 or position[row["file"]] > 0.7]
                test_description = ("TIME SPLIT (secondary): first 60% of every run's frames train, last 30% test, 10% gap; "
                                    "16 run-level predictions; the same recordings appear in training, so this is easier")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = load(rows)
            label_index = {label: i for i, label in enumerate(LABELS)}
            y = np.asarray([label_index[row["class"]] for row in rows])
            names = np.asarray([row["class"] for row in rows])
            units = np.asarray([row["unit"] for row in rows])
            groups = np.asarray([row["group"] for row in rows])
            rng = np.random.default_rng(SEED)
            unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
            segment_actual, segment_predicted, per_recording = [], [], {}
            folds = ([np.char.endswith(units.astype(str), "_late")] if args.split == "time"
                     else [groups == s for s in ("F5", "F15", "F50", "F60")])
            for number, test in enumerate(folds, 1):
                train = ~test
                if set(units[train]) & set(units[test]):
                    raise ValueError("A test recording also appears in training")
                val_units = [rng.choice(sorted(set(units[train & (names == label)]))) for label in LABELS]
                val = train & np.isin(units, val_units)
                fit_part = train & ~val
                _, history = fit(x[fit_part], y[fit_part], MAX_EPOCHS, (x[val], names[val], units[val]))
                best_epochs = int(np.argmax(history)) + 1
                model, _ = fit(x[train], y[train], best_epochs)
                p = probabilities(model, x[test])
                a, pr = unit_predictions(p, np.asarray(LABELS), names[test], units[test])
                unit_actual += a
                unit_predicted += pr
                image_actual += list(names[test])
                image_predicted += [LABELS[i] for i in p.argmax(axis=1)]
                # more test units from the same unseen recordings: split each held-out recording, in time order,
                # into SEGMENTS consecutive parts and give every part its own prediction (still never seen in training)
                for unit in sorted(set(units[test])):
                    rows_u = units[test] == unit
                    pu, true = p[rows_u], names[test][rows_u][0]
                    for part in np.array_split(np.arange(len(pu)), min(SEGMENTS, len(pu))):
                        segment_actual.append(true)
                        segment_predicted.append(LABELS[int(pu[part].mean(axis=0).argmax())])
                    per_recording[str(unit)] = {"class": str(true), "images": int(len(pu)),
                                                "images_predicted": {l: int((pu.argmax(axis=1) == i).sum())
                                                                     for i, l in enumerate(LABELS)},
                                                "recording_prediction": LABELS[int(pu.mean(axis=0).argmax())]}
                epochs_chosen.append(best_epochs)
                print(f"fold {number}: validation chose {best_epochs} epochs; tested on {int(test.sum())} images", flush=True)
            result = {
                "purpose": "Week 6: transfer learning (ImageNet-pretrained ResNet18) on named images, 4 classes",
                "images": int(len(rows)), "classes": LABELS, "test": test_description,
                "settings": {"model": "ResNet18, ImageNet weights, all layers fine-tuned", "max_epochs": MAX_EPOCHS,
                             "batch": BATCH, "learning_rate": LEARNING_RATE, "optimiser": "AdamW", "seed": SEED,
                             "epochs_chosen_per_round": epochs_chosen,
                             "epoch_selection": "one training recording per class held out for validation; best validation macro F1",
                             "augmentation": "random horizontal flip, shift up to 8 px, brightness/contrast x0.8-1.2, Gaussian noise 0.05"},
                "selection_rule": "the test set is never used to choose epochs or settings",
                "recording_level": score(unit_actual, unit_predicted, LABELS),
                "image_level": score(image_actual, image_predicted, LABELS),
                "segment_level": {**score(segment_actual, segment_predicted, LABELS),
                                  "segments_per_recording": SEGMENTS,
                                  "note": "each held-out recording split in time order into consecutive parts, one "
                                          "prediction per part; parts of one recording are related, so the "
                                          "recording-level score stays the main result"},
                "per_recording": per_recording,
            }
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["segment_level"]
            plot_matrix(s["confusion_matrix"], LABELS,
                        f"{MACHINE}: pretrained ResNet18 per segment, accuracy {s['accuracy']:.3f} "
                        f"({s['test_units']} test segments)", FIGURE.with_name(FIGURE.stem + "_segments.png"))
            print(f"Segment-level accuracy {s['accuracy']:.3f} on {s['test_units']} test segments "
                  f"({SEGMENTS} per held-out recording)")
            r = result["recording_level"]
            plot_levels(result, f"{MACHINE}: pretrained ResNet18 on {result['images']:,} images in total; "
                                f"{r['test_units']} test recordings" + (" (time split, secondary: later frames of training recordings)"
                                if "timesplit" in FIGURE.stem else ", each never seen in training"), FIGURE)
            print(f"Pretrained CNN recording-level accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}; "
                  f"image-level accuracy {result['image_level']['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # train-final / predict: use the trained sensor model on any CSV
    # ====================================================================================================

    def train_final(run: bool = True) -> dict:
        """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

        Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
        Model: random forest on the "vibration" features (fixed settings).
        """
        import argparse
        import json

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "wheel_loader_sensor_model.joblib"

        def main():
            parser = argparse.ArgumentParser(description=train_final.__doc__)
            parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. U_F15_S")
            args = parser.parse_args()
            table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            features = feature_sets["vibration"]
            unknown = set(args.exclude) - set(table["unit"])
            if unknown:
                raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
            train = ~table["unit"].isin(args.exclude)
            model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "wheel_loader",
                         "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                        MODEL_PATH)
            print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
                  f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    def predict(run: bool = True) -> dict:
        """Predict the condition from any CSV and, if the true class is known, show the accuracy.

        The CSV can be (1) a feature table with the same columns as Processed_Data/wheel_loader_sensor_features_4class.csv
        (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of CORA Vx vibration values sampled at 3 kHz (give --speed).
        Give the true class with --label (raw recordings) or a 'class' column (feature tables) to get accuracy.
        Run `train-final` first (it runs automatically if no saved model exists).
        """
        import argparse
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix

        MODEL_PATH = ROOT / "Processed_Data" / "models" / "wheel_loader_sensor_model.joblib"

        def read_any(path):
            table = load_csv(path)
            if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
                table = load_csv(path, header=None)
            return table

        def _is_number(text):
            try:
                float(text)
                return True
            except (TypeError, ValueError):
                return False

        def raw_features(values, args):
            if not args.speed:
                raise SystemExit("Raw recordings need --speed (F5, F15, F50 or F60)")
            ns = csv(run=False)
            return ns["vibration_frame"](values, args.speed)[ns["VIB_NAMES"]].to_numpy(dtype=np.float64)


        def main():
            parser = argparse.ArgumentParser(description=predict.__doc__)
            parser.add_argument("--csv", required=True, help="path to the CSV file")
            parser.add_argument("--label", help="true class of a raw recording, one of: healthy, bearing, half_broken_bar, broken_bar")
            parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")
            parser.add_argument("--speed", choices=("F5", "F15", "F50", "F60"), help="speed setting of a raw recording")
            args = parser.parse_args()
            if not MODEL_PATH.is_file():
                print("No saved model yet: training it now (train-final).")
                train_final(run=False)["main"]()
            saved = joblib.load(MODEL_PATH)
            model, features = saved["model"], saved["features"]
            path = Path(args.csv)
            table = read_any(path)
            if set(features) <= set(table.columns):
                mode = "feature table"
                x = table[features].to_numpy(dtype=np.float64)
                units = table["unit"].astype(str).to_numpy() if "unit" in table.columns else np.full(len(table), path.stem)
                truth = table["class"].astype(str).to_numpy() if "class" in table.columns else np.full(len(table), args.label)
            else:
                mode = "raw recording"
                column = args.column if args.column is not None else table.select_dtypes("number").columns[0]
                values = table[column if column in table.columns else int(column)].to_numpy(dtype=np.float64)
                x = raw_features(values, args)
                units = np.full(len(x), path.stem)
                truth = np.full(len(x), args.label)
            probabilities = model.predict_proba(x)
            rows = []
            for unit in dict.fromkeys(units):
                mask = units == unit
                mean = probabilities[mask].mean(axis=0)
                rows.append({"recording": unit, "predicted": model.classes_[int(mean.argmax())],
                             "confidence": round(float(mean.max()), 3), "true": truth[mask][0], "rows": int(mask.sum())})
            report = pd.DataFrame(rows)
            print(f"Model: {saved['machine']} random forest | input: {mode} | {len(report)} recording(s)")
            print(report.to_string(index=False))
            seen = sorted(set(report["recording"]) & set(saved["trained_units"]))
            if seen:
                print(f"WARNING: {seen} were used to train this model, so their accuracy is optimistic. "
                      "Use train-final --exclude ... to test fairly.")
            known = report["true"].notna() & (report["true"].astype(str) != "None")
            if known.any():
                k = report[known]
                print(f"\nAccuracy: {accuracy_score(k['true'], k['predicted']):.3f} "
                      f"({int((k['true'] == k['predicted']).sum())}/{len(k)} recordings)")
                labels = saved["classes"]
                print("Confusion matrix (rows = true, columns = predicted):", labels)
                print(confusion_matrix(k["true"], k["predicted"], labels=labels))
            else:
                print("\nNo true class given (use --label or a 'class' column) so accuracy cannot be computed.")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Proposal Week 6: advanced ensemble models (XGBoost / LightGBM comparison table)
    # ====================================================================================================

    def ensembles(run: bool = True) -> dict:
        """Proposal Week 6: compare advanced ensemble models on the same CSV sensor features and held-out tests.

        Models (settings fixed in advance, nothing tuned on the test): majority-class control, random forest,
        extra trees, scikit-learn gradient boosting, XGBoost and LightGBM. Same feature set and the same held-out
        recordings as the Week 6 CSV main test (or --split time). One prediction per recording (mean class
        probabilities of its windows). Also records how long each model takes to train.
        """
        import argparse
        import json
        import time
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("Install the ensemble libraries first:  python -m pip install xgboost lightgbm")

        MACHINE = "wheel_loader"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_ensembles.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_ensembles.png"
        MODELS = {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": lambda: LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }

        def main():
            parser = argparse.ArgumentParser(description=ensembles.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair test) or time (secondary: first 60%% of every recording trains, "
                                     "last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): first 60% of every recording trains, last 30% tests, 10% gap; "
                                    "the same recordings appear in training, so this is easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            code = {label: i for i, label in enumerate(labels)}
            y = table["class"].map(code).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            result = {"purpose": "Proposal Week 6: advanced ensemble comparison (XGBoost / LightGBM)",
                      "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                      "classes": list(labels), "test": test_description,
                      "selection_rule": "settings fixed in advance; no model or setting chosen from these test results",
                      "notes": notes, "models": {}}
            for name, make in MODELS.items():
                actual, predicted, seconds = [], [], 0.0
                for test in folds(table):
                    model = make()
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[~test], y[~test])
                    seconds += time.perf_counter() - start
                    probabilities = np.zeros((int(test.sum()), len(labels)))
                    probabilities[:, model.classes_] = model.predict_proba(x[test])
                    test_units = units[test]
                    for unit in dict.fromkeys(test_units):
                        rows = test_units == unit
                        actual.append(int(y[test][rows][0]))
                        predicted.append(int(probabilities[rows].mean(axis=0).argmax()))
                result["models"][name] = {
                    "test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, average="macro", labels=list(range(len(labels))),
                                                     zero_division=0)), 4),
                    "train_seconds": round(seconds, 2),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
                }
                block = result["models"][name]
                print(f"{name:33s} accuracy {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
                      f"({block['test_units']} recordings, {seconds:.1f} s to train)")
            names = list(result["models"])
            figure, axis = plt.subplots(figsize=(8, 3.6))
            positions = np.arange(len(names))
            axis.bar(positions - 0.2, [result["models"][n]["accuracy"] for n in names], 0.4, label="accuracy")
            axis.bar(positions + 0.2, [result["models"][n]["macro_f1"] for n in names], 0.4, label="macro F1")
            axis.axhline(0.9, color="grey", ls="--", lw=1)
            axis.set_xticks(positions, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=8)
            axis.set_title(f"{MACHINE.replace('_', ' ').title()}: ensemble comparison"
                           + (" - TIME SPLIT" if args.split == "time" else " (main test)"), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: anomaly detection (proposal Week 7: Isolation Forest / One-Class SVM)
    # ====================================================================================================

    def anomaly(run: bool = True) -> dict:
        """Week 7: anomaly detection. Learn what HEALTHY looks like, then flag anything unusual as a possible fault.

        Isolation Forest and One-Class SVM are trained on healthy rows of the training recordings only (no fault
        labels are used for training). Same held-out recordings as the CSV main test (or --split time). Each test
        recording gets one score: the mean window anomaly score. The alarm threshold is fixed before testing:
        each healthy training recording is scored by a model fitted on the other healthy training recordings, and
        the highest of those scores is the threshold. Reported: ROC-AUC (healthy vs fault, no threshold needed),
        fault detection rate (recall), false-alarm rate on healthy recordings, and detection rate per fault class.
        """
        import argparse
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        MACHINE = "wheel_loader"
        RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_anomaly.json"
        FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_anomaly.png"
        DETECTORS = {
            "Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        }
        MAX_FIT_ROWS = 4000   # One-Class SVM gets slow on many rows; a fixed, evenly spaced subset is used

        def fit(make, x):
            rows = np.linspace(0, len(x) - 1, min(len(x), MAX_FIT_ROWS)).astype(int)
            return make().fit(x[rows])

        def recording_scores(model, x, units):
            window = -model.decision_function(x)   # higher = more unusual
            return {unit: float(window[units == unit].mean()) for unit in dict.fromkeys(units)}

        def threshold(make, x, units, healthy_units):
            """Highest score of any healthy training recording when it is left out of fitting (no test data used)."""
            if len(healthy_units) < 2:
                raise SystemExit("Need at least two healthy training recordings to set the threshold")
            scores = []
            for unit in healthy_units:
                others = np.isin(units, [u for u in healthy_units if u != unit])
                model = fit(make, x[others])
                scores.append(recording_scores(model, x[units == unit], units[units == unit])[unit])
            return float(max(scores))

        def main():
            parser = argparse.ArgumentParser(description=anomaly.__doc__)
            parser.add_argument("--split", choices=("main", "time"), default="main",
                                help="main (fair: held-out recordings) or time (secondary: first 60%% of every recording "
                                     "trains, last 30%% tests, 10%% gap)")
            args = parser.parse_args()
            nonlocal RESULT, FIGURE
            table, feature_sets, labels, folds, test_description, notes = csv(run=False)["build_table"]()
            feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
            set_name, features = next(iter(feature_sets.items()))
            if args.split == "time":
                if "window" not in table.columns:
                    table["window"] = table.groupby("unit").cumcount()
                position = table.groupby("unit")["window"].rank(pct=True)
                table = table[(position <= 0.6) | (position > 0.7)].copy()
                late = position.loc[table.index] > 0.7
                table["unit"] = table["unit"] + np.where(late, "_late", "_early")
                folds = lambda t: [t["unit"].str.endswith("_late").to_numpy()]
                test_description = ("TIME SPLIT (secondary): healthy rows from the first 60% of every recording train, "
                                    "the last 30% of every recording is tested (10% gap); easier than the main test")
                RESULT = RESULT.with_name(RESULT.stem + "_timesplit.json")
                FIGURE = FIGURE.with_name(FIGURE.stem + "_timesplit.png")
            x = table[features].to_numpy(dtype=np.float64)
            y = table["class"].astype(str).to_numpy()
            units = table["unit"].astype(str).to_numpy()
            fault_classes = [c for c in labels if c != "healthy"]
            result = {
                "purpose": "Week 7 (proposal): anomaly detection trained on healthy data only",
                "machine": MACHINE, "features": f"{set_name} ({len(features)} sensor features, same CSV as Week 6)",
                "test": test_description,
                "threshold_rule": "highest leave-one-recording-out score among the healthy TRAINING recordings; "
                                  "no test data and no fault labels are used to train or to set it",
                "notes": notes, "detectors": {},
            }
            figure, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 3.6))
            for axis, (name, make) in zip(np.atleast_1d(axes), DETECTORS.items()):
                rows = []
                for test in folds(table):
                    train_healthy = sorted(set(units[~test & (y == "healthy")]))
                    cut = threshold(make, x, units, train_healthy)
                    model = fit(make, x[np.isin(units, train_healthy)])
                    scores = recording_scores(model, x[test], units[test])
                    for unit, score in scores.items():
                        rows.append({"unit": unit, "class": y[units == unit][0], "margin": score - cut})
                truth = np.array([r["class"] != "healthy" for r in rows])
                margin = np.array([r["margin"] for r in rows])
                alarm = margin > 0
                per_class = {c: round(float(alarm[[r["class"] == c for r in rows]].mean()), 4)
                             for c in fault_classes if any(r["class"] == c for r in rows)}
                block = {
                    "test_recordings": len(rows), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                    "roc_auc": round(float(roc_auc_score(truth, margin)), 4) if 0 < truth.sum() < len(truth) else None,
                    "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                    "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                    "accuracy": round(float((alarm == truth).mean()), 4),
                    "detection_rate_per_fault": per_class,
                    "recordings": [{**r, "margin": round(r["margin"], 4)} for r in rows],
                }
                result["detectors"][name] = block
                print(f"{name:17s} ROC-AUC {block['roc_auc']}  faults detected {block['fault_detection_rate']:.3f}  "
                      f"false alarms {block['false_alarm_rate']:.3f}  ({len(rows)} test recordings)")
                order = ["healthy", *fault_classes]
                for k, c in enumerate(order):
                    values = margin[[r["class"] == c for r in rows]]
                    axis.scatter(np.full(len(values), k) + np.linspace(-0.15, 0.15, len(values)), values, s=18,
                                 color="tab:green" if c == "healthy" else "tab:red")
                axis.axhline(0, color="black", lw=1, ls="--")
                axis.set_xticks(range(len(order)), order, rotation=20)
                axis.set_ylabel("score minus alarm threshold")
                axis.set_title(f"{name}: AUC {block['roc_auc']}, detect {block['fault_detection_rate']:.2f}, "
                               f"false alarm {block['false_alarm_rate']:.2f}", fontsize=8.5)
            figure.suptitle(f"{MACHINE.replace('_', ' ').title()} anomaly detection (above the line = alarm)"
                            + (" - TIME SPLIT" if args.split == "time" else ""), fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=110)
            plt.close(figure)
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()


    # ====================================================================================================
    # Week 7: full-data sensor experiment (every CORA sensor, window and recording)
    # ====================================================================================================

    def fulldata(run: bool = True) -> dict:
        """Week 7: every CORA sensor (vibration X/Y/Z, current C1-C3, temperatures T1-T6), every 1 s window, and the
        five start-up test repeats per condition and speed. See Models/cora_fulldata.py. Saves *_fulldata.json."""
        import argparse
        import importlib.util

        def main():
            parser = argparse.ArgumentParser(description=fulldata.__doc__)
            parser.add_argument("--part", choices=("classify", "ensembles", "anomaly", "all"), default="classify",
                                help="classify (default), ensembles (XGBoost/LightGBM comparison), anomaly, or all")
            part = parser.parse_args().part
            engine = get_model("cora_fulldata")
            conditions = {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}
            for name in ("classify", "ensembles", "anomaly"):
                if part in (name, "all"):
                    (engine.run if name == "classify" else getattr(engine, name))("wheel_loader", conditions)

        if run:
            main()
        return locals()


    COMMANDS = {
        "fulldata": fulldata,
        "ensembles": ensembles,
        "anomaly": anomaly,
        "train-final": train_final,
        "predict": predict,
        "intake": intake,
        "csv": csv,
        "export-images": export_images,
        "cnn": cnn,
        "cnn-pretrained": cnn_pretrained,
    }
    RUN_ALL = ['csv', 'export-images', 'cnn-pretrained']


    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"wheel_loader.py {name}", *(options if command != "all" else [])]
            COMMANDS[name]()



    return SimpleNamespace(**locals())


# DRILL_RIG MODEL IMPLEMENTATION
def _make_drill_rig():
    """Drill Rig (machine 7): rotary-head reduction gear, gearbox vibration. SENSOR ONLY, separate data.

    Data: MCC5-THU gearbox fault diagnosis datasets (Chen et al., Tsinghua University, Mendeley Data
    p92gj2732w v2, CC BY 4.0). A two-stage parallel gearbox; the faulty gear is the 36-tooth gear on the
    intermediate shaft. This is NOT the CORA rig, so the Drill Rig does not reuse another machine's data.
    No public gear-wear image dataset could be verified, so there is no camera model for this machine.

    4 classes: healthy, gear wear, gear pitting, tooth crack. Every fault is present at three severities
    (light, medium, high), each a separately machined gear; every gear is run under 12 working conditions
    (speed varying at 10 or 20 Nm, or load varying at 1000, 2000 or 3000 rpm). 120 recordings, 60 s each,
    12.8 kHz, six vibration channels.

    Fair test: leave one speed out (1000 / 2000 / 3000 rpm). Inside each training set the other two speeds
    are used, one at a time, to choose the model. Each held-out recording gets one answer (mean of its
    1 s windows).

    Usage:
        python Notebook/notebook_helpers.py drill_rig <command>

    Commands:
        intake   check the downloaded zip against the published SHA-256 and list the recordings used
        csv      4-class gearbox-vibration model, leave one speed out (validation-selected model)
        all      run intake, then csv

    Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
    """
    MACHINE = "drill_rig"
    ZIP = ROOT / "Raw_Data" / "mcc5_thu_gearbox" / "MCC5-THU gearbox fault diagnosis datasets.zip"
    SHA256 = "dcf7ba4b2d375c951ea57baf9bf90db4b3196982b324a13d85b187f286116b86"
    SOURCE = "MCC5-THU, Mendeley Data p92gj2732w v2 (doi 10.17632/p92gj2732w.2), CC BY 4.0"
    PREFIX = {"health": "healthy", "gear_wear": "gear_wear", "gear_pitting": "gear_pitting", "teeth_crack": "tooth_crack"}
    LABELS = ["healthy", "gear_wear", "gear_pitting", "tooth_crack"]
    CHANNELS = ["motor_vibration_x", "motor_vibration_y", "motor_vibration_z",
                "gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"]
    RATE, WINDOW_SECONDS, SEED = 12_800, 1.0, 613
    FEATURES = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
    RESULT = ROOT / "Results" / MACHINE / f"{MACHINE}_csv_4class.json"
    INTAKE = ROOT / "Results" / MACHINE / f"{MACHINE}_mcc5_intake.json"
    FIGURE = ROOT / "Figures" / MACHINE / f"{MACHINE}_csv_4class.png"

    def parse(name):
        """Class, severity and working condition from a file name such as gear_wear_M_speed_circulation_10Nm-1000rpm.csv."""
        import re
        base = name.rsplit("/", 1)[-1][:-4]
        for prefix, label in PREFIX.items():
            if base.startswith(prefix + "_"):
                rest = base[len(prefix) + 1:]
                break
        else:
            return None
        severity = "none"
        if label != "healthy":
            severity, rest = rest.split("_", 1)
        if rest.startswith(("bearing", "and")):          # compound faults are not used
            return None
        mode = "speed varying" if rest.startswith("speed_circulation") else "load varying"
        rpm = int(re.search(r"(\d+)rpm", rest).group(1))
        torque = int(re.search(r"(\d+)Nm", rest).group(1))
        return {"file": name, "recording": base, "class": label, "severity": severity, "mode": mode,
                "rpm": rpm, "torque_nm": torque}

    def members():
        import zipfile
        with zipfile.ZipFile(ZIP) as archive:
            rows = [parse(i.filename) for i in archive.infolist()
                    if i.filename.endswith(".csv") and "__MACOSX" not in i.filename]
        return [r for r in rows if r]

    def intake(run: bool = True) -> dict:
        """Check the zip's SHA-256 against the published value and record which recordings are used."""
        import hashlib
        import json

        def main():
            if not ZIP.is_file():
                raise SystemExit(f"Missing {ZIP}: download {SOURCE}")
            digest = hashlib.sha256()
            with ZIP.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            ok = digest.hexdigest() == SHA256
            rows = members()
            counts = {label: sum(r["class"] == label for r in rows) for label in LABELS}
            INTAKE.parent.mkdir(parents=True, exist_ok=True)
            INTAKE.write_text(json.dumps({"source": SOURCE, "sha256_expected": SHA256,
                                          "sha256_matches": ok, "zip_bytes": ZIP.stat().st_size,
                                          "recordings_used": counts, "recordings": rows}, indent=2) + "\n",
                              encoding="utf-8")
            print(f"SHA-256 {'matches' if ok else 'DOES NOT MATCH'} the published value; recordings used: {counts}")
            if not ok:
                raise SystemExit(2)

        if run:
            main()
        return locals()

    def features(signal):
        """Fixed-size features of one 1 s window, per channel: time-domain statistics and 8 log band energies."""
        import numpy as np
        out = []
        spectrum = np.abs(np.fft.rfft(signal - signal.mean(axis=0), axis=0)) ** 2
        edges = np.linspace(0, spectrum.shape[0], 9).astype(int)
        for c in range(signal.shape[1]):
            x = signal[:, c]
            rms = np.sqrt(np.mean(x ** 2))
            centred = x - x.mean()
            sd = centred.std() + 1e-12
            out += [rms, sd, np.abs(x).max(), np.abs(x).max() / (rms + 1e-12),
                    np.mean(centred ** 3) / sd ** 3, np.mean(centred ** 4) / sd ** 4]
            total = spectrum[:, c].sum() + 1e-12
            out += [np.log(spectrum[a:b, c].sum() / total + 1e-12) for a, b in zip(edges[:-1], edges[1:])]
        return out

    def feature_names():
        names = []
        for c in CHANNELS:
            names += [f"{c}_{k}" for k in ("rms", "std", "peak", "crest", "skew", "kurtosis")]
            names += [f"{c}_band{b}" for b in range(8)]
        return names

    def build_features():
        import zipfile
        import numpy as np
        import pandas as pd
        rows, size = [], int(RATE * WINDOW_SECONDS)
        with zipfile.ZipFile(ZIP) as archive:
            recordings = members()
            for n, r in enumerate(recordings, 1):
                with archive.open(r["file"]) as handle:
                    signal = pd.read_csv(handle, usecols=CHANNELS, dtype="float32")[CHANNELS].to_numpy()
                for w in range(len(signal) // size):
                    rows.append({**{k: r[k] for k in ("recording", "class", "severity", "mode", "rpm", "torque_nm")},
                                 "window": w, **dict(zip(feature_names(), features(signal[w * size:(w + 1) * size])))})
                print(f"  {n}/{len(recordings)} {r['recording']}: {len(signal) // size} windows", flush=True)
        table = pd.DataFrame(rows)
        FEATURES.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(FEATURES, index=False)
        return table

    def csv(run: bool = True) -> dict:
        """4-class gearbox-vibration model, leave one speed out, model chosen on the training speeds only."""
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def candidates():
            return {
                "Majority class (control)": DummyClassifier(strategy="most_frequent"),
                "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
                "Random forest": RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
                "Extra trees": ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
                "Gradient boosting": HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
            }

        def per_recording(model, table, rows):
            """One answer per recording: the class with the highest mean window probability."""
            p = model.predict_proba(table.loc[rows, feature_names()].to_numpy())
            frame = pd.DataFrame(p, columns=model.classes_, index=table.index[rows])
            frame["recording"] = table.loc[rows, "recording"].to_numpy()
            mean = frame.groupby("recording").mean()
            truth = table.loc[rows].groupby("recording")["class"].first().loc[mean.index]
            return list(truth), list(mean.idxmax(axis=1))

        def score(actual, predicted):
            return {"test_units": len(actual),
                    "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
                    "macro_f1": round(float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)), 4),
                    "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist()}

        def main():
            table = pd.read_csv(FEATURES) if FEATURES.is_file() else build_features()
            speeds = sorted(table["rpm"].unique())
            print(f"{table['recording'].nunique()} recordings, {len(table)} windows; held-out speeds {speeds}")
            models_actual, models_predicted = {name: [] for name in candidates()}, {name: [] for name in candidates()}
            chosen_actual, chosen_predicted, folds = [], [], []
            for speed in speeds:
                train, test = (table["rpm"] != speed).to_numpy(), (table["rpm"] == speed).to_numpy()
                # inner validation: each remaining speed held out once, inside the training data only
                inner = {name: [] for name in candidates()}
                for check in sorted(set(table.loc[train, "rpm"])):
                    fit_rows, check_rows = train & (table["rpm"] != check).to_numpy(), (table["rpm"] == check).to_numpy()
                    for name, model in candidates().items():
                        model.fit(table.loc[fit_rows, feature_names()].to_numpy(), table.loc[fit_rows, "class"])
                        a, p = per_recording(model, table, check_rows)
                        inner[name].append(f1_score(a, p, labels=LABELS, average="macro", zero_division=0))
                choice = max((n for n in inner if n != "Majority class (control)"), key=lambda n: np.mean(inner[n]))
                for name, model in candidates().items():
                    model.fit(table.loc[train, feature_names()].to_numpy(), table.loc[train, "class"])
                    a, p = per_recording(model, table, test)
                    models_actual[name] += a
                    models_predicted[name] += p
                    if name == choice:
                        chosen_actual += a
                        chosen_predicted += p
                        fold_accuracy = accuracy_score(a, p)
                folds.append({"held_out_rpm": int(speed), "test_recordings": int(table.loc[test, "recording"].nunique()),
                              "chosen_model": choice, "accuracy": round(float(fold_accuracy), 4),
                              "inner_macro_f1": {n: round(float(np.mean(v)), 4) for n, v in inner.items()}})
                print(f"held-out {speed} rpm: chose {choice}; accuracy {fold_accuracy:.3f}", flush=True)
            result = {
                "machine": MACHINE, "source": SOURCE, "modality": "sensor only (no matching gear image dataset found)",
                "classes": LABELS, "recordings": int(table["recording"].nunique()), "windows": int(len(table)),
                "test": "leave one speed out (1000 / 2000 / 3000 rpm); model chosen by inner leave-one-speed-out "
                        "on the training speeds; one answer per recording (mean of its 1 s windows)",
                "features": f"{len(feature_names())} per 1 s window: RMS, std, peak, crest, skewness, kurtosis and "
                            "8 relative log band energies for each of 6 vibration channels",
                "folds": folds,
                "validation_selected": {"score": score(chosen_actual, chosen_predicted)},
                "models": {name: score(models_actual[name], models_predicted[name]) for name in candidates()},
                "caveat": "one laboratory gearbox; one healthy gear but three separately machined gears per fault, "
                          "so the healthy class has fewer independent examples; recordings are 60 s each",
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            s = result["validation_selected"]["score"]
            matrix = np.asarray(s["confusion_matrix"])
            figure, axis = plt.subplots(figsize=(5.6, 4.8))
            axis.imshow(matrix, cmap="Blues", vmin=0)
            axis.set_xticks(range(len(LABELS)), LABELS, rotation=25)
            axis.set_yticks(range(len(LABELS)), LABELS)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            for i in range(len(LABELS)):
                for j in range(len(LABELS)):
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                              color="white" if matrix[i, j] > matrix.max() / 2 else "black")
            axis.set_title(f"Drill Rig gearbox vibration (MCC5-THU), unseen speed\n"
                           f"{s['test_units']} test recordings, accuracy {s['accuracy']:.3f}", fontsize=10)
            figure.tight_layout()
            FIGURE.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE, dpi=130)
            plt.close(figure)
            print(f"Validation-selected model: accuracy {s['accuracy']:.3f}, macro F1 {s['macro_f1']:.3f} "
                  f"on {s['test_units']} unseen-speed recordings")
            for name, block in result["models"].items():
                print(f"  {name:26s} accuracy {block['accuracy']:.3f}")
            print(f"Saved {RESULT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")

        if run:
            main()
        return locals()

    COMMANDS = {"intake": intake, "csv": csv}
    RUN_ALL = ["intake", "csv"]

    def cli() -> None:
        if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
            print(_make_drill_rig.__doc__)
            raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
        command, options = sys.argv[1], sys.argv[2:]
        for name in (RUN_ALL if command == "all" else [command]):
            sys.argv = [f"drill_rig {name}", *(options if command != "all" else [])]
            COMMANDS[name]()

    return SimpleNamespace(**locals())


# CORA_FULLDATA MODEL IMPLEMENTATION
def _make_cora_fulldata():
    __doc__ = "Week 7 full-data sensor experiment for the CORA machines (Belt Conveyor, Dozer, Excavator, Wheel Loader).\n\nShared engine; each machine file has a `fulldata` command that calls `run(machine, conditions)`.\nUses every CORA sensor and every recording that exists for the machine's conditions:\n\n  - stationary runs (one 45-50 min run per condition and speed): vibration X/Y/Z (3 kHz), motor current\n    C1-C3 (4 kHz) and six temperature probes T1-T6 (1 kHz), EVERY 1-second window;\n  - start-up transient tests (FIVE 30 s repeats per condition and speed): vibration X/Y/Z and current\n    C1-C3; the steady part (10-30 s) is used, 20 one-second windows per repeat.\n\nMain test: each whole speed is held out in turn (4 rounds). Training uses all stationary and transient\nrecordings of the other three speeds; the held-out speed gives 6 test recordings per class (1 stationary\n+ 5 transient). One prediction per recording (mean class probabilities of its windows); every training\nrecording gets equal weight. The feature set and model are chosen by leave-one-speed-out validation\ninside the training speeds only. All features are computed per window from that window alone (vibration\nin shaft orders, current in supply-frequency orders), so no preprocessing looks at the held-out speed.\n\nSecondary (easier) time split: stationary runs only, first 60% of every run trains, last 30% tests.\nTemperature turned out to identify the recording session (it separates the conditions already in the\nfirst 5 minutes, before any fault heating), so temperature sets are reported but excluded from selection.\nTemperature is offered two ways: the pattern only (each probe relative to the others and to the run's first\nminute) and including absolute temperature, which mostly follows the room temperature of that recording day.\nResults go to Results/<machine>/<machine>_fulldata.json; the Week 6 results are not changed."
    """Week 7 full-data sensor experiment for the CORA machines (Belt Conveyor, Dozer, Excavator, Wheel Loader).

    Shared engine; each machine file has a `fulldata` command that calls `run(machine, conditions)`.
    Uses every CORA sensor and every recording that exists for the machine's conditions:

      - stationary runs (one 45-50 min run per condition and speed): vibration X/Y/Z (3 kHz), motor current
        C1-C3 (4 kHz) and six temperature probes T1-T6 (1 kHz), EVERY 1-second window;
      - start-up transient tests (FIVE 30 s repeats per condition and speed): vibration X/Y/Z and current
        C1-C3; the steady part (10-30 s) is used, 20 one-second windows per repeat.

    Main test: each whole speed is held out in turn (4 rounds). Training uses all stationary and transient
    recordings of the other three speeds; the held-out speed gives 6 test recordings per class (1 stationary
    + 5 transient). One prediction per recording (mean class probabilities of its windows); every training
    recording gets equal weight. The feature set and model are chosen by leave-one-speed-out validation
    inside the training speeds only. All features are computed per window from that window alone (vibration
    in shaft orders, current in supply-frequency orders), so no preprocessing looks at the held-out speed.

    Secondary (easier) time split: stationary runs only, first 60% of every run trains, last 30% tests.
    Temperature turned out to identify the recording session (it separates the conditions already in the
    first 5 minutes, before any fault heating), so temperature sets are reported but excluded from selection.
    Temperature is offered two ways: the pattern only (each probe relative to the others and to the run's first
    minute) and including absolute temperature, which mostly follows the room temperature of that recording day.
    Results go to Results/<machine>/<machine>_fulldata.json; the Week 6 results are not changed.
    """


    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd


    FULL = ROOT / "Raw_Data" / "cora_full"
    VX = ROOT / "Raw_Data" / "cora_vibration"
    CACHE = ROOT / "Processed_Data" / "cora_full_features"
    SPEEDS = ("F5", "F15", "F50", "F60")
    RATE = {"V": 3000, "C": 4000, "T": 1000}
    ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
    VIB = ["log_rms", "kurtosis", "skewness", "crest", "peak_order", "log_2x_1x",
           *(f"order_{lo}_{hi}" for lo, hi in ORDERS)]
    CUR = ["log_rms", "kurtosis", "crest", "log_thd", "log_sideband", "spectral_entropy",
           *(f"order_{lo}_{hi}" for lo, hi in ORDERS)]
    TRANSIENT_WINDOWS = range(10, 30)   # steady part of the 30 s start-up test


    # ---------------------------------------------------------------------------------------------- features

    def _read(path: Path) -> np.ndarray:
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}: download the CORA files first")
        return load_csv(path, header=None, dtype=np.float64, engine="c").iloc[:, 0].to_numpy()


    def _windows(x: np.ndarray, rate: int) -> np.ndarray:
        n = len(x) // rate
        w = x[: n * rate].reshape(n, rate)
        return w - w.mean(axis=1, keepdims=True)


    def _order_shares(power, freq, base):
        order = freq / base
        usable = (order >= 0.5) & (order < 20.5)
        share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
        o = order[usable]
        return share, o, [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in ORDERS]


    def vibration_features(x, speed):
        rate, shaft = RATE["V"], float(speed[1:])
        w = _windows(x, rate)
        std = np.maximum(w.std(axis=1), 1e-12)
        freq = np.fft.rfftfreq(rate, d=1 / rate)
        power = np.abs(np.fft.rfft(w * np.hanning(rate), axis=1)) ** 2
        share, o, bands = _order_shares(power, freq, shaft)
        return np.column_stack([np.log(std), (w ** 4).mean(axis=1) / std ** 4, (w ** 3).mean(axis=1) / std ** 3,
                                np.abs(w).max(axis=1) / std, o[np.argmax(share, axis=1)],
                                np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands])


    def current_features(x, speed):
        rate, supply = RATE["C"], float(speed[1:])
        w = _windows(x, rate)
        std = np.maximum(w.std(axis=1), 1e-12)
        freq = np.fft.rfftfreq(rate, d=1 / rate)            # 1 Hz bins
        power = np.abs(np.fft.rfft(w * np.hanning(rate), axis=1)) ** 2
        f0 = int(round(supply))
        fundamental = power[:, max(f0 - 1, 1): f0 + 2].sum(axis=1) + 1e-12
        harmonics = sum(power[:, k * f0 - 1: k * f0 + 2].sum(axis=1) for k in range(2, 10) if k * f0 + 2 < power.shape[1])
        side = np.r_[max(f0 - 6, 1): max(f0 - 1, 1), f0 + 2: f0 + 7]
        sideband = power[:, side].sum(axis=1)
        p = power[:, 1:] / np.maximum(power[:, 1:].sum(axis=1, keepdims=True), 1e-12)
        entropy = -(p * np.log(p + 1e-15)).sum(axis=1)
        _, _, bands = _order_shares(power, freq, supply)
        return np.column_stack([np.log(std), (w ** 4).mean(axis=1) / std ** 4, np.abs(w).max(axis=1) / std,
                                np.log(harmonics / fundamental + 1e-12), np.log(sideband / fundamental + 1e-12),
                                entropy, *bands])


    def temperature_features(x):
        n = len(x) // RATE["T"]
        return x[: n * RATE["T"]].reshape(n, RATE["T"]).mean(axis=1)[:, None]


    def cached(name: str, compute):
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}.npy"
        if path.is_file():
            return load_numpy(path)
        value = compute()
        np.save(path, value)
        return value


    def recording(code: str, speed: str, regime: str, rep: int | None = None) -> pd.DataFrame:
        """All features of one recording, one row per 1 s window."""
        tag = f"{code}_{speed}_{regime}"
        suffix = "" if rep is None else f"_{rep}"
        parts = {}
        for axis in ("x", "y", "z"):
            if regime == "S" and axis == "x":
                path = VX / f"{tag}_Vx.csv"
            else:
                path = FULL / f"{tag}_V{axis}{suffix}.csv"
            parts[f"vib_{axis}"] = (cached(f"{tag}_V{axis}{suffix}", lambda p=path: vibration_features(_read(p), speed)), VIB)
        for phase in ("1", "2", "3"):
            path = FULL / f"{tag}_C{phase}{suffix}.csv"
            parts[f"cur_{phase}"] = (cached(f"{tag}_C{phase}{suffix}", lambda p=path: current_features(_read(p), speed)), CUR)
        if regime == "S":
            for probe in range(1, 7):
                path = FULL / f"{tag}_T{probe}.csv"
                parts[f"temp_{probe}"] = (cached(f"{tag}_T{probe}", lambda p=path: temperature_features(_read(p))), ["mean"])
        n = min(len(v) for v, _ in parts.values())
        frame = pd.DataFrame({f"{key}_{name}": values[:n, i] for key, (values, names) in parts.items()
                              for i, name in enumerate(names)})
        if regime == "S":   # temperature relative to the probes' own mean and to the run's first minute (same recording only)
            probes = [f"temp_{p}_mean" for p in range(1, 7)]
            frame[[f"{c}_vs_probes" for c in probes]] = frame[probes].to_numpy() - frame[probes].mean(axis=1).to_numpy()[:, None]
            frame[[f"{c}_rise" for c in probes]] = frame[probes].to_numpy() - frame[probes].iloc[:60].mean().to_numpy()
        else:
            frame = frame.iloc[[i for i in TRANSIENT_WINDOWS if i < len(frame)]]
        frame.insert(0, "window", np.arange(len(frame)))
        frame.insert(0, "regime", "stationary" if regime == "S" else "transient")
        frame.insert(0, "speed", speed)
        frame.insert(0, "unit", tag + suffix)
        return frame.reset_index(drop=True)


    def build(conditions: dict[str, str]) -> pd.DataFrame:
        frames = []
        for code, label in conditions.items():
            for speed in SPEEDS:
                for frame in [recording(code, speed, "S")] + [recording(code, speed, "T", r) for r in range(1, 6)]:
                    frame.insert(1, "class", label)
                    frames.append(frame)
                print(f"  features ready: {code}_{speed} (stationary + 5 start-up tests)", flush=True)
        return pd.concat(frames, ignore_index=True)


    def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
        cols = [c for c in table.columns if c not in ("unit", "class", "speed", "regime", "window")]
        vx = [c for c in cols if c.startswith("vib_x_")]
        vib = [c for c in cols if c.startswith("vib_")]
        cur = [c for c in cols if c.startswith("cur_")]
        temp = [c for c in cols if c.startswith("temp_")]
        pattern = [c for c in temp if c.endswith(("_vs_probes", "_rise"))]   # no absolute temperature (room-dependent)
        return {"vibration X (as Week 6, all windows)": vx, "vibration X/Y/Z": vib, "current C1-C3": cur,
                "vibration + current": vib + cur,
                "vibration + current + temperature pattern (stationary only)": vib + cur + pattern,
                "vibration + current + temperature incl. absolute (stationary only)": vib + cur + temp}


    # ---------------------------------------------------------------------------------------------- models

    def models():
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        return {
            "Random forest": lambda: RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2,
                                                            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            "LightGBM": lambda: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
                                               subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                                               verbose=-1),
        }


    def _fit_predict(make, train: pd.DataFrame, test: pd.DataFrame, features, labels):
        import warnings
        weight = 1.0 / train.groupby("unit")["unit"].transform("size").to_numpy()
        model = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[features].to_numpy(), train["class"].to_numpy(), sample_weight=weight)
            proba = model.predict_proba(test[features].to_numpy())
        out = pd.DataFrame(proba, columns=list(model.classes_)).reindex(columns=labels, fill_value=0.0)
        out["unit"], out["class"], out["regime"] = test["unit"].to_numpy(), test["class"].to_numpy(), test["regime"].to_numpy()
        per_unit = out.groupby("unit").agg({**{c: "mean" for c in labels}, "class": "first", "regime": "first"})
        per_unit["predicted"] = per_unit[labels].idxmax(axis=1)
        return per_unit


    def _score(per_unit: pd.DataFrame, labels):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
        block = {}
        for name, part in (("all", per_unit), ("stationary", per_unit[per_unit.regime == "stationary"]),
                           ("transient", per_unit[per_unit.regime == "transient"])):
            if len(part):
                block[name] = {"test_recordings": int(len(part)),
                               "accuracy": round(float(accuracy_score(part["class"], part["predicted"])), 4),
                               "balanced_accuracy": round(float(balanced_accuracy_score(part["class"], part["predicted"])), 4),
                               "macro_f1": round(float(f1_score(part["class"], part["predicted"], labels=labels,
                                                                average="macro", zero_division=0)), 4)}
        block["confusion_matrix_all"] = confusion_matrix(per_unit["class"], per_unit["predicted"], labels=labels).tolist()
        return block


    def _subset(table, name):
        return table[table.regime == "stationary"] if "stationary only" in name else table


    def run(machine: str, conditions: dict[str, str], train_step: int = 3) -> dict:
        """Run the full-data experiment for one machine and save Results/<machine>/<machine>_fulldata.json."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        start = time.perf_counter()
        labels = list(conditions.values())
        print(f"{machine}: building features from every CORA sensor and recording", flush=True)
        table = build(conditions)
        sets = feature_sets(table)
        makers = models()
        # train on every train_step-th window of the long stationary runs (equal recording weights anyway)
        thin = lambda t: t[(t.regime == "transient") | (t.window % train_step == 0)]

        grid, chosen, outer_units = {}, [], []
        for held in SPEEDS:
            train_all, test_all = table[table.speed != held], table[table.speed == held]
            # inner leave-one-speed-out validation on the three training speeds: choose feature set + model
            best, best_score = None, -1.0
            for set_name, features in sets.items():
                if "temperature" in set_name:
                    continue   # confounded with the recording session (see temperature_diagnostic); never selected
                for model_name, make in makers.items():
                    scores = []
                    for inner in sorted(set(train_all.speed)):
                        tr = _subset(train_all[train_all.speed != inner], set_name)
                        va = _subset(train_all[train_all.speed == inner], set_name)
                        pu = _fit_predict(make, thin(tr), va, features, labels)
                        scores.append(_score(pu, labels)["all"]["macro_f1"])
                    if np.mean(scores) > best_score:
                        best, best_score = (set_name, model_name), float(np.mean(scores))
            chosen.append({"held_out_speed": held, "feature_set": best[0], "model": best[1],
                           "validation_macro_f1": round(best_score, 4)})
            # every feature set x model on the held-out speed (reported for comparison, not used to choose)
            for set_name, features in sets.items():
                for model_name, make in makers.items():
                    pu = _fit_predict(make, thin(_subset(train_all, set_name)), _subset(test_all, set_name), features, labels)
                    grid.setdefault((set_name, model_name), []).append(pu)
                    if (set_name, model_name) == best:
                        outer_units.append(pu)
            print(f"  held-out {held}: validation chose {best[0]} / {best[1]}", flush=True)

        result = {
            "purpose": "Week 7 full-data sensor experiment: every CORA sensor, every window, stationary + start-up recordings",
            "machine": machine, "classes": labels,
            "test": "four rounds, each holding out one whole speed; test recordings per round = 1 stationary + 5 start-up "
                    "tests per class; one prediction per recording",
            "selection_rule": "feature set and model chosen by leave-one-speed-out validation inside the training speeds; "
                              "test speeds never used for any choice or preprocessing; temperature sets excluded from "
                              "selection (session confound, see temperature_diagnostic)",
            "recordings": {"stationary": int(table[table.regime == "stationary"].unit.nunique()),
                           "transient": int(table[table.regime == "transient"].unit.nunique())},
            "windows": int(len(table)),
            "validation_selected": {"choices": chosen, "score": _score(pd.concat(outer_units), labels)},
            "all_combinations": {f"{s} | {m}": _score(pd.concat(v), labels) for (s, m), v in grid.items()},
        }

        # temperature diagnostic: does the probe pattern identify the condition before fault heating can build up?
        stat_all = table[table.regime == "stationary"]
        probe_pattern = [c for c in table.columns if c.endswith("_vs_probes")]
        warming = [c for c in table.columns if c.endswith("_rise")]
        diag = {}
        for name, part, cols in (("probe pattern, first 5 minutes only", stat_all[stat_all.window < 300], probe_pattern),
                                 ("probe pattern, whole run", stat_all[stat_all.window % train_step == 0], probe_pattern),
                                 ("warming since run start, whole run", stat_all[stat_all.window % train_step == 0], warming)):
            units = pd.concat([_fit_predict(makers["Random forest"], part[part.speed != h], part[part.speed == h], cols, labels)
                               for h in SPEEDS])
            diag[name] = _score(units, labels)["all"]
        result["temperature_diagnostic"] = {
            "finding": "if the probe pattern already identifies the condition in the first 5 minutes, it reflects the "
                       "recording session (probe placement / residual heat after each rebuild), not fault heating; "
                       "temperature feature sets are therefore reported but never selected",
            "unseen_speed_scores": diag}

        # secondary time split: stationary runs only
        stat = table[table.regime == "stationary"].copy()
        position = stat.groupby("unit")["window"].rank(pct=True)
        early, late = stat[position <= 0.6], stat[position > 0.7]
        result["time_split_stationary"] = {
            f"{s} | Random forest": _score(_fit_predict(makers["Random forest"], early.iloc[::train_step], late, f, labels), labels)
            for s, f in sets.items()}

        out = ROOT / "Results" / machine / f"{machine}_fulldata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        names = list(result["all_combinations"])
        figure, axis = plt.subplots(figsize=(9, 4))
        pos = np.arange(len(names))
        axis.bar(pos - 0.2, [result["all_combinations"][n]["all"]["accuracy"] for n in names], 0.4, label="all test recordings")
        axis.bar(pos + 0.2, [result["all_combinations"][n].get("stationary", {}).get("accuracy", 0) for n in names], 0.4,
                 label="stationary runs only")
        axis.axhline(1 / len(labels), color="grey", ls="--", lw=1)
        axis.set_xticks(pos, [n.replace(" | ", "\n").replace(" (", "\n(") for n in names], fontsize=6)
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8)
        v = result["validation_selected"]["score"]["all"]
        axis.set_title(f"{machine.replace('_', ' ').title()} full data, unseen speed: validation-selected "
                       f"{v['accuracy']:.3f} on {v['test_recordings']} recordings (dashed = chance)", fontsize=9)
        figure.tight_layout()
        fig = ROOT / "Figures" / machine / f"{machine}_fulldata.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(fig, dpi=110)
        plt.close(figure)
        s = result["validation_selected"]["score"]
        parts = ", ".join(f"{k} {s[k]['accuracy']:.3f}" for k in ("stationary", "transient") if k in s)
        print(f"Validation-selected: all {s['all']['accuracy']:.3f} ({s['all']['test_recordings']} recordings); {parts}")
        print(f"Saved {out.relative_to(ROOT)} and {fig.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
        save_recording_confusion(result)
        return result


    # ---------------------------------------------------------------------------------------------- Week 7 extras
    # The ensemble comparison and the anomaly detection rerun on the full data, with the same fair main test:
    # one whole speed held out per round, nothing chosen or fitted on the held-out speed.

    FIXED_SET = "vibration + current"   # every non-temperature sensor; fixed in advance, not chosen on any test


    def ensemble_models():
        from lightgbm import LGBMClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from xgboost import XGBClassifier
        return {
            "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
            "Random forest": models()["Random forest"],
            "Extra trees": lambda: ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                                        class_weight="balanced", random_state=42, n_jobs=-1),
            "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                                       random_state=42),
            "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1,
                                             verbosity=0),
            "LightGBM": models()["LightGBM"],
        }


    class _Encoded:
        """Wrap a classifier so it trains on integer labels (XGBoost needs them) and reports the class names."""

        def __init__(self, model, labels):
            self.model, self.labels = model, labels

        def fit(self, x, y, sample_weight=None):
            code = {label: i for i, label in enumerate(self.labels)}
            self.model.fit(x, np.array([code[v] for v in y]), sample_weight=sample_weight)
            self.classes_ = np.array([self.labels[i] for i in self.model.classes_])
            return self

        def predict_proba(self, x):
            return self.model.predict_proba(x)


    def ensembles(machine: str, conditions: dict[str, str], train_step: int = 3) -> dict:
        """Ensemble comparison on the full data; saves Results/<machine>/<machine>_fulldata_ensembles.json."""
        start = time.perf_counter()
        labels = list(conditions.values())
        print(f"{machine}: ensemble comparison on the full data", flush=True)
        table = build(conditions)
        features = feature_sets(table)[FIXED_SET]
        thin = lambda t: t[(t.regime == "transient") | (t.window % train_step == 0)]
        stat = table[table.regime == "stationary"]
        position = stat.groupby("unit")["window"].rank(pct=True)
        early, late = stat[position <= 0.6], stat[position > 0.7]
        result = {"purpose": "Week 7: ensemble comparison (proposal Week 6) rerun on every CORA sensor, window and recording",
                  "machine": machine, "classes": labels, "features": f"{FIXED_SET} ({len(features)} features)",
                  "test": "MAIN: four rounds, each holding out one whole speed (24 test recordings per round, 96 in total); "
                          "one prediction per recording. TIME SPLIT (secondary, easier): stationary runs, first 60% "
                          "trains, last 30% tests",
                  "selection_rule": "feature set and model settings fixed in advance; nothing chosen on any test result",
                  "models": {}}
        for name, make in ensemble_models().items():
            wrapped = lambda make=make: _Encoded(make(), labels)
            t0 = time.perf_counter()
            main = pd.concat([_fit_predict(wrapped, thin(table[table.speed != held]), table[table.speed == held],
                                           features, labels) for held in SPEEDS])
            seconds = time.perf_counter() - t0
            result["models"][name] = {"main": _score(main, labels),
                                      "time_split_stationary": _score(_fit_predict(wrapped, early.iloc[::train_step], late,
                                                                                   features, labels), labels),
                                      "seconds": round(seconds, 1)}
            m = result["models"][name]
            print(f"  {name:33s} main {m['main']['all']['accuracy']:.3f} (stationary "
                  f"{m['main']['stationary']['accuracy']:.3f}, start-up {m['main']['transient']['accuracy']:.3f})  "
                  f"time split {m['time_split_stationary']['all']['accuracy']:.3f}  [{seconds:.0f} s]", flush=True)
        out = ROOT / "Results" / machine / f"{machine}_fulldata_ensembles.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _bar_figure(machine, "ensembles", {n: (m["main"]["all"]["accuracy"], m["time_split_stationary"]["all"]["accuracy"])
                                           for n, m in result["models"].items()},
                    ("main test (unseen speed)", "time split (secondary)"), 1 / len(labels),
                    f"{machine.replace('_', ' ').title()} full data: ensemble comparison (dashed = chance)")
        print(f"Saved {out.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
        return result


    def _detectors():
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        return {"Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                    n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
                "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))}


    def _detector_fit(make, frame, features, max_rows=4000):
        # every healthy recording gets an equal share of the (at most max_rows) training rows
        per_unit = max(1, max_rows // frame.unit.nunique())
        keep = np.concatenate([idx[np.linspace(0, len(idx) - 1, min(len(idx), per_unit)).astype(int)]
                               for idx in frame.groupby("unit").indices.values()])
        return make().fit(frame[features].to_numpy()[keep])


    def _recording_scores(model, frame, features):
        window = -model.decision_function(frame[features].to_numpy())   # higher = more unusual
        return pd.Series(window, index=frame.index).groupby(frame.unit).mean()


    def _anomaly_block(rows: pd.DataFrame, labels) -> dict:
        from sklearn.metrics import roc_auc_score
        truth = (rows["class"] != "healthy").to_numpy()
        alarm = rows["margin"].to_numpy() > 0
        return {"test_recordings": int(len(rows)), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
                "roc_auc": round(float(roc_auc_score(truth, rows["margin"])), 4) if 0 < truth.sum() < len(truth) else None,
                "fault_detection_rate": round(float(alarm[truth].mean()), 4),
                "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
                "accuracy": round(float((alarm == truth).mean()), 4),
                "detection_rate_per_fault": {c: round(float(alarm[(rows["class"] == c).to_numpy()].mean()), 4)
                                             for c in labels if c != "healthy"}}


    def anomaly(machine: str, conditions: dict[str, str]) -> dict:
        """Anomaly detection on the full data; saves Results/<machine>/<machine>_fulldata_anomaly.json."""
        start = time.perf_counter()
        labels = list(conditions.values())
        print(f"{machine}: anomaly detection on the full data", flush=True)
        table = build(conditions)
        features = feature_sets(table)[FIXED_SET]
        result = {"purpose": "Week 7: anomaly detection rerun on every CORA sensor, window and recording "
                             "(trained on healthy data only)",
                  "machine": machine, "features": f"{FIXED_SET} ({len(features)} features)",
                  "test": "MAIN: four rounds, each holding out one whole speed (24 test recordings per round: 6 healthy, "
                          "18 faulty). TIME SPLIT (secondary): stationary runs, healthy first 60% trains, last 30% tests",
                  "threshold_rule": "MAIN: leave-one-speed-out inside the training speeds - fit on the healthy recordings "
                                    "of two training speeds, score the healthy recordings of the third; the highest score "
                                    "is the alarm threshold. TIME SPLIT: leave-one-healthy-run-out on the early parts. "
                                    "No test data and no fault labels are used to train or to set the threshold",
                  "detectors": {}}
        stat = table[table.regime == "stationary"]
        position = stat.groupby("unit")["window"].rank(pct=True)
        early, late = stat[position <= 0.6], stat[position > 0.7]
        for name, make in _detectors().items():
            rows = []
            for held in SPEEDS:
                train = table[(table.speed != held) & (table["class"] == "healthy")]
                cut = max(_recording_scores(_detector_fit(make, train[train.speed != inner], features),
                                            train[train.speed == inner], features).max()
                          for inner in sorted(set(train.speed)))
                test = table[table.speed == held]
                scores = _recording_scores(_detector_fit(make, train, features), test, features)
                rows.append(test.groupby("unit")[["class", "regime"]].first().assign(margin=scores - cut))
            rows = pd.concat(rows)
            healthy_early = early[early["class"] == "healthy"]
            cut = max(_recording_scores(_detector_fit(make, healthy_early[healthy_early.unit != u], features),
                                        healthy_early[healthy_early.unit == u], features).max()
                      for u in sorted(set(healthy_early.unit)))
            ts = _recording_scores(_detector_fit(make, healthy_early, features), late, features)
            ts_rows = late.groupby("unit")[["class", "regime"]].first().assign(margin=ts - cut)
            block = {"main": _anomaly_block(rows, labels),
                     "main_stationary": _anomaly_block(rows[rows.regime == "stationary"], labels),
                     "main_transient": _anomaly_block(rows[rows.regime == "transient"], labels),
                     "time_split_stationary": _anomaly_block(ts_rows, labels)}
            result["detectors"][name] = block
            m, t = block["main"], block["time_split_stationary"]
            print(f"  {name:17s} main ROC-AUC {m['roc_auc']}  detected {m['fault_detection_rate']:.3f}  false alarms "
                  f"{m['false_alarm_rate']:.3f} ({m['test_recordings']} recordings) | time split ROC-AUC {t['roc_auc']}",
                  flush=True)
        out = ROOT / "Results" / machine / f"{machine}_fulldata_anomaly.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _bar_figure(machine, "anomaly", {n: (b["main"]["roc_auc"], b["time_split_stationary"]["roc_auc"])
                                         for n, b in result["detectors"].items()},
                    ("main test ROC-AUC (unseen speed)", "time split ROC-AUC (secondary)"), 0.5,
                    f"{machine.replace('_', ' ').title()} full data: anomaly detection (dashed = chance)")
        print(f"Saved {out.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
        return result


    def _bar_figure(machine, kind, values, legend, chance, title):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(values)
        figure, axis = plt.subplots(figsize=(8, 3.6))
        pos = np.arange(len(names))
        for k, offset in enumerate((-0.2, 0.2)):
            axis.bar(pos + offset, [values[n][k] or 0 for n in names], 0.4, label=legend[k])
        axis.axhline(chance, color="grey", ls="--", lw=1)
        axis.set_xticks(pos, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8)
        axis.set_title(title, fontsize=10)
        figure.tight_layout()
        figure.savefig(ROOT / "Figures" / machine / f"{machine}_fulldata_{kind}.png", dpi=110)
        plt.close(figure)


    def save_recording_confusion(result: dict) -> Path:
        """Plot real held-out recording counts and row percentages from saved results."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matrix = np.asarray(result["validation_selected"]["score"]["confusion_matrix_all"])
        labels = result["classes"]
        support = matrix.sum(axis=1)
        total = int(matrix.sum())
        score = result["validation_selected"]["score"]["all"]
        figure, axis = plt.subplots(figsize=(8, 7))
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks(range(len(labels)), labels, rotation=25)
        axis.set_yticks(range(len(labels)), [f"{label} (n={n})" for label, n in zip(labels, support)])
        axis.set_xlabel("Predicted condition")
        axis.set_ylabel("Actual condition")
        for i in range(len(labels)):
            for j in range(len(labels)):
                percentage = 100 * matrix[i, j] / support[i] if support[i] else 0
                axis.text(j, i, f"{matrix[i, j]}\n({percentage:.1f}%)", ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=11)
        axis.set_title(f"{result['machine'].replace('_', ' ').title()}: full sensor evaluation\n"
                       f"{total} held-out recordings | accuracy {score['accuracy']:.1%} | four unseen-speed folds")
        figure.text(.5, .025, "Each cell counts whole recordings; percentages are within each actual class.\n"
                    "0 means no recordings in that combination. Start-up repeats share one laboratory rig.",
                    ha="center", fontsize=9)
        figure.tight_layout(rect=(0, .08, 1, 1))
        output = ROOT / "Figures" / result["machine"] / f"{result['machine']}_fulldata_confusion.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        return output

    return SimpleNamespace(**locals())


@lru_cache(maxsize=None)
def get_model(equipment: str):
    """Construct one equipment namespace lazily, avoiding conflicting model names."""
    factories = {"pump": _make_pump, "haul_truck": _make_haul_truck, "belt_conveyor": _make_belt_conveyor, "excavator": _make_excavator, "dozer": _make_dozer, "wheel_loader": _make_wheel_loader, "drill_rig": _make_drill_rig, "cora_fulldata": _make_cora_fulldata}
    if equipment not in factories:
        raise ValueError(f"Unknown equipment: {equipment}")
    return factories[equipment]()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python Notebook/notebook_helpers.py <equipment> <command> [options]")
        print("Equipment: pump, haul_truck, belt_conveyor, excavator, dozer, wheel_loader, drill_rig")
    else:
        equipment = sys.argv.pop(1)
        get_model(equipment).cli()
