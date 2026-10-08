"""Audit CORA sensor and thermal files without loading the large measurements.

The filename provides an experiment-level key, not a one-to-one pairing of
individual sensor samples and thermal frames. Run from the repository root:

    python Preprocessing/cora_manifest.py /path/to/CORA/files

Pass either a directory of downloaded files or the repository's
``Information_about_database_files.txt`` guide. No files are downloaded,
modified, or committed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Local/cloud-mounted CORA folder or official file guide .txt")
    parser.add_argument("--frequency", choices=sorted(FREQUENCIES), default="F15")
    parser.add_argument("--regime", choices=("S", "T"), default="S")
    args = parser.parse_args()
    audit(args.source, args.frequency, args.regime)


if __name__ == "__main__":
    main()
