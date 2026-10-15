"""Belt Conveyor: geared drive, CORA T1 temperature vs infrared camera.

All of this machine's code in one file. The notebook Notebook/Belt_Conveyor.ipynb
runs these commands and shows the results.

Usage:
    python Models/belt_conveyor.py <command> [options]
    python Models/belt_conveyor.py <command> --help

Commands:
    intake          download CORA infrared stacks and bounded T1 samples
    inspect-file    helper: inspect a local CORA publisher file
    preprocess      helper: CORA sensor/thermal preprocessing smoke test
    experiment      T1 temperature and infrared models, report and figure
    baselines       Week 5: compare majority class, logistic regression, decision tree and random forest
    all             run experiment

Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    INTAKE = CAPSTONE / "Results" / "belt_conveyor_cora_intake.json"
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
        reader = pd.read_csv(path, header=None, chunksize=samples_per_window)
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

    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat
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
    INTAKE = CAPSTONE / "Results" / "belt_conveyor_cora_intake.json"
    RESULT = CAPSTONE / "Results" / "belt_conveyor_cora.json"
    REPORT = CAPSTONE / "Results" / "belt_conveyor_sensor_visual_comparison.md"
    FIGURE = CAPSTONE / "Figures" / "belt_conveyor_cora_confusions.png"
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
            f"![Separate run-level confusion matrices](../Figures/{FIGURE.name})",
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
        samples = np.load(PROCESSED / "t1_bounded_samples.npz")
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

    RESULT = ROOT / "Results" / "belt_conveyor_baselines.json"
    FIGURE = ROOT / "Figures" / "belt_conveyor_baselines.png"
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
        samples = np.load(e["PROCESSED"] / "t1_bounded_samples.npz")
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


COMMANDS = {
    "intake": intake,
    "inspect-file": inspect_file,
    "preprocess": preprocess,
    "experiment": experiment,
    "baselines": baselines,
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


if __name__ == "__main__":
    cli()
