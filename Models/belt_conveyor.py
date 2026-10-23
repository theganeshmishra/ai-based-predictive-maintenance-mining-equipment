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
    export-images   Week 6: save the camera images as named files in Images/<machine>/ (local only)
    csv             Week 6: 4-class sensor model from a CSV feature table
    cnn             Week 6: CNN with data augmentation on the named images (run export-images first)
    cnn-pretrained  Week 6: transfer learning, pretrained ResNet18 on the named images
    intake-vibration  check selectively downloaded CORA vibration files (MD5) and file them
    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
    ensembles       proposal Week 6: XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
    anomaly         Week 7: anomaly detection trained on healthy data only (--split time for the secondary test)
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
    from scipy.io import loadmat

    MACHINE = "belt_conveyor"
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
    from scipy.io import loadmat
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
            return pd.read_csv(cache)
        source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
        if not source.is_file():
            raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
        values = pd.read_csv(source, header=None, dtype=np.float64).iloc[:, 0].to_numpy()
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
        e = experiment(run=False)
        samples = np.load(e["PROCESSED"] / "t1_bounded_samples.npz")
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
        normalised = per_speed_normalised(table, VIB_NAMES + ["t1_mean", "t1_std", "t1_slope"])
        return (table, {"vibration": VIB_NAMES, "vibration+temperature": VIB_NAMES + ["t1_mean", "t1_std", "t1_slope"],
                        "vibration+temperature (per-speed normalised)": normalised},
                ["healthy", "wear25", "wear50", "wear75"], folds,
                "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                ["CORA Vx vibration (3 kHz, 1 s windows, every 10th window) in shaft orders, optionally with run-level T1 temperature",
                 "only one run per wear level and speed"])
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
            image = cv2.imread(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
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

    _small = cnn(run=False)
    read_index, augment = _small["read_index"], _small["augment"]

    def load(rows):
        images = []
        for row in rows:
            image = cv2.imread(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)
            images.append((image - image.mean()) / (image.std() + 1e-6))
        return np.stack(images)

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
        }
        RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        r = result["recording_level"]
        plot_matrix(r["confusion_matrix"], LABELS,
                    f"{MACHINE}: pretrained ResNet18, accuracy {r['accuracy']:.3f} ({r['test_units']} test units)", FIGURE)
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
        table = pd.read_csv(path)
        if all(_is_number(c) for c in table.columns):   # no header row: plain column(s) of numbers
            table = pd.read_csv(path, header=None)
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


COMMANDS = {
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


if __name__ == "__main__":
    cli()
