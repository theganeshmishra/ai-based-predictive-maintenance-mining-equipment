"""All capstone code in one file (backup of the former Models/ and Preprocessing/ scripts).

The notebooks in Notebook/ are self-contained and do not need this file. This file
regenerates the saved results in Results/ and Figures/.

Usage:
    python capstone_code.py <command> [options]
    python capstone_code.py <command> --help

Commands:
    pump-sensor-intake   Pump: download the 48 Ottawa vibration recordings
    pump-sensor          Pump: vibration sensor model
    pump-camera          Pump: XJTU event-camera model
    pump-compare         Pump: sensor vs camera report and figure
    pump-original-xlsx   Pump: original centrifugal-pump XLSX diagnostic (sensor only)
    belt-intake          Belt Conveyor: download CORA infrared stacks and T1 samples
    belt-manifest        Belt Conveyor: inspect a local CORA file (helper)
    belt-preprocess      Belt Conveyor: CORA preprocessing helper
    belt-conveyor        Belt Conveyor: T1 temperature vs infrared models
    haul-truck-intake    Haul Truck: download UODS vibration and the XJTU-Rotor zip
    haul-truck           Haul Truck: vibration vs event-camera models
    uci-hydraulic        Supporting: UCI hydraulic pump leakage (sensor only)
    espset               Supporting: ESPset V3 pump vibration (sensor only)
    rock-drill           Supporting: PHM 2022 rock drill (sensor only)

Each command is one of the original scripts, unchanged, wrapped in a function.
Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Saved results: one JSON file per machine in Results/, one section per experiment.
MACHINE_SECTIONS = {
    "pump": ('sensor_intake', 'sensor', 'camera', 'original_xlsx_diagnostic', 'uci_hydraulic', 'espset'),
    "belt_conveyor": ('intake', 'experiment'),
    "haul_truck": ('intake', 'experiment'),
    "drill_rig": ('rock_drill_sensor_only',),
}


class _NoParent:
    def mkdir(self, *args, **kwargs) -> None:
        pass


class ResultSection:
    """One section of Results/<machine>.json.

    It offers the few pathlib.Path methods the experiment code uses (read_text, write_text,
    exists, is_file), so each experiment still reads and writes its own JSON document,
    which is now stored as one section of its machine's file.
    """

    parent = _NoParent()

    def __init__(self, machine: str, section: str) -> None:
        self.machine, self.section = machine, section
        self.file = ROOT / "Results" / f"{machine}.json"

    def _load(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8")) if self.file.is_file() else {}

    def exists(self) -> bool:
        return self.section in self._load()

    is_file = exists

    def read_text(self, encoding: str = "utf-8") -> str:
        data = self._load()
        if self.section not in data:
            raise FileNotFoundError(f"{self} has not been created yet")
        return json.dumps(data[self.section])

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        data = self._load()
        data[self.section] = json.loads(text)
        order = MACHINE_SECTIONS[self.machine]
        ordered = {key: data[key] for key in order if key in data}
        ordered.update({key: value for key, value in data.items() if key not in ordered})
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def relative_to(self, *args) -> "ResultSection":
        return self

    def __str__(self) -> str:
        return f"Results/{self.machine}.json [{self.section}]"

# ==================================================================================================
# pump-sensor-intake: Pump: download the 48 Ottawa vibration recordings
# (formerly Preprocessing/pump_bearing_independent_intake.py)
# ==================================================================================================

def pump_sensor_intake(run: bool = True) -> dict:
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
    MANIFEST = ResultSection("pump", "sensor_intake")
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
# pump-sensor: Pump: vibration sensor model
# (formerly Models/pump_bearing_independent_baseline.py)
# ==================================================================================================

def pump_sensor(run: bool = True) -> dict:
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
    from scipy.io import loadmat
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


    CAPSTONE = ROOT
    INTAKE = ResultSection("pump", "sensor_intake")
    RAW = CAPSTONE / "Raw_Data" / "uottawa_bearing"
    CACHE = CAPSTONE / "Processed_Data" / "pump_bearing_independent_features.npz"
    RESULT = ResultSection("pump", "sensor")
    FIGURE = CAPSTONE / "Figures" / "pump_bearing_independent_confusions.png"
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
            signal = loadmat(RAW / name, variable_names=["Channel_1"])["Channel_1"].ravel()
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
            with np.load(CACHE, allow_pickle=False) as saved:
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
# pump-camera: Pump: XJTU event-camera model
# (formerly Models/pump_event_visual_baseline.py)
# ==================================================================================================

def pump_camera(run: bool = True) -> dict:
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
    RESULT = ResultSection("pump", "camera")
    FIGURE = CAPSTONE / "Figures" / "pump_event_visual_confusion.png"
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
            with np.load(cached_run, allow_pickle=False) as saved:
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
            with np.load(CACHE, allow_pickle=False) as saved:
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
# pump-compare: Pump: sensor vs camera report and figure
# (formerly Models/pump_compare_modalities.py)
# ==================================================================================================

def pump_compare(run: bool = True) -> dict:
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
    SENSOR = ResultSection("pump", "sensor")
    SENSOR_DIAGNOSTIC = ResultSection("pump", "original_xlsx_diagnostic")
    VISUAL = ResultSection("pump", "camera")
    REPORT = CAPSTONE / "Results" / "pump_sensor_visual_comparison.md"
    FIGURE = CAPSTONE / "Figures" / "pump_sensor_visual_confusions.png"
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
            "- **Sensor source:** [Huang and Baddour, University of Ottawa bearing vibration V2](https://data.mendeley.com/datasets/v43hmbwxpm/2), DOI 10.17632/v43hmbwxpm.2, CC BY 4.0. Four single-condition classes, four speed profiles and three trials per profile yield 48 source MAT files. The [verified intake record](pump.json) contains publisher SHA-256 values for every file. The compound-fault fifth class is excluded.",
            "- **Sensor preprocessing:** only accelerometer `Channel_1` at 200 kHz is used; encoder `Channel_2` and file names are not predictors. Each 10 s run becomes 100 non-overlapping 0.1 s windows, with 16 time/spectral features. A fixed random forest is fit only on A/B runs. Window probabilities are averaged before **one prediction per held-out C/D recording**; no window from a test recording appears in training. Hyperparameters were not selected on the test runs.",
            "- **Visual source and unchanged experiment:** [XJTU-DV-Pump](https://zenodo.org/records/17447083), CC BY 4.0. The first two seconds of each of 48 relevant released recordings become eight 250 ms polarity-count event images. OpenCV HOG feeds a fixed random forest trained on 20 Hz runs; average frame probabilities produce one prediction per separate 30 Hz recording. These are event-camera motion images, **not RGB photographs of visible surface damage**. Full-file CRCs were not checked because only bounded intervals were read remotely.",
            "- **Original pump-vibration source:** [Jiang centrifugal-pump multi-fault V1](https://data.mendeley.com/datasets/63gjxpn7s6/1) was retained as a pump-domain pipeline diagnostic, but it has one file per selected class. Its within-file window score is **excluded from the independent comparison**. The Ottawa rig supplies repeat-run evidence at the same bearing-fault taxonomy, not direct pump-domain validation.",
            "- **Limits:** none of these evaluations holds out a new physical pump or mine. The Ottawa data do not establish independent fault-bearing specimens across all trials. Different source rigs and operating shifts preclude a causal claim that one modality is better. No matching timestamps, sample-level pairing, multimodal fusion, maintenance dates, failure horizon, or remaining-useful-life target exist.",
            "",
            "## Professor-requirement decision",
            "",
            "**Yes, for the professor's explicitly allowed component/fault-level comparison using sufficiently similar machines:** real vibration and real event-camera images both classify healthy/inner/outer/ball bearing conditions, and both now have recording-held-out Accuracy, Precision, Recall, F1, per-class results and confusion matrices. **No**, if the requirement means paired sensor-plus-camera observations from the *same physical mining pump*, a controlled modality-superiority experiment, image–sensor fusion, or prediction of a future repair date. The sensor proxy is a bearing rig, so this is a defensible component-level Pump proof of concept, **not validated mining-pump deployment**. Confirm that scope with the professor before describing the project as fully predictive maintenance.",
            "",
            f"![Separate recording-level confusion matrices](../Figures/{FIGURE.name})",
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
# pump-original-xlsx: Pump: original centrifugal-pump XLSX diagnostic (sensor only)
# (formerly Models/pump_bearing_baseline.py)
# ==================================================================================================

def pump_original_xlsx(run: bool = True) -> dict:
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
    OUTPUT = ResultSection("pump", "original_xlsx_diagnostic")
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
            frame = pd.read_excel(workbook, sheet_name=sheet, header=None)
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
# belt-intake: Belt Conveyor: download CORA infrared stacks and T1 samples
# (formerly Preprocessing/cora_gear_intake.py)
# ==================================================================================================

def belt_intake(run: bool = True) -> dict:
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
    from scipy.io import whosmat


    CAPSTONE = ROOT
    RAW = CAPSTONE / "Raw_Data" / "cora_gear_wear"
    PROCESSED = CAPSTONE / "Processed_Data" / "cora_gear_wear"
    INTAKE = ResultSection("belt_conveyor", "intake")
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
        parser = argparse.ArgumentParser(description=belt_intake.__doc__)
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
# belt-manifest: Belt Conveyor: inspect a local CORA file (helper)
# (formerly Preprocessing/cora_manifest.py)
# ==================================================================================================

def belt_manifest(run: bool = True) -> dict:
    """Audit CORA sensor and thermal files without loading the large measurements.

    The filename provides an experiment-level key, not a one-to-one pairing of
    individual sensor samples and thermal frames. Run from the repository root:

        python capstone_code.py belt-manifest /path/to/CORA/files

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
        parser = argparse.ArgumentParser(description=belt_manifest.__doc__)
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
# belt-preprocess: Belt Conveyor: CORA preprocessing helper
# (formerly Preprocessing/cora_preprocessing.py)
# ==================================================================================================

def belt_preprocess(run: bool = True) -> dict:
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

    _manifest = belt_manifest(run=False)
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
        parser = argparse.ArgumentParser(description=belt_preprocess.__doc__)
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
# belt-conveyor: Belt Conveyor: T1 temperature vs infrared models
# (formerly Models/belt_conveyor_cora_baseline.py)
# ==================================================================================================

def belt_conveyor(run: bool = True) -> dict:
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
    INTAKE = ResultSection("belt_conveyor", "intake")
    RESULT = ResultSection("belt_conveyor", "experiment")
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
            "- [CORA version 2.1](https://doi.org/10.34810/DATA2500), CC BY 4.0. The [intake record](belt_conveyor.json) lists exact publisher file IDs, local SHA-256 checksums, source sizes and selected sensor byte ranges. All 16 MAT stacks passed filename/structure checks; H_F5_S also matched the publisher's MD5. The other 15 local SHA-256 values establish future local integrity but were not compared with publisher checksums. All source MAT stacks and sampled values remain in Git-ignored `Raw_Data/` and `Processed_Data/`.",
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


# ==================================================================================================
# haul-truck-intake: Haul Truck: download UODS vibration and the XJTU-Rotor zip
# (formerly Preprocessing/haul_truck_intake.py)
# ==================================================================================================

def haul_truck_intake(run: bool = True) -> dict:
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
    MANIFEST = ResultSection("haul_truck", "intake")
    MENDELEY = "https://data.mendeley.com/public-api/datasets/y2px5tg92h"
    FOLDERS = {"H": "1_Healthy", "I": "2_Inner_Race_Faults", "O": "3_Outer_Race_Faults"}
    NAME = re.compile(r"^([HIO])_(\d+)_([012])\.mat$")
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
        return 6 <= bearing <= 10 and stage == 2


    def sensor_inventory() -> list[dict]:
        by_name = {item["name"]: item["id"] for item in get_json(f"{MENDELEY}/folders/1")}
        if not set(FOLDERS.values()).issubset(by_name):
            raise ValueError("Expected publisher fault folders are missing")
        items = []
        for code, folder in FOLDERS.items():
            members = get_json(f"{MENDELEY}/files?folder_id={by_name[folder]}&version=1&%24start=0&%24limit=1000")
            for member in members:
                name = member["filename"]
                if not selected(name):
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
        counts = {code: sum(item["class_code"] == code for item in items) for code in FOLDERS}
        if counts != {"H": 10, "I": 5, "O": 5}:
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
# haul-truck: Haul Truck: vibration vs event-camera models
# (formerly Models/haul_truck_wheel_motor_bearing.py)
# ==================================================================================================

def haul_truck(run: bool = True) -> dict:
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
    from scipy.io import loadmat
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    _pump = pump_camera(run=False)  # reuse the Pump event-camera code
    BIN_US, END_US, START_US = _pump["BIN_US"], _pump["END_US"], _pump["START_US"]
    event_images, frame_features = _pump["event_images"], _pump["frame_features"]


    CAPSTONE = ROOT
    INTAKE = ResultSection("haul_truck", "intake")
    SENSOR_RAW = CAPSTONE / "Raw_Data" / "uods_vafdc"
    ROTOR_ZIP = CAPSTONE / "Raw_Data" / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip"
    ROTOR_URL = "https://zenodo.org/api/records/17447083/files/XJTU-DV-Rotor.zip/content"
    SENSOR_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_sensor_features.npz"
    RUN_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_event_runs"
    RESULT = ResultSection("haul_truck", "experiment")
    REPORT = CAPSTONE / "Results" / "haul_truck_sensor_visual_comparison.md"
    FIGURE = CAPSTONE / "Figures" / "haul_truck_confusions.png"
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
            with np.load(SENSOR_CACHE, allow_pickle=False) as saved:
                return saved["x"], saved["y"], saved["bearing"], saved["run"], json.loads(str(saved["provenance"]))
        files = json.loads(INTAKE.read_text(encoding="utf-8"))["sensor"]["files"]
        xs, ys, bearings, runs, provenance = [], [], [], [], []
        for item in files:
            name = item["name"]
            signal = loadmat(SENSOR_RAW / name)[name[:-4]][:, 0].astype(np.float64)  # column 0 = accelerometer
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
        return zipfile.ZipFile(ROTOR_ZIP) if ROTOR_ZIP.is_file() else RemoteZip(ROTOR_URL)


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
            with np.load(cached, allow_pickle=False) as saved:
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
            "Generated by `capstone_code.py haul-truck`. Target: present bearing condition "
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
            "![Confusion matrices](../Figures/haul_truck_confusions.png)",
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


# ==================================================================================================
# uci-hydraulic: Supporting: UCI hydraulic pump leakage (sensor only)
# (formerly Preprocessing/hydraulic_baseline.py)
# ==================================================================================================

def uci_hydraulic(run: bool = True) -> dict:
    """Reproduce a pump-leakage baseline on the UCI hydraulic test rig.

    This is a subsystem benchmark, not data recorded on a mining vehicle. The
    original ZIP stays under ignored Raw_Data. Features and results are derived
    locally; no raw observations are committed to Git.

    Run from the repository root:
        .venv/bin/python capstone_code.py uci-hydraulic
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
    RESULTS = ResultSection("pump", "uci_hydraulic")
    FIGURE = PROJECT / "Figures/uci_hydraulic_pump_leakage.png"
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
        return zipfile.ZipFile(ARCHIVE)


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
# espset: Supporting: ESPset V3 pump vibration (sensor only)
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
    RESULTS = ResultSection("pump", "espset")
    FIGURE = PROJECT / "Figures/espset_v3_binary_fault.png"
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
        with zipfile.ZipFile(ARCHIVE) as source:
            frame = pd.read_csv(source.open("features.csv"), sep=";")
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


# ==================================================================================================
# rock-drill: Supporting: PHM 2022 rock drill (sensor only)
# (formerly Models/rock_drill_baseline.py)
# ==================================================================================================

def rock_drill(run: bool = True) -> dict:
    """Reproducible 11-condition rock-drill pressure-signal baseline.

    The official PHM 2022 training archive contains test-cell rock-drill cycles,
    not underground fleet recordings. Its three sensor files are paired by row
    within each configuration. This experiment never joins them to photographs.
    """


    import hashlib
    import io
    import json
    import zipfile
    from contextlib import ExitStack
    from itertools import zip_longest
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


    PROJECT = ROOT
    ARCHIVE = PROJECT / "Raw_Data/phm2022_rock_drill/training_data.zip"
    PROCESSED = PROJECT / "Processed_Data/phm2022_rock_drill_features.npz"
    RESULTS = ResultSection("drill_rig", "rock_drill_sensor_only")
    FIGURE = PROJECT / "Figures/phm2022_rock_drill.png"
    EXPECTED_SHA256 = "0a49a31af500e40a00d4949020e41c6b41422ae3fc4895aeb0675c0002427719"
    SOURCE_FOLDER = "Data_Challenge_PHM2022_training_data"
    SENSORS = ("pin", "pdmp", "po")  # Exact names used in the published CSV archive.
    CONFIGURATIONS = (1, 2, 4, 5, 6)
    TRAIN_CONFIGS = (1, 2, 4)
    VALIDATION_CONFIG = 5
    TEST_CONFIG = 6
    FAULT_LABELS = tuple(range(1, 12))


    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def feature_names() -> list[str]:
        names = []
        for sensor in SENSORS:
            names.extend(f"{sensor}_shape_{index:02d}" for index in range(32))
            names.extend(f"{sensor}_{name}" for name in (
                "q10", "median", "q90", "min", "max", "difference_rms",
                "spectral_low", "spectral_mid", "spectral_high",
            ))
        return names


    def describe_cycle(signal: np.ndarray) -> np.ndarray:
        # Each source cycle is already independently normalized by the publisher.
        positions = np.linspace(0, len(signal) - 1, 32)
        shape = np.interp(positions, np.arange(len(signal)), signal)
        spectrum = np.abs(np.fft.rfft(shape - shape.mean())) ** 2
        spectral = np.array([
            spectrum[1:5].sum(), spectrum[5:11].sum(), spectrum[11:].sum(),
        ], dtype=np.float64) / max(float(spectrum[1:].sum()), 1e-12)
        summary = np.array([
            *np.quantile(signal, [0.1, 0.5, 0.9]),
            signal.min(), signal.max(), np.sqrt(np.mean(np.diff(signal) ** 2)),
            *spectral,
        ], dtype=np.float64)
        return np.concatenate((shape, summary))


    def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        if not ARCHIVE.is_file():
            raise FileNotFoundError(f"Missing {ARCHIVE}; see the official PHM source link in the README.")
        if file_sha256(ARCHIVE) != EXPECTED_SHA256:
            raise ValueError("The PHM training ZIP differs from the archive inspected for this experiment.")

        features: list[np.ndarray] = []
        labels: list[int] = []
        configurations: list[int] = []
        cycle_lengths = {sensor: [10**9, 0] for sensor in SENSORS}
        counts: dict[str, dict[str, int]] = {}
        with zipfile.ZipFile(ARCHIVE) as source:
            for config in CONFIGURATIONS:
                with ExitStack() as stack:
                    streams = []
                    for sensor in SENSORS:
                        name = f"{SOURCE_FOLDER}/data_{sensor}{config}.csv"
                        streams.append(stack.enter_context(io.TextIOWrapper(source.open(name), encoding="utf-8")))
                    class_counts = {str(label): 0 for label in FAULT_LABELS}
                    for row_number, lines in enumerate(zip_longest(*streams), start=1):
                        if any(line is None for line in lines):
                            raise ValueError(f"The three sensor files have different row counts for configuration {config}.")
                        cycle_features = []
                        row_labels = []
                        for sensor, line in zip(SENSORS, lines):
                            values = np.fromstring(line, sep=",", dtype=np.float64)
                            if len(values) < 500 or not np.isfinite(values).all():
                                raise ValueError(f"Missing/non-finite or short {sensor} cycle at {config}:{row_number}.")
                            label = int(values[0])
                            if label not in FAULT_LABELS or values[0] != label:
                                raise ValueError(f"Unexpected fault label at {config}:{row_number}.")
                            row_labels.append(label)
                            signal = values[1:]
                            cycle_lengths[sensor][0] = min(cycle_lengths[sensor][0], len(signal))
                            cycle_lengths[sensor][1] = max(cycle_lengths[sensor][1], len(signal))
                            cycle_features.append(describe_cycle(signal))
                        if len(set(row_labels)) != 1:
                            raise ValueError(f"Sensor labels do not agree at {config}:{row_number}.")
                        features.append(np.concatenate(cycle_features))
                        labels.append(row_labels[0])
                        configurations.append(config)
                        class_counts[str(row_labels[0])] += 1
                    counts[str(config)] = class_counts

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int16)
        groups = np.asarray(configurations, dtype=np.int8)
        names = feature_names()
        if x.shape != (len(y), len(names)) or not np.isfinite(x).all():
            raise ValueError("Extracted feature matrix has the wrong shape or non-finite values.")
        if any(set(y[groups == config]) != set(FAULT_LABELS) for config in CONFIGURATIONS):
            raise ValueError("A published configuration lacks one or more expected fault classes.")
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PROCESSED, x=x, y=y, configuration=groups, feature_names=np.asarray(names))
        details = {
            "cycles": int(len(y)),
            "sensors": list(SENSORS),
            "features_per_cycle": len(names),
            "configuration_class_counts": counts,
            "cycle_signal_length_min_max": cycle_lengths,
            "missing_or_nonfinite_values": 0,
            "archive_sha256": EXPECTED_SHA256,
        }
        return x, y, groups, details


    def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_by_label": {
                str(label): float(value) for label, value in zip(
                    FAULT_LABELS, recall_score(y_true, y_pred, labels=FAULT_LABELS, average=None, zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=FAULT_LABELS).tolist(),
        }


    def forest(seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )


    def main() -> None:
        x, y, groups, details = load_features()
        train = np.isin(groups, TRAIN_CONFIGS)
        validation = groups == VALIDATION_CONFIG
        test = groups == TEST_CONFIG
        if sum(map(np.count_nonzero, (train, validation, test))) != len(y):
            raise AssertionError("The configuration split dropped or duplicated cycles.")

        full = forest(42).fit(x[train], y[train])
        importance = full.feature_importances_
        selected = np.argsort(importance)[-32:][::-1]
        reduced = forest(43).fit(x[train][:, selected], y[train])
        validation_scores = {
            "full_features": score(y[validation], full.predict(x[validation])),
            "top_32_train_selected_features": score(y[validation], reduced.predict(x[validation][:, selected])),
        }
        chosen = max(validation_scores, key=lambda name: (
            validation_scores[name]["macro_f1"], validation_scores[name]["balanced_accuracy"]
        ))
        train_val = train | validation
        columns = selected if chosen.startswith("top_32") else np.arange(x.shape[1])
        model = forest(44).fit(x[train_val][:, columns], y[train_val])
        predictions = model.predict(x[test][:, columns])
        dummy = DummyClassifier(strategy="most_frequent").fit(x[train_val], y[train_val])
        result = {
            "dataset": "PHM 2022 rock-drill fault-classification training data (rockdrill11)",
            "source": "https://data.phmsociety.org/2022-phm-conference-data-challenge/",
            "paper_doi": "10.36001/phmconf.2022.v14i1.3144",
            "provenance": "Epiroc hydraulic rock drill in a controlled test cell; mining-relevant equipment, not field recordings",
            "task": "Classify current condition among 11 classes (no fault plus 10 induced faults); not future failure prediction",
            "data": details,
            "split": {
                "method": "Hold out whole source configurations (the paper's 'individuals'); no row-level random split",
                "train": list(TRAIN_CONFIGS), "validation": [VALIDATION_CONFIG], "test": [TEST_CONFIG],
                "rows": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
                "official_blind_test_used": False,
            },
            "features": {
                "normalization": "Source cycles are already individually z-normalized; no fit on validation/test rows",
                "extraction": "32 interpolated shape points, quantiles, extrema, difference RMS and three spectral-energy bands per sensor",
                "selection": "Random-forest importance on training configurations only; compare top 32 with all 123 features on validation configuration",
                "top_32_train_only_feature_names": [feature_names()[index] for index in selected],
            },
            "validation": validation_scores,
            "selection_rule": "Maximum validation macro F1, then balanced accuracy",
            "selected_model": chosen,
            "test_selected": score(y[test], predictions),
            "test_most_frequent": score(y[test], dummy.predict(x[test])),
            "limitations": [
                "Only five labelled source configurations were public in the official training archive; this local test is not the PHM competition blind benchmark.",
                "Adjacent cycles from a configuration can be strongly related, so all cycles of test configuration 6 are held out together.",
                "The publisher normalized each impact cycle before release; absolute pressure levels are unavailable.",
                "Controlled induced faults on one rock-drill test-cell setup do not establish performance on working underground drill rigs.",
                "No paired maintenance images, dated repairs, or remaining-useful-life labels are provided.",
                "The public download page does not clearly state a separate data-file reuse licence; raw files are kept locally and are not redistributed.",
            ],
        }
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        counts = np.bincount(y, minlength=12)[1:]
        axes[0].bar(FAULT_LABELS, counts, color="#416D91")
        axes[0].set_xticks(FAULT_LABELS)
        axes[0].set_xlabel("Source condition label")
        axes[0].set_ylabel("Impact cycles")
        axes[0].set_title("Published training archive")

        matrix = np.asarray(result["test_selected"]["confusion_matrix"])
        row_total = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        heat = axes[1].imshow(matrix / row_total, vmin=0, vmax=1, cmap="Blues")
        axes[1].set_xticks(np.arange(11), FAULT_LABELS, fontsize=8)
        axes[1].set_yticks(np.arange(11), FAULT_LABELS, fontsize=8)
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("Actual label")
        axes[1].set_title("Held-out configuration 6: 32 selected features" if chosen.startswith("top_32")
                          else "Held-out configuration 6: all features")
        fig.colorbar(heat, ax=axes[1], fraction=0.045, pad=0.04, label="Within-class share")

        top = np.argsort(importance)[-10:]
        names = feature_names()
        axes[2].barh([names[index] for index in top], importance[top], color="#607D4C")
        axes[2].set_xlabel("Training-only feature importance")
        axes[2].set_title("Most informative pressure features")
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=160)
        plt.close(fig)
        print(f"Inspected {len(y):,} cycles from configurations {CONFIGURATIONS}.")
        print(f"Selected {chosen}; held-out macro F1 {result['test_selected']['macro_f1']:.3f}; accuracy {result['test_selected']['accuracy']:.3f}.")
        print(f"Wrote {RESULTS} and {FIGURE}")

    if run:
        main()
    return locals()


COMMANDS = {
    "pump-sensor-intake": pump_sensor_intake,
    "pump-sensor": pump_sensor,
    "pump-camera": pump_camera,
    "pump-compare": pump_compare,
    "pump-original-xlsx": pump_original_xlsx,
    "belt-intake": belt_intake,
    "belt-manifest": belt_manifest,
    "belt-preprocess": belt_preprocess,
    "belt-conveyor": belt_conveyor,
    "haul-truck-intake": haul_truck_intake,
    "haul-truck": haul_truck,
    "uci-hydraulic": uci_hydraulic,
    "espset": espset,
    "rock-drill": rock_drill,
}


def cli() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    command = sys.argv[1]
    sys.argv = [f"capstone_code.py {command}", *sys.argv[2:]]
    COMMANDS[command]()


if __name__ == "__main__":
    cli()
