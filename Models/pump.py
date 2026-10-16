"""Pump: support bearing, vibration sensor vs XJTU event camera.

All of this machine's code in one file. The notebook Notebook/Pump.ipynb
runs these commands and shows the results.

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
    all             run sensor, then camera, then compare

Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    from scipy.io import loadmat
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
            with np.load(s["CACHE"], allow_pickle=False) as saved:
                x, y, profile, units = saved["x"], saved["y"], saved["profile"], saved["run"]
        else:
            x, y, profile, units, _ = s["preprocess"]()
        branches["Sensor: vibration"] = {
            "test": "train on speed profiles A/B; test on 24 separate C/D recordings",
            "classes": list(s["CLASSES"]),
            "models": evaluate(x, y, units, [np.isin(profile, ["C", "D"])], list(s["CLASSES"]))}
        c = camera(run=False)
        if c["CACHE"].is_file():
            with np.load(c["CACHE"], allow_pickle=False) as saved:
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
        folders = {"healthy": "Healthy", "inner": "Inner", "outer": "Outer", "ball": "Ball"}
        for hz, part in (("20Hz", "part1"), ("30Hz", "part2")):
            with zipfile.ZipFile(ROOT / "Raw_Data" / "xjtu_dv_pump" / f"XJTU-DV-Pump_{part}.zip") as archive:
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
                                write(rows, label, unit, hz, f"{MACHINE}_{label}_{hz}_{setup}_t{k:02d}.png", event_png(frame))
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
            with np.load(s["CACHE"], allow_pickle=False) as saved:
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
            image = cv2.imread(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
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
        parser.parse_args()
        rows = read_index()
        x = load(rows)
        label_index = {label: i for i, label in enumerate(LABELS)}
        y = np.asarray([label_index[row["class"]] for row in rows])
        names = np.asarray([row["class"] for row in rows])
        units = np.asarray([row["unit"] for row in rows])
        groups = np.asarray([row["group"] for row in rows])
        rng = np.random.default_rng(SEED)
        unit_actual, unit_predicted, image_actual, image_predicted, epochs_chosen = [], [], [], [], []
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


COMMANDS = {
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


if __name__ == "__main__":
    cli()
