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

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared data readers live in the notebook helper file.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Notebook.notebook_helpers import load_csv, load_excel, load_image, load_mat as loadmat, load_numpy, open_archive


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

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pump as _pump_file  # noqa: E402  (reuse the Pump event-camera code)
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
        images = []
        for row in rows:
            image = load_image(str(IMAGES / row["file"]), cv2.IMREAD_COLOR)
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


if __name__ == "__main__":
    cli()
