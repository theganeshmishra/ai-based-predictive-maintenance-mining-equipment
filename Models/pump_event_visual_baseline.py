"""Pump event-camera fault classifier using independent XJTU-DV-Pump runs.

Read selected members of the public Zenodo ZIPs by HTTP range. Convert the
first two seconds of each event stream into 250 ms polarity-count images,
extract OpenCV HOG features, train on 20 Hz runs, and test on 30 Hz runs.
Each source recording is one evaluation unit; no sensor row is paired to it.
"""

from __future__ import annotations

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


CAPSTONE = Path(__file__).resolve().parents[1]
CACHE = CAPSTONE / "Processed_Data" / "pump_event_features.npz"
RUN_CACHE = CAPSTONE / "Processed_Data" / "pump_event_runs"
RESULT = CAPSTONE / "Results" / "pump_event_visual.json"
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


if __name__ == "__main__":
    main()
