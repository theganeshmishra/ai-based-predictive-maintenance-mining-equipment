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

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pump_event_visual_baseline import BIN_US, END_US, START_US, event_images, frame_features  # noqa: E402


CAPSTONE = Path(__file__).resolve().parents[1]
INTAKE = CAPSTONE / "Results" / "haul_truck_intake.json"
SENSOR_RAW = CAPSTONE / "Raw_Data" / "uods_vafdc"
ROTOR_ZIP = CAPSTONE / "Raw_Data" / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip"
ROTOR_URL = "https://zenodo.org/api/records/17447083/files/XJTU-DV-Rotor.zip/content"
SENSOR_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_sensor_features.npz"
RUN_CACHE = CAPSTONE / "Processed_Data" / "haul_truck_event_runs"
RESULT = CAPSTONE / "Results" / "haul_truck_wheel_motor_bearing.json"
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
        "Generated by `Models/haul_truck_wheel_motor_bearing.py`. Target: present bearing condition "
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
    global REFRESH_EVENTS
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


if __name__ == "__main__":
    main()
