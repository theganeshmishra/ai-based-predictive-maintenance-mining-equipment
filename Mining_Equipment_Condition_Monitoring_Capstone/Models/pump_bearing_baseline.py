"""Exploratory pump-bearing vibration baseline for the locked P1 comparison.

Four official spreadsheets contain one continuous recording per condition. A
within-recording split is useful for a code/feature smoke test, but its score
is *not* independent-run, independent-pump, or image-versus-sensor accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


CAPSTONE = Path(__file__).resolve().parents[1]
RAW = CAPSTONE / "Raw_Data" / "pump_bearing"
OUTPUT = CAPSTONE / "Results" / "pump_bearing_exploratory.json"
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


if __name__ == "__main__":
    main()
