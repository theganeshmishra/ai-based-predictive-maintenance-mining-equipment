"""Reproduce a pump-leakage baseline on the UCI hydraulic test rig.

This is a subsystem benchmark, not data recorded on a mining vehicle. The
original ZIP stays under ignored Raw_Data. Features and results are derived
locally; no raw observations are committed to Git.

Run from the repository root:
    .venv/bin/python Mining_Equipment_Condition_Monitoring_Capstone/Preprocessing/hydraulic_baseline.py
"""

from __future__ import annotations

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


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "Raw_Data/uci_hydraulic/condition_monitoring_of_hydraulic_systems.zip"
FEATURES = PROJECT / "Processed_Data/uci_hydraulic_cycle_features.npz"
RESULTS = PROJECT / "Results/uci_hydraulic_pump_leakage.json"
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


if __name__ == "__main__":
    main()
