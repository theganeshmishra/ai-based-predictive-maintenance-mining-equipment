"""Group-held-out fault baseline on ESPset V3's published feature file.

ESPset records offshore oil-production pumps, not mining pumps. No images are
paired with these sensor rows. Run from the repository root with the project's
Python environment after placing the official features.zip under Raw_Data.
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
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "Raw_Data/espset_v3/features.zip"
PROCESSED = PROJECT / "Processed_Data/espset_v3_features.npz"
RESULTS = PROJECT / "Results/espset_v3_binary_fault.json"
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


if __name__ == "__main__":
    main()
