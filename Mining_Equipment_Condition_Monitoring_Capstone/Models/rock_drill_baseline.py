"""Reproducible 11-condition rock-drill pressure-signal baseline.

The official PHM 2022 training archive contains test-cell rock-drill cycles,
not underground fleet recordings. Its three sensor files are paired by row
within each configuration. This experiment never joins them to photographs.
"""

from __future__ import annotations

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


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "Raw_Data/phm2022_rock_drill/training_data.zip"
PROCESSED = PROJECT / "Processed_Data/phm2022_rock_drill_features.npz"
RESULTS = PROJECT / "Results/phm2022_rock_drill.json"
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


if __name__ == "__main__":
    main()
