"""Recording-held-out vibration benchmark for the Pump bearing-fault taxonomy.

Source recordings are from a University of Ottawa bearing rig, not a pump.
The professor allows comparable components on different physical machines.
This model uses whole-run evaluation and never joins its rows to Pump images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


CAPSTONE = Path(__file__).resolve().parents[1]
INTAKE = CAPSTONE / "Results" / "pump_uottawa_intake.json"
RAW = CAPSTONE / "Raw_Data" / "uottawa_bearing"
CACHE = CAPSTONE / "Processed_Data" / "pump_bearing_independent_features.npz"
RESULT = CAPSTONE / "Results" / "pump_bearing_independent.json"
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


if __name__ == "__main__":
    main()
