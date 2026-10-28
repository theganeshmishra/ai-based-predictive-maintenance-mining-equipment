"""Dozer (machine 5): track final-drive gear wear, CORA vibration/temperature vs infrared camera.

All of this machine's code in one file. The combined notebook
Notebook/Mining_Equipment_Condition_Monitoring.ipynb runs these commands and shows the results.

IMPORTANT: there is no public dozer dataset with matching sensor and camera data. As planned in the
dataset manifest, the Dozer final drive uses the SAME 16 CORA gear-wear runs as the Belt Conveyor
(not new data), with the same 4 classes: healthy, wear25, wear50, wear75. Its scores are
therefore expected to match the Belt Conveyor's; it maps that rig to the dozer track final drive.
The data loading is shared with Models/belt_conveyor.py.

Usage:
    python Models/dozer.py <command> [options]
    python Models/dozer.py <command> --help

Commands:
    csv             4-class sensor model from a CSV feature table (--split time for the secondary test)
    export-images   save the infrared images as named files in Images/dozer/<class>/ (local only)
    cnn             small CNN with data augmentation on the named images (run export-images first)
    cnn-pretrained  pretrained ResNet18 on the named images (--split time for the secondary test)
    ensembles       XGBoost / LightGBM vs random forest etc. (--split time for the secondary test)
    anomaly         anomaly detection trained on healthy data only (--split time for the secondary test)
    train-final     train the sensor model on all data and save it (--exclude to leave recordings out)
    predict         predict from any CSV (feature table or raw recording) and show accuracy if labels are known
    fulldata        Week 7: every CORA sensor, window and start-up recording (Models/cora_fulldata.py)
                    (--part ensembles / anomaly / all: XGBoost-LightGBM comparison, anomaly detection)
    all             run csv, ensembles and anomaly

Raw data stays in Raw_Data/ and Processed_Data/ (not in Git).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The Dozer reuses the Belt Conveyor's CORA data loading (same rig, same runs).
_spec = importlib.util.spec_from_file_location("belt_conveyor", ROOT / "Models" / "belt_conveyor.py")
BELT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BELT)


def export_images(run: bool = True) -> dict:
    """Week 6: save this machine's camera images as named PNG files for the CNN.

    Files go to Images/dozer/<class>/ and are named <machine>_<class>_<recording>_<frame>.png,
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
    from scipy.io import loadmat

    MACHINE = "dozer"
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
        labels = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
        for condition, label in labels.items():
            for speed in ("F5", "F15", "F50", "F60"):
                unit = f"{condition}_{speed}_S"
                frames = loadmat(ROOT / "Raw_Data" / "cora_gear_wear" / f"{unit}.mat",
                                 variable_names=["imagenes_celda"])["imagenes_celda"]
                for k, frame in enumerate(frames.flat):
                    if k % 3:
                        continue  # every 3rd frame: 150 images per run
                    gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                    write(rows, label, unit, speed, f"{MACHINE}_{label}_{speed}_f{k:03d}.png", gray)
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
    from scipy.io import loadmat
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    MACHINE = "dozer"
    CSV_PATH = ROOT / "Processed_Data" / f"{MACHINE}_sensor_features_4class.csv"
    RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_csv_4class.json"
    FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_csv_4class.png"
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


    VIB_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
    VIB_NAMES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                 *(f"order_power_{lo}_{hi}" for lo, hi in VIB_ORDERS)]

    def vibration_rows(run_name, label, speed):
        """CORA Vx vibration (3 kHz) in 1 s windows (every 10th window), measured in shaft orders.

        Orders = frequency / speed setting, so 1x, 2x and higher multiples line up across speeds:
        unbalance shows at 1x, misalignment at 2x, bearing and gear damage at higher orders and spikes.
        """
        cache = ROOT / "Processed_Data" / "cora_vibration" / f"{run_name}_Vx_features.csv"
        if cache.is_file():
            return pd.read_csv(cache)
        source = ROOT / "Raw_Data" / "cora_vibration" / f"{run_name}_Vx.csv"
        if not source.is_file():
            raise SystemExit(f"Missing {source.relative_to(ROOT)}: download it and run the intake command")
        values = pd.read_csv(source, header=None, dtype=np.float64).iloc[:, 0].to_numpy()
        frame = vibration_frame(values, speed)
        frame.insert(0, "window", np.arange(len(frame)))
        frame.insert(0, "class", label)
        frame.insert(0, "speed", speed)
        frame.insert(0, "unit", run_name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
        return frame

    def vibration_frame(values, speed):
        """Order-domain vibration features for a raw 3 kHz Vx signal at speed setting F5/F15/F50/F60."""
        rate, shaft = 3000, float(speed[1:])
        n = len(values) // rate
        windows = values[: n * rate].reshape(n, rate)[::10]
        centered = windows - windows.mean(axis=1, keepdims=True)
        std = np.maximum(centered.std(axis=1), 1e-12)
        freq = np.fft.rfftfreq(rate, d=1 / rate)
        power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
        order = freq / shaft
        usable = (order >= 0.5) & (order < 20.5)
        share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
        o = order[usable]
        bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in VIB_ORDERS]
        frame = pd.DataFrame(np.column_stack([
            np.log(std), (centered ** 4).mean(axis=1) / std ** 4, (centered ** 3).mean(axis=1) / std ** 3,
            np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
            np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]), columns=VIB_NAMES)
        return frame

    def per_speed_normalised(table, names):
        """Rescale each feature within each speed using all recordings at that speed together (no labels used).

        Removes what each speed 'sounds like', so the model compares machines running at the same speed.
        It assumes several machines are recorded at each speed, as in this dataset.
        """
        out = []
        for name in names:
            column = f"{name}_per_speed"
            grouped = table.groupby("speed")[name]
            table[column] = (table[name] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
            out.append(column)
        return out

    def build_table():
        """Same CORA runs, features and 4 wear classes as the Belt Conveyor (no public dozer dataset exists)."""
        table, feature_sets, labels, folds, _, _ = BELT.csv(run=False)["build_table"]()
        return (table, feature_sets, labels, folds,
                "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
                ["CORA motor/gearbox rig, the SAME 16 runs, features and 4 wear classes as the Belt Conveyor (not new "
                 "data), so the scores are expected to equal the Belt Conveyor's; mapped to the Dozer track final drive"])
    INNER_GROUP = "speed"

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

    MACHINE = "dozer"
    IMAGES = ROOT / "Images" / MACHINE
    RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_cnn_4class.json"
    FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_cnn_4class.png"
    LABELS = ["healthy", "wear25", "wear50", "wear75"]
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
            channels = [0]  # infrared images are grayscale
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
        for number, test in enumerate([groups == s for s in ("F5", "F15", "F50", "F60")], 1):
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
            "images": int(len(rows)), "classes": LABELS, "test": "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions",
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

    MACHINE = "dozer"
    IMAGES = ROOT / "Images" / MACHINE
    RESULT = ROOT / "Results" / "dozer" / f"{MACHINE}_cnn_pretrained_4class.json"
    FIGURE = ROOT / "Figures" / "dozer" / f"{MACHINE}_cnn_pretrained_4class.png"
    LABELS = ["healthy", "wear25", "wear50", "wear75"]
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
        parser.add_argument("--split", choices=("main", "time"), default="main",
                            help="main (fair test) or time (secondary, easier: first 60%% of each recording's frames "
                                 "train, last 30%% test, 10%% gap)")
        args = parser.parse_args()
        nonlocal RESULT, FIGURE
        rows = read_index()
        test_description = "four rounds, each holding out one whole speed (4 runs, one per wear level); 16 run-level predictions"
        if args.split == "time":
            import re
            position = {row["file"]: int(re.search(r"_f(\d{3})\.png$", row["file"]).group(1)) / 450 for row in rows}
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
                 else [groups == s for s in ("F5", "F15", "F50", "F60")])
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



def train_final(run: bool = True) -> dict:
    """Train the sensor model on all of this machine's data and save it (Processed_Data/models/, not in Git).

    Use --exclude to leave recordings out of training (for example to test on them fairly with `predict`).
    Model: random forest on the "vibration" features (fixed settings).
    """
    import argparse
    import json

    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    MODEL_PATH = ROOT / "Processed_Data" / "models" / "dozer_sensor_model.joblib"

    def main():
        parser = argparse.ArgumentParser(description=train_final.__doc__)
        parser.add_argument("--exclude", nargs="*", default=[], help="recording names to leave out, e.g. W50_F15_S")
        args = parser.parse_args()
        table, feature_sets, labels, *_ = csv(run=False)["build_table"]()
        feature_sets = feature_sets if isinstance(feature_sets, dict) else {"base": feature_sets}
        features = feature_sets["vibration"]
        unknown = set(args.exclude) - set(table["unit"])
        if unknown:
            raise SystemExit(f"Unknown recording names: {sorted(unknown)}")
        train = ~table["unit"].isin(args.exclude)
        model = RandomForestClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
                                       min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(table.loc[train, features].to_numpy(dtype=np.float64), table.loc[train, "class"])
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": features, "classes": list(labels), "machine": "dozer",
                     "trained_units": sorted(set(table.loc[train, "unit"])), "excluded_units": sorted(args.exclude)},
                    MODEL_PATH)
        print(f"Trained on {int(train.sum())} rows from {table.loc[train, 'unit'].nunique()} recordings "
              f"(excluded: {args.exclude or 'none'}); saved {MODEL_PATH.relative_to(ROOT)}")

    if run:
        main()
    return locals()



def predict(run: bool = True) -> dict:
    """Predict the condition from any CSV and, if the true class is known, show the accuracy.

    The CSV can be (1) a feature table with the same columns as Processed_Data/dozer_sensor_features_4class.csv
    (optional 'unit' and 'class' columns), or (2) a raw sensor recording: one column of CORA Vx vibration values sampled at 3 kHz (give --speed).
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

    MODEL_PATH = ROOT / "Processed_Data" / "models" / "dozer_sensor_model.joblib"

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
        if not args.speed:
            raise SystemExit("Raw recordings need --speed (F5, F15, F50 or F60)")
        ns = csv(run=False)
        return ns["vibration_frame"](values, args.speed)[ns["VIB_NAMES"]].to_numpy(dtype=np.float64)


    def main():
        parser = argparse.ArgumentParser(description=predict.__doc__)
        parser.add_argument("--csv", required=True, help="path to the CSV file")
        parser.add_argument("--label", help="true class of a raw recording, one of: healthy, wear25, wear50, wear75")
        parser.add_argument("--column", help="column holding the raw signal (default: first numeric column)")
        parser.add_argument("--speed", choices=("F5", "F15", "F50", "F60"), help="speed setting of a raw recording")
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

    MACHINE = "dozer"
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

    MACHINE = "dozer"
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


# ====================================================================================================
# Week 7: full-data sensor experiment (every CORA sensor, window and recording)
# ====================================================================================================

def fulldata(run: bool = True) -> dict:
    """Week 7: every CORA sensor (vibration X/Y/Z, current C1-C3, temperatures T1-T6), every 1 s window, and the
    five start-up test repeats per condition and speed. See Models/cora_fulldata.py. Saves *_fulldata.json."""
    import argparse
    import importlib.util

    def main():
        parser = argparse.ArgumentParser(description=fulldata.__doc__)
        parser.add_argument("--part", choices=("classify", "ensembles", "anomaly", "all"), default="classify",
                            help="classify (default), ensembles (XGBoost/LightGBM comparison), anomaly, or all")
        part = parser.parse_args().part
        spec = importlib.util.spec_from_file_location("cora_fulldata", ROOT / "Models" / "cora_fulldata.py")
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        conditions = {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}
        for name in ("classify", "ensembles", "anomaly"):
            if part in (name, "all"):
                (engine.run if name == "classify" else getattr(engine, name))("dozer", conditions)

    if run:
        main()
    return locals()


COMMANDS = {
    "fulldata": fulldata,
    "csv": csv,
    "export-images": export_images,
    "cnn": cnn,
    "cnn-pretrained": cnn_pretrained,
    "ensembles": ensembles,
    "anomaly": anomaly,
    "train-final": train_final,
    "predict": predict,
}
RUN_ALL = ["csv", "ensembles", "anomaly"]


def cli() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    command, options = sys.argv[1], sys.argv[2:]
    for name in (RUN_ALL if command == "all" else [command]):
        sys.argv = [f"dozer.py {name}", *(options if command != "all" else [])]
        COMMANDS[name]()


if __name__ == "__main__":
    cli()
