"""Run-level CORA gear-wear sensor/infrared comparison for Equipment #2.

The same 16 stationary motor/gearbox experiments feed separate T1 temperature
and infrared-image classifiers. Leave-one-speed-out folds hold out all four
condition runs at each speed. Windows/frames are never split across folds.
This is a laboratory geared-drive analogue, not mining-conveyor validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CAPSTONE = Path(__file__).resolve().parents[1]
RAW = CAPSTONE / "Raw_Data" / "cora_gear_wear"
PROCESSED = CAPSTONE / "Processed_Data" / "cora_gear_wear"
INTAKE = CAPSTONE / "Results" / "belt_conveyor_cora_intake.json"
RESULT = CAPSTONE / "Results" / "belt_conveyor_cora.json"
REPORT = CAPSTONE / "Results" / "belt_conveyor_sensor_visual_comparison.md"
FIGURE = CAPSTONE / "Figures" / "belt_conveyor_cora_confusions.png"
CLASSES = ("healthy", "gear_worn")
CONDITIONS = ("H", "W25", "W50", "W75")
SPEEDS = ("F5", "F15", "F50", "F60")
RUNS = tuple(f"{condition}_{speed}_S" for condition in CONDITIONS for speed in SPEEDS)


def sensor_features(values: np.ndarray) -> list[float]:
    if values.shape != (8, 6_000) or not np.isfinite(values).all():
        raise ValueError(f"Unexpected bounded T1 sample shape: {values.shape}")
    means = values.mean(axis=1)
    within_std = values.std(axis=1)
    low = np.percentile(values, 10, axis=1)
    high = np.percentile(values, 90, axis=1)
    trend = float(np.polyfit(np.arange(8), means, 1)[0])
    return [
        float(means.mean()), float(means.std()), float(np.median(means)),
        float(means.min()), float(means.max()), trend,
        float(within_std.mean()), float(np.mean(high - low)),
    ]


def infrared_features(path: Path) -> list[float]:
    cells = loadmat(path, variable_names=["imagenes_celda"])["imagenes_celda"]
    if cells.size != 450:
        raise ValueError(f"Expected 450 infrared frames in {path.name}")
    rows = []
    for frame in cells.flat:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid infrared frame in {path.name}")
        gray = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_RGB2GRAY)
        center = gray[32:96, 32:96].astype(np.float32)
        full = gray.astype(np.float32)
        rows.append([
            float(full.mean()), float(full.std()),
            float(np.percentile(full, 90)), float(np.percentile(full, 99)),
            float(np.mean(full >= 220)),
            float(center.mean() - full.mean()),
            float(center.std()),
            float(np.mean(np.abs(np.diff(full, axis=1)))),
        ])
    rows_array = np.asarray(rows)
    if not np.isfinite(rows_array).all():
        raise ValueError(f"Non-finite infrared descriptor in {path.name}")
    return np.concatenate([rows_array.mean(axis=0), rows_array.std(axis=0)]).tolist()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    correct = int(np.trace(matrix))
    interval = binomtest(correct, len(y_true)).proportion_ci(method="exact")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "correct_runs": correct,
        "test_runs": int(len(y_true)),
        "exploratory_accuracy_interval_95_percent": [float(interval.low), float(interval.high)],
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(CLASSES)
        },
    }


def evaluate(x: np.ndarray, y: np.ndarray, speeds: np.ndarray) -> dict:
    model_predictions = np.full(len(y), -1, dtype=int)
    dummy_predictions = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for speed in SPEEDS:
        test = speeds == speed
        train = ~test
        if train.sum() != 12 or test.sum() != 4 or sorted(np.bincount(y[test])) != [1, 3]:
            raise ValueError(f"Invalid whole-speed fold: {speed}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                               max_iter=1000, random_state=42),
        )
        model.fit(x[train], y[train])
        model_predictions[test] = model.predict(x[test])
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(x[train], y[train])
        dummy_predictions[test] = dummy.predict(x[test])
        fold_rows.append({
            "held_out_speed": speed,
            "train_runs": int(train.sum()),
            "test_runs": int(test.sum()),
            "actual": y[test].tolist(),
            "predicted": model_predictions[test].tolist(),
            "accuracy": float(accuracy_score(y[test], model_predictions[test])),
        })
    if np.any(model_predictions < 0) or np.any(dummy_predictions < 0):
        raise ValueError("Some runs lack an out-of-fold prediction")
    return {"model": metrics(y, model_predictions),
            "most_frequent_control": metrics(y, dummy_predictions),
            "folds": fold_rows,
            "predictions": {run: int(prediction)
                            for run, prediction in zip(RUNS, model_predictions)}}


def draw_figures(sensor: dict, visual: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for axis, record, title in zip(
        axes, (sensor, visual), ("T1 sensor (held-out speed)", "Infrared images (held-out speed)")
    ):
        matrix = np.asarray(record["model"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks([0, 1], CLASSES, rotation=20)
        axis.set_yticks([0, 1], CLASSES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(title)
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.suptitle("CORA gear-wear analogue: 16 whole-run out-of-fold predictions")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.78, wspace=0.45)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


def metric_row(label: str, record: dict) -> str:
    return (f"| {label} | {record['accuracy']:.3f} | {record['balanced_accuracy']:.3f} | "
            f"{record['macro_precision']:.3f} | {record['macro_recall']:.3f} | "
            f"{record['macro_f1']:.3f} |")


def render_report(result: dict) -> None:
    sensor = result["sensor"]
    visual = result["infrared"]
    lines = [
        "# Equipment #2 — Belt Conveyor geared-drive condition comparison",
        "",
        "## Scope and target",
        "",
        "The source is one controlled CORA motor/gearbox **laboratory rig**, not a mining conveyor. A geared conveyor drive reducer is mechanically comparable at the component level only; this result says nothing about belt cracks, idlers, or gearless drives. The binary target is **healthy (`H`) versus imposed gear-tooth wear (`W25`, `W50`, `W75`)**. It classifies present condition, not a future maintenance date.",
        "",
        "CORA supplies T1 temperature readings and infrared frames from the **same 16 stationary experiment runs**. We train separate models and compare their predictions at the run/fault-label level. We do **not** claim one-to-one frame/sensor-window synchronization, sample-level fusion, or two independent physical machines.",
        "",
        "## Leakage-safe evaluation",
        "",
        "Each of four folds holds out one complete speed setting (F5, F15, F50 or F60), containing one healthy and three worn runs. Training uses the other 12 whole runs. The run—not a sensor range or infrared frame—is the test unit. Both modalities use the **same four held-out-speed folds and same 16 test runs**. A fixed class-balanced logistic regression is fitted separately inside each fold, with feature scaling learned only on training runs; no test-selected features or hyperparameters. The majority-class control predicts worn in every fold.",
        "",
        "| Model / control | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
        metric_row("T1 sensor", sensor["model"]),
        metric_row("Infrared / OpenCV", visual["model"]),
        metric_row("Most-frequent control", sensor["most_frequent_control"]),
        "",
        "The scores above use **16 distinct experiment-level out-of-fold predictions per modality**, not thousands of independent frames/windows. Runs reuse one rig and may share a worn gear across speeds, so they are not 16 independent physical machines. The exploratory exact 95% intervals for accuracy are wide: sensor "
        + f"{sensor['model']['exploratory_accuracy_interval_95_percent'][0]:.3f}–{sensor['model']['exploratory_accuracy_interval_95_percent'][1]:.3f}; infrared "
        + f"{visual['model']['exploratory_accuracy_interval_95_percent'][0]:.3f}–{visual['model']['exploratory_accuracy_interval_95_percent'][1]:.3f}. These binomial intervals assume independent trials, an approximation here, and must not be interpreted as mine-fleet confidence bounds.",
        "",
        "Both learnt models are **below the 0.750 majority-class accuracy control**. The sensor model's macro F1 is above that control, while the infrared model's macro F1 is below it. Neither result supports reliable deployment or a claim that one modality is inherently superior.",
        "",
        "| Held-out speed | Sensor accuracy | Infrared accuracy | Test runs |",
        "|---|---:|---:|---:|",
    ]
    for sensor_fold, visual_fold in zip(sensor["folds"], visual["folds"]):
        if sensor_fold["held_out_speed"] != visual_fold["held_out_speed"]:
            raise ValueError("Modality fold mismatch")
        lines.append(f"| {sensor_fold['held_out_speed']} | {sensor_fold['accuracy']:.3f} | {visual_fold['accuracy']:.3f} | {sensor_fold['test_runs']} |")
    lines.extend([
        "",
        "## Per-class and per-condition performance",
        "",
        "| Modality | Actual class | Precision | Recall | F1 | Runs |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name, record in (("T1 sensor", sensor["model"]), ("Infrared", visual["model"])):
        for label in CLASSES:
            item = record["per_class"][label]
            lines.append(f"| {name} | {label} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |")
    lines.extend(["", "| Original CORA state | Runs | Sensor correctly classified | Infrared correctly classified |",
                  "|---|---:|---:|---:|"])
    for state in CONDITIONS:
        selected = [run for run in RUNS if run.startswith(state + "_")]
        actual = 0 if state == "H" else 1
        sensor_correct = sum(sensor["predictions"][run] == actual for run in selected)
        visual_correct = sum(visual["predictions"][run] == actual for run in selected)
        lines.append(f"| {state} | 4 | {sensor_correct}/4 | {visual_correct}/4 |")
    lines.extend([
        "",
        "## Run-level confusion matrices",
        "",
        "Rows are actual and columns predicted, in the order **healthy, gear-worn**.",
        "",
        "| Modality / actual | Predicted healthy | Predicted gear-worn |",
        "|---|---:|---:|",
    ])
    for name, record in (("Sensor healthy", sensor["model"]),
                         ("Sensor worn", sensor["model"]),
                         ("Infrared healthy", visual["model"]),
                         ("Infrared worn", visual["model"])):
        row = 0 if name.endswith("healthy") else 1
        values = record["confusion_matrix"][row]
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines.extend([
        "",
        f"![Separate run-level confusion matrices](../Figures/{FIGURE.name})",
        "",
        "## Data and reproducibility",
        "",
        "- [CORA version 2.1](https://doi.org/10.34810/DATA2500), CC BY 4.0. The [intake record](belt_conveyor_cora_intake.json) lists exact publisher file IDs, local SHA-256 checksums, source sizes and selected sensor byte ranges. All 16 MAT stacks passed filename/structure checks; H_F5_S also matched the publisher's MD5. The other 15 local SHA-256 values establish future local integrity but were not compared with publisher checksums. All source MAT stacks and sampled values remain in Git-ignored `Raw_Data/` and `Processed_Data/`.",
        "- **Sensor branch:** T1 is a source temperature channel sampled at 1 kHz. For each run we request eight deterministic, equally spaced 256-KiB ranges from the original CSV and retain the first 6,000 complete readings of each range. This is about 4.3% of source bytes and is **not full-signal modelling**. Eight predeclared time/variation features summarize each run.",
        "- **Visual branch:** all 450 published infrared frames per run are converted with OpenCV to grayscale intensity/spatial descriptors and summarized to one run-level vector. Pixel levels are not calibrated degrees Celsius. Publisher MATLAB files are used rather than equipment-identification photographs.",
        "- **No artificial pairing:** the two modalities share an experiment ID and fault condition, but their individual samples are not joined. Both branches have the same held-out speed/experiment units, making the run-level comparison more directly aligned than the earlier Pump cross-rig comparison.",
        "",
        "## Limitations and professor-requirement decision",
        "",
        "Only four speed settings and **one run per wear state/speed** are available. All 16 runs come from one laboratory rig; there is no unseen gearbox, mine, conveyor, load regime, or dated repair outcome. Reusing this CORA evidence for excavator, dozer, loader, drill, or grader later will **not** create additional independent equipment validations. Scores can compare how these fixed pipelines behave on the same held-out experiments, but are too small to establish sensor/infrared superiority or deployment performance.",
        "",
        "**Yes for completing the requested experimental comparison, conditional on the professor accepting a geared-drive laboratory reducer as a mechanically similar conveyor component.** Both real modalities address the same present gear-wear target using whole-run held-out-speed tests, and all requested metrics are reported. **No for an effective or validated predictive-maintenance model:** both accuracies are below the simple majority-class control, infrared healthy-class recall is 0.250, and there is no mining-conveyor test, sample-level fusion, or future-repair outcome. These weak results must be shown to the professor rather than described as a successful detector.",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    intake = json.loads(INTAKE.read_text(encoding="utf-8"))
    if set(intake["runs"]) != set(RUNS):
        raise ValueError("Intake must contain all 16 CORA experiments")
    samples = np.load(PROCESSED / "t1_bounded_samples.npz")
    if set(samples.files) != set(RUNS):
        raise ValueError("Sensor cache must contain exactly 16 runs")
    sensor_rows = []
    visual_rows = []
    for index, run in enumerate(RUNS, 1):
        if set(intake["runs"][run]) != {"infrared", "temperature_sensor"}:
            raise ValueError(f"Incomplete modality provenance: {run}")
        sensor_rows.append(sensor_features(samples[run]))
        visual_rows.append(infrared_features(RAW / f"{run}.mat"))
        print(f"Extracted run {index}/16: {run}", flush=True)
    sensor_x = np.asarray(sensor_rows)
    visual_x = np.asarray(visual_rows)
    y = np.asarray([0 if run.startswith("H_") else 1 for run in RUNS])
    speeds = np.asarray([run.split("_")[1] for run in RUNS])
    sensor = evaluate(sensor_x, y, speeds)
    visual = evaluate(visual_x, y, speeds)
    result = {
        "status": "complete_component_analogue_held_out_speed_evaluation",
        "equipment": "Belt Conveyor with a geared drive reducer",
        "source_domain": "Single controlled CORA motor/gearbox laboratory rig, not a mining conveyor",
        "source": "https://doi.org/10.34810/DATA2500",
        "target": "healthy versus imposed gear-tooth wear (W25/W50/W75)",
        "class_order": list(CLASSES),
        "original_conditions": list(CONDITIONS),
        "runs": list(RUNS),
        "split": "Four leave-one-speed-out folds; one held-out full run per original state in each fold",
        "independent_physical_machines": 1,
        "sensor": sensor,
        "infrared": visual,
        "same_experiment_level_ids": True,
        "sample_level_alignment_verified": False,
        "future_maintenance_prediction": False,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw_figures(sensor, visual)
    render_report(result)
    print(f"Saved {RESULT}, {REPORT}, and {FIGURE}")


if __name__ == "__main__":
    main()
