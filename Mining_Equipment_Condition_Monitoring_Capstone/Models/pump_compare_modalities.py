"""Report aligned bearing-fault targets without pairing unrelated source rows.

Both branches have recording-held-out tests; different rigs and shift tests mean
their accuracy values are descriptive and cannot establish modality superiority.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


CAPSTONE = Path(__file__).resolve().parents[1]
SENSOR = CAPSTONE / "Results" / "pump_bearing_independent.json"
SENSOR_DIAGNOSTIC = CAPSTONE / "Results" / "pump_bearing_exploratory.json"
VISUAL = CAPSTONE / "Results" / "pump_event_visual.json"
REPORT = CAPSTONE / "Results" / "pump_sensor_visual_comparison.md"
FIGURE = CAPSTONE / "Figures" / "pump_sensor_visual_confusions.png"
CLASSES = ("healthy", "inner", "outer", "ball")


def fmt(value: float) -> str:
    return f"{value:.3f}"


def metric_line(name: str, unit: str, record: dict) -> str:
    return (f"| {name} | {unit} | {fmt(record['accuracy'])} | "
            f"{fmt(record['macro_precision'])} | {fmt(record['macro_recall'])} | "
            f"{fmt(record['macro_f1'])} |")


def per_class(lines: list[str], record: dict, unit: str) -> None:
    lines.extend([f"| Condition | Precision | Recall | F1 | {unit} |",
                  "|---|---:|---:|---:|---:|"])
    for label in CLASSES:
        row = record["per_class"][label]
        lines.append(f"| {label} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
                     f"{fmt(row['f1'])} | {row['support']} |")


def matrix_table(lines: list[str], title: str, record: dict) -> None:
    lines.extend([f"### {title}", "",
                  "| Actual \\ predicted | Healthy | Inner | Outer | Ball |",
                  "|---|---:|---:|---:|---:|"])
    for label, row in zip(CLASSES, record["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def draw_matrices(records: list[dict], titles: list[str]) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(19, 5.7))
    for axis, record, title in zip(axes, records, titles):
        array = np.asarray(record["confusion_matrix"])
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
    figure.suptitle("Separate recording-level evaluations — not a paired or controlled modality trial", fontsize=13)
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.20, top=0.84, wspace=0.40)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)


def main() -> None:
    sensor = json.loads(SENSOR.read_text(encoding="utf-8"))
    diagnostic = json.loads(SENSOR_DIAGNOSTIC.read_text(encoding="utf-8"))
    visual = json.loads(VISUAL.read_text(encoding="utf-8"))
    if tuple(sensor["class_order"]) != CLASSES or tuple(visual["class_order"]) != CLASSES:
        raise ValueError("Sensor and visual results must use the same four fault classes")
    if sensor["status"] != "independent_recording_cross_speed_profile_evaluation":
        raise ValueError("Sensor result lacks a full-recording holdout")
    if visual["status"] != "cross_speed_independent_recording_evaluation":
        raise ValueError("Visual result lacks a full-recording holdout")
    if diagnostic["independent_run_accuracy"] is not None:
        raise ValueError("Original Pump sensor score must remain a same-file diagnostic")
    sm = sensor["random_forest_run_level"]
    vm = visual["random_forest_run_level"]
    sp = sensor["held_out_speed_profile_run_level"]
    vc = visual["held_out_camera_configuration_run_level"]
    for name, record, expected in (("sensor primary", sm, 24), ("visual primary", vm, 24),
                                   ("sensor profile folds", sp, 48), ("visual camera folds", vc, 48)):
        if sum(map(sum, record["confusion_matrix"])) != expected:
            raise ValueError(f"Incorrect recording-level confusion support: {name}")
    sensor_correct = sum(sm["confusion_matrix"][i][i] for i in range(4))
    visual_correct = sum(vm["confusion_matrix"][i][i] for i in range(4))
    sensor_ci = binomtest(sensor_correct, 24).proportion_ci(method="exact")
    visual_ci = binomtest(visual_correct, 24).proportion_ci(method="exact")

    lines = [
        "# Pump support-bearing sensor–visual fault comparison",
        "",
        "## What is—and is not—being compared",
        "",
        "The models predict the same four **present-condition** labels: healthy, inner-race, outer-race, and rolling-element/ball fault. The sensor input is real vibration from a University of Ottawa **bearing test rig**, used as a mechanically comparable *pump support-bearing proxy*. The visual input is real event-camera motion from the XJTU **pump-bearing housing**. The rigs, physical bearings, samples and capture processes differ. No sensor recording is matched to, joined with, or fused with an image recording.",
        "",
        "Both primary tests now hold out **entire source recordings**, with 24 train and 24 test recordings and six test recordings per class. That closes the earlier same-file sensor-window leakage gap. It does **not** make this a controlled modality-superiority trial: the sensor split transfers across speed *patterns* (A/B to C/D), while the visual split transfers across fixed speeds (20 to 30 Hz). Scores are descriptive evidence for each branch; **do not subtract them, rank modalities, or call the difference a fusion gain**.",
        "",
        "| Branch | Source reality | Training/test recordings | Test unit | Physical-machine holdout? |",
        "|---|---|---:|---|---|",
        "| Vibration | University of Ottawa laboratory rolling-bearing rig, **not a pump** | 24 / 24 | One 10 s recording | No; separate runs, shared rig and possibly same fault specimen |",
        "| Event camera / OpenCV | XJTU laboratory pump support-bearing housing, **not a mining pump** | 24 / 24 | One event recording | No; separate runs, shared rig |",
        "",
        "## Sensor branch: recording-held-out test",
        "",
        "| Evaluation | Test unit | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---:|---:|---:|---:|",
        metric_line("A/B speed profiles train → C/D test", "24 distinct recordings", sm),
        metric_line("Four-fold held-out speed profile", "48 fold-held-out run predictions", sp),
        metric_line("Most-frequent-class control", "Same 24 primary test runs", sensor["dummy_run_level"]),
        metric_line("Shuffled training-run labels", "Same 24 primary test runs", sensor["shuffled_training_run_labels_control"]),
        "",
        f"Primary test: **{sensor_correct}/24** correctly classified; an exploratory exact 95% binomial interval is **{sensor_ci.low:.3f}–{sensor_ci.high:.3f}**. It treats run outcomes as independent, which may be optimistic on one rig/possibly shared fault specimens, and does not include uncertainty from changing pumps or sites.",
        "",
        "### Sensor per-class results on held-out C/D recordings",
        "",
    ]
    per_class(lines, sm, "Test runs")
    lines.extend([
        "",
        "## Visual branch: recording-held-out test",
        "",
        "| Evaluation | Test unit | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---:|---:|---:|---:|",
        metric_line("20 Hz train → 30 Hz test", "24 distinct recordings", vm),
        metric_line("Six-fold held-out camera configuration", "48 fold-held-out run predictions", vc),
        metric_line("Most-frequent-class control", "Same 24 primary test runs", visual["dummy_run_level"]),
        metric_line("Shuffled training-run labels", "Same 24 primary test runs", visual["shuffled_training_run_labels_control"]),
        "",
        f"Primary test: **{visual_correct}/24** correctly classified; an exploratory exact 95% binomial interval is **{visual_ci.low:.3f}–{visual_ci.high:.3f}**. It treats run outcomes as independent despite one shared rig; the harder camera-configuration check is not a second independent dataset.",
        "",
        "### Visual per-class results on held-out 30 Hz recordings",
        "",
    ])
    per_class(lines, vm, "Test runs")
    lines.extend(["", "### Visual per-class sensitivity to held-out camera configurations", ""])
    per_class(lines, vc, "Fold-held-out runs")
    lines.extend([
        "",
        "## Confusion matrices",
        "",
        "Rows are actual classes and columns predicted classes, ordered healthy / inner / outer / ball. The secondary fold matrices reuse the same 48 source recordings across different training fits; they are sensitivity checks, not extra independent tests.",
        "",
    ])
    matrix_table(lines, "Sensor: held-out C/D speed-profile recordings", sm)
    matrix_table(lines, "Visual: held-out 30 Hz event recordings", vm)
    matrix_table(lines, "Sensor: held-out speed-profile folds", sp)
    matrix_table(lines, "Visual: held-out camera-configuration folds", vc)
    lines.extend([
        "## Method and provenance",
        "",
        "- **Sensor source:** [Huang and Baddour, University of Ottawa bearing vibration V2](https://data.mendeley.com/datasets/v43hmbwxpm/2), DOI 10.17632/v43hmbwxpm.2, CC BY 4.0. Four single-condition classes, four speed profiles and three trials per profile yield 48 source MAT files. The [verified intake record](pump_uottawa_intake.json) contains publisher SHA-256 values for every file. The compound-fault fifth class is excluded.",
        "- **Sensor preprocessing:** only accelerometer `Channel_1` at 200 kHz is used; encoder `Channel_2` and file names are not predictors. Each 10 s run becomes 100 non-overlapping 0.1 s windows, with 16 time/spectral features. A fixed random forest is fit only on A/B runs. Window probabilities are averaged before **one prediction per held-out C/D recording**; no window from a test recording appears in training. Hyperparameters were not selected on the test runs.",
        "- **Visual source and unchanged experiment:** [XJTU-DV-Pump](https://zenodo.org/records/17447083), CC BY 4.0. The first two seconds of each of 48 relevant released recordings become eight 250 ms polarity-count event images. OpenCV HOG feeds a fixed random forest trained on 20 Hz runs; average frame probabilities produce one prediction per separate 30 Hz recording. These are event-camera motion images, **not RGB photographs of visible surface damage**. Full-file CRCs were not checked because only bounded intervals were read remotely.",
        "- **Original pump-vibration source:** [Jiang centrifugal-pump multi-fault V1](https://data.mendeley.com/datasets/63gjxpn7s6/1) was retained as a pump-domain pipeline diagnostic, but it has one file per selected class. Its within-file window score is **excluded from the independent comparison**. The Ottawa rig supplies repeat-run evidence at the same bearing-fault taxonomy, not direct pump-domain validation.",
        "- **Limits:** none of these evaluations holds out a new physical pump or mine. The Ottawa data do not establish independent fault-bearing specimens across all trials. Different source rigs and operating shifts preclude a causal claim that one modality is better. No matching timestamps, sample-level pairing, multimodal fusion, maintenance dates, failure horizon, or remaining-useful-life target exist.",
        "",
        "## Professor-requirement decision",
        "",
        "**Yes, for the professor's explicitly allowed component/fault-level comparison using sufficiently similar machines:** real vibration and real event-camera images both classify healthy/inner/outer/ball bearing conditions, and both now have recording-held-out Accuracy, Precision, Recall, F1, per-class results and confusion matrices. **No**, if the requirement means paired sensor-plus-camera observations from the *same physical mining pump*, a controlled modality-superiority experiment, image–sensor fusion, or prediction of a future repair date. The sensor proxy is a bearing rig, so this is a defensible component-level Pump proof of concept, **not validated mining-pump deployment**. Confirm that scope with the professor before describing the project as fully predictive maintenance.",
        "",
        f"![Separate recording-level confusion matrices](../Figures/{FIGURE.name})",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    draw_matrices([sm, vm, sp, vc],
                  ["Sensor: held-out C/D runs", "Event camera: held-out 30 Hz",
                   "Sensor: held-out profiles", "Event camera: held-out configs"])
    print(f"Saved {REPORT} and {FIGURE}")


if __name__ == "__main__":
    main()
