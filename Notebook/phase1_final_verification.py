"""Phase 1 final verification, as planned in Documentation/phase_1_final_verification_plan.md.

Run from the project root with the project interpreter:

    python Notebook/phase1_final_verification.py [stage ...]

Stages: reproduce (sensor fair tests, calibration, leakage, robustness), images, transient, phase1, diagnostics,
benchmark. With no stage, all run. Each stage is saved into Results/experiments/phase_1/final_verification.json
as soon as it finishes. Nothing here writes to Results/<machine>/, to the model folder or to the trust list.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

import pdm

OUT = pdm.RESULTS / "experiments" / "phase_1"
RESULT_FILE = OUT / "final_verification.json"
FIGURES = pdm.FIGURES / "experiments" / "phase_1"
SEED = 613
CASES = ("clean", "noise", "amplitude_half", "amplitude_double", "missing_5pct", "half_length", "rate_90pct")
ROBUSTNESS_MACHINES = ("haul_truck", "drill_rig", "grader")


def save_stage(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULT_FILE.read_text(encoding="utf-8")) if RESULT_FILE.is_file() else {}
    data[name] = value
    data["environment"] = {"platform": platform.platform(), "python": platform.python_version(),
                           "numpy": np.__version__, "pandas": pd.__version__,
                           "scikit-learn": __import__("sklearn").__version__, "seed": SEED}
    RESULT_FILE.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"saved stage '{name}'", flush=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


# --- statistics ------------------------------------------------------------------------------------

def wilson(correct, total, z=1.96):
    """Wilson 95% interval for a proportion."""
    if total == 0:
        return None
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def calibration(actual, probabilities, labels, bins=5):
    """Brier score, top-label ECE and confidence summaries for recording-level probabilities."""
    actual = np.asarray(actual)
    onehot = (actual[:, None] == np.asarray(labels)[None, :]).astype(float)
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    correct = predicted == actual
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if inside.any():
            gap = abs(correct[inside].mean() - confidence[inside].mean())
            ece += inside.mean() * gap
            table.append({"bin": [round(lo, 2), round(hi, 2)], "recordings": int(inside.sum()),
                          "mean_confidence": round(float(confidence[inside].mean()), 4),
                          "accuracy": round(float(correct[inside].mean()), 4)})
    return {
        "recordings": int(len(actual)),
        "brier": round(float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))), 4),
        "ece_top_label_5_bins": round(float(ece), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "mean_confidence_when_correct": round(float(confidence[correct].mean()), 4) if correct.any() else None,
        "mean_confidence_when_wrong": round(float(confidence[~correct].mean()), 4) if (~correct).any() else None,
        "wrong_with_confidence_over_0_9": int(((~correct) & (confidence > 0.9)).sum()),
        "reliability": table,
    }


# --- leakage audit of a sensor table ---------------------------------------------------------------

def audit_sensor_table(table, m, folds, feature_columns):
    units = table["unit"].to_numpy()
    per_unit = table.groupby("unit")[list(dict.fromkeys(["class", m.sensor_group, m.inner_group]))].nunique()
    held = [sorted(map(str, set(table.loc[test, m.sensor_group]))) for test in folds]
    declared = [sorted(map(str, g)) for g in m.sensor_held_out]
    tested_once = pd.Series(np.concatenate([np.unique(units[test]) for test in folds])).value_counts()
    rows = table[feature_columns].round(12)
    digest = pd.util.hash_pandas_object(rows, index=False).to_numpy()
    owners = pd.DataFrame({"digest": digest, "unit": units}).drop_duplicates()
    shared = owners.groupby("digest")["unit"].nunique()
    cross_split = 0
    for test in folds:
        test_digests = set(digest[test])
        cross_split += len(test_digests & set(digest[~test]))
    return {
        "recordings": int(table["unit"].nunique()),
        "one_class_per_recording": bool((per_unit["class"] == 1).all()),
        "one_holdout_group_per_recording": bool((per_unit[m.sensor_group] == 1).all()),
        "one_inner_group_per_recording": bool((per_unit[m.inner_group] == 1).all()),
        "held_out_groups_match_declared": held == declared,
        "no_recording_on_both_sides": all(not (set(units[t]) & set(units[~t])) for t in folds),
        "every_recording_tested_once": bool(len(folds) == 1 or (tested_once == 1).all()),
        "identical_feature_rows_in_more_than_one_recording": int((shared > 1).sum()),
        "identical_feature_rows_across_train_and_test": int(cross_split),
        "test_recordings_per_round": [int(len(set(units[t]))) for t in folds],
        "class_counts_per_round": [table.loc[t].groupby("class")["unit"].nunique().to_dict() for t in folds],
    }


# --- 1. reproduce the archived fair tests with probabilities ----------------------------------------

def fair_test_with_probabilities(machine, table, feature_sets):
    m = pdm.MACHINES[machine]
    y, units = table["class"].to_numpy(), table["unit"].to_numpy()
    folds = pdm.fair_test_folds(table[m.sensor_group], m.sensor_held_out)
    records, chosen, fitted = [], [], []
    for k, test in enumerate(folds):
        pdm.check_no_overlap(units, test)
        train_idx = np.flatnonzero(~test)
        assert not set(units[train_idx]) & set(units[test])  # selection below only sees training rows
        (set_name, model_name), score = pdm.choose_model(table, train_idx, feature_sets, m.labels, m.inner_group)
        names = feature_sets[set_name]
        x = table[names].to_numpy(dtype=np.float64)
        model = pdm.sensor_candidates()[model_name]().fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[test])
        a, p = pdm.predict_recordings(proba, model.classes_, y[test], units[test])
        test_units, test_y = units[test], y[test]
        mine = []
        for unit in sorted(set(test_units)):
            rows = test_units == unit
            mean = proba[rows].mean(axis=0)
            mine.append((test_y[rows][0], model.classes_[int(np.argmax(mean))]))
            records.append({"unit": unit, "round": k, "actual": test_y[rows][0], "predicted": mine[-1][1],
                            "probabilities": {c: round(float(v), 6) for c, v in zip(model.classes_, mean)}})
        assert mine == list(zip(a, p))  # identical to the archived scoring function
        chosen.append({"feature_set": set_name, "model": model_name, "validation_macro_f1": round(score, 4)})
        fitted.append((test, names, model))
    return folds, records, chosen, fitted


def reproduce(machine):
    m = pdm.MACHINES[machine]
    started = time.perf_counter()
    table, feature_sets = m.load_sensor()
    folds, records, chosen, fitted = fair_test_with_probabilities(machine, table, feature_sets)
    actual = [r["actual"] for r in records]
    predicted = [r["predicted"] for r in records]
    score = pdm.classification_metrics(actual, predicted, m.labels)
    archived = json.loads((pdm.RESULTS / machine / "sensor.json").read_text(encoding="utf-8"))
    archived_choices = [(c["feature_set"], c["model"]) for c in archived["chosen_per_round"]]
    probabilities = np.array([[r["probabilities"].get(c, 0.0) for c in m.labels] for r in records])
    correct = sum(a == p for a, p in zip(actual, predicted))
    all_features = sorted({c for names in feature_sets.values() for c in names})
    result = {
        "reproduced": {
            "confusion_matrix_identical": score["confusion_matrix"] == archived["score"]["confusion_matrix"],
            "choices_identical": [(c["feature_set"], c["model"]) for c in chosen] == archived_choices,
            "accuracy": score["accuracy"], "archived_accuracy": archived["score"]["accuracy"],
            "balanced_accuracy": score["balanced_accuracy"], "macro_f1": score["macro_f1"],
            "chosen_per_round": chosen,
        },
        "accuracy_wilson_95": wilson(correct, len(actual)),
        "calibration": calibration(actual, probabilities, m.labels),
        "leakage": audit_sensor_table(table, m, folds, all_features),
        "recordings": records,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return result, table, fitted


# --- 3. robustness on held-out recordings, archived choices refitted per round ----------------------

def perturb(x, kind, rng):
    """x: (samples, channels). One removal mask for all channels."""
    if kind == "clean":
        return x
    if kind == "noise":
        return x + rng.normal(0, 1, x.shape) * (0.1 * np.sqrt(np.mean(x**2, axis=0)))[None, :]
    if kind == "amplitude_half":
        return x * 0.5
    if kind == "amplitude_double":
        return x * 2
    if kind == "missing_5pct":
        return x[rng.random(len(x)) >= 0.05]
    if kind == "half_length":
        return x[: len(x) // 2]
    if kind == "rate_90pct":
        return resample_poly(x, 9, 10, axis=0)
    raise ValueError(kind)


def raw_recordings(machine):
    """Yield (unit, samples x channels array, feature function) for every recording of the machine."""
    if machine == "haul_truck":
        for item in json.loads((pdm.MANIFEST / "haul_truck_uods_files.json").read_text("utf-8")):
            name = item["name"]
            data = loadmat(pdm.RAW / "uods_vafdc" / name)[name[:-4]][:, :2].astype(np.float64)
            yield name[:-4], data, pdm.uods_array_features
    elif machine == "drill_rig":
        def features(x):
            frame = pd.DataFrame(x, columns=pdm.MCC5_CHANNELS)
            with patch.object(pdm.pd, "read_csv", lambda *args, **kwargs: frame):
                return pdm.mcc5_recording(None)
        with zipfile.ZipFile(pdm.MCC5_ZIP) as archive:
            for r in pdm.mcc5_recordings():
                with archive.open(r["file"]) as handle:
                    x = pd.read_csv(handle, usecols=pdm.MCC5_CHANNELS, dtype=np.float64)[pdm.MCC5_CHANNELS].to_numpy()
                yield r["unit"], x, features
    elif machine == "grader":
        def features(x):
            frame = pd.DataFrame(x.astype(np.float32), columns=pdm.JUST_CHANNELS)  # files are read as float32
            with patch.object(pdm.pd, "read_csv", lambda *args, **kwargs: frame):
                return pdm.just_recording(None)
        for r in pdm.just_recordings():
            with zipfile.ZipFile(pdm.JUST_DIR / f"{r['zip']}.zip") as archive, archive.open(r["file"]) as handle:
                x = pd.read_csv(handle, usecols=pdm.JUST_CHANNELS, dtype="float32")[pdm.JUST_CHANNELS]
            yield r["unit"], x.to_numpy(np.float64), features
    else:
        raise ValueError(machine)


def robustness(machine, table, fitted):
    m = pdm.MACHINES[machine]
    model_of = {}
    for test, names, model in fitted:
        for unit in set(table.loc[test, "unit"]):
            model_of[unit] = (names, model)
    label_of = table.groupby("unit")["class"].first().to_dict()
    rng = np.random.default_rng(SEED)
    predictions = {case: {} for case in CASES}
    rejected = {case: 0 for case in CASES}
    for unit, x, features in raw_recordings(machine):
        if unit not in model_of:
            continue
        names, model = model_of[unit]
        for case in CASES:
            try:
                values = features(perturb(x, case, rng))[names].to_numpy(dtype=np.float64)
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("no usable windows")
                predictions[case][unit] = model.classes_[int(model.predict_proba(values).mean(axis=0).argmax())]
            except ValueError:
                rejected[case] += 1
    clean = predictions["clean"]
    out = {}
    for case in CASES:
        units = sorted(predictions[case])
        score = pdm.classification_metrics([label_of[u] for u in units], [predictions[case][u] for u in units],
                                           m.labels)
        out[case] = {k: score[k] for k in ("test_units", "accuracy", "balanced_accuracy", "macro_f1",
                                           "confusion_matrix")}
        out[case]["rejected_recordings"] = rejected[case]
        out[case]["same_class_as_clean"] = round(float(np.mean([predictions[case][u] == clean.get(u)
                                                                for u in units])), 4) if units else None
    return out


def stage_reproduce(machines=None):
    summary = {}
    for machine in machines or pdm.MACHINES:
        print(f"reproducing {machine} ...", flush=True)
        result, table, fitted = reproduce(machine)
        if machine in ROBUSTNESS_MACHINES:
            print(f"  robustness {machine} ...", flush=True)
            result["robustness"] = robustness(machine, table, fitted)
            clean = result["robustness"]["clean"]["confusion_matrix"]
            result["robustness_clean_matches_fair_test"] = clean == pdm.classification_metrics(
                [r["actual"] for r in result["recordings"]], [r["predicted"] for r in result["recordings"]],
                pdm.MACHINES[machine].labels)["confusion_matrix"]
        summary[machine] = result
        save_stage(f"sensor_{machine}", result)
    return summary


# --- 2b. image leakage --------------------------------------------------------------------------------

def stage_images():
    out = {}
    for machine, m in pdm.MACHINES.items():
        folder = pdm.IMAGES / machine
        if not (folder / "index.csv").is_file() or m.image_held_out is None:
            out[machine] = {"status": "no local image index" if m.image_held_out else
                            "Grader tooth crops: split by source set inside the archive; not exported as an index"}
            continue
        rows = pdm.read_index(folder)
        digest = np.asarray([hashlib.sha256((folder / r["file"]).read_bytes()).hexdigest() for r in rows])
        groups = np.asarray([r["group"] for r in rows])
        units = np.asarray([r["unit"] for r in rows])
        classes = np.asarray([r["class"] for r in rows])
        folds = pdm.fair_test_folds(groups, m.image_held_out)
        frame = pd.DataFrame({"digest": digest, "class": classes, "unit": units})
        out[machine] = {
            "images": int(len(rows)), "units": int(len(set(units))),
            "unit_in_one_group": bool(pd.DataFrame({"u": units, "g": groups}).groupby("u")["g"].nunique().max() == 1),
            "identical_files": int(len(digest) - len(set(digest))),
            "identical_files_with_different_classes": int((frame.groupby("digest")["class"].nunique() > 1).sum()),
            "identical_files_across_train_and_test": [int(len(set(digest[t]) & set(digest[~t]))) for t in folds],
            "units_across_train_and_test": [int(len(set(units[t]) & set(units[~t]))) for t in folds],
        }
        print(f"images {machine}: {out[machine]}", flush=True)
    save_stage("image_leakage", out)
    return out


# --- 4. experiment E -----------------------------------------------------------------------------------

def stage_transient():
    codes = {"belt_conveyor": pdm.GEAR_WEAR_CODES,
             "excavator": {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"},
             "wheel_loader": {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}}
    out = {"plan": "Documentation/phase_1_final_verification_plan.md, section 4 (experiment E)",
           "test": "hold out one speed with all five transient repetitions (20 runs per round, 80 in all); "
                   "model chosen inside the training speeds; one prediction per run",
           "dozer": "uses the Belt Conveyor runs, so its result is the Belt Conveyor's"}
    for machine, classes in codes.items():
        m = pdm.MACHINES[machine]
        table = pdm.cora_transient_table(classes)
        folds = pdm.fair_test_folds(table["speed"], pdm.CORA_SPEEDS)
        scores = pdm.sensor_fair_test(table, {"vibration": pdm.CORA_FEATURES}, m.labels, folds, "speed")
        s = scores["score"]
        correct = int(np.trace(np.asarray(s["confusion_matrix"])))
        out[machine] = {"runs": int(table["unit"].nunique()), **scores,
                        "accuracy_wilson_95": wilson(correct, s["test_units"])}
        print(f"E {machine}: accuracy {s['accuracy']:.3f}, balanced {s['balanced_accuracy']:.3f}, macro F1 "
              f"{s['macro_f1']:.3f} on {s['test_units']} runs; majority {scores['majority_control']['accuracy']:.3f}",
              flush=True)
        pdm.save_confusion_figure([("unseen speed, transient runs", s), ("majority control", scores["majority_control"])],
                                  m.labels, f"{m.title}: CORA transient runs, unseen speed (experiment E)",
                                  FIGURES / f"{machine}_cora_transient_unseen_speed.png")
    save_stage("experiment_e_cora_transient_unseen_speed", out)
    return out


# --- 5. rerun the committed Phase 1 runner and diagnostics --------------------------------------------

def same_scores(a, b):
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix")
    return all(a.get(k) == b.get(k) for k in keys)


def stage_phase1():
    import phase1_validation as v

    committed = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    cora = v.cora_candidates()
    pump = v.pump_robustness()
    probes = v.deployment_robustness()
    out = {
        "cora_candidates_identical": {k: same_scores(cora[k]["score"], committed["cora_candidates"][k]["score"])
                                      and cora[k]["choices"] == committed["cora_candidates"][k]["choices"]
                                      for k in cora},
        "pump_robustness_identical": {k: same_scores(pump[k], committed["pump_robustness"][k]) for k in pump},
        "pump_robustness_here": {k: {x: pump[k][x] for x in ("accuracy", "balanced_accuracy", "macro_f1")}
                                 for k in pump},
        "deployment_probes_identical": {k: probes[k]["status"] == committed["deployment_robustness"][k]["status"]
                                        for k in probes},
        "deployment_probes_here": probes,
        "make_figures_input_missing_here": not (pdm.PROCESSED / "deployment_samples" / "grader" /
                                                "20221208-O-2rpm-0N-1.csv").is_file(),
    }
    save_stage("phase1_runner_rerun", out)
    return out


def stage_diagnostics():
    """Recompute the reported JUST diagnostic values from the archive member; measure sampling rates."""
    name = "20221208-O-2rpm-0N-1.csv"
    channel = pdm.JUST_CHANNELS[0]
    found = None
    for n in range(1, 7):
        with zipfile.ZipFile(pdm.JUST_DIR / f"condition{n}.zip") as archive:
            member = next((x for x in archive.namelist() if x.endswith(name)), None)
            if member:
                with archive.open(member) as handle:
                    head = pd.read_csv(handle, nrows=50_000)
                found = (f"condition{n}.zip", member, head)
                break
    out = {}
    if found:
        zip_name, member, head = found
        analysis = pdm.signal_diagnostics(head[channel], 50_000, "m/s²", rpm=2,
                                          bands=[(0, 1000), (1000, 5000), (5000, 25000)])
        out["just_example"] = {
            "archive": zip_name, "member": member, "columns": list(head.columns),
            "rms": analysis["rms"], "peak_to_peak": analysis["peak_to_peak"],
            "fft_peak_hz": float(analysis["frequency_hz"][np.argmax(analysis["amplitude"])]),
            "frequency_resolution_hz": float(analysis["frequency_hz"][1]),
            "reported": {"rms": 0.001658, "peak_to_peak": 0.016288, "fft_peak_hz": 30.0},
        }
        if "Time (s)" in head.columns or "time" in head.columns:
            t = head["Time (s)" if "Time (s)" in head.columns else "time"].to_numpy(np.float64)
            out["just_example"]["measured_rate_hz"] = round(float(1 / np.median(np.diff(t))), 2)
    aalto = next(iter(sorted((pdm.AALTO / "250RPM" / "0_12Nm" / "Healthy").glob("GP*.csv"))), None)
    if aalto:
        t = pd.read_csv(aalto, usecols=["time"], nrows=30_000)["time"].to_numpy(np.float64)
        out["aalto_measured_rate_hz"] = round(float(1 / np.median(np.diff(t))), 2)
        out["aalto_declared_rate_hz"] = pdm.AALTO_RATE
    out["rates_from_publisher_metadata_only"] = {
        "Ottawa": "200 kHz (no time column in the .mat)", "UODS": "42 kHz (no time column)",
        "CORA Vx": "3 kHz (single column)", "MCC5-THU": "12.8 kHz (no time column)"}
    save_stage("diagnostics_check", out)
    return out


def stage_benchmark():
    code = ("import json, sys; sys.path.insert(0, 'Notebook'); import psutil, phase1_validation as v; "
            "r = v.benchmark(sys.argv[1]); r['peak_working_set_mib'] = round(psutil.Process().memory_info().peak_wset / 2**20, 1); "
            "print('JSON' + json.dumps(r, default=str))")
    out = {}
    for route in ("pump_sensor", "pump_image", "aalto"):
        done = subprocess.run([sys.executable, "-c", code, route], capture_output=True, text=True, cwd=pdm.ROOT)
        line = next((x for x in done.stdout.splitlines() if x.startswith("JSON")), None)
        out[route] = json.loads(line[4:]) if line else {"error": done.stderr[-2000:]}
        print(f"benchmark {route} done", flush=True)
    save_stage("benchmark_windows", out)
    return out


if __name__ == "__main__":
    stages = {"reproduce": stage_reproduce, "images": stage_images, "transient": stage_transient,
              "phase1": stage_phase1, "diagnostics": stage_diagnostics, "benchmark": stage_benchmark}
    chosen = sys.argv[1:] or list(stages)
    for stage in chosen:
        if stage.startswith("reproduce="):
            stage_reproduce(stage.split("=", 1)[1].split(","))
            continue
        stages[stage]()
