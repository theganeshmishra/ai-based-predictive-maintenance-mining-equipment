"""Week 7 full-data sensor experiment for the CORA machines (Belt Conveyor, Dozer, Excavator, Wheel Loader).

Shared engine; each machine file has a `fulldata` command that calls `run(machine, conditions)`.
Uses every CORA sensor and every recording that exists for the machine's conditions:

  - stationary runs (one 45-50 min run per condition and speed): vibration X/Y/Z (3 kHz), motor current
    C1-C3 (4 kHz) and six temperature probes T1-T6 (1 kHz), EVERY 1-second window;
  - start-up transient tests (FIVE 30 s repeats per condition and speed): vibration X/Y/Z and current
    C1-C3; the steady part (10-30 s) is used, 20 one-second windows per repeat.

Main test: each whole speed is held out in turn (4 rounds). Training uses all stationary and transient
recordings of the other three speeds; the held-out speed gives 6 test recordings per class (1 stationary
+ 5 transient). One prediction per recording (mean class probabilities of its windows); every training
recording gets equal weight. The feature set and model are chosen by leave-one-speed-out validation
inside the training speeds only. All features are computed per window from that window alone (vibration
in shaft orders, current in supply-frequency orders), so no preprocessing looks at the held-out speed.

Secondary (easier) time split: stationary runs only, first 60% of every run trains, last 30% tests.
Temperature turned out to identify the recording session (it separates the conditions already in the
first 5 minutes, before any fault heating), so temperature sets are reported but excluded from selection.
Temperature is offered two ways: the pattern only (each probe relative to the others and to the run's first
minute) and including absolute temperature, which mostly follows the room temperature of that recording day.
Results go to Results/<machine>/<machine>_fulldata.json; the Week 6 results are not changed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "Raw_Data" / "cora_full"
VX = ROOT / "Raw_Data" / "cora_vibration"
CACHE = ROOT / "Processed_Data" / "cora_full_features"
SPEEDS = ("F5", "F15", "F50", "F60")
RATE = {"V": 3000, "C": 4000, "T": 1000}
ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
VIB = ["log_rms", "kurtosis", "skewness", "crest", "peak_order", "log_2x_1x",
       *(f"order_{lo}_{hi}" for lo, hi in ORDERS)]
CUR = ["log_rms", "kurtosis", "crest", "log_thd", "log_sideband", "spectral_entropy",
       *(f"order_{lo}_{hi}" for lo, hi in ORDERS)]
TRANSIENT_WINDOWS = range(10, 30)   # steady part of the 30 s start-up test


# ---------------------------------------------------------------------------------------------- features

def _read(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}: download the CORA files first")
    return pd.read_csv(path, header=None, dtype=np.float64, engine="c").iloc[:, 0].to_numpy()


def _windows(x: np.ndarray, rate: int) -> np.ndarray:
    n = len(x) // rate
    w = x[: n * rate].reshape(n, rate)
    return w - w.mean(axis=1, keepdims=True)


def _order_shares(power, freq, base):
    order = freq / base
    usable = (order >= 0.5) & (order < 20.5)
    share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
    o = order[usable]
    return share, o, [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in ORDERS]


def vibration_features(x, speed):
    rate, shaft = RATE["V"], float(speed[1:])
    w = _windows(x, rate)
    std = np.maximum(w.std(axis=1), 1e-12)
    freq = np.fft.rfftfreq(rate, d=1 / rate)
    power = np.abs(np.fft.rfft(w * np.hanning(rate), axis=1)) ** 2
    share, o, bands = _order_shares(power, freq, shaft)
    return np.column_stack([np.log(std), (w ** 4).mean(axis=1) / std ** 4, (w ** 3).mean(axis=1) / std ** 3,
                            np.abs(w).max(axis=1) / std, o[np.argmax(share, axis=1)],
                            np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands])


def current_features(x, speed):
    rate, supply = RATE["C"], float(speed[1:])
    w = _windows(x, rate)
    std = np.maximum(w.std(axis=1), 1e-12)
    freq = np.fft.rfftfreq(rate, d=1 / rate)            # 1 Hz bins
    power = np.abs(np.fft.rfft(w * np.hanning(rate), axis=1)) ** 2
    f0 = int(round(supply))
    fundamental = power[:, max(f0 - 1, 1): f0 + 2].sum(axis=1) + 1e-12
    harmonics = sum(power[:, k * f0 - 1: k * f0 + 2].sum(axis=1) for k in range(2, 10) if k * f0 + 2 < power.shape[1])
    side = np.r_[max(f0 - 6, 1): max(f0 - 1, 1), f0 + 2: f0 + 7]
    sideband = power[:, side].sum(axis=1)
    p = power[:, 1:] / np.maximum(power[:, 1:].sum(axis=1, keepdims=True), 1e-12)
    entropy = -(p * np.log(p + 1e-15)).sum(axis=1)
    _, _, bands = _order_shares(power, freq, supply)
    return np.column_stack([np.log(std), (w ** 4).mean(axis=1) / std ** 4, np.abs(w).max(axis=1) / std,
                            np.log(harmonics / fundamental + 1e-12), np.log(sideband / fundamental + 1e-12),
                            entropy, *bands])


def temperature_features(x):
    n = len(x) // RATE["T"]
    return x[: n * RATE["T"]].reshape(n, RATE["T"]).mean(axis=1)[:, None]


def cached(name: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.is_file():
        return np.load(path)
    value = compute()
    np.save(path, value)
    return value


def recording(code: str, speed: str, regime: str, rep: int | None = None) -> pd.DataFrame:
    """All features of one recording, one row per 1 s window."""
    tag = f"{code}_{speed}_{regime}"
    suffix = "" if rep is None else f"_{rep}"
    parts = {}
    for axis in ("x", "y", "z"):
        if regime == "S" and axis == "x":
            path = VX / f"{tag}_Vx.csv"
        else:
            path = FULL / f"{tag}_V{axis}{suffix}.csv"
        parts[f"vib_{axis}"] = (cached(f"{tag}_V{axis}{suffix}", lambda p=path: vibration_features(_read(p), speed)), VIB)
    for phase in ("1", "2", "3"):
        path = FULL / f"{tag}_C{phase}{suffix}.csv"
        parts[f"cur_{phase}"] = (cached(f"{tag}_C{phase}{suffix}", lambda p=path: current_features(_read(p), speed)), CUR)
    if regime == "S":
        for probe in range(1, 7):
            path = FULL / f"{tag}_T{probe}.csv"
            parts[f"temp_{probe}"] = (cached(f"{tag}_T{probe}", lambda p=path: temperature_features(_read(p))), ["mean"])
    n = min(len(v) for v, _ in parts.values())
    frame = pd.DataFrame({f"{key}_{name}": values[:n, i] for key, (values, names) in parts.items()
                          for i, name in enumerate(names)})
    if regime == "S":   # temperature relative to the probes' own mean and to the run's first minute (same recording only)
        probes = [f"temp_{p}_mean" for p in range(1, 7)]
        frame[[f"{c}_vs_probes" for c in probes]] = frame[probes].to_numpy() - frame[probes].mean(axis=1).to_numpy()[:, None]
        frame[[f"{c}_rise" for c in probes]] = frame[probes].to_numpy() - frame[probes].iloc[:60].mean().to_numpy()
    else:
        frame = frame.iloc[[i for i in TRANSIENT_WINDOWS if i < len(frame)]]
    frame.insert(0, "window", np.arange(len(frame)))
    frame.insert(0, "regime", "stationary" if regime == "S" else "transient")
    frame.insert(0, "speed", speed)
    frame.insert(0, "unit", tag + suffix)
    return frame.reset_index(drop=True)


def build(conditions: dict[str, str]) -> pd.DataFrame:
    frames = []
    for code, label in conditions.items():
        for speed in SPEEDS:
            for frame in [recording(code, speed, "S")] + [recording(code, speed, "T", r) for r in range(1, 6)]:
                frame.insert(1, "class", label)
                frames.append(frame)
            print(f"  features ready: {code}_{speed} (stationary + 5 start-up tests)", flush=True)
    return pd.concat(frames, ignore_index=True)


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in table.columns if c not in ("unit", "class", "speed", "regime", "window")]
    vx = [c for c in cols if c.startswith("vib_x_")]
    vib = [c for c in cols if c.startswith("vib_")]
    cur = [c for c in cols if c.startswith("cur_")]
    temp = [c for c in cols if c.startswith("temp_")]
    pattern = [c for c in temp if c.endswith(("_vs_probes", "_rise"))]   # no absolute temperature (room-dependent)
    return {"vibration X (as Week 6, all windows)": vx, "vibration X/Y/Z": vib, "current C1-C3": cur,
            "vibration + current": vib + cur,
            "vibration + current + temperature pattern (stationary only)": vib + cur + pattern,
            "vibration + current + temperature incl. absolute (stationary only)": vib + cur + temp}


# ---------------------------------------------------------------------------------------------- models

def models():
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    return {
        "Random forest": lambda: RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2,
                                                        class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                                           verbose=-1),
    }


def _fit_predict(make, train: pd.DataFrame, test: pd.DataFrame, features, labels):
    import warnings
    weight = 1.0 / train.groupby("unit")["unit"].transform("size").to_numpy()
    model = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[features].to_numpy(), train["class"].to_numpy(), sample_weight=weight)
        proba = model.predict_proba(test[features].to_numpy())
    out = pd.DataFrame(proba, columns=list(model.classes_)).reindex(columns=labels, fill_value=0.0)
    out["unit"], out["class"], out["regime"] = test["unit"].to_numpy(), test["class"].to_numpy(), test["regime"].to_numpy()
    per_unit = out.groupby("unit").agg({**{c: "mean" for c in labels}, "class": "first", "regime": "first"})
    per_unit["predicted"] = per_unit[labels].idxmax(axis=1)
    return per_unit


def _score(per_unit: pd.DataFrame, labels):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    block = {}
    for name, part in (("all", per_unit), ("stationary", per_unit[per_unit.regime == "stationary"]),
                       ("transient", per_unit[per_unit.regime == "transient"])):
        if len(part):
            block[name] = {"test_recordings": int(len(part)),
                           "accuracy": round(float(accuracy_score(part["class"], part["predicted"])), 4),
                           "balanced_accuracy": round(float(balanced_accuracy_score(part["class"], part["predicted"])), 4),
                           "macro_f1": round(float(f1_score(part["class"], part["predicted"], labels=labels,
                                                            average="macro", zero_division=0)), 4)}
    block["confusion_matrix_all"] = confusion_matrix(per_unit["class"], per_unit["predicted"], labels=labels).tolist()
    return block


def _subset(table, name):
    return table[table.regime == "stationary"] if "stationary only" in name else table


def run(machine: str, conditions: dict[str, str], train_step: int = 3) -> dict:
    """Run the full-data experiment for one machine and save Results/<machine>/<machine>_fulldata.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    labels = list(conditions.values())
    print(f"{machine}: building features from every CORA sensor and recording", flush=True)
    table = build(conditions)
    sets = feature_sets(table)
    makers = models()
    # train on every train_step-th window of the long stationary runs (equal recording weights anyway)
    thin = lambda t: t[(t.regime == "transient") | (t.window % train_step == 0)]

    grid, chosen, outer_units = {}, [], []
    for held in SPEEDS:
        train_all, test_all = table[table.speed != held], table[table.speed == held]
        # inner leave-one-speed-out validation on the three training speeds: choose feature set + model
        best, best_score = None, -1.0
        for set_name, features in sets.items():
            if "temperature" in set_name:
                continue   # confounded with the recording session (see temperature_diagnostic); never selected
            for model_name, make in makers.items():
                scores = []
                for inner in sorted(set(train_all.speed)):
                    tr = _subset(train_all[train_all.speed != inner], set_name)
                    va = _subset(train_all[train_all.speed == inner], set_name)
                    pu = _fit_predict(make, thin(tr), va, features, labels)
                    scores.append(_score(pu, labels)["all"]["macro_f1"])
                if np.mean(scores) > best_score:
                    best, best_score = (set_name, model_name), float(np.mean(scores))
        chosen.append({"held_out_speed": held, "feature_set": best[0], "model": best[1],
                       "validation_macro_f1": round(best_score, 4)})
        # every feature set x model on the held-out speed (reported for comparison, not used to choose)
        for set_name, features in sets.items():
            for model_name, make in makers.items():
                pu = _fit_predict(make, thin(_subset(train_all, set_name)), _subset(test_all, set_name), features, labels)
                grid.setdefault((set_name, model_name), []).append(pu)
                if (set_name, model_name) == best:
                    outer_units.append(pu)
        print(f"  held-out {held}: validation chose {best[0]} / {best[1]}", flush=True)

    result = {
        "purpose": "Week 7 full-data sensor experiment: every CORA sensor, every window, stationary + start-up recordings",
        "machine": machine, "classes": labels,
        "test": "four rounds, each holding out one whole speed; test recordings per round = 1 stationary + 5 start-up "
                "tests per class; one prediction per recording",
        "selection_rule": "feature set and model chosen by leave-one-speed-out validation inside the training speeds; "
                          "test speeds never used for any choice or preprocessing; temperature sets excluded from "
                          "selection (session confound, see temperature_diagnostic)",
        "recordings": {"stationary": int(table[table.regime == "stationary"].unit.nunique()),
                       "transient": int(table[table.regime == "transient"].unit.nunique())},
        "windows": int(len(table)),
        "validation_selected": {"choices": chosen, "score": _score(pd.concat(outer_units), labels)},
        "all_combinations": {f"{s} | {m}": _score(pd.concat(v), labels) for (s, m), v in grid.items()},
    }

    # temperature diagnostic: does the probe pattern identify the condition before fault heating can build up?
    stat_all = table[table.regime == "stationary"]
    probe_pattern = [c for c in table.columns if c.endswith("_vs_probes")]
    warming = [c for c in table.columns if c.endswith("_rise")]
    diag = {}
    for name, part, cols in (("probe pattern, first 5 minutes only", stat_all[stat_all.window < 300], probe_pattern),
                             ("probe pattern, whole run", stat_all[stat_all.window % train_step == 0], probe_pattern),
                             ("warming since run start, whole run", stat_all[stat_all.window % train_step == 0], warming)):
        units = pd.concat([_fit_predict(makers["Random forest"], part[part.speed != h], part[part.speed == h], cols, labels)
                           for h in SPEEDS])
        diag[name] = _score(units, labels)["all"]
    result["temperature_diagnostic"] = {
        "finding": "if the probe pattern already identifies the condition in the first 5 minutes, it reflects the "
                   "recording session (probe placement / residual heat after each rebuild), not fault heating; "
                   "temperature feature sets are therefore reported but never selected",
        "unseen_speed_scores": diag}

    # secondary time split: stationary runs only
    stat = table[table.regime == "stationary"].copy()
    position = stat.groupby("unit")["window"].rank(pct=True)
    early, late = stat[position <= 0.6], stat[position > 0.7]
    result["time_split_stationary"] = {
        f"{s} | Random forest": _score(_fit_predict(makers["Random forest"], early.iloc[::train_step], late, f, labels), labels)
        for s, f in sets.items()}

    out = ROOT / "Results" / machine / f"{machine}_fulldata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    names = list(result["all_combinations"])
    figure, axis = plt.subplots(figsize=(9, 4))
    pos = np.arange(len(names))
    axis.bar(pos - 0.2, [result["all_combinations"][n]["all"]["accuracy"] for n in names], 0.4, label="all test recordings")
    axis.bar(pos + 0.2, [result["all_combinations"][n].get("stationary", {}).get("accuracy", 0) for n in names], 0.4,
             label="stationary runs only")
    axis.axhline(1 / len(labels), color="grey", ls="--", lw=1)
    axis.set_xticks(pos, [n.replace(" | ", "\n").replace(" (", "\n(") for n in names], fontsize=6)
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    v = result["validation_selected"]["score"]["all"]
    axis.set_title(f"{machine.replace('_', ' ').title()} full data, unseen speed: validation-selected "
                   f"{v['accuracy']:.3f} on {v['test_recordings']} recordings (dashed = chance)", fontsize=9)
    figure.tight_layout()
    fig = ROOT / "Figures" / machine / f"{machine}_fulldata.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fig, dpi=110)
    plt.close(figure)
    s = result["validation_selected"]["score"]
    parts = ", ".join(f"{k} {s[k]['accuracy']:.3f}" for k in ("stationary", "transient") if k in s)
    print(f"Validation-selected: all {s['all']['accuracy']:.3f} ({s['all']['test_recordings']} recordings); {parts}")
    print(f"Saved {out.relative_to(ROOT)} and {fig.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
    save_recording_confusion(result)
    return result


# ---------------------------------------------------------------------------------------------- Week 7 extras
# The ensemble comparison and the anomaly detection rerun on the full data, with the same fair main test:
# one whole speed held out per round, nothing chosen or fitted on the held-out speed.

FIXED_SET = "vibration + current"   # every non-temperature sensor; fixed in advance, not chosen on any test


def ensemble_models():
    from lightgbm import LGBMClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    return {
        "Majority class (control)": lambda: DummyClassifier(strategy="most_frequent"),
        "Random forest": models()["Random forest"],
        "Extra trees": lambda: ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                                    class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient boosting (scikit-learn)": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                                   random_state=42),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1,
                                         verbosity=0),
        "LightGBM": models()["LightGBM"],
    }


class _Encoded:
    """Wrap a classifier so it trains on integer labels (XGBoost needs them) and reports the class names."""

    def __init__(self, model, labels):
        self.model, self.labels = model, labels

    def fit(self, x, y, sample_weight=None):
        code = {label: i for i, label in enumerate(self.labels)}
        self.model.fit(x, np.array([code[v] for v in y]), sample_weight=sample_weight)
        self.classes_ = np.array([self.labels[i] for i in self.model.classes_])
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)


def ensembles(machine: str, conditions: dict[str, str], train_step: int = 3) -> dict:
    """Ensemble comparison on the full data; saves Results/<machine>/<machine>_fulldata_ensembles.json."""
    start = time.perf_counter()
    labels = list(conditions.values())
    print(f"{machine}: ensemble comparison on the full data", flush=True)
    table = build(conditions)
    features = feature_sets(table)[FIXED_SET]
    thin = lambda t: t[(t.regime == "transient") | (t.window % train_step == 0)]
    stat = table[table.regime == "stationary"]
    position = stat.groupby("unit")["window"].rank(pct=True)
    early, late = stat[position <= 0.6], stat[position > 0.7]
    result = {"purpose": "Week 7: ensemble comparison (proposal Week 6) rerun on every CORA sensor, window and recording",
              "machine": machine, "classes": labels, "features": f"{FIXED_SET} ({len(features)} features)",
              "test": "MAIN: four rounds, each holding out one whole speed (24 test recordings per round, 96 in total); "
                      "one prediction per recording. TIME SPLIT (secondary, easier): stationary runs, first 60% "
                      "trains, last 30% tests",
              "selection_rule": "feature set and model settings fixed in advance; nothing chosen on any test result",
              "models": {}}
    for name, make in ensemble_models().items():
        wrapped = lambda make=make: _Encoded(make(), labels)
        t0 = time.perf_counter()
        main = pd.concat([_fit_predict(wrapped, thin(table[table.speed != held]), table[table.speed == held],
                                       features, labels) for held in SPEEDS])
        seconds = time.perf_counter() - t0
        result["models"][name] = {"main": _score(main, labels),
                                  "time_split_stationary": _score(_fit_predict(wrapped, early.iloc[::train_step], late,
                                                                               features, labels), labels),
                                  "seconds": round(seconds, 1)}
        m = result["models"][name]
        print(f"  {name:33s} main {m['main']['all']['accuracy']:.3f} (stationary "
              f"{m['main']['stationary']['accuracy']:.3f}, start-up {m['main']['transient']['accuracy']:.3f})  "
              f"time split {m['time_split_stationary']['all']['accuracy']:.3f}  [{seconds:.0f} s]", flush=True)
    out = ROOT / "Results" / machine / f"{machine}_fulldata_ensembles.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _bar_figure(machine, "ensembles", {n: (m["main"]["all"]["accuracy"], m["time_split_stationary"]["all"]["accuracy"])
                                       for n, m in result["models"].items()},
                ("main test (unseen speed)", "time split (secondary)"), 1 / len(labels),
                f"{machine.replace('_', ' ').title()} full data: ensemble comparison (dashed = chance)")
    print(f"Saved {out.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
    return result


def _detectors():
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    return {"Isolation Forest": lambda: make_pipeline(StandardScaler(), IsolationForest(
                n_estimators=300, max_samples=256, random_state=42, n_jobs=-1)),
            "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))}


def _detector_fit(make, frame, features, max_rows=4000):
    # every healthy recording gets an equal share of the (at most max_rows) training rows
    per_unit = max(1, max_rows // frame.unit.nunique())
    keep = np.concatenate([idx[np.linspace(0, len(idx) - 1, min(len(idx), per_unit)).astype(int)]
                           for idx in frame.groupby("unit").indices.values()])
    return make().fit(frame[features].to_numpy()[keep])


def _recording_scores(model, frame, features):
    window = -model.decision_function(frame[features].to_numpy())   # higher = more unusual
    return pd.Series(window, index=frame.index).groupby(frame.unit).mean()


def _anomaly_block(rows: pd.DataFrame, labels) -> dict:
    from sklearn.metrics import roc_auc_score
    truth = (rows["class"] != "healthy").to_numpy()
    alarm = rows["margin"].to_numpy() > 0
    return {"test_recordings": int(len(rows)), "healthy": int((~truth).sum()), "faulty": int(truth.sum()),
            "roc_auc": round(float(roc_auc_score(truth, rows["margin"])), 4) if 0 < truth.sum() < len(truth) else None,
            "fault_detection_rate": round(float(alarm[truth].mean()), 4),
            "false_alarm_rate": round(float(alarm[~truth].mean()), 4),
            "accuracy": round(float((alarm == truth).mean()), 4),
            "detection_rate_per_fault": {c: round(float(alarm[(rows["class"] == c).to_numpy()].mean()), 4)
                                         for c in labels if c != "healthy"}}


def anomaly(machine: str, conditions: dict[str, str]) -> dict:
    """Anomaly detection on the full data; saves Results/<machine>/<machine>_fulldata_anomaly.json."""
    start = time.perf_counter()
    labels = list(conditions.values())
    print(f"{machine}: anomaly detection on the full data", flush=True)
    table = build(conditions)
    features = feature_sets(table)[FIXED_SET]
    result = {"purpose": "Week 7: anomaly detection rerun on every CORA sensor, window and recording "
                         "(trained on healthy data only)",
              "machine": machine, "features": f"{FIXED_SET} ({len(features)} features)",
              "test": "MAIN: four rounds, each holding out one whole speed (24 test recordings per round: 6 healthy, "
                      "18 faulty). TIME SPLIT (secondary): stationary runs, healthy first 60% trains, last 30% tests",
              "threshold_rule": "MAIN: leave-one-speed-out inside the training speeds - fit on the healthy recordings "
                                "of two training speeds, score the healthy recordings of the third; the highest score "
                                "is the alarm threshold. TIME SPLIT: leave-one-healthy-run-out on the early parts. "
                                "No test data and no fault labels are used to train or to set the threshold",
              "detectors": {}}
    stat = table[table.regime == "stationary"]
    position = stat.groupby("unit")["window"].rank(pct=True)
    early, late = stat[position <= 0.6], stat[position > 0.7]
    for name, make in _detectors().items():
        rows = []
        for held in SPEEDS:
            train = table[(table.speed != held) & (table["class"] == "healthy")]
            cut = max(_recording_scores(_detector_fit(make, train[train.speed != inner], features),
                                        train[train.speed == inner], features).max()
                      for inner in sorted(set(train.speed)))
            test = table[table.speed == held]
            scores = _recording_scores(_detector_fit(make, train, features), test, features)
            rows.append(test.groupby("unit")[["class", "regime"]].first().assign(margin=scores - cut))
        rows = pd.concat(rows)
        healthy_early = early[early["class"] == "healthy"]
        cut = max(_recording_scores(_detector_fit(make, healthy_early[healthy_early.unit != u], features),
                                    healthy_early[healthy_early.unit == u], features).max()
                  for u in sorted(set(healthy_early.unit)))
        ts = _recording_scores(_detector_fit(make, healthy_early, features), late, features)
        ts_rows = late.groupby("unit")[["class", "regime"]].first().assign(margin=ts - cut)
        block = {"main": _anomaly_block(rows, labels),
                 "main_stationary": _anomaly_block(rows[rows.regime == "stationary"], labels),
                 "main_transient": _anomaly_block(rows[rows.regime == "transient"], labels),
                 "time_split_stationary": _anomaly_block(ts_rows, labels)}
        result["detectors"][name] = block
        m, t = block["main"], block["time_split_stationary"]
        print(f"  {name:17s} main ROC-AUC {m['roc_auc']}  detected {m['fault_detection_rate']:.3f}  false alarms "
              f"{m['false_alarm_rate']:.3f} ({m['test_recordings']} recordings) | time split ROC-AUC {t['roc_auc']}",
              flush=True)
    out = ROOT / "Results" / machine / f"{machine}_fulldata_anomaly.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _bar_figure(machine, "anomaly", {n: (b["main"]["roc_auc"], b["time_split_stationary"]["roc_auc"])
                                     for n, b in result["detectors"].items()},
                ("main test ROC-AUC (unseen speed)", "time split ROC-AUC (secondary)"), 0.5,
                f"{machine.replace('_', ' ').title()} full data: anomaly detection (dashed = chance)")
    print(f"Saved {out.relative_to(ROOT)} in {(time.perf_counter() - start) / 60:.1f} min")
    return result


def _bar_figure(machine, kind, values, legend, chance, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(values)
    figure, axis = plt.subplots(figsize=(8, 3.6))
    pos = np.arange(len(names))
    for k, offset in enumerate((-0.2, 0.2)):
        axis.bar(pos + offset, [values[n][k] or 0 for n in names], 0.4, label=legend[k])
    axis.axhline(chance, color="grey", ls="--", lw=1)
    axis.set_xticks(pos, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(ROOT / "Figures" / machine / f"{machine}_fulldata_{kind}.png", dpi=110)
    plt.close(figure)


def save_recording_confusion(result: dict) -> Path:
    """Plot real held-out recording counts and row percentages from saved results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(result["validation_selected"]["score"]["confusion_matrix_all"])
    labels = result["classes"]
    support = matrix.sum(axis=1)
    total = int(matrix.sum())
    score = result["validation_selected"]["score"]["all"]
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(labels)), labels, rotation=25)
    axis.set_yticks(range(len(labels)), [f"{label} (n={n})" for label, n in zip(labels, support)])
    axis.set_xlabel("Predicted condition")
    axis.set_ylabel("Actual condition")
    for i in range(len(labels)):
        for j in range(len(labels)):
            percentage = 100 * matrix[i, j] / support[i] if support[i] else 0
            axis.text(j, i, f"{matrix[i, j]}\n({percentage:.1f}%)", ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=11)
    axis.set_title(f"{result['machine'].replace('_', ' ').title()}: full sensor evaluation\n"
                   f"{total} held-out recordings | accuracy {score['accuracy']:.1%} | four unseen-speed folds")
    figure.text(.5, .025, "Each cell counts whole recordings; percentages are within each actual class.\n"
                "0 means no recordings in that combination. Start-up repeats share one laboratory rig.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .08, 1, 1))
    output = ROOT / "Figures" / result["machine"] / f"{result['machine']}_fulldata_confusion.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
