"""Bounded Phase 1 experiments. Run from the project root with the project interpreter."""
import json
import platform
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pdm


def cora_candidates():
    """Evaluate one fixed candidate family on each distinct CORA task."""
    codes = dict(H="healthy", W25="wear25", W50="wear50", W75="wear75", BD="bearing",
                 U="unbalance", M="misalignment", HB="half_broken_bar", OB="broken_bar")
    raw, _ = pdm.cora_rig(codes)
    # Describe distribution within each recording using quantiles and no fitted preprocessing.
    features = [c for c in pdm.CORA_FEATURES if c != "log_rms"]
    group = ["unit", "speed", "class"]
    table = raw.groupby(group)[features].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    table.columns = ["_".join(str(x) for x in col if x != "") for col in table.columns]
    values = table.drop(columns=group).to_numpy()
    factories = {
        "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RBF SVM": lambda: make_pipeline(StandardScaler(), SVC(C=1)),
        "Random forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42),
        "Extra trees": lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, random_state=42),
    }
    result = {}
    for machine in ("belt_conveyor", "excavator", "wheel_loader"):
        config = pdm.MACHINES[machine]
        active = table["class"].isin(config.labels).to_numpy()
        groups = table["speed"].to_numpy()[active]
        units = table["unit"].to_numpy()[active]
        y = table["class"].to_numpy()[active]
        x = values[active]
        predictions = np.empty_like(y)
        choices = []
        for test in pdm.fair_test_folds(groups, config.sensor_held_out):
            pdm.check_no_overlap(units, test)
            training_speeds = sorted(set(groups[~test]))

            def validation(name):
                scores = []
                for speed in training_speeds:
                    valid = groups == speed
                    fit = ~test & ~valid
                    model = factories[name]().fit(x[fit], y[fit])
                    scores.append(f1_score(y[valid], model.predict(x[valid]), labels=config.labels,
                                           average="macro", zero_division=0))
                return np.mean(scores)

            chosen = max(factories, key=validation)
            predictions[test] = factories[chosen]().fit(x[~test], y[~test]).predict(x[test])
            choices.append(chosen)
        result[machine] = {"classes": config.labels, "held_out_groups": 4, "choices": choices,
                           "score": pdm.classification_metrics(y, predictions, config.labels)}
    return result


def perturb_signal(signal, kind, rng):
    if kind == "noise":
        return signal + rng.normal(0, 0.1 * np.sqrt(np.mean(signal**2)), len(signal))
    if kind == "amplitude_half":
        return signal * 0.5
    if kind == "amplitude_double":
        return signal * 2
    if kind == "missing_5pct":
        return signal[rng.random(len(signal)) >= 0.05]
    if kind == "half_length":
        return signal[:len(signal) // 2]
    if kind == "rate_90pct":
        return resample_poly(signal, 9, 10)
    if kind == "clean":
        return signal
    raise ValueError(f"Unknown perturbation {kind!r}")


def pump_robustness():
    """Fit the archived model on A/B only and perturb the 24 held-out C/D recordings."""
    table, features = pdm.ottawa_bearing()
    config = pdm.MACHINES["pump"]
    test = pdm.fair_test_folds(table["speed_profile"], config.sensor_held_out)[0]
    pdm.check_no_overlap(table["unit"].to_numpy(), test)
    names = features["vibration"]
    model = pdm.sensor_candidates()["Extra trees"]().fit(table.loc[~test, names].to_numpy(dtype=float),
                                                         table.loc[~test, "class"])
    cases = ("clean", "noise", "amplitude_half", "amplitude_double", "missing_5pct", "half_length", "rate_90pct")
    rng = np.random.default_rng(613)
    predictions = {key: [] for key in cases}
    actual = []
    recordings = table.loc[test, ["unit", "class"]].drop_duplicates().sort_values("unit")
    for unit, label in recordings.itertuples(index=False, name=None):
        recording = pdm.RAW / "uottawa_bearing" / unit
        signal = loadmat(recording, variable_names=["Channel_1"])["Channel_1"].ravel().astype(float)
        actual.append(label)
        for case in cases:
            values = perturb_signal(signal, case, rng)
            features = pdm.ottawa_signal_features(values)[names].to_numpy(dtype=float)
            index = model.predict_proba(features).mean(axis=0).argmax()
            predictions[case].append(model.classes_[index])
    return {k: pdm.classification_metrics(actual, v, config.labels) for k, v in predictions.items()}


def deployment_robustness():
    """Probe real deployment inputs and isolated malformed cases with no checkpoint writes."""
    sensor, _, images = pdm.route_samples()
    image = images["pump"]["healthy"]
    pixels = cv2.imread(str(image))
    output = {}

    def probe(name, file, machine=None):
        try:
            result = pdm.predict(file, machine)
            output[name] = {"status": result.get("status", "prediction"),
                            "condition": result.get("condition", result.get("Predicted tooth condition"))}
        except (ValueError, FileNotFoundError) as problem:
            output[name] = {"status": "rejected", "reason": str(problem)}

    with TemporaryDirectory() as folder:
        root = Path(folder)
        for name, samples in (("short", ["1"] * 10), ("nan", ["nan"] * 20000),
                              ("inf", ["inf"] * 20000), ("corrupt", ["xyz"])):
            path = root / f"{name}.csv"
            path.write_text(pdm.OTTAWA_CSV_CHANNEL + "\n" + "\n".join(samples) + "\n")
            probe(name, path)
        short_drill = root / "short_drill.csv"
        short_drill.write_text(",".join(pdm.MCC5_CHANNELS) + "\n" + ",".join(["1"] * len(pdm.MCC5_CHANNELS)))
        probe("short_drill", short_drill)
        for name in ("broken.mat", "broken.png", "unsupported.txt"):
            path = root / name
            path.write_bytes(b"broken")
            probe(name, path)
        jpeg = root / "compressed.jpg"
        cv2.imwrite(str(jpeg), pixels, [cv2.IMWRITE_JPEG_QUALITY, 65])
        probe("jpeg_auto", jpeg)
        probe("jpeg_confirmed", jpeg, "pump")
        resized = root / "resized.png"
        cv2.imwrite(str(resized), cv2.resize(pixels, (64, 48)))
        probe("resized_image", resized, "pump")
        old_models, old_manifest = pdm.MODELS, pdm.TRUSTED_MODELS
        try:
            pdm.MODELS = root
            pdm.TRUSTED_MODELS = root / "trusted_sha256.json"
            probe("missing_checkpoint", sensor["pump"])
            (root / "pump_sensor.joblib").write_bytes(b"untrusted fixture")
            probe("untrusted_checkpoint", sensor["pump"])
        finally:
            pdm.MODELS, pdm.TRUSTED_MODELS = old_models, old_manifest
    return output


def make_figures(output, robustness):
    import matplotlib.pyplot as plt

    example = pdm.PROCESSED / "deployment_samples" / "grader" / "20221208-O-2rpm-0N-1.csv"
    channel = pdm.JUST_CHANNELS[0]
    values = pdm.pd.read_csv(example, usecols=[channel], nrows=50_000)[channel]
    analysis = pdm.signal_diagnostics(values, 50_000, "m/s²", rpm=2,
                                      bands=[(0, 1000), (1000, 5000), (5000, 25000)])
    pdm.save_signal_diagnostic_plot(analysis, output / "just_signal_diagnostics.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    cases = list(robustness)
    names = ["Clean", "Noise", "Half amplitude", "Double amplitude", "5% removed", "Half length", "Wrong rate"]
    index = np.arange(len(cases))
    ax.bar(index - 0.18, [robustness[k]["balanced_accuracy"] * 100 for k in cases], 0.36,
           label="Accuracy / balanced accuracy")
    ax.bar(index + 0.18, [robustness[k]["macro_f1"] * 100 for k in cases], 0.36, label="Macro F1")
    ax.set(ylabel="Held-out recording score (%)", title="Pump sensor robustness on 24 C/D recordings", ylim=(0, 105))
    ax.set_xticks(index, names, rotation=18, ha="right")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "pump_sensor_robustness.png", dpi=140)
    plt.close(fig)
    summary = {k: analysis[k] for k in ("rms", "variance", "std", "skewness", "excess_kurtosis",
                                        "crest_factor", "peak_to_peak")}
    summary["fft_peak_hz"] = float(analysis["frequency_hz"][np.argmax(analysis["amplitude"])])
    summary["band_energy"] = {f"{lo}–{hi} Hz": value for (lo, hi), value in analysis["band_energy"].items()}
    return summary


def benchmark(route=None):
    """CPU-only representative single-file measurements, including load and preprocessing hooks."""
    from unittest.mock import patch

    import torch

    torch.set_num_threads(2)
    sensor, _, images = pdm.route_samples()
    entries = {
        "pump_sensor": (sensor["pump"], "pump", pdm.MODELS / "pump_sensor.joblib"),
        "pump_image": (images["pump"]["healthy"], "pump", pdm.MODELS / "pump_images.pt"),
        "aalto": (pdm.AALTO / "250RPM" / "0_12Nm" / "Healthy" / "GP1.csv", "grader", pdm.GRADER_GEAR_MODEL),
    }
    result = {}
    if route is not None:
        entries = {route: entries[route]}
    for key, (path, mapping, checkpoint) in entries.items():
        loads = []
        owner, method = (torch, "load") if key == "pump_image" else (pdm, "load_trusted_joblib")
        original = getattr(owner, method)

        def timed_load(*args, **kwargs):
            start = time.perf_counter()
            value = original(*args, **kwargs)
            loads.append(time.perf_counter() - start)
            return value

        with patch.object(owner, method, timed_load):
            cold_start = time.perf_counter()
            expected = pdm.predict(path, mapping)
            cold = time.perf_counter() - cold_start
            warm = []
            for _ in range(5):
                start = time.perf_counter()
                assert pdm.predict(path, mapping) == expected
                warm.append(time.perf_counter() - start)
        source, _, details = pdm.identify(path)
        start = time.perf_counter()
        if key == "pump_image":
            pdm.load_images(path.parent, [{"file": path.name}], "per_image")
        else:
            pdm.sensor_features(path, source, details)
        preprocessing = time.perf_counter() - start
        result[key] = {"cold_s": cold, "warm_median_s": float(np.median(warm)),
                       "preprocessing_s": preprocessing, "checkpoint_bytes": checkpoint.stat().st_size,
                       "first_load_call_s": loads[0], "load_calls": len(loads), "prediction": expected}
    try:
        import resource
    except ImportError:
        peak = None
    else:
        scale = 1024 ** 2 if sys.platform == "darwin" else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu_torch_threads": 2,
            "peak_process_rss_mib": peak, "routes": result}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--benchmark":
        print(json.dumps(benchmark(sys.argv[2]), indent=2))
        raise SystemExit
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else pdm.PROCESSED / "phase1"
    target.mkdir(parents=True, exist_ok=True)
    data = {"cora_candidates": cora_candidates()}
    print("Completed grouped CORA comparisons", flush=True)
    data["pump_robustness"] = pump_robustness()
    print("Completed Pump disturbances", flush=True)
    data["deployment_robustness"] = deployment_robustness()
    data["diagnostics"] = make_figures(target, data["pump_robustness"])
    data["benchmark"] = benchmark()
    (target / "verification.json").write_text(json.dumps(data, indent=2) + "\n")
