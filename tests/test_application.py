"""Application tests in one file, grouped by the feature they protect.

Run all checks: python -m pytest tests -q
Sections: shared setup; signals; splitting; routes; identification; trusted models;
uploads; dashboard/history; archived experiment results.
Real-file checks skip only when their documented local data/model files are absent.
"""

import csv
import http.client
import io
import json
import os
import socket
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Notebook"))

import predictive_maintenance  # noqa: E402


# ========================================================================================
# Shared setup and fixtures
# ========================================================================================

@pytest.fixture
def isolated_models(tmp_path, monkeypatch):
    """Point MODELS and the trust list at an empty temporary folder."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(predictive_maintenance, "MODELS", models)
    monkeypatch.setattr(predictive_maintenance, "TRUSTED_MODELS", models / "trusted_sha256.json")
    monkeypatch.setattr(predictive_maintenance, "GRADER_GEAR_MODEL", models / "grader_gear_sensor.joblib")
    return models


def needs(*paths):
    """Skip a test when local data it reads is not on this computer (data is never in git)."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"local data missing: {', '.join(missing)}")


@pytest.fixture(autouse=True, scope="session")
def isolated_history(tmp_path_factory):
    """Tests never write to the real prediction history in Processed_Data/."""
    original = predictive_maintenance.HISTORY
    predictive_maintenance.HISTORY = tmp_path_factory.mktemp("history") / "prediction_history.jsonl"
    yield predictive_maintenance.HISTORY
    predictive_maintenance.HISTORY = original


# ========================================================================================
# Signal diagnostics and invalid measurements
# ========================================================================================

@pytest.mark.parametrize("size", [999, 1000])
def test_sinusoid_amplitude_and_diagnostics(size):
    t = np.arange(size) / size
    values = 2 * np.sin(2 * np.pi * 50 * t)
    result = predictive_maintenance.signal_diagnostics(values, size, "m/s²", rpm=1500, bands=[(40, 60), (100, 150)])
    peak = np.argmax(result["amplitude"])
    assert result["frequency_hz"][peak] == pytest.approx(50)
    assert result["shaft_orders"][peak] == pytest.approx(2)
    assert result["amplitude"][peak] == pytest.approx(2)
    assert result["rms"] == pytest.approx(np.sqrt(2))
    assert result["variance"] == pytest.approx(2)
    assert result["std"] == pytest.approx(np.sqrt(2))
    assert result["skewness"] == pytest.approx(0, abs=1e-12)
    assert result["excess_kurtosis"] == pytest.approx(-1.5)
    assert result["crest_factor"] == pytest.approx(np.sqrt(2), rel=0.02)
    assert result["peak_to_peak"] == pytest.approx(4, rel=0.06)
    assert result["band_energy"][(40, 60)] == pytest.approx(2)
    assert result["band_energy"][(100, 150)] == pytest.approx(0, abs=1e-12)


def test_zero_signal_is_well_defined():
    result = predictive_maintenance.signal_diagnostics(np.zeros(32), 1000, "m/s²")
    assert result["rms"] == 0 and result["variance"] == 0
    assert result["crest_factor"] is None and result["excess_kurtosis"] is None
    assert result["shaft_orders"] is None
    assert np.isfinite(result["psd"]).all()


@pytest.mark.parametrize("samples,rate,units,options", [
    ([0, 1], 100, "g", {}),
    ([0, 1, np.nan, 3], 100, "g", {}),
    ([0, 1, np.inf, 3], 100, "g", {}),
    ([0, 1, 2, 3], 0, "g", {}),
    ([0, 1, 2, 3], 100, "", {}),
    ([0, 1, 2, 3], 100, "g", {"rpm": 0}),
    ([0, 1, 2, 3], 100, "g", {"bands": [(40, 60)]}),
])
def test_bad_metadata_or_samples_are_rejected(samples, rate, units, options):
    with pytest.raises(ValueError):
        predictive_maintenance.signal_diagnostics(samples, rate, units, **options)


def test_diagnostic_figure_is_written(tmp_path):
    path = tmp_path / "waveform.png"
    result = predictive_maintenance.signal_diagnostics(np.arange(100), 500, "m/s²", rpm=600)
    predictive_maintenance.save_signal_diagnostic_plot(result, path)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_controlled_perturbations_are_repeatable_and_have_expected_lengths():
    perturb_signal = predictive_maintenance.perturb_signal

    x = np.arange(1000, dtype=float)
    assert np.array_equal(perturb_signal(x, "amplitude_half", None), x / 2)
    assert np.array_equal(perturb_signal(x, "amplitude_double", None), x * 2)
    assert len(perturb_signal(x, "half_length", None)) == 500
    assert len(perturb_signal(x, "rate_90pct", None)) == 900
    for case in ("noise", "missing_5pct"):
        a = perturb_signal(x, case, np.random.default_rng(613))
        b = perturb_signal(x, case, np.random.default_rng(613))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, x)
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb_signal(x, "unknown", None)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_ottawa_sample_is_rejected_before_feature_extraction(bad):
    signal = np.zeros(20_000)
    signal[-1] = bad
    with pytest.raises(ValueError, match="invalid Ottawa signal values"):
        predictive_maintenance.ottawa_signal_features(signal)


def test_short_mcc5_has_clear_error():
    from io import StringIO

    contents = ",".join(predictive_maintenance.MCC5_CHANNELS) + "\n" + ",".join(["1"] * len(predictive_maintenance.MCC5_CHANNELS))
    with pytest.raises(ValueError, match="at least 12,800 samples"):
        predictive_maintenance.mcc5_recording(StringIO(contents))


# ========================================================================================
# Train/test separation and metric calculations
# ========================================================================================

def test_fair_test_folds_hold_out_whole_groups():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    folds = predictive_maintenance.fair_test_folds(groups, [["a"], ["b"], ["c"]])
    assert [f.tolist() for f in folds] == [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    # every row is tested exactly once across the rounds
    assert np.sum(folds, axis=0).tolist() == [1] * 6


def test_check_no_overlap_stops_a_leak():
    units = np.array(["r1", "r1", "r2", "r2"])
    predictive_maintenance.check_no_overlap(units, np.array([False, False, True, True]))
    with pytest.raises(ValueError, match="also appears in training"):
        predictive_maintenance.check_no_overlap(units, np.array([False, True, True, True]))


def test_time_split_drops_the_gap_and_names_halves():
    units = np.repeat(["r1", "r2"], 10)
    order = np.tile(np.arange(10), 2)
    keep, test, new_units = predictive_maintenance.time_split(units, order)
    assert keep.sum() == 18  # position 0.7 (window 6) is in the dropped gap of each recording
    assert test.sum() == 6  # last 30% of each recording
    assert set(new_units[test]) == {"r1_late", "r2_late"}
    assert set(new_units[~test]) == {"r1_early", "r2_early"}
    predictive_maintenance.check_no_overlap(new_units, test)


def test_predict_recordings_averages_windows():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    y = np.array(["h", "h", "f", "f"])
    units = np.array(["u1", "u1", "u2", "u2"])
    actual, predicted = predictive_maintenance.predict_recordings(probabilities, np.array(["h", "f"]), y, units)
    assert actual == ["h", "f"]
    assert predicted == ["h", "f"]  # u1 mean [0.55, 0.45]; u2 mean [0.45, 0.55]


def test_predict_recordings_refuses_mixed_labels():
    with pytest.raises(ValueError, match="Mixed labels"):
        predictive_maintenance.predict_recordings(np.ones((2, 2)) / 2, np.array(["h", "f"]), np.array(["h", "f"]),
                               np.array(["u1", "u1"]))


def test_classification_metrics_known_case():
    m = predictive_maintenance.classification_metrics(["h", "h", "f", "f"], ["h", "f", "f", "f"], ["h", "f"])
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class_recall"] == {"h": 0.5, "f": 1.0}


def test_sensor_fair_test_learns_a_clear_signal_without_leaking():
    rng = np.random.default_rng(613)
    rows = []
    for group in range(4):
        for k, label in enumerate(["healthy", "fault"]):
            for window in range(6):
                rows.append({"unit": f"g{group}_{label}", "group": group, "class": label,
                             "x1": k * 3 + rng.normal(0, 0.3), "x2": rng.normal(0, 1)})
    table = pd.DataFrame(rows)
    folds = predictive_maintenance.fair_test_folds(table["group"], [[g] for g in range(4)])
    result = predictive_maintenance.sensor_fair_test(table, {"both": ["x1", "x2"]}, ["healthy", "fault"], folds, "group")
    assert result["score"]["test_units"] == 8
    assert result["score"]["accuracy"] == 1.0
    assert result["majority_control"]["balanced_accuracy"] == 0.5
    assert len(result["chosen_per_round"]) == 4


# ========================================================================================
# Feature extraction and prediction routes
# ========================================================================================

def synthetic_aalto_csv(path, seconds=13, rpm=600):
    n = seconds * predictive_maintenance.AALTO_RATE
    t = np.arange(n) / predictive_maintenance.AALTO_RATE
    rng = np.random.default_rng(613)
    data = {"time": t, "enc1_ang": (360 * rpm / 60 * t) % 360}
    for c in predictive_maintenance.AALTO_ACC:
        data[c] = np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.1, n)
    for c in predictive_maintenance.AALTO_TORQUE:
        data[c] = 5 + rng.normal(0, 0.05, n)
    pd.DataFrame(data).to_csv(path)
    return path


def test_aalto_features_shape_matches_the_model_input(tmp_path):
    features, order_names = predictive_maintenance.aalto_run_features(synthetic_aalto_csv(tmp_path / "run.csv"))
    assert len(features) == 3  # 13 s - 10 s skipped = 3 one-second windows
    assert features.shape[1] == 100  # 4 acc x 12 + 4 acc x 7 orders + 2 torque x 12
    assert len(order_names) == 28
    assert np.isfinite(features.to_numpy()).all()


def test_short_aalto_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="shorter than"):
        predictive_maintenance.aalto_run_features(synthetic_aalto_csv(tmp_path / "short.csv", seconds=10))


def test_aalto_prediction_uses_gear_checkpoint(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier

    path = synthetic_aalto_csv(tmp_path / "run.csv")
    model_path = tmp_path / "grader_gear_sensor.joblib"
    monkeypatch.setattr(predictive_maintenance, "GRADER_GEAR_MODEL", model_path)
    monkeypatch.setattr(predictive_maintenance, "TRUSTED_MODELS", tmp_path / "trusted_sha256.json")
    with pytest.raises(FileNotFoundError, match="No Grader gear sensor model"):
        predictive_maintenance.predict(path)

    features, _ = predictive_maintenance.aalto_run_features(path)
    estimator = DummyClassifier(strategy="constant", constant="healthy")
    estimator.fit(features.to_numpy(), ["healthy", "healthy", "tooth_damage"])
    checkpoint = {"model": estimator, "features": list(features.columns), "model_name": "Test classifier"}
    predictive_maintenance.save_trusted_joblib(checkpoint, model_path)

    answer = predictive_maintenance.predict(path)
    assert answer["condition"] == "healthy"
    assert set(answer["all classes"]) == {"healthy", "tooth_damage"}
    assert answer["model used"] == "Test classifier"

    checkpoint["features"] = ["unavailable_feature"]
    predictive_maintenance.save_trusted_joblib(checkpoint, model_path)
    with pytest.raises(ValueError, match="Checkpoint features"):
        predictive_maintenance.predict(path)


MODELS = ROOT / "Processed_Data" / "models"


RAW = ROOT / "Raw_Data"


@needs(RAW / "uottawa_bearing" / "I-C-1.mat", MODELS / "pump_sensor.joblib", MODELS / "trusted_sha256.json")
def test_pump_sensor_route():
    answer = predictive_maintenance.predict(RAW / "uottawa_bearing" / "I-C-1.mat")
    assert answer["machine"] == "Pump" and answer["condition"] == "inner"


AALTO_HEALTHY = RAW / "aalto_gear" / "3012Hz" / "250RPM" / "0_12Nm" / "Healthy" / "GP1.csv"


@needs(AALTO_HEALTHY, MODELS / "grader_gear_sensor.joblib", MODELS / "trusted_sha256.json")
def test_grader_gear_route():
    answer = predictive_maintenance.predict(AALTO_HEALTHY)
    assert answer["machine"] == "Grader" and answer["condition"] == "healthy"
    assert set(answer["all classes"]) == {"healthy", "tooth_damage"}
    assert "not the slewing bearing" in answer["task"]


@pytest.mark.parametrize(("status", "exit_code"), [("PASS", None), ("FAIL", 1), ("no local sample", 2)])
def test_route_check_exit_status(monkeypatch, status, exit_code):
    monkeypatch.setattr("sys.argv", ["predictive_maintenance.py", "route-check"])
    monkeypatch.setattr(predictive_maintenance, "verify_routes", lambda: {"routes": [{"status": status}], "input_checks": []})
    if exit_code is None:
        assert predictive_maintenance.main() is None
    else:
        with pytest.raises(SystemExit) as error:
            predictive_maintenance.main()
        assert error.value.code == exit_code


def test_route_check_preserves_archived_results(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    local = tmp_path / "local"
    archived_report = archive / "deployment" / "route_check.json"
    archived_report.parent.mkdir(parents=True)
    archived_report.write_text("archived")
    monkeypatch.setattr(predictive_maintenance, "RESULTS", archive)
    monkeypatch.setattr(predictive_maintenance, "PROCESSED", local)
    monkeypatch.setattr(predictive_maintenance, "MODELS", tmp_path / "models")
    monkeypatch.setattr(predictive_maintenance, "AALTO", tmp_path / "aalto")
    samples = {machine: tmp_path / f"{machine}.csv" for machine in predictive_maintenance.MACHINES}
    monkeypatch.setattr(predictive_maintenance, "route_samples", lambda: (samples, dict.fromkeys(samples, "healthy"), {}))
    result = predictive_maintenance.verify_routes()
    assert all(row["status"] == "no local sample" for row in result["routes"])
    assert archived_report.read_text() == "archived"
    assert (local / "deployment" / "route_check.json").is_file()


def test_image_model_cache_reuses_weights_and_invalidates_replaced_file(isolated_models, monkeypatch):
    import os

    import torch
    from torch import nn

    path = isolated_models / "pump_images.pt"
    path.write_bytes(b"first")
    loaded = []
    predictive_maintenance.trust_model(path)

    def read_weights(file, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only
        loaded.append(file.read())
        return {}

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 4)

        def load_state_dict(self, state_dict, *args, **kwargs):
            assert state_dict == {}

    monkeypatch.setattr(torch, "load", read_weights)
    monkeypatch.setattr("torchvision.models.resnet18", lambda weights: TinyModel())
    monkeypatch.setattr(predictive_maintenance, "load_images", lambda *args: None)
    monkeypatch.setattr(predictive_maintenance, "image_probabilities", lambda *args: np.array([[.1, .2, .3, .4]]))
    sample = isolated_models / "example.png"
    try:
        first = predictive_maintenance.image_probability(sample, "pump")
        assert predictive_maintenance.image_probability(sample, "pump") == first
        assert loaded == [b"first"]
        stat = path.stat()
        replacement = path.with_suffix(".tmp")
        replacement.write_bytes(b"other")
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(path)
        with pytest.raises(ValueError, match="has changed since it was trusted"):
            predictive_maintenance.image_probability(sample, "pump")
        assert loaded == [b"first"]
        with pytest.raises(ValueError, match="Previously trusted checkpoints have changed"):
            predictive_maintenance.trust_models()
        predictive_maintenance.trust_model(path)  # operator explicitly verifies and registers the replacement
        assert predictive_maintenance.image_probability(sample, "pump") == first
        assert loaded == [b"first", b"other"]
        predictive_maintenance.TRUSTED_MODELS.unlink()
        with pytest.raises(ValueError, match="not in the trusted model list"):
            predictive_maintenance.image_probability(sample, "pump")
        path.unlink()
        with pytest.raises(FileNotFoundError, match="No trained image model"):
            predictive_maintenance.image_probability(sample, "pump")
    finally:
        predictive_maintenance.cached_image_model.cache_clear()


# ========================================================================================
# Input identification and ambiguous equipment
# ========================================================================================

def write_csv(path, header, rows=2):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(["0.1"] * len(header)) for _ in range(rows)) + "\n",
                    encoding="utf-8")
    return path


def test_just_csv_is_grader(tmp_path):
    source, machines, details = predictive_maintenance.identify(write_csv(tmp_path / "x.csv", predictive_maintenance.JUST_CHANNELS))
    assert source == "JUST slewing bearing" and machines == ["grader"] and not details["mixed_schema"]


def test_mcc5_csv_is_drill_rig(tmp_path):
    source, machines, _ = predictive_maintenance.identify(write_csv(tmp_path / "x.csv", predictive_maintenance.MCC5_CHANNELS))
    assert source == "MCC5-THU gearbox" and machines == ["drill_rig"]


def test_aalto_csv_is_grader_gear_task(tmp_path):
    header = ["", "time", "enc1_ang", "enc2_ang", "acc1", "acc2", "acc3", "acc4", "Torq1", "Torq2"]
    source, machines, _ = predictive_maintenance.identify(write_csv(tmp_path / "run.csv", header))
    assert source.startswith("Aalto") and machines == ["grader"]


def test_two_schemas_in_one_csv_are_flagged(tmp_path):
    _, machines, details = predictive_maintenance.identify(write_csv(tmp_path / "x.csv", predictive_maintenance.JUST_CHANNELS + predictive_maintenance.MCC5_CHANNELS))
    assert set(machines) == {"grader", "drill_rig"} and details["mixed_schema"]


def test_cora_file_is_ambiguous_across_four_machines(tmp_path):
    path = tmp_path / "W50_F50_S_Vx.csv"
    path.write_text("0.25\n0.5\n", encoding="utf-8")
    source, machines, details = predictive_maintenance.identify(path)
    assert source.startswith("CORA") and machines == predictive_maintenance.CORA and details["speed"] == "F50"
    answer = predictive_maintenance.predict(path)
    assert answer["status"] == "Ambiguous Equipment"


@pytest.mark.parametrize("name, text", [("other_name.csv", "0.25\n"), ("W50_F50_S_Vx.csv", "abc\n"),
                                        ("x.csv", "a,b,c\n1,2,3\n")])
def test_unknown_csv_is_rejected(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        predictive_maintenance.identify(path)


def test_unsupported_file_is_rejected():
    with pytest.raises(ValueError, match="none of the known dataset formats"):
        predictive_maintenance.predict(ROOT / "README.md")


def test_wrong_machine_is_refused(tmp_path):
    path = write_csv(tmp_path / "x.csv", predictive_maintenance.JUST_CHANNELS)
    with pytest.raises(ValueError, match="does not belong to pump"):
        predictive_maintenance.predict(path, "pump")


def test_blank_and_unindexed_images(tmp_path):
    import cv2

    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((224, 224, 3), 128, np.uint8))
    with pytest.raises(ValueError, match="no spatial detail"):
        predictive_maintenance.identify(blank)
    render = tmp_path / "render.png"
    cv2.imwrite(str(render), np.random.default_rng(613).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="provenance could not be verified"):
        predictive_maintenance.identify(render)


def test_event_frame_needs_a_mapping(tmp_path):
    import cv2

    frame = np.zeros((96, 128, 3), np.uint8)
    frame[10:40, 20:60, 1:] = 200  # blue channel stays empty, as in the event-camera exports
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)
    source, machines, _ = predictive_maintenance.identify(path)
    assert source.startswith("event-camera") and set(machines) == {"pump", "haul_truck"}


@pytest.mark.parametrize("content", [b"", b"not a MAT file", b"X" * 130])
def test_corrupt_mat_reports_a_readable_error(tmp_path, content):
    path = tmp_path / "broken.mat"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Invalid or unsupported MAT recording"):
        predictive_maintenance.identify(path)


# ========================================================================================
# Checkpoint trust and model loading
# ========================================================================================

def save(models, name, value):
    path = models / name
    joblib.dump(value, path)
    return path


def test_untrusted_model_is_refused(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    with pytest.raises(ValueError, match="not in the trusted model list"):
        predictive_maintenance.load_trusted_joblib(path)


def test_trusted_model_loads_and_a_changed_one_is_refused(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    predictive_maintenance.trust_model(path)
    assert predictive_maintenance.load_trusted_joblib(path) == {"x": 1}
    with path.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="has changed since it was trusted"):
        predictive_maintenance.load_trusted_joblib(path)


def test_trust_models_covers_only_models_predict_uses(isolated_models):
    save(isolated_models, "pump_sensor.joblib", 1)
    save(isolated_models, "grader_gear_sensor.joblib", 2)
    save(isolated_models, "someone_elses_model.joblib", 3)
    (isolated_models / "pump_images.pt").write_bytes(b"test weights")
    predictive_maintenance.trust_models()
    assert set(predictive_maintenance.trusted_hashes()) == {"pump_sensor.joblib", "grader_gear_sensor.joblib", "pump_images.pt"}


def test_missing_gear_model_gives_a_clear_message(isolated_models, tmp_path):
    header = ["", "time", "enc1_ang", "acc1", "acc2", "acc3", "acc4", "Torq1", "Torq2"]
    path = tmp_path / "run.csv"
    path.write_text(",".join(header) + "\n" + ",".join(["0"] * len(header)) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="grader-gear train-final"):
        predictive_maintenance.predict(path, "grader")


def test_other_grader_photo_export_needs_its_hash_typed(isolated_models, tmp_path, monkeypatch):
    archive = tmp_path / "export.zip"
    archive.write_bytes(b"not the evaluated export")
    monkeypatch.setattr(predictive_maintenance, "TOOTH_ZIP", archive)
    with pytest.raises(ValueError, match="--accept-archive"):
        predictive_maintenance.run_train_grader_images()
    with pytest.raises(ValueError, match="--accept-archive"):
        predictive_maintenance.run_train_grader_images(accept_archive="0" * 64)


def test_cached_model_detects_replacement_with_same_size_and_mtime(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    predictive_maintenance.trust_model(path)
    assert predictive_maintenance.load_trusted_joblib(path) == {"x": 1}
    stat = path.stat()
    replacement = save(isolated_models, "replacement.joblib", {"x": 2})
    assert replacement.stat().st_size == stat.st_size
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(ValueError, match="has changed since it was trusted"):
        predictive_maintenance.load_trusted_joblib(path)


@pytest.mark.parametrize("manifest", ["{", "[]"])
def test_invalid_trust_manifest_is_rejected(isolated_models, manifest):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    predictive_maintenance.TRUSTED_MODELS.write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="trusted_sha256.json"):
        predictive_maintenance.load_trusted_joblib(path)


def test_unregistered_and_corrupt_image_checkpoint(isolated_models, monkeypatch):
    import torch

    path = isolated_models / "pump_images.pt"
    path.write_bytes(b"")
    calls = []
    original = torch.load

    def checked_load(stream, **options):
        calls.append(options)
        return original(stream, **options)

    monkeypatch.setattr(torch, "load", checked_load)
    with pytest.raises(ValueError, match="not in the trusted model list"):
        predictive_maintenance.image_probability("unused.png", "pump")
    assert not calls
    predictive_maintenance.trust_model(path)
    with pytest.raises(ValueError, match="not a readable weights-only checkpoint"):
        predictive_maintenance.image_probability("unused.png", "pump")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


# ========================================================================================
# Upload interface and request validation
# ========================================================================================

@pytest.fixture(scope="module")
def server():
    srv = predictive_maintenance.upload_server(0)
    port = srv.server_port
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield port
    srv.shutdown()
    srv.server_close()


def request(port, method="GET", body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest(method, "/", skip_host=True, skip_accept_encoding=True)
    for key, value in ({"Host": f"127.0.0.1:{port}"} | (headers or {})).items():
        if value is not None:
            connection.putheader(key, value)
    connection.endheaders(body or None)
    response = connection.getresponse()
    length = response.getheader("Content-Length")
    assert length is not None
    content = response.read()
    assert len(content) == int(length)
    text = content.decode("utf-8", "replace")
    connection.close()
    return response.status, text


def form(filename, content, machine=""):
    boundary = "testboundary613"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="machine"\r\n\r\n{machine}\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n",
             f"--{boundary}--\r\n".encode()]
    body = b"".join(parts)
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))}


def test_page_opens(server):
    status, text = request(server)
    assert status == 200 and "<form" in text


def test_other_host_names_are_refused(server):
    assert request(server, headers={"Host": f"evil.example:{server}"})[0] == 403


def test_uploads_from_other_websites_are_refused(server):
    body, headers = form("x.csv", b"a,b\n1,2\n")
    assert request(server, "POST", body, headers | {"Origin": "https://evil.example"})[0] == 403


def test_same_page_origin_is_accepted(server):
    body, headers = form("x.csv", b"a,b\n1,2\n")
    status, text = request(server, "POST", body, headers | {"Origin": f"http://127.0.0.1:{server}"})
    assert status == 200 and "No prediction" in text


@pytest.mark.parametrize("headers, status", [
    ({"Content-Type": "multipart/form-data; boundary=x", "Content-Length": "-5"}, 400),
    ({"Content-Type": "multipart/form-data; boundary=x"}, 411),
    ({"Content-Type": "application/x-www-form-urlencoded", "Content-Length": "3"}, 400),
])
def test_malformed_requests_are_refused(server, headers, status):
    body = b"a=b" if headers.get("Content-Length") == "3" else b""
    assert request(server, "POST", body, headers)[0] == status


@pytest.mark.parametrize("filename", ["..", ".hidden.csv", "a<b>.csv"])
def test_unsafe_file_names_get_an_error_page(server, filename):
    body, headers = form(filename, b"0.1\n")
    status, text = request(server, "POST", body, headers)
    assert status == 200 and "No prediction" in text


def test_damaged_file_still_gets_an_answer(server):
    body, headers = form("I-C-1.mat", b"not really a mat file")
    status, text = request(server, "POST", body, headers)
    assert status == 200 and "No prediction" in text


def test_upload_name_rule():
    assert predictive_maintenance.upload_name_ok("W50_F50_S_Vx.csv")
    assert predictive_maintenance.upload_name_ok("20221208-O-2rpm-0N-1.csv")
    assert predictive_maintenance.upload_name_ok("pump_outer_20Hz_15_-30_t000.png")
    assert not predictive_maintenance.upload_name_ok("..")
    assert not predictive_maintenance.upload_name_ok("a/b.csv")


def test_truncated_upload_is_rejected(server):
    body, headers = form("sample.csv", b"1,2")
    connection = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    try:
        connection.request("POST", "/", body, headers | {"Content-Length": str(len(body) + 10)})
        connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        assert response.status == 400
        assert b"ended before the full file arrived" in response.read()
    finally:
        connection.close()


def test_missing_multipart_closing_boundary_is_rejected(server):
    body, headers = form("sample.csv", b"1,2")
    body = body.rsplit(b"--testboundary613--", 1)[0]
    status, text = request(server, "POST", body, headers | {"Content-Length": str(len(body))})
    assert status == 400
    assert "incomplete or malformed" in text


# ========================================================================================
# Dashboard, prediction history and demonstration
# ========================================================================================

@pytest.fixture
def history(tmp_path, monkeypatch):
    path = tmp_path / "prediction_history.jsonl"
    monkeypatch.setattr(predictive_maintenance, "HISTORY", path)
    return path


def get(port, path, host=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
    connection.putheader("Host", host or f"127.0.0.1:{port}")
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), body.decode("utf-8", "replace")


def sample(tmp_path, name="sample.csv", content=b"1,2\n"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_history_roundtrip_keeps_no_file_contents(history, tmp_path):
    path = sample(tmp_path, content=b"secret-looking contents")
    answer = {"machine": "Pump", "condition": "inner", "model probability": 0.9, "model used": "pump_sensor.joblib"}
    predictive_maintenance.record_prediction(path, "command line", None, answer)
    predictive_maintenance.record_prediction(path, "upload page", "pump", problem=ValueError("too short"))
    predictive_maintenance.record_prediction(path, "upload page", None, {"status": "Ambiguous Equipment", "candidates": ["Pump", "Haul Truck"]})
    entries, skipped = predictive_maintenance.read_history()
    assert skipped == 0 and [e["status"] for e in entries] == ["ambiguous", "rejected", "prediction"]
    newest_first_last = entries[-1]
    assert newest_first_last["machine"] == "Pump" and newest_first_last["probability"] == 0.9
    assert newest_first_last["mapping"] == "automatic" and newest_first_last["bytes"] == 23
    assert entries[1]["message"] == "too short" and entries[0]["message"] == "Pump, Haul Truck"
    assert b"secret-looking" not in history.read_bytes()


def test_corrupt_history_lines_are_skipped(history, tmp_path):
    predictive_maintenance.record_prediction(sample(tmp_path), "command line", None, {"machine": "Grader", "condition": "healthy"})
    with history.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n[1, 2]\n\n")
    entries, skipped = predictive_maintenance.read_history()
    assert len(entries) == 1 and skipped == 2


def test_history_csv_neutralises_formulas(history, tmp_path):
    predictive_maintenance.record_prediction(sample(tmp_path, "=cmd.csv"), "upload page", None, problem=ValueError("+1 bad"))
    rows = list(csv.reader(io.StringIO(predictive_maintenance.history_csv())))
    assert rows[0] == predictive_maintenance.HISTORY_FIELDS
    record = dict(zip(rows[0], rows[1]))
    assert record["file"] == "'=cmd.csv" and record["message"] == "'+1 bad"


def test_dashboard_escapes_history_text(history, tmp_path):
    predictive_maintenance.record_prediction(sample(tmp_path, "x.csv"), "upload page", None, problem=ValueError("<script>alert(1)</script>"))
    page = predictive_maintenance.dashboard_page().decode("utf-8")
    assert "<script>alert" not in page and "&lt;script&gt;alert" in page
    assert "Fair-test accuracy" in page and '<div class="v">1</div><div class="l">files checked' in page


def test_dashboard_and_csv_are_served(server):
    status, kind, text = get(server, "/dashboard")
    assert status == 200 and "Prediction routes" in text
    status, kind, text = get(server, "/history.csv")
    assert status == 200 and kind.startswith("text/csv") and text.startswith("time_utc,")
    assert get(server, "/nothing-here")[0] == 404


@pytest.mark.parametrize("path", ["/dashboard", "/history.csv"])
def test_dashboard_refuses_other_host_names(server, path):
    assert get(server, path, host=f"evil.example:{server}")[0] == 403


def post_form(port, path, filename, content, machine=""):
    boundary = "testboundary613"
    crlf = "\r\n"
    body = (f'--{boundary}{crlf}Content-Disposition: form-data; name="machine"{crlf}{crlf}{machine}{crlf}'
            f'--{boundary}{crlf}Content-Disposition: form-data; name="file"; filename="{filename}"{crlf}'
            f'Content-Type: application/octet-stream{crlf}{crlf}').encode() + content + f'{crlf}--{boundary}--{crlf}'.encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("POST", path, body, {"Host": f"127.0.0.1:{port}", "Content-Length": str(len(body)),
                                            "Content-Type": f"multipart/form-data; boundary={boundary}"})
    response = connection.getresponse()
    text = response.read().decode("utf-8", "replace")
    connection.close()
    return response.status, text


def test_dashboard_upload_shows_result_and_records_it(server, history):
    status, text = post_form(server, "/dashboard", "notes.txt", b"hello")
    assert status == 200 and "Fleet view" in text and 'class="result no"' in text and "notes.txt" in text
    entry = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["file"] == "notes.txt" and entry["status"] == "rejected"


def test_dashboard_upload_from_other_website_is_refused(server):
    connection = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    connection.request("POST", "/dashboard", b"x", {"Host": f"127.0.0.1:{server}", "Origin": "https://evil.example",
                                                     "Content-Length": "1", "Content-Type": "multipart/form-data; boundary=x"})
    assert connection.getresponse().status == 403
    connection.close()


def test_answer_cards_escape_and_show_classes():
    card = predictive_maintenance.answer_card("<b>.csv", {"file format": "x", "machine": "Pump", "condition": "<i>inner",
                                       "model probability": 0.9, "model used": "RF",
                                       "all classes": {"inner": 0.9, "outer": 0.1}})
    assert "<b>" not in card and "<i>" not in card and "&lt;i&gt;inner" in card and "90.0%" in card
    amb = predictive_maintenance.answer_card("a.csv", {"status": "Ambiguous Equipment", "candidates": ["Dozer"], "file format": "CORA"})
    assert "Which machine" in amb and "Dozer" in amb
    assert 'class="result no"' in predictive_maintenance.answer_card("a.csv", problem=ValueError("bad"))


def test_upload_is_recorded(server, history):
    boundary = "testboundary613"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="machine"\r\n\r\n\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.csv"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\na,b\r\n--{boundary}--\r\n").encode()
    connection = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    connection.request("POST", "/", body, {"Host": f"127.0.0.1:{server}", "Content-Length": str(len(body)),
                                           "Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert connection.getresponse().status == 200
    connection.close()
    entry = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["interface"] == "upload page" and entry["file"] == "x.csv" and entry["status"] == "rejected"


def test_demo_steps_cover_the_script():
    steps = predictive_maintenance.demo_steps()
    assert [s[0] for s in steps] == sorted(s[0] for s in steps)
    assert {s[0] for s in steps} == {2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert steps[-1][3] == "rejected"


def test_dashboard_renders_when_local_registry_is_malformed(server, history, isolated_models):
    predictive_maintenance.TRUSTED_MODELS.write_text("[]")
    status, _, body = get(server, "/dashboard")
    assert status == 200 and "Prediction routes" in body
    with pytest.raises(ValueError, match="trusted_sha256.json"):
        predictive_maintenance.trusted_hashes()


def test_history_csv_neutralises_whitespace_prefixed_formulas(history, tmp_path):
    predictive_maintenance.record_prediction(sample(tmp_path), "upload page", None, problem=ValueError("  =1+1"))
    rows = list(csv.DictReader(io.StringIO(predictive_maintenance.history_csv())))
    assert rows[0]["message"] == "'  =1+1"


def test_upload_to_unknown_route_creates_no_history(server, history):
    status, _ = post_form(server, "/missing", "notes.txt", b"hello")
    assert status == 404
    assert not history.exists()


def test_demo_steps_reuse_indexed_images_and_deployment_sensor_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(predictive_maintenance, "IMAGES", tmp_path / "Images")
    pump_frame = tmp_path / "outer.png"
    drill_sample = tmp_path / "gear_pitting_recording.csv"
    grader_sample = tmp_path / "bearing_outer_ring.csv"
    monkeypatch.setattr(
        predictive_maintenance, "route_samples",
        lambda: ({"drill_rig": drill_sample, "grader": grader_sample}, {},
                 {"pump": {"outer": pump_frame}}),
    )
    steps = predictive_maintenance.demo_steps()
    assert [(path, expected) for step, path, _, expected in steps if step == 3] == [
        (pump_frame, "Ambiguous Equipment"), (pump_frame, ("Pump", "outer"))]
    assert next(path for step, path, _, _ in steps if step == 7) == drill_sample
    assert next(path for step, path, _, _ in steps if step == 8) == grader_sample


def test_demo_accepts_automatic_pump_only_when_image_source_is_identified(tmp_path, monkeypatch):
    monkeypatch.setattr(predictive_maintenance, "IMAGES", tmp_path)
    frame = tmp_path / "pump" / "outer" / "pump_outer_20Hz_15_-30_t000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"indexed image fixture")
    monkeypatch.setattr(predictive_maintenance, "identify", lambda path: ("event frame", ["pump"], {"indexed_image": True}))
    automatic = next(expected for step, _, mapping, expected in predictive_maintenance.demo_steps()
                     if step == 3 and mapping is None)
    assert automatic == ("Pump", "outer")
    monkeypatch.setattr(predictive_maintenance, "identify", lambda path: ("event frame", ["pump", "haul_truck"], {}))
    ambiguous = next(expected for step, _, mapping, expected in predictive_maintenance.demo_steps()
                     if step == 3 and mapping is None)
    assert ambiguous == "Ambiguous Equipment"


# ========================================================================================
# Archived results and final regression checks
# ========================================================================================

def metric_blocks(value, path=""):
    if isinstance(value, dict):
        if "confusion_matrix" in value and "accuracy" in value:
            yield path, value
        for key, item in value.items():
            yield from metric_blocks(item, f"{path}/{key}")
    elif isinstance(value, list):
        for n, item in enumerate(value):
            yield from metric_blocks(item, f"{path}[{n}]")


def saved_blocks():
    for file in sorted((ROOT / "Results").rglob("*.json")):
        for where, block in metric_blocks(json.loads(file.read_text(encoding="utf-8"))):
            yield pytest.param(block, id=f"{file.relative_to(ROOT / 'Results').as_posix()}{where}")


@pytest.mark.parametrize("block", list(saved_blocks()))
def test_saved_confusion_matrix_matches_reported_metrics(block):
    cm = np.asarray(block["confusion_matrix"], dtype=float)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1] and (cm >= 0).all()
    total, true, predicted, hits = cm.sum(), cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    assert block["accuracy"] == pytest.approx(hits.sum() / total, abs=6e-4)
    if "test_units" in block:
        assert block["test_units"] == total
    if "balanced_accuracy" in block:
        assert block["balanced_accuracy"] == pytest.approx(np.mean(hits[true > 0] / true[true > 0]), abs=6e-4)
    if "macro_f1" in block:
        precision = np.divide(hits, predicted, out=np.zeros_like(hits), where=predicted > 0)
        recall = np.divide(hits, true, out=np.zeros_like(hits), where=true > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(hits),
                       where=precision + recall > 0)
        assert block["macro_f1"] == pytest.approx(f1.mean(), abs=6e-4)


def test_there_are_saved_results_to_check():
    assert len(list(saved_blocks())) > 100


def test_scaling_is_fitted_inside_each_model_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    for name, factory in predictive_maintenance.sensor_candidates().items():
        model = factory()
        if name in ("SVM (RBF)", "Logistic regression"):
            assert isinstance(model, Pipeline) and isinstance(model.steps[0][1], StandardScaler)
        else:
            assert not isinstance(model, Pipeline)  # trees need no scaling


def test_model_selection_never_sees_test_recordings(monkeypatch):
    rng = np.random.default_rng(613)
    table = pd.DataFrame([{"unit": f"g{g}_{label}", "group": g, "class": label,
                           "x": k + rng.normal(0, 0.2)}
                          for g in range(3) for k, label in enumerate(["a", "b"]) for _ in range(4)])
    folds = predictive_maintenance.fair_test_folds(table["group"], [[0], [1], [2]])
    seen = []
    real = predictive_maintenance.choose_model

    def spy(table_, train_idx, *args):
        seen.append(set(table_["unit"].to_numpy()[train_idx]))
        return real(table_, train_idx, *args)

    monkeypatch.setattr(predictive_maintenance, "choose_model", spy)
    predictive_maintenance.sensor_fair_test(table, {"x": ["x"]}, ["a", "b"], folds, "group")
    for test, train_units in zip(folds, seen):
        assert not train_units & set(table.loc[test, "unit"])


def test_image_cache_keeps_routes_apart_and_holds_at_most_two_models(isolated_models, monkeypatch):
    import torch
    from torch import nn

    loads = []

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 4)

        def load_state_dict(self, state_dict, *args, **kwargs):
            self.source = state_dict["source"]

    def read_weights(file, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only
        source = file.read().decode() + "_images.pt"
        loads.append(source)
        return {"source": source}

    monkeypatch.setattr(torch, "load", read_weights)
    monkeypatch.setattr("torchvision.models.resnet18", lambda weights: TinyModel())
    monkeypatch.setattr(predictive_maintenance, "load_images", lambda *args: None)
    used = []
    monkeypatch.setattr(predictive_maintenance, "image_probabilities", lambda model, x: used.append(model.source) or
                        np.array([[0.1, 0.2, 0.3, 0.4]]))
    predictive_maintenance.cached_image_model.cache_clear()
    for machine in ("pump", "haul_truck", "drill_rig"):
        path = isolated_models / f"{machine}_images.pt"
        path.write_bytes(machine.encode())
        predictive_maintenance.trust_model(path)
    sample = isolated_models / "frame.png"
    try:
        for machine in ("pump", "haul_truck", "pump", "haul_truck", "drill_rig", "pump"):
            predictive_maintenance.image_probability(sample, machine)
            assert predictive_maintenance.cached_image_model.cache_info().currsize <= 2
        assert used == [f"{m}_images.pt" for m in ("pump", "haul_truck", "pump", "haul_truck", "drill_rig", "pump")]
        # pump and haul_truck load once each; drill_rig evicts pump, so pump loads again
        assert loads == ["pump_images.pt", "haul_truck_images.pt", "drill_rig_images.pt", "pump_images.pt"]
    finally:
        predictive_maintenance.cached_image_model.cache_clear()


def test_square_wave_moments_crest_and_peak_to_peak():
    values = np.tile([3.0, -3.0], 500)
    result = predictive_maintenance.signal_diagnostics(values, 1000, "g")
    assert result["rms"] == pytest.approx(3)
    assert result["crest_factor"] == pytest.approx(1)
    assert result["peak_to_peak"] == pytest.approx(6)
    assert result["skewness"] == pytest.approx(0, abs=1e-12)
    assert result["excess_kurtosis"] == pytest.approx(-2)


def test_skewness_and_kurtosis_of_a_known_asymmetric_signal():
    values = np.array([0.0] * 9 + [10.0])  # Bernoulli(0.1) scaled: skew (1-2p)/sqrt(pq), excess kurt (1-6pq)/pq
    p, q = 0.1, 0.9
    result = predictive_maintenance.signal_diagnostics(values, 100, "g")
    assert result["skewness"] == pytest.approx((1 - 2 * p) / np.sqrt(p * q))
    assert result["excess_kurtosis"] == pytest.approx((1 - 6 * p * q) / (p * q))
    assert result["variance"] == pytest.approx(100 * p * q)


def test_welch_band_power_adds_up_to_the_variance():
    rng = np.random.default_rng(613)
    values = rng.normal(0, 2, 200_000)
    rate = 10_000
    result = predictive_maintenance.signal_diagnostics(values, rate, "g", bands=[(0, 1000), (1000, 5000), (0, 5000)])
    bands = result["band_energy"]
    assert bands[(0, 5000)] == pytest.approx(4, rel=0.03)
    assert bands[(0, 1000)] + bands[(1000, 5000)] == pytest.approx(bands[(0, 5000)], rel=1e-9)
    assert bands[(0, 1000)] == pytest.approx(0.8, rel=0.06)  # white noise: share = bandwidth / Nyquist


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_tiny_and_huge_signals_give_the_same_shape_statistics(scale):
    rng = np.random.default_rng(613)
    base = rng.normal(0, 1, 4096) + 0.3 * rng.exponential(1, 4096)
    reference = predictive_maintenance.signal_diagnostics(base, 1000, "g")
    scaled = predictive_maintenance.signal_diagnostics(base * scale, 1000, "g")
    for key in ("skewness", "excess_kurtosis", "crest_factor"):
        assert scaled[key] == pytest.approx(reference[key], rel=1e-9)
    assert scaled["rms"] == pytest.approx(reference["rms"] * scale, rel=1e-9)
    assert np.isfinite(scaled["psd"]).all()


def test_uods_trims_partial_blocks_the_same_way_for_every_format():
    rng = np.random.default_rng(613)
    data = rng.normal(0, 1, (42_000 + 5_000, 2))
    trimmed = predictive_maintenance.uods_array_features(data)
    whole = predictive_maintenance.uods_array_features(data[:42_000])
    pd.testing.assert_frame_equal(trimmed, whole)


@pytest.mark.parametrize("length,value,message", [(20_999, 0.0, "too short"), (42_000, np.nan, "invalid"),
                                                  (42_000, np.inf, "invalid")])
def test_uods_short_or_invalid_recordings_get_a_clear_message(length, value, message):
    data = np.zeros((length, 2))
    data[-1, 0] = value
    with pytest.raises(ValueError, match=message):
        predictive_maintenance.uods_array_features(data)
