"""Feature extraction on synthetic signals, and the real prediction routes when local data and models exist."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs, pdm


def synthetic_aalto_csv(path, seconds=13, rpm=600):
    n = seconds * pdm.AALTO_RATE
    t = np.arange(n) / pdm.AALTO_RATE
    rng = np.random.default_rng(613)
    data = {"time": t, "enc1_ang": (360 * rpm / 60 * t) % 360}
    for c in pdm.AALTO_ACC:
        data[c] = np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.1, n)
    for c in pdm.AALTO_TORQUE:
        data[c] = 5 + rng.normal(0, 0.05, n)
    pd.DataFrame(data).to_csv(path)
    return path


def test_aalto_features_shape_matches_the_model_input(tmp_path):
    features, order_names = pdm.aalto_run_features(synthetic_aalto_csv(tmp_path / "run.csv"))
    assert len(features) == 3  # 13 s - 10 s skipped = 3 one-second windows
    assert features.shape[1] == 100  # 4 acc x 12 + 4 acc x 7 orders + 2 torque x 12
    assert len(order_names) == 28
    assert np.isfinite(features.to_numpy()).all()


def test_short_aalto_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="shorter than"):
        pdm.aalto_run_features(synthetic_aalto_csv(tmp_path / "short.csv", seconds=10))


def test_aalto_prediction_uses_gear_checkpoint(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier

    path = synthetic_aalto_csv(tmp_path / "run.csv")
    model_path = tmp_path / "grader_gear_sensor.joblib"
    monkeypatch.setattr(pdm, "GRADER_GEAR_MODEL", model_path)
    monkeypatch.setattr(pdm, "TRUSTED_MODELS", tmp_path / "trusted_sha256.json")
    with pytest.raises(FileNotFoundError, match="No Grader gear sensor model"):
        pdm.predict(path)

    features, _ = pdm.aalto_run_features(path)
    estimator = DummyClassifier(strategy="constant", constant="healthy")
    estimator.fit(features.to_numpy(), ["healthy", "healthy", "tooth_damage"])
    checkpoint = {"model": estimator, "features": list(features.columns), "model_name": "Test classifier"}
    pdm.save_trusted_joblib(checkpoint, model_path)

    answer = pdm.predict(path)
    assert answer["condition"] == "healthy"
    assert set(answer["all classes"]) == {"healthy", "tooth_damage"}
    assert answer["model used"] == "Test classifier"

    checkpoint["features"] = ["unavailable_feature"]
    pdm.save_trusted_joblib(checkpoint, model_path)
    with pytest.raises(ValueError, match="Checkpoint features"):
        pdm.predict(path)


MODELS = ROOT / "Processed_Data" / "models"
RAW = ROOT / "Raw_Data"


@needs(RAW / "uottawa_bearing" / "I-C-1.mat", MODELS / "pump_sensor.joblib", MODELS / "trusted_sha256.json")
def test_pump_sensor_route():
    answer = pdm.predict(RAW / "uottawa_bearing" / "I-C-1.mat")
    assert answer["machine"] == "Pump" and answer["condition"] == "inner"


AALTO_HEALTHY = RAW / "aalto_gear" / "3012Hz" / "250RPM" / "0_12Nm" / "Healthy" / "GP1.csv"


@needs(AALTO_HEALTHY, MODELS / "grader_gear_sensor.joblib", MODELS / "trusted_sha256.json")
def test_grader_gear_route():
    answer = pdm.predict(AALTO_HEALTHY)
    assert answer["machine"] == "Grader" and answer["condition"] == "healthy"
    assert set(answer["all classes"]) == {"healthy", "tooth_damage"}
    assert "not the slewing bearing" in answer["task"]
