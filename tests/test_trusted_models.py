"""Saved .joblib models load only when their SHA-256 is on the local trust list."""
import joblib
import pytest

from conftest import pdm


def save(models, name, value):
    path = models / name
    joblib.dump(value, path)
    return path


def test_untrusted_model_is_refused(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    with pytest.raises(ValueError, match="not in the trusted model list"):
        pdm.load_trusted_joblib(path)


def test_trusted_model_loads_and_a_changed_one_is_refused(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    pdm.trust_model(path)
    assert pdm.load_trusted_joblib(path) == {"x": 1}
    with path.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="has changed since it was trusted"):
        pdm.load_trusted_joblib(path)


def test_trust_models_covers_only_models_predict_uses(isolated_models):
    save(isolated_models, "pump_sensor.joblib", 1)
    save(isolated_models, "grader_gear_sensor.joblib", 2)
    save(isolated_models, "someone_elses_model.joblib", 3)
    pdm.trust_models()
    assert set(pdm.trusted_hashes()) == {"pump_sensor.joblib", "grader_gear_sensor.joblib"}


def test_missing_gear_model_gives_a_clear_message(isolated_models, tmp_path):
    header = ["", "time", "enc1_ang", "acc1", "acc2", "acc3", "acc4", "Torq1", "Torq2"]
    path = tmp_path / "run.csv"
    path.write_text(",".join(header) + "\n" + ",".join(["0"] * len(header)) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="grader-gear train-final"):
        pdm.predict(path, "grader")
