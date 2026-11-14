"""Saved prediction checkpoints require a trusted SHA-256."""
import os

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
    (isolated_models / "pump_images.pt").write_bytes(b"test weights")
    pdm.trust_models()
    assert set(pdm.trusted_hashes()) == {"pump_sensor.joblib", "grader_gear_sensor.joblib", "pump_images.pt"}


def test_missing_gear_model_gives_a_clear_message(isolated_models, tmp_path):
    header = ["", "time", "enc1_ang", "acc1", "acc2", "acc3", "acc4", "Torq1", "Torq2"]
    path = tmp_path / "run.csv"
    path.write_text(",".join(header) + "\n" + ",".join(["0"] * len(header)) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="grader-gear train-final"):
        pdm.predict(path, "grader")


def test_other_grader_photo_export_needs_its_hash_typed(isolated_models, tmp_path, monkeypatch):
    archive = tmp_path / "export.zip"
    archive.write_bytes(b"not the evaluated export")
    monkeypatch.setattr(pdm, "TOOTH_ZIP", archive)
    with pytest.raises(ValueError, match="--accept-archive"):
        pdm.run_train_grader_images()
    with pytest.raises(ValueError, match="--accept-archive"):
        pdm.run_train_grader_images(accept_archive="0" * 64)

def test_cached_model_detects_replacement_with_same_size_and_mtime(isolated_models):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    pdm.trust_model(path)
    assert pdm.load_trusted_joblib(path) == {"x": 1}
    stat = path.stat()
    replacement = save(isolated_models, "replacement.joblib", {"x": 2})
    assert replacement.stat().st_size == stat.st_size
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(ValueError, match="has changed since it was trusted"):
        pdm.load_trusted_joblib(path)


@pytest.mark.parametrize("manifest", ["{", "[]"])
def test_invalid_trust_manifest_is_rejected(isolated_models, manifest):
    path = save(isolated_models, "pump_sensor.joblib", {"x": 1})
    pdm.TRUSTED_MODELS.write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="trusted_sha256.json"):
        pdm.load_trusted_joblib(path)


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
        pdm.image_probability("unused.png", "pump")
    assert not calls
    pdm.trust_model(path)
    with pytest.raises(ValueError, match="not a readable weights-only checkpoint"):
        pdm.image_probability("unused.png", "pump")
    assert calls == [{"map_location": "cpu", "weights_only": True}]
