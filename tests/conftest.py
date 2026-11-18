"""Shared setup: import Notebook/predictive_maintenance.py and keep tests away from the real model folder."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Notebook"))

import predictive_maintenance  # noqa: E402


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
