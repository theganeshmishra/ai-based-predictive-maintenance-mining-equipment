"""Shared setup: import Notebook/pdm.py and keep tests away from the real model folder."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Notebook"))

import pdm  # noqa: E402


@pytest.fixture
def isolated_models(tmp_path, monkeypatch):
    """Point MODELS and the trust list at an empty temporary folder."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(pdm, "MODELS", models)
    monkeypatch.setattr(pdm, "TRUSTED_MODELS", models / "trusted_sha256.json")
    monkeypatch.setattr(pdm, "GRADER_GEAR_MODEL", models / "grader_gear_sensor.joblib")
    return models


def needs(*paths):
    """Skip a test when local data it reads is not on this computer (data is never in git)."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"local data missing: {', '.join(missing)}")
