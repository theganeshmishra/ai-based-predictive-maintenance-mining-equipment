"""Phase 1 final verification: saved metrics, leakage guards, cache bounds, diagnostics and UODS trimming."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, pdm


# --- saved results: every confusion matrix agrees with the metrics reported next to it ---------------

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


# --- leakage guards -----------------------------------------------------------------------------------

def test_scaling_is_fitted_inside_each_model_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    for name, factory in pdm.sensor_candidates().items():
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
    folds = pdm.fair_test_folds(table["group"], [[0], [1], [2]])
    seen = []
    real = pdm.choose_model

    def spy(table_, train_idx, *args):
        seen.append(set(table_["unit"].to_numpy()[train_idx]))
        return real(table_, train_idx, *args)

    monkeypatch.setattr(pdm, "choose_model", spy)
    pdm.sensor_fair_test(table, {"x": ["x"]}, ["a", "b"], folds, "group")
    for test, train_units in zip(folds, seen):
        assert not train_units & set(table.loc[test, "unit"])


# --- image model cache: routes kept apart, memory bounded --------------------------------------------

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
    monkeypatch.setattr(pdm, "load_images", lambda *args: None)
    used = []
    monkeypatch.setattr(pdm, "image_probabilities", lambda model, x: used.append(model.source) or
                        np.array([[0.1, 0.2, 0.3, 0.4]]))
    pdm.cached_image_model.cache_clear()
    for machine in ("pump", "haul_truck", "drill_rig"):
        path = isolated_models / f"{machine}_images.pt"
        path.write_bytes(machine.encode())
        pdm.trust_model(path)
    sample = isolated_models / "frame.png"
    try:
        for machine in ("pump", "haul_truck", "pump", "haul_truck", "drill_rig", "pump"):
            pdm.image_probability(sample, machine)
            assert pdm.cached_image_model.cache_info().currsize <= 2
        assert used == [f"{m}_images.pt" for m in ("pump", "haul_truck", "pump", "haul_truck", "drill_rig", "pump")]
        # pump and haul_truck load once each; drill_rig evicts pump, so pump loads again
        assert loads == ["pump_images.pt", "haul_truck_images.pt", "drill_rig_images.pt", "pump_images.pt"]
    finally:
        pdm.cached_image_model.cache_clear()


# --- signal diagnostics: analytic cases not covered by the sinusoid test -------------------------------

def test_square_wave_moments_crest_and_peak_to_peak():
    values = np.tile([3.0, -3.0], 500)
    result = pdm.signal_diagnostics(values, 1000, "g")
    assert result["rms"] == pytest.approx(3)
    assert result["crest_factor"] == pytest.approx(1)
    assert result["peak_to_peak"] == pytest.approx(6)
    assert result["skewness"] == pytest.approx(0, abs=1e-12)
    assert result["excess_kurtosis"] == pytest.approx(-2)


def test_skewness_and_kurtosis_of_a_known_asymmetric_signal():
    values = np.array([0.0] * 9 + [10.0])  # Bernoulli(0.1) scaled: skew (1-2p)/sqrt(pq), excess kurt (1-6pq)/pq
    p, q = 0.1, 0.9
    result = pdm.signal_diagnostics(values, 100, "g")
    assert result["skewness"] == pytest.approx((1 - 2 * p) / np.sqrt(p * q))
    assert result["excess_kurtosis"] == pytest.approx((1 - 6 * p * q) / (p * q))
    assert result["variance"] == pytest.approx(100 * p * q)


def test_welch_band_power_adds_up_to_the_variance():
    rng = np.random.default_rng(613)
    values = rng.normal(0, 2, 200_000)
    rate = 10_000
    result = pdm.signal_diagnostics(values, rate, "g", bands=[(0, 1000), (1000, 5000), (0, 5000)])
    bands = result["band_energy"]
    assert bands[(0, 5000)] == pytest.approx(4, rel=0.03)
    assert bands[(0, 1000)] + bands[(1000, 5000)] == pytest.approx(bands[(0, 5000)], rel=1e-9)
    assert bands[(0, 1000)] == pytest.approx(0.8, rel=0.06)  # white noise: share = bandwidth / Nyquist


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_tiny_and_huge_signals_give_the_same_shape_statistics(scale):
    rng = np.random.default_rng(613)
    base = rng.normal(0, 1, 4096) + 0.3 * rng.exponential(1, 4096)
    reference = pdm.signal_diagnostics(base, 1000, "g")
    scaled = pdm.signal_diagnostics(base * scale, 1000, "g")
    for key in ("skewness", "excess_kurtosis", "crest_factor"):
        assert scaled[key] == pytest.approx(reference[key], rel=1e-9)
    assert scaled["rms"] == pytest.approx(reference["rms"] * scale, rel=1e-9)
    assert np.isfinite(scaled["psd"]).all()


# --- UODS recordings that are not a whole number of 0.5 s blocks -------------------------------------

def test_uods_trims_partial_blocks_the_same_way_for_every_format():
    rng = np.random.default_rng(613)
    data = rng.normal(0, 1, (42_000 + 5_000, 2))
    trimmed = pdm.uods_array_features(data)
    whole = pdm.uods_array_features(data[:42_000])
    pd.testing.assert_frame_equal(trimmed, whole)


@pytest.mark.parametrize("length,value,message", [(20_999, 0.0, "too short"), (42_000, np.nan, "invalid"),
                                                  (42_000, np.inf, "invalid")])
def test_uods_short_or_invalid_recordings_get_a_clear_message(length, value, message):
    data = np.zeros((length, 2))
    data[-1, 0] = value
    with pytest.raises(ValueError, match=message):
        pdm.uods_array_features(data)
