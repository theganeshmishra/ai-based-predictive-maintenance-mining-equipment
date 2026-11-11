import numpy as np
import pytest

from conftest import pdm


@pytest.mark.parametrize("size", [999, 1000])
def test_sinusoid_amplitude_and_diagnostics(size):
    t = np.arange(size) / size
    values = 2 * np.sin(2 * np.pi * 50 * t)
    result = pdm.signal_diagnostics(values, size, "m/s²", rpm=1500, bands=[(40, 60), (100, 150)])
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
    result = pdm.signal_diagnostics(np.zeros(32), 1000, "m/s²")
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
        pdm.signal_diagnostics(samples, rate, units, **options)


def test_diagnostic_figure_is_written(tmp_path):
    path = tmp_path / "waveform.png"
    result = pdm.signal_diagnostics(np.arange(100), 500, "m/s²", rpm=600)
    pdm.save_signal_diagnostic_plot(result, path)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_controlled_perturbations_are_repeatable_and_have_expected_lengths():
    from phase1_validation import perturb_signal

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
        pdm.ottawa_signal_features(signal)


def test_short_mcc5_has_clear_error():
    from io import StringIO

    contents = ",".join(pdm.MCC5_CHANNELS) + "\n" + ",".join(["1"] * len(pdm.MCC5_CHANNELS))
    with pytest.raises(ValueError, match="at least 12,800 samples"):
        pdm.mcc5_recording(StringIO(contents))
