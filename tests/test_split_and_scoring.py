"""The fair split, the time split and one-answer-per-recording scoring."""
import numpy as np
import pandas as pd
import pytest

from conftest import pdm


def test_fair_test_folds_hold_out_whole_groups():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    folds = pdm.fair_test_folds(groups, [["a"], ["b"], ["c"]])
    assert [f.tolist() for f in folds] == [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    # every row is tested exactly once across the rounds
    assert np.sum(folds, axis=0).tolist() == [1] * 6


def test_check_no_overlap_stops_a_leak():
    units = np.array(["r1", "r1", "r2", "r2"])
    pdm.check_no_overlap(units, np.array([False, False, True, True]))
    with pytest.raises(ValueError, match="also appears in training"):
        pdm.check_no_overlap(units, np.array([False, True, True, True]))


def test_time_split_drops_the_gap_and_names_halves():
    units = np.repeat(["r1", "r2"], 10)
    order = np.tile(np.arange(10), 2)
    keep, test, new_units = pdm.time_split(units, order)
    assert keep.sum() == 18  # position 0.7 (window 6) is in the dropped gap of each recording
    assert test.sum() == 6  # last 30% of each recording
    assert set(new_units[test]) == {"r1_late", "r2_late"}
    assert set(new_units[~test]) == {"r1_early", "r2_early"}
    pdm.check_no_overlap(new_units, test)


def test_predict_recordings_averages_windows():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    y = np.array(["h", "h", "f", "f"])
    units = np.array(["u1", "u1", "u2", "u2"])
    actual, predicted = pdm.predict_recordings(probabilities, np.array(["h", "f"]), y, units)
    assert actual == ["h", "f"]
    assert predicted == ["h", "f"]  # u1 mean [0.55, 0.45]; u2 mean [0.45, 0.55]


def test_predict_recordings_refuses_mixed_labels():
    with pytest.raises(ValueError, match="Mixed labels"):
        pdm.predict_recordings(np.ones((2, 2)) / 2, np.array(["h", "f"]), np.array(["h", "f"]),
                               np.array(["u1", "u1"]))


def test_classification_metrics_known_case():
    m = pdm.classification_metrics(["h", "h", "f", "f"], ["h", "f", "f", "f"], ["h", "f"])
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
    folds = pdm.fair_test_folds(table["group"], [[g] for g in range(4)])
    result = pdm.sensor_fair_test(table, {"both": ["x1", "x2"]}, ["healthy", "fault"], folds, "group")
    assert result["score"]["test_units"] == 8
    assert result["score"]["accuracy"] == 1.0
    assert result["majority_control"]["balanced_accuracy"] == 0.5
    assert len(result["chosen_per_round"]) == 4
