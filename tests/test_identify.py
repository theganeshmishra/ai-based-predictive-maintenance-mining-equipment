"""The format check that decides which machine a file belongs to (no machine learning)."""
import numpy as np
import pytest

from conftest import ROOT, pdm


def write_csv(path, header, rows=2):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(["0.1"] * len(header)) for _ in range(rows)) + "\n",
                    encoding="utf-8")
    return path


def test_just_csv_is_grader(tmp_path):
    source, machines, details = pdm.identify(write_csv(tmp_path / "x.csv", pdm.JUST_CHANNELS))
    assert source == "JUST slewing bearing" and machines == ["grader"] and not details["mixed_schema"]


def test_mcc5_csv_is_drill_rig(tmp_path):
    source, machines, _ = pdm.identify(write_csv(tmp_path / "x.csv", pdm.MCC5_CHANNELS))
    assert source == "MCC5-THU gearbox" and machines == ["drill_rig"]


def test_aalto_csv_is_grader_gear_task(tmp_path):
    header = ["", "time", "enc1_ang", "enc2_ang", "acc1", "acc2", "acc3", "acc4", "Torq1", "Torq2"]
    source, machines, _ = pdm.identify(write_csv(tmp_path / "run.csv", header))
    assert source.startswith("Aalto") and machines == ["grader"]


def test_two_schemas_in_one_csv_are_flagged(tmp_path):
    _, machines, details = pdm.identify(write_csv(tmp_path / "x.csv", pdm.JUST_CHANNELS + pdm.MCC5_CHANNELS))
    assert set(machines) == {"grader", "drill_rig"} and details["mixed_schema"]


def test_cora_file_is_ambiguous_across_four_machines(tmp_path):
    path = tmp_path / "W50_F50_S_Vx.csv"
    path.write_text("0.25\n0.5\n", encoding="utf-8")
    source, machines, details = pdm.identify(path)
    assert source.startswith("CORA") and machines == pdm.CORA and details["speed"] == "F50"
    answer = pdm.predict(path)
    assert answer["status"] == "Ambiguous Equipment"


@pytest.mark.parametrize("name, text", [("other_name.csv", "0.25\n"), ("W50_F50_S_Vx.csv", "abc\n"),
                                        ("x.csv", "a,b,c\n1,2,3\n")])
def test_unknown_csv_is_rejected(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        pdm.identify(path)


def test_unsupported_file_is_rejected():
    with pytest.raises(ValueError, match="none of the known dataset formats"):
        pdm.predict(ROOT / "README.md")


def test_wrong_machine_is_refused(tmp_path):
    path = write_csv(tmp_path / "x.csv", pdm.JUST_CHANNELS)
    with pytest.raises(ValueError, match="does not belong to pump"):
        pdm.predict(path, "pump")


def test_blank_and_unindexed_images(tmp_path):
    import cv2

    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((224, 224, 3), 128, np.uint8))
    with pytest.raises(ValueError, match="no spatial detail"):
        pdm.identify(blank)
    render = tmp_path / "render.png"
    cv2.imwrite(str(render), np.random.default_rng(613).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="provenance could not be verified"):
        pdm.identify(render)


def test_event_frame_needs_a_mapping(tmp_path):
    import cv2

    frame = np.zeros((96, 128, 3), np.uint8)
    frame[10:40, 20:60, 1:] = 200  # blue channel stays empty, as in the event-camera exports
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)
    source, machines, _ = pdm.identify(path)
    assert source.startswith("event-camera") and set(machines) == {"pump", "haul_truck"}


@pytest.mark.parametrize("content", [b"", b"not a MAT file", b"X" * 130])
def test_corrupt_mat_reports_a_readable_error(tmp_path, content):
    path = tmp_path / "broken.mat"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Invalid or unsupported MAT recording"):
        pdm.identify(path)
