"""Prediction history, the dashboard and the demo check."""
import csv
import http.client
import io
import json
import threading

import pytest

from conftest import predictive_maintenance


@pytest.fixture
def history(tmp_path, monkeypatch):
    path = tmp_path / "prediction_history.jsonl"
    monkeypatch.setattr(predictive_maintenance, "HISTORY", path)
    return path


@pytest.fixture(scope="module")
def server():
    srv = predictive_maintenance.upload_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_port
    srv.shutdown()
    srv.server_close()


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
