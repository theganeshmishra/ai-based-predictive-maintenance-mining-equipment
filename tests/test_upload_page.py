"""The local upload page: refuses outside requests and malformed uploads, answers every upload."""
import http.client
import socket
import threading

import pytest

from conftest import pdm


@pytest.fixture(scope="module")
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = pdm.upload_server(port)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield port
    srv.shutdown()
    srv.server_close()


def request(port, method="GET", body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest(method, "/", skip_host=True, skip_accept_encoding=True)
    for key, value in ({"Host": f"127.0.0.1:{port}"} | (headers or {})).items():
        if value is not None:
            connection.putheader(key, value)
    connection.endheaders(body or None)
    response = connection.getresponse()
    text = response.read().decode("utf-8", "replace")
    connection.close()
    return response.status, text


def form(filename, content, machine=""):
    boundary = "testboundary613"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="machine"\r\n\r\n{machine}\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n",
             f"--{boundary}--\r\n".encode()]
    body = b"".join(parts)
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))}


def test_page_opens(server):
    status, text = request(server)
    assert status == 200 and "<form" in text


def test_other_host_names_are_refused(server):
    assert request(server, headers={"Host": f"evil.example:{server}"})[0] == 403


def test_uploads_from_other_websites_are_refused(server):
    body, headers = form("x.csv", b"a,b\n1,2\n")
    assert request(server, "POST", body, headers | {"Origin": "https://evil.example"})[0] == 403


def test_same_page_origin_is_accepted(server):
    body, headers = form("x.csv", b"a,b\n1,2\n")
    status, text = request(server, "POST", body, headers | {"Origin": f"http://127.0.0.1:{server}"})
    assert status == 200 and "No prediction" in text


@pytest.mark.parametrize("headers, status", [
    ({"Content-Type": "multipart/form-data; boundary=x", "Content-Length": "-5"}, 400),
    ({"Content-Type": "multipart/form-data; boundary=x"}, 411),
    ({"Content-Type": "application/x-www-form-urlencoded", "Content-Length": "3"}, 400),
])
def test_malformed_requests_are_refused(server, headers, status):
    body = b"a=b" if headers.get("Content-Length") == "3" else b""
    assert request(server, "POST", body, headers)[0] == status


@pytest.mark.parametrize("filename", ["..", ".hidden.csv", "a<b>.csv"])
def test_unsafe_file_names_get_an_error_page(server, filename):
    body, headers = form(filename, b"0.1\n")
    status, text = request(server, "POST", body, headers)
    assert status == 200 and "No prediction" in text


def test_damaged_file_still_gets_an_answer(server):
    body, headers = form("I-C-1.mat", b"not really a mat file")
    status, text = request(server, "POST", body, headers)
    assert status == 200 and "No prediction" in text


def test_upload_name_rule():
    assert pdm.upload_name_ok("W50_F50_S_Vx.csv")
    assert pdm.upload_name_ok("20221208-O-2rpm-0N-1.csv")
    assert pdm.upload_name_ok("pump_outer_20Hz_15_-30_t000.png")
    assert not pdm.upload_name_ok("..")
    assert not pdm.upload_name_ok("a/b.csv")
