"""Acquire the four-class University of Ottawa bearing-vibration proxy.

Source: Huang and Baddour, Mendeley Data V2, DOI 10.17632/v43hmbwxpm.2.
Only H/I/O/B MAT files are selected; the compound-fault class is excluded.
The files describe an independent bearing test rig, not the XJTU pump.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


CAPSTONE = Path(__file__).resolve().parents[1]
RAW = CAPSTONE / "Raw_Data" / "uottawa_bearing"
MANIFEST = CAPSTONE / "Results" / "pump_uottawa_intake.json"
BASE = "https://data.mendeley.com/public-api/datasets/v43hmbwxpm"
FOLDERS = {
    "H": "1 Data collected from a healthy bearing",
    "I": "2 Data collected from a bearing with inner race fault",
    "O": "3 Data collected from a bearing with outer race fault",
    "B": "4 Data collected from a bearing with ball fault",
}
NAME = re.compile(r"^([HIOB])-([ABCD])-([123])\.mat$")


def get_json(url: str):
    with urlopen(Request(url, headers={"User-Agent": "PROJ613-capstone/1.0"}), timeout=45) as response:
        return json.load(response)


def inventory() -> list[dict]:
    folders = get_json(f"{BASE}/folders/2")
    by_name = {item["name"]: item["id"] for item in folders}
    if not set(FOLDERS.values()).issubset(by_name):
        raise ValueError("Expected publisher fault folders are missing")
    selected = []
    for prefix, folder_name in FOLDERS.items():
        folder_id = by_name[folder_name]
        url = f"{BASE}/files?folder_id={folder_id}&version=2&%24start=0&%24limit=1000"
        members = get_json(url)
        if len(members) != 12:
            raise ValueError(f"Expected 12 separate files in {folder_name}; found {len(members)}")
        for item in members:
            name = item["filename"]
            match = NAME.fullmatch(name)
            if not match or match.group(1) != prefix:
                raise ValueError(f"Unexpected source file in {folder_name}: {name}")
            details = item["content_details"]
            digest = details["sha256_hash"]
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Missing publisher SHA-256 for {name}")
            selected.append({
                "name": name,
                "class_code": prefix,
                "speed_profile": match.group(2),
                "trial": int(match.group(3)),
                "source_file_id": item["id"],
                "source_sha256": digest,
                "source_size_bytes": int(details["size"]),
                "url": details["download_url"] + "?version=2",
            })
    if len({item["name"] for item in selected}) != 48:
        raise ValueError("Expected 48 unique H/I/O/B recordings")
    return sorted(selected, key=lambda item: item["name"])


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def acquire(item: dict) -> tuple[str, str]:
    destination = RAW / item["name"]
    if destination.is_file():
        if destination.stat().st_size != item["source_size_bytes"] or sha256(destination) != item["source_sha256"]:
            raise ValueError(f"Existing source file failed publisher hash check: {destination}")
        return item["name"], "cached-and-verified"
    temporary = destination.with_suffix(".mat.part")
    for attempt in range(3):
        try:
            request = Request(item["url"], headers={"User-Agent": "PROJ613-capstone/1.0"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if temporary.stat().st_size != item["source_size_bytes"] or sha256(temporary) != item["source_sha256"]:
                raise ValueError(f"Publisher hash/size mismatch for {item['name']}")
            os.replace(temporary, destination)
            return item["name"], "downloaded-and-verified"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError("Unreachable")


def main() -> None:
    items = inventory()
    RAW.mkdir(parents=True, exist_ok=True)
    status = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(acquire, item): item["name"] for item in items}
        for completed, future in enumerate(as_completed(futures), 1):
            name, outcome = future.result()
            status[name] = outcome
            print(f"{completed}/48 {name}: {outcome}", flush=True)
    for item in items:
        item.pop("url")
        item["local_status"] = status[item["name"]]
    result = {
        "source": "https://data.mendeley.com/datasets/v43hmbwxpm/2",
        "doi": "10.17632/v43hmbwxpm.2",
        "license": "CC BY 4.0",
        "source_domain": "University of Ottawa laboratory bearing rig, not a pump or mining machine",
        "selected": "H/I/O/B only; four fault classes × four speed profiles × three trials",
        "selected_recordings": len(items),
        "selected_bytes": sum(item["source_size_bytes"] for item in items),
        "files": items,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved verified file manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
