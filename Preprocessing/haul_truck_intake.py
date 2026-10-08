"""Acquire the Haul/Dump Truck wheel-motor bearing sources (matrix row B1).

Sensor: University of Ottawa UODS-VAFDC (Mendeley Data, DOI 10.17632/y2px5tg92h.1),
CC BY 4.0. Only bearings 1-10 are selected: their healthy stage-0 recordings and
their final stage-2 inner-race (bearings 1-5) or outer-race (bearings 6-10)
recordings. Ball faults are excluded because the publisher states they were
recorded under no load, while healthy/inner/outer used 400 N. Stage 1
(fault developing) has no XJTU counterpart and is also excluded.

Visual: XJTU-DV-Rotor event-camera ZIP (Zenodo record 17447083), CC BY 4.0,
downloaded in full and checked against Zenodo's published MD5.

Both are laboratory rigs, not haul-truck data. Run with --sensor-only or
--visual-only to fetch one source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


CAPSTONE = Path(__file__).resolve().parents[1]
SENSOR_RAW = CAPSTONE / "Raw_Data" / "uods_vafdc"
VISUAL_RAW = CAPSTONE / "Raw_Data" / "xjtu_dv_rotor"
MANIFEST = CAPSTONE / "Results" / "haul_truck_intake.json"
MENDELEY = "https://data.mendeley.com/public-api/datasets/y2px5tg92h"
FOLDERS = {"H": "1_Healthy", "I": "2_Inner_Race_Faults", "O": "3_Outer_Race_Faults"}
NAME = re.compile(r"^([HIO])_(\d+)_([012])\.mat$")
ROTOR_URL = "https://zenodo.org/api/records/17447083/files/XJTU-DV-Rotor.zip/content"
ROTOR_NAME = "XJTU-DV-Rotor.zip"
ROTOR_SIZE = 2_812_106_658
ROTOR_MD5 = "2dfed028950f3769ae62effa47faaccb"
HEADERS = {"User-Agent": "PROJ613-capstone/1.0"}


def get_json(url: str):
    with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def selected(name: str) -> bool:
    match = NAME.fullmatch(name)
    if not match:
        return False
    code, bearing, stage = match.group(1), int(match.group(2)), int(match.group(3))
    if code == "H":
        return 1 <= bearing <= 10 and stage == 0
    if code == "I":
        return 1 <= bearing <= 5 and stage == 2
    return 6 <= bearing <= 10 and stage == 2


def sensor_inventory() -> list[dict]:
    by_name = {item["name"]: item["id"] for item in get_json(f"{MENDELEY}/folders/1")}
    if not set(FOLDERS.values()).issubset(by_name):
        raise ValueError("Expected publisher fault folders are missing")
    items = []
    for code, folder in FOLDERS.items():
        members = get_json(f"{MENDELEY}/files?folder_id={by_name[folder]}&version=1&%24start=0&%24limit=1000")
        for member in members:
            name = member["filename"]
            if not selected(name):
                continue
            details = member["content_details"]
            if not re.fullmatch(r"[a-f0-9]{64}", details.get("sha256_hash", "")):
                raise ValueError(f"Missing publisher SHA-256 for {name}")
            match = NAME.fullmatch(name)
            items.append({
                "name": name,
                "class_code": code,
                "bearing": int(match.group(2)),
                "stage": int(match.group(3)),
                "source_file_id": member["id"],
                "source_sha256": details["sha256_hash"],
                "source_size_bytes": int(details["size"]),
                "url": details["download_url"] + "?version=1",
            })
    counts = {code: sum(item["class_code"] == code for item in items) for code in FOLDERS}
    if counts != {"H": 10, "I": 5, "O": 5}:
        raise ValueError(f"Unexpected UODS selection: {counts}")
    return sorted(items, key=lambda item: (item["bearing"], item["stage"]))


def digest(path: Path, algorithm: str) -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fetch_sensor(item: dict) -> str:
    destination = SENSOR_RAW / item["name"]
    if destination.is_file():
        if destination.stat().st_size != item["source_size_bytes"] or digest(destination, "sha256") != item["source_sha256"]:
            raise ValueError(f"Existing file failed publisher hash check: {destination}")
        return "cached-and-verified"
    temporary = destination.with_suffix(".mat.part")
    for attempt in range(3):
        try:
            with urlopen(Request(item["url"], headers=HEADERS), timeout=120) as response, temporary.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
            if temporary.stat().st_size != item["source_size_bytes"] or digest(temporary, "sha256") != item["source_sha256"]:
                raise ValueError(f"Publisher hash/size mismatch for {item['name']}")
            os.replace(temporary, destination)
            return "downloaded-and-verified"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError("Unreachable")


def fetch_rotor() -> str:
    """Download the full Rotor ZIP, resuming a partial file, then check Zenodo's MD5."""
    VISUAL_RAW.mkdir(parents=True, exist_ok=True)
    destination = VISUAL_RAW / ROTOR_NAME
    if destination.is_file():
        if destination.stat().st_size != ROTOR_SIZE or digest(destination, "md5") != ROTOR_MD5:
            raise ValueError(f"Existing Rotor ZIP failed the Zenodo MD5 check: {destination}")
        return "cached-and-verified"
    partial = destination.with_suffix(".zip.part")
    for attempt in range(20):
        done = partial.stat().st_size if partial.is_file() else 0
        if done >= ROTOR_SIZE:
            break
        try:
            request = Request(ROTOR_URL, headers={**HEADERS, "Range": f"bytes={done}-"})
            with urlopen(request, timeout=120) as response, partial.open("ab") as out:
                if done and response.status != 206:
                    raise ValueError("Server ignored the resume request")
                last = time.time()
                for block in iter(lambda: response.read(1 << 22), b""):
                    out.write(block)
                    done += len(block)
                    if time.time() - last > 30:
                        print(f"Rotor ZIP: {done / ROTOR_SIZE:.1%}", flush=True)
                        last = time.time()
        except Exception as error:
            print(f"Rotor ZIP interrupted at {done:,} bytes ({error}); resuming", flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))
    if partial.stat().st_size != ROTOR_SIZE or digest(partial, "md5") != ROTOR_MD5:
        raise ValueError("Rotor ZIP failed the Zenodo size/MD5 check; delete the .part file and retry")
    os.replace(partial, destination)
    return "downloaded-and-verified"


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sensor-only", action="store_true")
    group.add_argument("--visual-only", action="store_true")
    args = parser.parse_args()
    record = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}

    if not args.visual_only:
        items = sensor_inventory()
        SENSOR_RAW.mkdir(parents=True, exist_ok=True)
        for number, item in enumerate(items, 1):
            item["local_status"] = fetch_sensor(item)
            item.pop("url")
            print(f"{number}/{len(items)} {item['name']}: {item['local_status']}", flush=True)
        record["sensor"] = {
            "source": "https://data.mendeley.com/datasets/y2px5tg92h/1",
            "doi": "10.17632/y2px5tg92h.1",
            "license": "CC BY 4.0",
            "source_domain": "University of Ottawa laboratory bearing rig, not a haul truck",
            "selected": "bearings 1-10: healthy stage 0 for all ten, final stage 2 inner (1-5) or outer (6-10); ball, cage and fault-developing stage excluded",
            "selected_recordings": len(items),
            "selected_bytes": sum(item["source_size_bytes"] for item in items),
            "files": items,
        }

    if not args.sensor_only:
        status = fetch_rotor()
        print(f"{ROTOR_NAME}: {status}", flush=True)
        record["visual"] = {
            "source": "https://zenodo.org/records/17447083",
            "doi": "10.5281/zenodo.17447083",
            "license": "CC BY 4.0",
            "source_domain": "XJTU laboratory rotor-bearing rig, not a haul truck",
            "file": ROTOR_NAME,
            "size_bytes": ROTOR_SIZE,
            "zenodo_md5": ROTOR_MD5,
            "local_status": status,
        }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved intake record: {MANIFEST.relative_to(CAPSTONE)}")


if __name__ == "__main__":
    main()
