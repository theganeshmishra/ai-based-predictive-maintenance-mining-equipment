"""Acquire a bounded CORA gear-wear sensor/infrared experiment set.

The 16 stationary experiments are H/W25/W50/W75 at F5/F15/F50/F60.
Thermal MATLAB stacks are downloaded in full; only eight deterministic,
equally spaced byte ranges of original T1 CSV per run are retained locally.
The byte-range sample is a resource-bounded proxy, not the full sensor stream.
All raw/processed data are ignored by Git. No cross-run or frame-level pairing
is inferred from matching filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import whosmat


CAPSTONE = Path(__file__).resolve().parents[1]
RAW = CAPSTONE / "Raw_Data" / "cora_gear_wear"
PROCESSED = CAPSTONE / "Processed_Data" / "cora_gear_wear"
INTAKE = CAPSTONE / "Results" / "belt_conveyor_cora_intake.json"
BASE = "https://dataverse.csuc.cat/api/access/datafile"
CHUNK_BYTES = 256 * 1024
N_RANGES = 8
SOURCE = "https://doi.org/10.34810/DATA2500"
FILE_IDS = {
    "H_F5_S": (334968, 335123),
    "H_F15_S": (334964, 335059),
    "H_F50_S": (334886, 335054),
    "H_F60_S": (334901, 335165),
    "W25_F5_S": (334980, 335370),
    "W25_F15_S": (334958, 335494),
    "W25_F50_S": (334909, 335479),
    "W25_F60_S": (334991, 335458),
    "W50_F5_S": (334971, 335495),
    "W50_F15_S": (334982, 335438),
    "W50_F50_S": (334950, 335415),
    "W50_F60_S": (334937, 335423),
    "W75_F5_S": (334949, 335449),
    "W75_F15_S": (334977, 335391),
    "W75_F50_S": (334993, 335488),
    "W75_F60_S": (334918, 335400),
}
PUBLISHER_MD5 = {"H_F5_S": "18b4bf748275343a220cca7d4f854d60"}


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def request(file_id: int, byte_range: tuple[int, int] | None = None) -> Request:
    headers = {"User-Agent": "PROJ613-capstone/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return Request(f"{BASE}/{file_id}?format=original", headers=headers)


def download_thermal(run: str, file_id: int) -> dict:
    destination = RAW / f"{run}.mat"
    temporary = RAW / f"{run}.mat.part"
    if not destination.is_file():
        for attempt in range(3):
            try:
                with urlopen(request(file_id), timeout=180) as response, temporary.open("wb") as output:
                    content_name = response.headers.get("Content-Disposition", "")
                    if f'filename="{run}.mat"' not in content_name:
                        raise ValueError(f"Unexpected thermal response for {run}: {content_name}")
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                if temporary.stat().st_size < 1_000_000:
                    raise ValueError(f"Thermal response too small for {run}")
                os.replace(temporary, destination)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
    if run in PUBLISHER_MD5 and digest(destination, "md5") != PUBLISHER_MD5[run]:
        raise ValueError(f"Publisher MD5 mismatch: {run}")
    variables = whosmat(destination)
    if ("imagenes_celda", (1, 450), "cell") not in variables:
        raise ValueError(f"Unexpected MATLAB image stack: {run}: {variables}")
    return {
        "file_id": file_id,
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": digest(destination, "sha256"),
        "publisher_md5_checked": run in PUBLISHER_MD5,
        "image_frames": 450,
    }


def sensor_length(file_id: int, run: str) -> int:
    with urlopen(request(file_id, (0, 0)), timeout=90) as response:
        if response.status != 206:
            raise ValueError(f"Range request not honored for {run}")
        disposition = response.headers.get("Content-Disposition", "")
        if f'filename="{run}_T1.csv"' not in disposition:
            raise ValueError(f"Unexpected sensor response for {run}: {disposition}")
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            raise ValueError(f"Missing source size for {run}")
        response.read()
    return int(match.group(1))


def sample_sensor(run: str, file_id: int) -> tuple[np.ndarray, dict]:
    total = sensor_length(file_id, run)
    if total < CHUNK_BYTES * N_RANGES:
        raise ValueError(f"Unusually short T1 source for {run}: {total}")
    starts = np.linspace(0, total - CHUNK_BYTES, N_RANGES, dtype=np.int64)
    values: list[np.ndarray] = []
    ranges = []
    for start_value in starts:
        start = int(start_value)
        end = start + CHUNK_BYTES - 1
        with urlopen(request(file_id, (start, end)), timeout=90) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or content_range != f"bytes {start}-{end}/{total}":
                raise ValueError(f"Unexpected range response for {run}: {content_range}")
            block = response.read()
        if len(block) != CHUNK_BYTES:
            raise ValueError(f"Truncated range for {run}: {start}-{end}")
        lines = block.splitlines()
        complete = lines[1:-1] if start else lines[:-1]
        chunk = np.fromstring(b"\n".join(complete).decode("ascii"), sep="\n")
        if len(chunk) < 5_000 or not np.isfinite(chunk).all():
            raise ValueError(f"Invalid sensor readings for {run} at byte {start}")
        values.append(chunk.astype(np.float32))
        ranges.append({"first_byte": start, "last_byte": end, "readings": len(chunk),
                       "sha256": hashlib.sha256(block).hexdigest()})
    # Preserve one fixed six-second reading window at every selected position.
    fixed = np.stack([chunk[:6_000] for chunk in values])
    return fixed, {
        "file_id": file_id,
        "source_filename": f"{run}_T1.csv",
        "source_size_bytes": total,
        "sampled_bytes": N_RANGES * CHUNK_BYTES,
        "sampled_readings": int(fixed.size),
        "ranges": ranges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-only", action="store_true", help="Skip full infrared downloads")
    parser.add_argument("--thermal-only", action="store_true", help="Skip sensor range sampling")
    args = parser.parse_args()
    if args.sensor_only and args.thermal_only:
        parser.error("Choose at most one of --sensor-only and --thermal-only")
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    records = {}
    existing = json.loads(INTAKE.read_text(encoding="utf-8")) if INTAKE.exists() else {}
    if not args.sensor_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download_thermal, run, ids[0]): run
                       for run, ids in FILE_IDS.items()}
            for completed, future in enumerate(as_completed(futures), 1):
                run = futures[future]
                records.setdefault(run, {})["infrared"] = future.result()
                print(f"Infrared {completed}/16: {run}", flush=True)
    if not args.thermal_only:
        sensor_data = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(sample_sensor, run, ids[1]): run
                       for run, ids in FILE_IDS.items()}
            for completed, future in enumerate(as_completed(futures), 1):
                run = futures[future]
                values, metadata = future.result()
                sensor_data[run] = values
                records.setdefault(run, {})["temperature_sensor"] = metadata
                print(f"Sensor {completed}/16: {run}", flush=True)
        np.savez_compressed(PROCESSED / "t1_bounded_samples.npz", **sensor_data)
    previous = existing.get("runs", {})
    for run, values in previous.items():
        for modality, metadata in values.items():
            records.setdefault(run, {}).setdefault(modality, metadata)
    result = {
        "source": SOURCE,
        "version": "2.1",
        "license": "CC BY 4.0",
        "source_domain": "One controlled motor/gearbox laboratory rig, not a mining conveyor",
        "target": "healthy versus imposed gear-tooth wear (W25/W50/W75)",
        "selection": "16 stationary runs; four classes by four speeds; one run per class/speed",
        "sensor_sampling": "Eight equally spaced 256-KiB ranges per original T1 CSV; first 6,000 complete readings in each range",
        "sensor_sample_fraction_approx": "about 4.3% of bytes per run; not the full signal",
        "frame_alignment": "No claimed one-to-one frame/window alignment",
        "runs": dict(sorted(records.items())),
    }
    INTAKE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {INTAKE}")


if __name__ == "__main__":
    main()
