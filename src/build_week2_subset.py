"""Build the documented Week 2 TiNA pilot subset from official source files."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import polars as pl


UPSTREAM_COMMIT = "927b1584ea4aeac2ac7daa084a82f899b114419b"
EXPECTED_PARTITION_SHA256 = {
    "2017-09-08": "7389a5347a95b6a03018879380859b68ac811fec939be297732696dd46157977",
    "2017-09-09": "ca44e34ec06dcccb76be106d095b725a1926659911623bae9d0b907a1ddc3bb5",
    "2017-09-10": "a427206041323dfe4e5e61a0eb4a6ab204059540fbed056cffdfde4e3f196223",
}
DATES = ("2017-09-08", "2017-09-09", "2017-09-10")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = PROJECT_ROOT / "data" / "raw" / "tina_week2_parts"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "tina_sample_2017-09-08_to_10.parquet"


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_partition(date: str) -> Path:
    """Download one pinned daily partition unless it is already present."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = PARTS_DIR / f"{date}.parquet"
    if destination.exists():
        if sha256(destination) != EXPECTED_PARTITION_SHA256[date]:
            raise ValueError(
                f"Cached TiNA partition {destination} failed its source checksum. "
                "Remove that one cached file and rerun the script."
            )
        return destination

    url = (
        "https://raw.githubusercontent.com/ari-dasci/OD-TINA/"
        f"{UPSTREAM_COMMIT}/data/day%3D{date}/data.parquet"
    )
    print(f"Downloading {date} from the official TiNA repository...")
    staged_path = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(url, staged_path)
        if sha256(staged_path) != EXPECTED_PARTITION_SHA256[date]:
            raise ValueError(f"Downloaded TiNA partition for {date} failed its source checksum.")
        staged_path.replace(destination)
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    """Download, combine, validate, and summarize the Week 2 subset."""
    parts = [download_partition(date) for date in DATES]
    frames = [pl.read_parquet(path) for path in parts]
    subset = pl.concat(frames, how="vertical").sort("unix")

    expected_rows = 3 * 24 * 60 * 60
    if subset.height != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {subset.height:,}.")
    if not subset["unix"].is_sorted():
        raise ValueError("The combined subset is not sorted by unix time.")
    if subset["unix"].n_unique() != subset.height:
        raise ValueError("The selected period contains duplicate timestamps.")
    if subset["unix"].min() != 1504828800 or subset["unix"].max() != 1505087999:
        raise ValueError("The selected period does not cover the expected UTC timestamps.")
    if not (subset["unix"].diff().drop_nulls() == 1).all():
        raise ValueError("The selected period is not sampled every second.")

    maintenance_rows = subset.filter(pl.col("m_id") != "none").height
    alarm_rows = subset.filter(pl.col("alarms") != "none").height
    if maintenance_rows != 4_994 or alarm_rows != 26_719:
        raise ValueError("The selected period has unexpected maintenance or alarm row counts.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    staged_path = OUTPUT_PATH.with_suffix(".tmp")
    try:
        subset.write_parquet(staged_path)
        checksum = sha256(staged_path)
        staged_path.replace(OUTPUT_PATH)
    finally:
        staged_path.unlink(missing_ok=True)

    print(f"Created: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {subset.height:,}")
    print(f"Maintenance rows: {maintenance_rows:,}")
    print(f"Alarm rows: {alarm_rows:,}")
    print(f"Local output SHA-256 (informational): {checksum}")


if __name__ == "__main__":
    main()
