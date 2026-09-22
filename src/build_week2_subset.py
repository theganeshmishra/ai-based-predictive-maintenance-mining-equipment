"""Build the documented Week 2 TiNA pilot subset from official source files."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import polars as pl


UPSTREAM_COMMIT = "927b1584ea4aeac2ac7daa084a82f899b114419b"
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
        return destination

    url = (
        "https://raw.githubusercontent.com/ari-dasci/OD-TINA/"
        f"{UPSTREAM_COMMIT}/data/day%3D{date}/data.parquet"
    )
    print(f"Downloading {date} from the official TiNA repository...")
    urllib.request.urlretrieve(url, destination)
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.write_parquet(OUTPUT_PATH)

    maintenance_rows = subset.filter(pl.col("m_id") != "none").height
    alarm_rows = subset.filter(pl.col("alarms") != "none").height
    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {subset.height:,}")
    print(f"Maintenance rows: {maintenance_rows:,}")
    print(f"Alarm rows: {alarm_rows:,}")
    print(f"SHA-256: {sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
