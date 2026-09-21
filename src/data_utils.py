"""Small, reusable helpers for loading and validating TiNA data."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl


DEFAULT_DATA_PATH = Path("data/raw/tina_sample_2017-09-06.parquet")
EVENT_COLUMNS = ("m_id", "m_subid", "alarms")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_path(default: Path = DEFAULT_DATA_PATH) -> Path:
    """Return the TiNA path from TINA_DATA_PATH, or the project default."""
    data_path = Path(os.environ.get("TINA_DATA_PATH", default))
    return data_path if data_path.is_absolute() else PROJECT_ROOT / data_path


def scan_tina(path: str | Path | None = None) -> pl.LazyFrame:
    """Lazily scan a CSV or Parquet TiNA file without loading it all at once."""
    data_path = Path(path) if path is not None else resolve_data_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"TiNA data not found at {data_path}. See data/README.md or set "
            "the TINA_DATA_PATH environment variable."
        )

    suffix = data_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(data_path)
    if suffix in {".csv", ".txt"}:
        return pl.scan_csv(data_path, try_parse_dates=True, infer_schema_length=10_000)
    raise ValueError(f"Unsupported data format: {suffix}. Use CSV or Parquet.")


def validate_event_columns(frame: pl.LazyFrame) -> None:
    """Raise a clear error if expected TiNA event columns are missing."""
    available = set(frame.collect_schema().names())
    missing = [column for column in EVENT_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"Missing expected TiNA columns: {missing}")
