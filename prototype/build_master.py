"""Build the master catalogue and the per-dataset files the prototype uses.

Run from the repository root:  python prototype/build_master.py

Outputs (all git-ignored, under data/processed/):
- master_catalog.csv          one row per sample from every dataset, same columns
- tina_minutes_8state.parquet TiNA per-minute sensor features + 8-class machine state
- bearing_photos_master.csv   bearing photo metadata joined with heuristic scores

The datasets are listed together but never merged into one training table:
they come from different equipment and share no ids, columns or labels.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TINA_FILE = ROOT / "data" / "interim" / "tina_2017-09_combined.parquet"
NONASTREDA_DIR = Path(os.environ.get(
    "NONASTREDA_DIR", ROOT.parent / "multimodal-pdm-from-scratch" / "data" / "raw" / "nonastreda"))
MIN_SECONDS_PER_MINUTE = 30

# 8 machine states. The four maintenance types with 9+ episodes each get their
# own class; M2 (3 episodes) is kept separate; codes with one or two episodes
# (M4/none, M7, M11, M13) are grouped as "other_maintenance".
STATES = ["normal", "alarm", "M4/MS3", "M4/MS5", "M9/MS1", "M9/MS6", "M2", "other_maintenance"]
OWN_CLASS = {"M4/MS3", "M4/MS5", "M9/MS1", "M9/MS6"}


def maintenance_state(key: str) -> str:
    if key in OWN_CLASS:
        return key
    if key.startswith("M2/"):
        return "M2"
    return "other_maintenance"


def build_tina() -> pd.DataFrame:
    data = pl.read_parquet(TINA_FILE).sort("unix")
    if data["unix"].n_unique() != data.height:
        raise ValueError("TiNA has duplicate timestamps.")
    sensors = [n for n, t in data.schema.items() if n.startswith("FEATURE") and t.is_numeric()]
    data = data.with_columns(
        (pl.col("unix") // 60 * 60 + 59).alias("minute_end"),
        (pl.col("m_id") + "/" + pl.col("m_subid")).alias("m_key"),
    )
    minute = (
        data.group_by("minute_end")
        .agg(
            pl.len().alias("n_seconds"),
            (pl.col("m_id") != "none").sum().alias("maintenance_seconds"),
            (pl.col("alarms") != "none").sum().alias("alarm_seconds"),
            pl.col("m_key").filter(pl.col("m_id") != "none").mode().first().alias("m_key"),
            *[e for n in sensors for e in (pl.col(n).mean().alias(f"{n}_mean"),
                                           pl.col(n).std().alias(f"{n}_std"))],
        )
        .sort("minute_end")
        .filter(pl.col("n_seconds") >= MIN_SECONDS_PER_MINUTE)
        .to_pandas()
    )
    state = np.where(minute["maintenance_seconds"] > 0,
                     minute["m_key"].fillna("").map(maintenance_state),
                     np.where(minute["alarm_seconds"] > 0, "alarm", "normal"))
    minute.insert(1, "state", state)
    minute.insert(1, "day", pd.to_datetime(minute["minute_end"], unit="s").dt.strftime("%Y-%m-%d"))
    # maintenance episode id (consecutive maintenance minutes with the same state)
    maint = minute["maintenance_seconds"].to_numpy() > 0
    gap = np.r_[True, np.diff(minute["minute_end"].to_numpy()) > 60]
    new_ep = maint & (gap | np.r_[True, ~maint[:-1]] | np.r_[True, state[1:] != state[:-1]])
    ep = np.cumsum(new_ep)
    minute.insert(2, "episode", np.where(maint, ep, 0))
    feats = [c for c in minute.columns if c.endswith(("_mean", "_std"))]
    minute[feats] = minute[feats].astype("float32")
    if not set(minute["state"]) <= set(STATES):
        raise ValueError(f"Unexpected states: {set(minute['state']) - set(STATES)}")
    return minute


def build_bearing() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data" / "raw" / "opencv" / "image_labels.csv")
    meta["image_id"] = meta["image_path"].str.replace("photos/", "", regex=False).str.replace(".jpg", "", regex=False)
    scores = pd.read_csv(PROCESSED / "bearing_image_pseudo_labels.csv")
    out = meta.merge(scores.drop(columns=["filename"]), on="image_id", how="left", validate="one_to_one")
    if len(out) != len(meta) or out["pseudo_label"].isna().any():
        raise ValueError("Bearing photo metadata and heuristic scores do not match one-to-one.")
    return out


def catalogue(tina: pd.DataFrame, bearing: pd.DataFrame) -> pd.DataFrame:
    parts = [pd.DataFrame({
        "dataset": "TiNA",
        "sample_id": "TiNA_" + tina["minute_end"].astype(str),
        "equipment": "mining machine (ArcelorMittal, anonymised sensors)",
        "modality": "sensor time series, 1-minute window",
        "source_file": "data/processed/tina_minutes_8state.parquet",
        "label": tina["state"],
        "label_source": "derived from TiNA m_id/m_subid/alarms columns (8-state definition in prototype/README.md)",
        "split_group": tina["day"],
        "time_or_order": pd.to_datetime(tina["minute_end"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "licence_or_permission": "public TiNA release; see reports/week-02-data-acquisition.md",
    })]
    labels = NONASTREDA_DIR / "labels.csv"
    if labels.exists():
        nl = pd.read_csv(labels)
        parts.append(pd.DataFrame({
            "dataset": "Nonastreda",
            "sample_id": nl["id"],
            "equipment": "milling tool (machining, NOT mining)",
            "modality": "force signals + tool/chip/workpiece photos",
            "source_file": "multimodal-pdm-from-scratch/data/processed/nonastreda_paired.parquet",
            "label": nl["image_label"],
            "label_source": "dataset authors (labels.csv)",
            "split_group": "tool_" + nl["id"].str.extract(r"^T(\d+)", expand=False),
            "time_or_order": nl["id"],
            "licence_or_permission": "CC BY 4.0, DOI 10.17632/m892d2wtzh.1",
        }))
    else:
        print(f"Nonastreda labels not found at {labels}; skipped in catalogue.")
    parts.append(pd.DataFrame({
        "dataset": "Bearing photos",
        "sample_id": bearing["image_id"],
        "equipment": bearing["equipment_type"],
        "modality": "phone photo",
        "source_file": "data/raw/opencv/" + bearing["image_path"],
        "label": bearing["pseudo_label"],
        "label_source": "OpenCV heuristic pseudo-label, NOT verified by a person",
        "split_group": "session_" + bearing["capture_time"].str[:10],
        "time_or_order": bearing["capture_time"],
        "licence_or_permission": bearing["usage_permission"],
    }))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tina = build_tina()
    tina.to_parquet(PROCESSED / "tina_minutes_8state.parquet", index=False)
    bearing = build_bearing()
    bearing.to_csv(PROCESSED / "bearing_photos_master.csv", index=False)
    cat = catalogue(tina, bearing)
    cat.to_csv(PROCESSED / "master_catalog.csv", index=False)

    print(f"TiNA minutes: {len(tina):,} from {tina['day'].nunique()} days")
    episodes = tina[tina["episode"] > 0].groupby("state")["episode"].nunique()
    counts = pd.DataFrame({"minutes": tina["state"].value_counts(), "episodes": episodes}).reindex(STATES)
    print(counts.fillna(0).astype(int).to_string())
    print(f"Bearing photos: {len(bearing):,}")
    print(f"master_catalog.csv: {len(cat):,} rows")
    print(cat.groupby("dataset")["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
