"""Exploratory, leakage-aware TiNA Week 3 maintenance baseline.

Run ``python -m src.week3_baseline`` from the repository root. Source partitions
are kept locally and verified against the pinned official TiNA commit.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_week2_subset import download_partition


DATES = tuple(f"2017-09-{day:02d}" for day in range(7, 13))
HORIZON_SECONDS = 30 * 60
RECOVERY_SECONDS = 30 * 60
TRAIN_END = 1505088000  # 2017-09-11 00:00:00 UTC
VALIDATION_END = 1505174400  # 2017-09-12 00:00:00 UTC
TEST_END = 1505260800  # 2017-09-13 00:00:00 UTC


def load_six_days() -> pl.DataFrame:
    """Validate and sort six complete daily source partitions."""
    frames = [pl.read_parquet(download_partition(day)) for day in DATES]
    data = pl.concat(frames, how="vertical").sort("unix")
    times = data["unix"]
    if (
        data.height != 6 * 86400
        or times.min() != 1504742400
        or times.max() != TEST_END - 1
        or times.n_unique() != data.height
        or not (times.diff().drop_nulls() == 1).all()
    ):
        raise ValueError("The six-day source must have one unique row per second.")
    return data


def make_model_table(data: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Aggregate past/current-minute sensors and label future event onsets."""
    active = data["m_id"] != "none"
    starts = active & ~active.shift(1).fill_null(False)
    ends = active & ~active.shift(-1).fill_null(False)
    onsets = data.filter(starts)["unix"].to_numpy()
    event_ends = data.filter(ends)["unix"].to_numpy()
    if len(onsets) != len(event_ends) or len(onsets) == 0:
        raise ValueError("Maintenance episode boundaries are inconsistent.")

    sensor_columns = [
        name for name, dtype in data.schema.items()
        if name.startswith("FEATURE") and dtype.is_numeric()
    ]
    if not sensor_columns:
        raise ValueError("No numeric TiNA sensor columns were found.")
    expressions = [
        expression
        for name in sensor_columns
        for expression in (
            pl.col(name).mean().alias(f"{name}_mean"),
            pl.col(name).std().alias(f"{name}_std"),
        )
    ]
    minute = (
        data.with_columns((pl.col("unix") // 60 * 60 + 59).alias("minute_end"))
        .group_by("minute_end")
        .agg(
            (pl.col("m_id") != "none").sum().alias("maintenance_seconds"),
            *expressions,
        )
        .sort("minute_end")
    )
    times = minute["minute_end"].to_numpy()
    eligible = minute["maintenance_seconds"].to_numpy() == 0
    for end in event_ends:
        eligible &= ~((times > end) & (times <= end + RECOVERY_SECONDS))

    next_index = np.searchsorted(onsets, times, side="right")
    next_onset = np.r_[onsets, np.iinfo(np.int64).max][next_index]
    target = (next_onset > times) & (next_onset <= times + HORIZON_SECONDS)
    minute = minute.with_columns(
        pl.Series("eligible", eligible),
        pl.Series("target", target.astype(np.int8)),
    )
    model_features = [name for name in minute.columns if name.endswith(("_mean", "_std"))]
    if minute.height != 6 * 1440:
        raise ValueError("Expected exactly 1,440 complete minutes per day.")
    return minute, onsets, event_ends, model_features


def get_split(minute: pl.DataFrame, start: int, end: int, features: list[str]):
    """Drop the final horizon of a split to prevent label overlap/censoring."""
    split = minute.filter(
        (pl.col("minute_end") >= start)
        & (pl.col("minute_end") + HORIZON_SECONDS < end)
        & pl.col("eligible")
    )
    x = split.select(features).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Minute-level sensor features contain non-finite values.")
    return x, split["target"].to_numpy(), split["minute_end"].to_numpy()


def summarize(
    name: str, y: np.ndarray, times: np.ndarray, onsets: np.ndarray,
    scores: np.ndarray, start: int, end: int,
) -> dict[str, object]:
    """Score minute warnings and distinct maintenance-onset coverage."""
    predicted = scores >= 0.5
    split_onsets = onsets[(onsets >= start) & (onsets < end)]
    covered = sum(
        np.any(predicted & (times >= onset - HORIZON_SECONDS) & (times < onset))
        for onset in split_onsets
    )
    false_times = times[predicted & (y == 0)]
    false_runs = int(np.sum(np.r_[True, np.diff(false_times) > 60])) if len(false_times) else 0
    return {
        "model": name,
        "minutes": int(len(y)),
        "positive_minutes": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "event_coverage": f"{covered}/{len(split_onsets)}",
        "false_alarm_minutes": int(len(false_times)),
        "false_alarm_runs": false_runs,
    }


def run_analysis() -> dict[str, object]:
    """Fit on Sept 7-10, validate on Sept 11, and test once on Sept 12."""
    data = load_six_days()
    minute, onsets, event_ends, features = make_model_table(data)
    boundaries = {
        "train": (1504742400, TRAIN_END),
        "validation": (TRAIN_END, VALIDATION_END),
        "test": (VALIDATION_END, TEST_END),
    }
    splits = {name: get_split(minute, start, end, features)
              for name, (start, end) in boundaries.items()}
    x_train, y_train, _ = splits["train"]
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training data must contain both target classes.")

    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    # Feature filtering/selection is learned only from the training period.
    selector = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=20)),
        ("model", GaussianNB()),
    ]).fit(x_train, y_train)
    retained = np.asarray(features)[selector.named_steps["variance"].get_support()]
    selected = retained[selector.named_steps["select"].get_support()].tolist()

    results = {}
    for split_name in ("validation", "test"):
        x, y, times = splits[split_name]
        start, end = boundaries[split_name]
        results[split_name] = [
            summarize(model_name, y, times, onsets,
                      model.predict_proba(x)[:, 1], start, end)
            for model_name, model in (("Most-frequent dummy", dummy), ("Gaussian Naive Bayes", selector))
        ]
    episode_counts = {
        name: int(((onsets >= start) & (onsets < end)).sum())
        for name, (start, end) in boundaries.items()
    }
    return {
        "source_rows": data.height,
        "source_columns": data.width,
        "numeric_sensors": len({name.split("_")[0] for name in features}),
        "maintenance_episodes": len(onsets),
        "maintenance_rows": int((data["m_id"] != "none").sum()),
        "episode_counts": episode_counts,
        "training_minutes": len(y_train),
        "training_positive_minutes": int(y_train.sum()),
        "selected_features": selected,
        "results": results,
    }


if __name__ == "__main__":
    from pprint import pprint

    print(f"TiNA six-day baseline, generated {datetime.now(timezone.utc).isoformat()}")
    pprint(run_analysis(), sort_dicts=False)
