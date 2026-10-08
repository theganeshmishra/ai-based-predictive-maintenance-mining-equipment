"""Small-memory CORA preprocessing; no model training or cross-modal join.

Use original, headerless CSV downloads for sensors. Dataverse's generated
``.tab`` view may treat the first numeric reading as a column heading.
Thermal stacks are MATLAB-v5 ``.mat`` files with RGB frames. These functions
retain experiment and within-file indices but do not claim that a frame index
matches a sensor-window index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cora_manifest import load_thermal_stack, parse_file


SAMPLE_RATES_HZ = {
    **{name: 4000 for name in ("C1", "C2", "C3")},
    **{name: 3000 for name in ("VX", "VY", "VZ")},
    **{name: 1000 for name in ("T1", "T2", "T3", "T4", "T5", "T6")},
    "RPM": 1 / 30,
}


def sensor_window_features(
    path: Path, *, window_seconds: int = 6, max_windows: int | None = None,
) -> pd.DataFrame:
    """Stream one sensor channel into non-overlapping, fixed-duration features.

    A window index is relative to this file only. It is not a verified thermal
    timestamp or a guarantee that other channels start at the same instant.
    """
    record = parse_file(path)
    if record is None or record["modality"] != "sensor" or path.suffix.lower() != ".csv":
        raise ValueError("Use an original CORA sensor .csv file, not an ingested .tab file.")
    channel = record["channel"]
    if channel not in SAMPLE_RATES_HZ:
        raise ValueError(f"No verified sample rate for channel {channel}.")
    if window_seconds <= 0 or max_windows is not None and max_windows <= 0:
        raise ValueError("Window duration and maximum window count must be positive.")
    sample_rate = SAMPLE_RATES_HZ[channel]
    expected_samples = sample_rate * window_seconds
    samples_per_window = round(expected_samples)
    if samples_per_window < 1 or not np.isclose(expected_samples, samples_per_window):
        raise ValueError("Window duration is not an integer number of sensor samples.")

    rows: list[dict[str, float | int | str]] = []
    reader = pd.read_csv(path, header=None, chunksize=samples_per_window)
    for index, chunk in enumerate(reader):
        if max_windows is not None and index >= max_windows:
            break
        if chunk.shape != (samples_per_window, 1):
            raise ValueError(f"Incomplete or multi-column window {index} in {path.name}.")
        values = pd.to_numeric(chunk.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite reading in window {index} of {path.name}.")
        mean = float(values.mean())
        centered = values - mean
        standard_deviation = float(values.std())
        rms = float(np.sqrt(np.mean(values ** 2)))
        peak = float(np.max(np.abs(values)))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        if len(spectrum) > 1:
            dominant_bin = int(np.argmax(spectrum[1:]) + 1)
            dominant_hz = float(np.fft.rfftfreq(len(values), d=1 / sample_rate)[dominant_bin])
            spectral_energy = float(spectrum[1:].sum() / len(values) ** 2)
        else:
            dominant_hz = float("nan")
            spectral_energy = 0.0
        rows.append({
            "experiment": record["experiment"],
            "channel": channel,
            "window_index": index,
            "mean": mean,
            "std": standard_deviation,
            "rms": rms,
            "peak_to_peak": float(np.ptp(values)),
            "crest_factor": peak / rms if rms else 0.0,
            "dominant_hz": dominant_hz,
            "spectral_energy": spectral_energy,
        })
    return pd.DataFrame(rows)


def thermal_frame_features(path: Path, *, max_frames: int | None = None) -> pd.DataFrame:
    """Extract simple OpenCV image descriptors from one stationary .mat stack.

    RGB-shaped values may be repeated grayscale or false-colour renderings;
    do not interpret them as calibrated temperature readings. This is a
    preprocessing smoke test,
    not a nine-class image model.
    """
    record = parse_file(path)
    if record is None or record["modality"] != "thermal_stack":
        raise ValueError(f"Not a CORA thermal .mat stack: {path.name}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("Maximum frame count must be positive.")
    frames = load_thermal_stack(path)
    rows: list[dict[str, float | int | str]] = []
    for index, frame in enumerate(frames, start=1):
        if max_frames is not None and index > max_frames:
            break
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        rows.append({
            "experiment": record["experiment"],
            "frame_index": index,
            "gray_mean": float(gray.mean()),
            "gray_std": float(gray.std()),
            "gray_p95": float(np.percentile(gray, 95)),
            "edge_fraction": float(np.mean(edges > 0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", type=Path, help="Original CORA sensor .csv")
    parser.add_argument("--thermal", type=Path, help="Stationary CORA .mat stack")
    parser.add_argument("--window-seconds", type=int, default=6)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    if not args.sensor and not args.thermal:
        parser.error("Provide --sensor and/or --thermal.")
    if args.sensor:
        features = sensor_window_features(
            args.sensor, window_seconds=args.window_seconds,
            max_windows=args.max_windows,
        )
        print("Sensor window features (not paired to thermal frames):")
        print(features.to_string(index=False))
    if args.thermal:
        features = thermal_frame_features(args.thermal, max_frames=args.max_frames)
        print("Thermal image features (not paired to sensor windows):")
        print(features.to_string(index=False))


if __name__ == "__main__":
    main()
