"""All the code of the project in one file (PROJ 613, Group 9).

Sensor-vs-image fault classification for eight mining-machine mappings, using public laboratory data.
The sections follow the workflow, top to bottom:

    0   project folders
    1   load data: raw sensor files -> feature rows (1a), raw camera files -> PNG images (1b), data check (1c)
    2   split: the fair test and the time split
    3   train and score: one answer per recording (3a), sensor model (3b), image model (3c)
    4   the eight machines: their datasets, classes and what each fair test holds out
    5   run the workflow for one machine and save Results/ and Figures/
    6   predict a new file (6b: an upload page for it)
    7   tables and figures for the notebook
    8   command line

Use it from the notebook (import pdm) or from a terminal in the project folder:

    python Notebook/pdm.py <machine> sensor [--time-split]     sensor model -> Results/<machine>/sensor*.json
    python Notebook/pdm.py <machine> export-images             raw camera files -> Images/<machine>/
    python Notebook/pdm.py <machine> images [--time-split]     image model -> Results/<machine>/images*.json
    python Notebook/pdm.py <machine> train-final               sensor model on all data, for predict
    python Notebook/pdm.py <machine> train-final-images        image model on all images, for predict
    python Notebook/pdm.py predict <file> [--machine <name>]   classify one new recording or image
    python Notebook/pdm.py serve                               upload page at http://localhost:8613
    python Notebook/pdm.py data-check                          list missing or changed raw files

Machines: pump, haul_truck, belt_conveyor, excavator, dozer, wheel_loader, drill_rig, grader
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
import random
import re
import shutil
import warnings


# ==================================================================================================
# 0. Project folders (Raw_Data, Processed_Data and Images are not in Git)
# ==================================================================================================

ROOT = Path(__file__).resolve().parents[1]  # this file is in Notebook/
RAW = ROOT / "Raw_Data"
PROCESSED = ROOT / "Processed_Data"
IMAGES = ROOT / "Images"
RESULTS = ROOT / "Results"
FIGURES = ROOT / "Figures"
MODELS = PROCESSED / "models"
MANIFEST = ROOT / "Dataset_Manifest"  # file lists (with SHA-256) of the Pump and Haul Truck recordings


# ==================================================================================================
# 1a. Sensor data: one feature row per time window
# Columns: unit (recording), class, window (position, for the time split), the group the fair test
# holds out, and features computed from the signal only, never from a label.
# Sources: Ottawa (Pump), UODS (Haul Truck), CORA (Belt Conveyor, Excavator, Dozer, Wheel Loader),
# MCC5-THU (Drill Rig), JUST (Grader).
# ==================================================================================================

def window_statistics(windows, rate, bands_hz):
    """Time and frequency features of each row of `windows` (mean removed per window)."""
    centered = windows - windows.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    safe_std = np.maximum(std, 1e-12)
    frequency = np.fft.rfftfreq(windows.shape[1], d=1 / rate)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    relative = power / np.maximum(power.sum(axis=1), 1e-12)[:, None]
    return {
        "std": std,
        "crest_factor": np.max(np.abs(centered), axis=1) / safe_std,
        "skew": np.mean(centered**3, axis=1) / safe_std**3,
        "excess_kurtosis": np.mean(centered**4, axis=1) / safe_std**4 - 3,
        "spectral_centroid_hz": (relative * frequency[None, :]).sum(axis=1),
        "dominant_frequency_hz": frequency[np.argmax(power[:, 1:], axis=1) + 1],
        "spectral_entropy": -(relative * np.log(np.maximum(relative, 1e-12))).sum(axis=1),
        **{f"power_{lo}_{hi}_hz": relative[:, (frequency >= lo) & (frequency < hi)].sum(axis=1)
           for lo, hi in bands_hz},
    }


# --- Pump: University of Ottawa bearing rig, 200 kHz accelerometer, 48 recordings of 10 s ---

OTTAWA_BANDS = ((0, 2_000), (2_000, 5_000), (5_000, 10_000), (10_000, 20_000), (20_000, 50_000),
                (50_000, 100_000))
OTTAWA_FEATURES = ["mean", "std", "rms", "peak_to_peak", "crest_factor", "skew", "excess_kurtosis",
                   "spectral_centroid_hz", "dominant_frequency_hz", "spectral_entropy",
                   *(f"power_{lo}_{hi}_hz" for lo, hi in OTTAWA_BANDS)]


def ottawa_recording(path):
    """Features of every 100 ms window (20,000 samples at 200 kHz) of one .mat recording."""
    from scipy.io import loadmat

    signal = loadmat(path, variable_names=["Channel_1"])["Channel_1"].ravel().astype(np.float64)

    return ottawa_signal_features(signal)


def ottawa_signal_features(signal):
    size = 20_000
    windows = signal[: len(signal) // size * size].reshape(-1, size)
    stats = window_statistics(windows, 200_000, OTTAWA_BANDS)
    stats.update(mean=windows.mean(axis=1), rms=np.sqrt(np.mean(windows**2, axis=1)),
                 peak_to_peak=np.ptp(windows, axis=1))
    return pd.DataFrame({name: stats[name] for name in OTTAWA_FEATURES}).astype(np.float32)


def ottawa_bearing():
    """The fair test holds out speed profiles C and D."""
    files = json.loads((MANIFEST / "pump_ottawa_files.json").read_text("utf-8"))
    classes = {"H": "healthy", "I": "inner", "O": "outer", "B": "ball"}
    frames = []
    for item in files:
        frame = ottawa_recording(RAW / "uottawa_bearing" / item["name"])
        frame.insert(0, "window", np.arange(len(frame)))
        frame.insert(0, "class", classes[item["class_code"]])
        frame.insert(0, "speed_profile", item["speed_profile"])
        frame.insert(0, "unit", item["name"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), {"vibration": OTTAWA_FEATURES}


# --- Haul Truck: UODS bearing rig, 42 kHz accelerometer and microphone, 30 recordings of 10 s ---

UODS_BANDS = ((0, 1_000), (1_000, 2_000), (2_000, 5_000), (5_000, 10_000), (10_000, 21_000))
UODS_FEATURES = ["std", "peak_to_peak", "crest_factor", "skew", "excess_kurtosis",
                 "spectral_centroid_hz", "dominant_frequency_hz", "spectral_entropy",
                 *(f"power_{lo}_{hi}_hz" for lo, hi in UODS_BANDS)]
ENVELOPE_BANDS = ((5, 50), (50, 90), (90, 130), (130, 180), (180, 250), (250, 400), (400, 800))
ENVELOPE_FEATURES = ["envelope_kurtosis", "envelope_peak_hz",
                     *(f"envelope_power_{lo}_{hi}_hz" for lo, hi in ENVELOPE_BANDS)]


def uods_windows(signal):
    """100 ms windows (4,200 samples). The raw mean is left out on purpose: inner-race files carry
    a DC offset that would let a model classify by sensor offset instead of vibration."""
    size = 4_200
    windows = signal[: len(signal) // size * size].reshape(-1, size)
    stats = window_statistics(windows, 42_000, UODS_BANDS)
    stats["peak_to_peak"] = np.ptp(windows - windows.mean(axis=1, keepdims=True), axis=1)
    return np.column_stack([stats[name] for name in UODS_FEATURES]).astype(np.float32)


def envelope(signal, n_windows, rate=42_000, window=4_200):
    """Envelope analysis, the standard bearing-fault method: band-pass 1-15 kHz, take the Hilbert
    envelope and measure where its energy sits (fault 'knock' rates are 50-400 Hz)."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    sos = butter(4, (1_000, 15_000), btype="bandpass", fs=rate, output="sos")
    band = sosfiltfilt(sos, signal - signal.mean())
    env = np.abs(hilbert(band))[: n_windows * window].reshape(n_windows, window)
    env = env - env.mean(axis=1, keepdims=True)
    std = np.maximum(env.std(axis=1), 1e-12)
    freq = np.fft.rfftfreq(window, d=1 / rate)
    power = np.abs(np.fft.rfft(env, axis=1)) ** 2
    low = (freq >= 5) & (freq < 800)
    share = power[:, low] / np.maximum(power[:, low].sum(axis=1, keepdims=True), 1e-12)
    f = freq[low]
    return np.column_stack([(env**4).mean(axis=1) / std**4, f[np.argmax(share, axis=1)],
                            *[share[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in ENVELOPE_BANDS]])


LONG_ENVELOPE = [f"long_{c}" for c in ENVELOPE_FEATURES]
MICROPHONE = [f"mic_{c}" for c in UODS_FEATURES]


def uods_recording(path):
    """One .mat recording: column 0 accelerometer, column 1 microphone (speed and load unused)."""
    from scipy.io import loadmat

    data = loadmat(path)[Path(path).stem]
    signal, microphone = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    return uods_signal_features(signal, microphone)


def uods_signal_features(signal, microphone):
    frame = pd.DataFrame(uods_windows(signal), columns=UODS_FEATURES)
    frame[ENVELOPE_FEATURES] = envelope(signal, len(frame))
    # 0.5 s envelope windows (2 Hz resolution), repeated for their five 0.1 s rows
    frame[LONG_ENVELOPE] = np.repeat(envelope(signal, len(frame) // 5, window=21_000), 5, axis=0)[: len(frame)]
    frame[MICROPHONE] = uods_windows(microphone)[: len(frame)]
    return frame


def uods_bearing():
    """Five rounds, each holding out three whole bearings; vibration, envelope and microphone sets."""
    files = json.loads((MANIFEST / "haul_truck_uods_files.json").read_text("utf-8"))
    classes = {"H": "healthy", "I": "inner", "O": "outer", "B": "ball"}
    frames = []
    for item in files:
        name = item["name"]
        frame = uods_recording(RAW / "uods_vafdc" / name)
        frame.insert(0, "window", np.arange(len(frame)))
        frame.insert(0, "class", classes[item["class_code"]])
        frame.insert(0, "bearing", item["bearing"])
        frame.insert(0, "unit", name[:-4])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), {
        "vibration": UODS_FEATURES,
        "vibration+envelope": UODS_FEATURES + ENVELOPE_FEATURES,
        "vibration+long envelope": UODS_FEATURES + LONG_ENVELOPE,
        "vibration+microphone": UODS_FEATURES + MICROPHONE,
        "vibration+long envelope+microphone": UODS_FEATURES + LONG_ENVELOPE + MICROPHONE,
    }


# --- CORA motor and gearbox rig: 3 kHz Vx vibration, one run per class and speed (F5/F15/F50/F60) ---

CORA_ORDERS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 8.5), (8.5, 12.5), (12.5, 20.5))
CORA_FEATURES = ["log_rms", "kurtosis", "skewness", "crest_factor", "peak_order", "log_ratio_2x_1x",
                 *(f"order_power_{lo}_{hi}" for lo, hi in CORA_ORDERS)]
TEMPERATURE = ["t1_mean", "t1_std", "t1_slope"]


def cora_windows(values, speed):
    """1 s windows (every 10th), measured in shaft orders (frequency / speed setting) so 1x, 2x and
    higher multiples line up across speeds: unbalance shows at 1x, misalignment at 2x, bearing and
    gear damage at higher orders and in spikes."""
    rate, shaft = 3000, float(speed[1:])
    windows = values[: len(values) // rate * rate].reshape(-1, rate)[::10]
    centered = windows - windows.mean(axis=1, keepdims=True)
    std = np.maximum(centered.std(axis=1), 1e-12)
    freq = np.fft.rfftfreq(rate, d=1 / rate)
    power = np.abs(np.fft.rfft(centered * np.hanning(rate), axis=1)) ** 2
    order = freq / shaft
    usable = (order >= 0.5) & (order < 20.5)
    share = power[:, usable] / np.maximum(power[:, usable].sum(axis=1, keepdims=True), 1e-12)
    o = order[usable]
    bands = [share[:, (o >= lo) & (o < hi)].sum(axis=1) for lo, hi in CORA_ORDERS]
    columns = [np.log(std), (centered**4).mean(axis=1) / std**4, (centered**3).mean(axis=1) / std**3,
               np.abs(centered).max(axis=1) / std, o[np.argmax(share, axis=1)],
               np.log((bands[1] + 1e-9) / (bands[0] + 1e-9)), *bands]
    return pd.DataFrame(np.column_stack(columns), columns=CORA_FEATURES)


def cora_rig(classes, temperature=False):
    """classes: {file code: class}, e.g. {"H": "healthy", "BD": "bearing"}. The fair test holds out
    one whole speed at a time. temperature=True adds run-level T1 temperature features (Belt Conveyor)."""
    if temperature:
        # T1 was downloaded as a bounded sample (8 x 6000 readings per run), not the full stream
        t1 = np.load(PROCESSED / "cora_gear_wear" / "t1_bounded_samples.npz")
    frames = []
    for code, label in classes.items():
        for speed in ("F5", "F15", "F50", "F60"):
            run = f"{code}_{speed}_S"
            values = pd.read_csv(RAW / "cora_vibration" / f"{run}_Vx.csv", header=None,
                                 dtype=np.float64).iloc[:, 0].to_numpy()
            frame = cora_windows(values, speed)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", label)
            frame.insert(0, "speed", speed)
            frame.insert(0, "unit", run)
            if temperature:
                samples = t1[run].astype(np.float64)
                frame["t1_mean"], frame["t1_std"] = samples.mean(), samples.std()
                frame["t1_slope"] = np.polyfit(np.arange(8), samples.mean(axis=1), 1)[0]
            frames.append(frame)
    sets = {"vibration": CORA_FEATURES}
    if temperature:
        sets["vibration+temperature"] = CORA_FEATURES + TEMPERATURE
    return pd.concat(frames, ignore_index=True), sets


# --- Drill Rig: MCC5-THU gearbox, 12.8 kHz, six vibration channels, 120 recordings of 60 s ---

MCC5_ZIP = RAW / "mcc5_thu_gearbox" / "MCC5-THU gearbox fault diagnosis datasets.zip"
MCC5_CHANNELS = ["motor_vibration_x", "motor_vibration_y", "motor_vibration_z",
                 "gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"]
MCC5_FEATURES = [f"{c}_{k}" for c in MCC5_CHANNELS
                 for k in ("rms", "std", "peak", "crest", "skew", "kurtosis", *(f"band{b}" for b in range(8)))]


def mcc5_recordings():
    """Class and working condition of each single-fault recording, from its file name."""
    import re
    import zipfile

    prefixes = {"health": "healthy", "gear_wear": "gear_wear", "gear_pitting": "gear_pitting",
                "teeth_crack": "tooth_crack"}
    rows = []
    with zipfile.ZipFile(MCC5_ZIP) as archive:
        for name in archive.namelist():
            if not name.endswith(".csv") or "__MACOSX" in name:
                continue
            base = name.rsplit("/", 1)[-1][:-4]
            prefix = next((p for p in prefixes if base.startswith(p + "_")), None)
            if prefix is None:
                continue
            rest = base[len(prefix) + 1:]
            if prefixes[prefix] != "healthy":
                rest = rest.split("_", 1)[1]  # drop the severity (L/M/H)
            if rest.startswith(("bearing", "and")):  # compound faults are not used
                continue
            rows.append({"file": name, "unit": base, "class": prefixes[prefix],
                         "rpm": int(re.search(r"(\d+)rpm", rest).group(1))})
    return rows


def mcc5_recording(handle):
    """Features of every 1 s window (12,800 samples) of one recording's CSV, per vibration channel:
    RMS, std, peak, crest, skewness, kurtosis and 8 relative log band energies."""
    size = 12_800
    signal = pd.read_csv(handle, usecols=MCC5_CHANNELS, dtype=np.float64)[MCC5_CHANNELS].to_numpy()
    windows = signal[: len(signal) // size * size].reshape(-1, size, len(MCC5_CHANNELS))
    centered = windows - windows.mean(axis=1, keepdims=True)
    sd = centered.std(axis=1) + 1e-12
    rms = np.sqrt(np.mean(windows**2, axis=1))
    peak = np.abs(windows).max(axis=1)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    edges = np.linspace(0, spectrum.shape[1], 9).astype(int)
    total = spectrum.sum(axis=1) + 1e-12
    bands = [np.log(spectrum[:, a:b].sum(axis=1) / total + 1e-12) for a, b in zip(edges[:-1], edges[1:])]
    per_channel = [rms, sd, peak, peak / (rms + 1e-12), np.mean(centered**3, axis=1) / sd**3,
                   np.mean(centered**4, axis=1) / sd**4, *bands]
    values = np.stack(per_channel, axis=2).reshape(len(windows), -1)  # channel-major order
    return pd.DataFrame(values, columns=MCC5_FEATURES)


def mcc5_gearbox():
    """The fair test holds out one speed (1000, 2000 or 3000 rpm) at a time."""
    import zipfile

    frames = []
    with zipfile.ZipFile(MCC5_ZIP) as archive:
        for r in mcc5_recordings():
            with archive.open(r["file"]) as handle:
                frame = mcc5_recording(handle)
            frame.insert(0, "window", np.arange(len(frame)))
            frame.insert(0, "class", r["class"])
            frame.insert(0, "rpm", r["rpm"])
            frame.insert(0, "unit", r["unit"])
            frames.append(frame)
            print(f"  {r['unit']}: {len(frame)} windows", flush=True)
    return pd.concat(frames, ignore_index=True), {"vibration": MCC5_FEATURES}


# Grader: JUST slewing-bearing rig, six accelerometers at 50 kHz, 1 s windows.
JUST_DIR = RAW / "just_slewing"
JUST_SHA256 = {
    "condition1": "27b6add40a742668c52bfe683a23d33de398aaa59dbbd5fa9ec756c576de36e3",
    "condition2": "3692e2173577e0df08bc823702debfbee81a9756892d304f5b9ac9a1d3e46878",
    "condition3": "f31e147bdc9ed657604bab9c8cb114531777aa38d3b7a28257974611f7caa2ef",
    "condition4": "74ccb0d69c0f24b35a5823ed4ace5ffa597a201950489e201340f52f3e098e48",
    "condition5": "f94ec5334a33a84d908c9675a1d1acd4ad0c3d2f94ae75740af53dbd240caa7f",
    "condition6": "7c30f4bc7cae20669ae0337183ead3eaafb2dd72965d5c63ef28d5da3a696a86",
}
JUST_CLASS = {"N": "healthy", "I": "inner_ring", "O": "outer_ring", "B1": "rolling_element"}
JUST_CHANNELS = [f"AI {n} (m/s2)" for n in range(1, 7)]  # AI 7 is a microphone: not used
JUST_FEATURES = [f"AI{n}_{k}" for n in range(1, 7)
                 for k in ("rms", "std", "peak", "crest", "skew", "kurtosis", *(f"band{b}" for b in range(8)))]


def just_recordings():
    """The 120 recordings: 4 classes x 5 repeats x 6 working conditions (2 or 6 rpm, 0/30/60 N)."""
    import zipfile

    rows = []
    for n in range(1, 7):
        with zipfile.ZipFile(JUST_DIR / f"condition{n}.zip") as archive:
            for name in archive.namelist():
                match = re.search(r"(\d{8})-(N|I|O|B1)-(\d+)rpm-(\d+)N-(\d+)\.csv$", name)
                if match:
                    code, rpm, force, repeat = match.group(2), *map(int, match.groups()[2:])
                    rows.append({"zip": f"condition{n}", "file": name, "unit": f"condition{n}_{code}_{repeat}",
                                 "class": JUST_CLASS[code], "condition": f"condition{n}", "rpm": rpm,
                                 "force_n": force})
    return rows


def just_recording(handle, rate=50_000):
    """Every whole 1 s window of one recording: per channel 6 statistics and 8 relative log band energies."""
    signal = pd.read_csv(handle, usecols=JUST_CHANNELS, dtype="float32")[JUST_CHANNELS].to_numpy(np.float64)
    rows = []
    for w in range(len(signal) // rate):
        x = signal[w * rate:(w + 1) * rate]
        spectrum = np.abs(np.fft.rfft(x - x.mean(axis=0), axis=0)) ** 2
        edges = np.linspace(0, spectrum.shape[0], 9).astype(int)
        row = []
        for c in range(x.shape[1]):
            v = x[:, c]
            rms, centred = np.sqrt(np.mean(v ** 2)), v - v.mean()
            sd = centred.std() + 1e-12
            row += [rms, sd, np.abs(v).max(), np.abs(v).max() / (rms + 1e-12),
                    np.mean(centred ** 3) / sd ** 3, np.mean(centred ** 4) / sd ** 4]
            total = spectrum[:, c].sum() + 1e-12
            row += [np.log(spectrum[a:b, c].sum() / total + 1e-12) for a, b in zip(edges[:-1], edges[1:])]
        rows.append(row)
    return pd.DataFrame(rows, columns=JUST_FEATURES)


def _just_worker(r):
    """One recording -> its window rows (module level, so that Windows worker processes can run it)."""
    import zipfile

    with zipfile.ZipFile(JUST_DIR / f"{r['zip']}.zip") as archive, archive.open(r["file"]) as handle:
        frame = just_recording(handle)
    frame.insert(0, "window", np.arange(len(frame)))
    for key in ("force_n", "rpm", "condition", "class", "unit"):
        frame.insert(0, key, r[key])
    return frame


def just_slewing(workers=6):
    """The fair test holds out one working condition at a time (six rounds of 20 recordings)."""
    from concurrent.futures import ProcessPoolExecutor

    recordings = just_recordings()
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(_just_worker, recordings))
    print(f"  {len(recordings)} recordings, {sum(len(f) for f in frames)} windows", flush=True)
    return pd.concat(frames, ignore_index=True), {"vibration": JUST_FEATURES}


# ==================================================================================================
# 1b. Camera data: PNG images and index.csv in Images/<machine>/ (rebuilt from the raw files)
# index.csv columns: file, class, unit, group, order (frame number, for the time split).
# Sources: XJTU-DV event camera (Pump, Haul Truck), CORA infrared (the four CORA machines),
# MFGNet-Gear renders (Drill Rig), Roboflow gear photos (Grader).
# ==================================================================================================

INDEX_COLUMNS = ["file", "class", "unit", "group", "order"]


def write_index(folder, rows):
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    print(f"Saved {len(rows)} images to {folder} (per class: {counts})")


def read_index(folder):
    with (Path(folder) / "index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fresh(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


# --- Event camera: each pixel reports when its brightness changes; events are counted in 125 ms frames ---

EVENT = np.dtype([("t", "<u4"), ("xy", "<u4")])
HEIGHT, WIDTH = 96, 128


def event_counts(stream, seconds=30, bin_us=125_000):
    """Count events per frame, polarity (brighter / darker) and pixel, on a 96 x 128 grid."""
    while stream.read(1) == b"%":  # skip the text header
        stream.readline()
    stream.read(1)
    frames = seconds * 1_000_000 // bin_us
    counts = np.zeros((frames, 2, HEIGHT, WIDTH), dtype=np.uint32)
    end = seconds * 1_000_000
    while True:
        chunk = stream.read(8 * 500_000)
        if not chunk:
            break
        ev = np.frombuffer(chunk, dtype=EVENT)
        keep = ev["t"] < end
        t, xy = ev["t"][keep], ev["xy"][keep]
        x, y, polarity = xy & 0x3FFF, (xy >> 14) & 0x3FFF, (xy >> 28) & 1
        index = ((((t // bin_us) * 2 + (1 - polarity)) * HEIGHT + (y * HEIGHT // 480)) * WIDTH
                 + (x * WIDTH // 640)).astype(np.int64)
        counts += np.bincount(index, minlength=counts.size).reshape(counts.shape).astype(np.uint32)
        if len(ev) and ev["t"][-1] >= end:
            break
    return counts


def event_png(frame):
    """Brighter events in red, darker events in green, log-scaled to 0-255 (BGR for OpenCV)."""
    scaled = np.log1p(frame.astype(np.float32))
    scaled = (255 * scaled / max(float(scaled.max()), 1e-6)).astype(np.uint8)
    return np.dstack([np.zeros_like(scaled[0]), scaled[1], scaled[0]])


def event_camera(machine, out):
    """Pump: XJTU-DV-Pump at 20 Hz and 30 Hz. Haul Truck: XJTU-DV-Rotor at 1000 and 2000 rpm."""
    import zipfile

    import cv2

    folders = {"healthy": "Healthy", "inner": "Inner", "outer": "Outer", "ball": "Ball"}
    recordings = []  # (zip, member, class, unit, group, file name stem)
    if machine == "pump":
        for hz, part in (("20Hz", "part1"), ("30Hz", "part2")):
            archive = RAW / "xjtu_dv_pump" / f"XJTU-DV-Pump_{part}.zip"
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
            for label, folder in folders.items():
                for member in sorted(n for n in names if n.startswith(f"XJTU-DV-Pump/Data/{folder}/{hz[:2]}")
                                     and n.endswith(".dat")):
                    setup = "_".join(Path(member).stem.split("_")[-2:])
                    recordings.append((archive, member, label, f"{hz}_{label}_{setup}", hz,
                                       f"pump_{label}_{hz}_{setup}"))
    else:
        archive = RAW / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip"
        for label, folder in folders.items():
            for angle in (1, 2, 3):
                for rpm in (1000, 2000):
                    recordings.append((archive, f"Data/{folder}/angle{angle}_{folder}_{rpm}.dat", label,
                                       f"{rpm}rpm_{label}_angle{angle}", f"{rpm}rpm",
                                       f"haul_truck_{label}_{rpm}rpm_angle{angle}"))
    fresh(out)
    rows = []
    for archive, member, label, unit, group, stem in recordings:
        with zipfile.ZipFile(archive) as z, z.open(member) as stream:
            counts = event_counts(stream)
        (out / label).mkdir(exist_ok=True)
        for k, frame in enumerate(counts):
            if frame.sum() > 0:  # skip empty frames after the recording ends
                name = f"{label}/{stem}_t{k:03d}.png"
                cv2.imwrite(str(out / name), event_png(frame))
                rows.append({"file": name, "class": label, "unit": unit, "group": group, "order": k})
        print(f"{unit}: done", flush=True)
    write_index(out, rows)


# --- CORA infrared camera: 450 thermal frames per run, saved as 96 x 128 grey images ---

def cora_infrared(machine, classes, out, step=1):
    """classes: {file code: class}. step=3 keeps every 3rd frame (150 per run)."""
    import cv2
    from scipy.io import loadmat

    fresh(out)
    rows = []
    for code, label in classes.items():
        (out / label).mkdir(exist_ok=True)
        for speed in ("F5", "F15", "F50", "F60"):
            unit = f"{code}_{speed}_S"
            source = RAW / "cora_gear_wear" / f"{unit}.mat"
            if not source.is_file():
                source = RAW / "cora_drive_faults" / f"{unit}.mat"
            frames = loadmat(source, variable_names=["imagenes_celda"])["imagenes_celda"]
            for k, frame in enumerate(frames.flat):
                if k % step:
                    continue
                gray = cv2.cvtColor(cv2.resize(np.asarray(frame), (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY)
                name = f"{label}/{machine}_{label}_{speed}_f{k:03d}.png"
                cv2.imwrite(str(out / name), gray)
                rows.append({"file": name, "class": label, "unit": unit, "group": speed, "order": k})
            print(f"{unit}: done", flush=True)
    write_index(out, rows)


# --- MFGNet-Gear: synthetic CAD meshes of good, worn, pitted and root-broken gears ---

GEAR_ZIP = RAW / "mfgnet_gear" / "mesh_ply.zip"
GEAR_CLASS = {"G0": "healthy", "W0": "gear_wear", "P0": "gear_pitting", "R0": "tooth_crack"}
GEAR_LABELS = ["healthy", "gear_wear", "gear_pitting", "tooth_crack"]
GEAR_PER_CLASS, GEAR_SIZE, GEAR_SEED = 100, 224, 613
GEAR_VIEWS = [(90, 0), (-90, 0)]  # (elevation, azimuth): both gear faces, where the defects show


def read_ply(data):
    """Vertices (n, 3) and triangles (m, 3) from an ASCII or binary little-endian PLY file."""
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end : end + 2] == b"\r\n" else 1
    header = data[:end].decode("ascii", "replace").split("\n")
    kind = next(line.split()[1] for line in header if line.startswith("format"))
    types = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "char": "i1",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "short": "i2", "ushort": "u2"}
    elements, current = [], None
    for line in header:
        part = line.split()
        if part[:1] == ["element"]:
            current = {"name": part[1], "count": int(part[2]), "props": []}
            elements.append(current)
        elif part[:1] == ["property"]:
            current["props"].append(part[1:])
    body = data[end:]
    if kind == "ascii":
        lines, out, start = body.decode("ascii").split("\n"), {}, 0
        for e in elements:
            out[e["name"]] = lines[start : start + e["count"]]
            start += e["count"]
        return (np.array([l.split()[:3] for l in out["vertex"]], dtype=np.float64),
                np.array([l.split()[1:4] for l in out["face"]], dtype=np.int64))
    offset, result = 0, {}
    for e in elements:
        fields = []  # face colours are read past but never used
        for p in e["props"]:
            if p[0] == "list":  # triangle meshes only: fixed count of 3
                fields += [("n", "<" + types[p[1]]), ("index", "<" + types[p[2]], 3)]
            else:
                fields.append((p[1], "<" + types[p[0]]))
        dtype = np.dtype(fields)
        block = np.frombuffer(body, dtype, e["count"], offset)
        if "index" in dtype.names:
            result[e["name"]] = block["index"].astype(np.int64)
        else:
            result[e["name"]] = np.stack([block[k] for k in ("x", "y", "z")], 1).astype(np.float64)
        offset += dtype.itemsize * e["count"]
    return result["vertex"], result["face"]


def render_gear(vertices, faces, seed):
    """One 3-channel image per view: height, local surface relief (pits go dark) and shading.

    1.5 million points are sampled over the surface by area (so mesh resolution is not visible),
    projected with a z-buffer like a depth camera and lit by a low raking light.
    """
    import cv2

    size, rng = GEAR_SIZE, np.random.default_rng(seed)
    v = vertices - (vertices.max(0) + vertices.min(0)) / 2
    v /= np.linalg.norm(v[:, :2], axis=1).max()
    tri = v[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    keep = area > 0
    tri, cross, area = tri[keep], cross[keep], area[keep]
    normal = cross / area[:, None]
    pick = rng.choice(len(tri), 1_500_000, p=area / area.sum())
    a, b = rng.random((2, len(pick)))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    points = tri[pick, 0] + a[:, None] * (tri[pick, 1] - tri[pick, 0]) + b[:, None] * (tri[pick, 2] - tri[pick, 0])
    normals = normal[pick]
    images = []
    for elevation, azimuth in GEAR_VIEWS:
        e, z = np.radians(90 - elevation), np.radians(azimuth)
        turn = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        tilt = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        rotation = tilt @ turn
        p, n = points @ rotation.T, normals @ rotation.T
        n = np.where(n[:, 2:3] < 0, -n, n)  # face the camera (camera on +z)
        light = np.array([0.75, -0.40, 0.45])
        shade = 0.15 + 0.85 * np.clip(n @ (light / np.linalg.norm(light)), 0, 1)
        px = np.clip(((p[:, 0] / 1.15 + 1) / 2 * size).astype(int), 0, size - 1)
        py = np.clip(((1 - p[:, 1] / 1.15) / 2 * size).astype(int), 0, size - 1)
        pixel = py * size + px
        # each pixel averages the visible surface within 0.01 (about 0.3 mm) of the nearest point
        front = np.full(size * size, -10.0)
        np.maximum.at(front, pixel, p[:, 2])
        seen = p[:, 2] > front[pixel] - 0.01
        count = np.bincount(pixel[seen], minlength=size * size)
        filled = count > 0
        image = np.zeros(size * size, np.float32)
        depth = np.full(size * size, -10.0, np.float32)
        image[filled] = np.bincount(pixel[seen], shade[seen], size * size)[filled] / count[filled]
        depth[filled] = np.bincount(pixel[seen], p[seen, 2], size * size)[filled] / count[filled]
        image, depth = image.reshape(size, size), depth.reshape(size, size)
        # a pixel that missed the front surface shows a hidden surface: take its neighbour's value
        near = cv2.dilate(depth, np.ones((3, 3), np.uint8))
        hole = depth < near - 0.05
        image = np.where(hole, cv2.dilate(image * ~hole, np.ones((3, 3), np.uint8)), image)
        depth = np.where(hole, near, depth)
        solid = (depth > -5).astype(np.float32)
        local = cv2.blur(depth * solid, (9, 9)) / np.maximum(cv2.blur(solid, (9, 9)), 1e-6)
        relief = np.clip(0.5 + 400 * (depth - local), 0, 1) * solid
        height = np.clip((depth + 1) / 2, 0, 1) * solid
        images.append((np.dstack([height, relief, image]) * 255).astype(np.uint8))  # saved as BGR
    return images


def gear_meshes():
    import re
    import zipfile

    rows = []
    with zipfile.ZipFile(GEAR_ZIP) as archive:
        for name in archive.namelist():
            match = re.search(r"(T\d+ID\d+)(G0|P0|W0|R0)_(\d+)\.ply$", name, re.IGNORECASE)
            if match and "__MACOSX" not in name:
                rows.append({"member": name, "design": match.group(1), "class": GEAR_CLASS[match.group(2)],
                             "instance": int(match.group(3))})
    return rows


def render_batch(job):
    """Render one batch of gears (runs in a worker process)."""
    import zipfile

    import cv2

    out_folder, chosen = job
    rows = []
    with zipfile.ZipFile(GEAR_ZIP) as archive:
        for row in chosen:
            vertices, faces = read_ply(archive.read(row["member"]))
            gear = f"{row['design']}_{row['class']}_{row['instance']:05d}"
            for view, image in enumerate(render_gear(vertices, faces, GEAR_SEED + row["instance"])):
                cv2.imwrite(str(Path(out_folder) / f"{gear}_v{view}.png"), image)
                rows.append({"file": f"{gear}_v{view}.png", "class": row["class"], "unit": gear,
                             "group": row["design"], "order": row["instance"]})
    return rows


def gear_renders(out, workers=10):
    """100 randomly chosen gears per design and class (seed 613), two views each."""
    from concurrent.futures import ProcessPoolExecutor

    meshes = gear_meshes()
    rng = np.random.default_rng(GEAR_SEED)
    chosen = []
    for design in sorted({r["design"] for r in meshes}):
        for label in GEAR_LABELS:
            pool = [r for r in meshes if r["design"] == design and r["class"] == label]
            chosen += [pool[i] for i in sorted(rng.choice(len(pool), min(GEAR_PER_CLASS, len(pool)),
                                                          replace=False))]
    print(f"{len(meshes)} meshes; rendering {len(chosen)} gears x {len(GEAR_VIEWS)} views", flush=True)
    fresh(out)
    jobs = [(str(out), chosen[i : i + 50]) for i in range(0, len(chosen), 50)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, part in enumerate(pool.map(render_batch, jobs), 1):
            rows += part
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n * 50}/{len(chosen)} gears", flush=True)
    write_index(out, rows)


# Grader images: real photos of small gears (Roboflow), cut into one 224 px crop per labelled tooth.
TOOTH_ZIP = RAW / "roboflow_gear_photos" / "GearBox_adddingImg.v1i.coco.zip"
TOOTH_LABELS = ["good_tooth", "damaged_tooth"]
DAMAGED_TOOTH = {"badTeeth", "tooth failure", "teethCrack", "Gear tooth breakage", "Breakage",
                 "damage", "Wear and Tear", "pitting"}


def tooth_source(original):
    """The video or photo set an image came from, e.g. dot_on_top_1 from dot_on_top_1_front_frame_30_jpg."""
    match = re.match(r"(.+?_\d+)_(front|back)_frame_\d+", original)
    return match.group(1) if match else re.sub(r"[-_ ]?\d+(_jpe?g|_png)?$|_jpg$", "", original)


def tooth_crops(margin=0.15):
    """(crops (N, 224, 224, 3) BGR, rows) for every good or damaged tooth box; margin is a share of the box."""
    import zipfile

    import cv2

    crops, rows = [], []
    with zipfile.ZipFile(TOOTH_ZIP) as archive:
        for split in ("train", "valid", "test"):
            document = json.loads(archive.read(f"{split}/_annotations.coco.json"))
            names = {c["id"]: c["name"] for c in document["categories"]}
            images = {i["id"]: i for i in document["images"]}
            for box in document["annotations"]:
                name = names[box["category_id"]]
                if name != "goodTeeth" and name not in DAMAGED_TOOTH:
                    continue
                info = images[box["image_id"]]
                picture = cv2.imdecode(np.frombuffer(archive.read(f"{split}/{info['file_name']}"), np.uint8),
                                       cv2.IMREAD_COLOR)
                x, y, w, h = box["bbox"]
                pad = margin * max(w, h)
                x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
                x1, y1 = int(min(picture.shape[1], x + w + pad)), int(min(picture.shape[0], y + h + pad))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                original = info["file_name"].split(".rf.")[0]
                crops.append(cv2.resize(picture[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA))
                rows.append({"unit": tooth_source(original), "frame": original, "box_label": name,
                             "class": "good_tooth" if name == "goodTeeth" else "damaged_tooth"})
    return np.stack(crops), rows


# ==================================================================================================
# 1c. Data check: raw files present in Raw_Data/ with the right SHA-256 (python Notebook/pdm.py data-check)
# Download by hand from the sources in Dataset_Manifest/manifest.md.
# ==================================================================================================

CORA_RUNS = {
    "cora_gear_wear": [f"{c}_{s}_S.mat" for c in ("H", "W25", "W50", "W75") for s in ("F5", "F15", "F50", "F60")],
    "cora_drive_faults": [f"{c}_{s}_S.mat" for c in ("BD", "U", "M", "HB", "OB") for s in ("F5", "F15", "F50", "F60")],
    "cora_vibration": [f"{c}_{s}_S_Vx.csv" for c in ("H", "W25", "W50", "W75", "BD", "U", "M", "HB", "OB")
                       for s in ("F5", "F15", "F50", "F60")],
}


def required_files():
    """(path, expected SHA-256 or None) for every raw file used."""
    files = []
    for name, folder in (("pump_ottawa_files.json", "uottawa_bearing"),
                         ("haul_truck_uods_files.json", "uods_vafdc")):
        for item in json.loads((MANIFEST / name).read_text("utf-8")):
            files.append((RAW / folder / item["name"], item["source_sha256"]))
    for folder, names in CORA_RUNS.items():
        files += [(RAW / folder / n, None) for n in names]
    files += [(PROCESSED / "cora_gear_wear" / "t1_bounded_samples.npz", None),
              (RAW / "xjtu_dv_pump" / "XJTU-DV-Pump_part1.zip", None),
              (RAW / "xjtu_dv_pump" / "XJTU-DV-Pump_part2.zip", None),
              (RAW / "xjtu_dv_rotor" / "XJTU-DV-Rotor.zip", None),
              (RAW / "mcc5_thu_gearbox" / "MCC5-THU gearbox fault diagnosis datasets.zip",
               "dcf7ba4b2d375c951ea57baf9bf90db4b3196982b324a13d85b187f286116b86"),
              (RAW / "mfgnet_gear" / "mesh_ply.zip", None),
              (TOOTH_ZIP, "daae26100160b88554bd3c311e1b445376d59bcce2e4bd5ed48c41e2089136f6")]
    files += [(JUST_DIR / f"{name}.zip", digest) for name, digest in JUST_SHA256.items()]
    return files


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def check_data():
    missing, wrong = [], []
    for path, expected in required_files():
        if not path.is_file():
            missing.append(path)
        elif expected and path.stat().st_size < 200_000_000 and sha256(path) != expected:
            wrong.append(path)
    print(f"{len(required_files())} files checked: {len(missing)} missing, {len(wrong)} with a wrong SHA-256")
    for path in missing + wrong:
        print(f"  {'missing' if path in missing else 'WRONG SHA-256'}: {path.relative_to(ROOT)}")
    return 1 if missing or wrong else 0


# ==================================================================================================
# 2. Split
# Fair test (main): whole groups (e.g. one speed) are held out; no part of a test recording is trained on.
# Time split (easier, for comparison): first 60% of each recording trains, last 30% tests, 10% gap.
# ==================================================================================================

def fair_test_folds(groups, held_out):
    """One test mask per round: round k tests every row whose group is in held_out[k]."""
    groups = np.asarray(groups)
    return [np.isin(groups, values) for values in held_out]


def check_no_overlap(units, test):
    """Stop if any recording appears in both training and test (it would leak)."""
    if set(units[~test]) & set(units[test]):
        raise ValueError("A test recording also appears in training")


def time_split(units, order, train_until=0.6, test_after=0.7):
    """Cut every recording in time.

    units: recording of each row. order: position of the row in its recording (window or frame number).
    Returns (keep, test, new_units): keep drops the 10% gap; test marks the last 30%; new_units names
    each half separately (<recording>_early / <recording>_late) so the scoring treats them as two units.
    """
    import pandas as pd

    position = pd.Series(np.asarray(order)).groupby(np.asarray(units)).rank(pct=True).to_numpy()
    keep = (position <= train_until) | (position > test_after)
    late = position > test_after
    new_units = np.char.add(np.asarray(units).astype(str), np.where(late, "_late", "_early"))
    return keep, late[keep], new_units[keep]


# ==================================================================================================
# 3a. Scoring: one answer per recording, metrics and a confusion-matrix figure
# ==================================================================================================

def predict_recordings(probabilities, classes, y, units):
    """One prediction per test unit (recording, run or gear): average the class probabilities of its rows."""
    actual, predicted = [], []
    for unit in sorted(set(units)):
        rows = units == unit
        if len(set(y[rows])) != 1:
            raise ValueError(f"Mixed labels within {unit}")
        actual.append(y[rows][0])
        predicted.append(classes[int(np.argmax(probabilities[rows].mean(axis=0)))])
    return actual, predicted


def classification_metrics(actual, predicted, labels):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    return {
        "test_units": len(actual),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "per_class_recall": {label: round(float(r), 4) for label, r in zip(labels, recall)},
    }


def save_confusion_figure(panels, labels, title, path):
    """panels: [(name, score block)]; one confusion matrix per panel, saved as a small PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    for axis, (name, block) in zip(axes[0], panels):
        matrix = np.asarray(block["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(
            f"{name}\n{block['test_units']:,} tested, accuracy {block['accuracy']:.3f}", fontsize=9
        )
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=8,
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


# ==================================================================================================
# 3b. Sensor model
# Fair test: five candidate models and the feature sets are compared on validation groups inside the
# training data; the best is retrained and tested once. Time split: a fixed random forest.
# Both are compared with a majority-class control.
# ==================================================================================================

# silence scikit-learn 1.9's SVC(probability=True) deprecation notice
warnings.filterwarnings("ignore", message=r".*`probability` parameter was deprecated.*")


def sensor_candidates():
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=250, max_features="sqrt", class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "Extra trees": lambda: ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=42
        ),
        "SVM (RBF)": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        ),
        "Logistic regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000,
                               random_state=42),
        ),
    }


def majority_control():
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")


def choose_model(table, train_idx, feature_sets, labels, inner_group):
    """Pick (feature set, model) by macro F1 on validation groups inside the training rows only."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    y, units = table["class"].to_numpy(), table["unit"].to_numpy()
    groups = table[inner_group].to_numpy()
    n_splits = min(4, len(set(groups[train_idx])))
    inner = list(GroupKFold(n_splits=n_splits).split(train_idx, groups=groups[train_idx]))
    best, best_score = None, -1.0
    for set_name, names in feature_sets.items():
        x = table[names].to_numpy(dtype=np.float64)
        for model_name, factory in sensor_candidates().items():
            f1s = []
            for fit_part, val_part in inner:
                fit_rows, val_rows = train_idx[fit_part], train_idx[val_part]
                model = factory().fit(x[fit_rows], y[fit_rows])
                a, p = predict_recordings(model.predict_proba(x[val_rows]), model.classes_,
                                          y[val_rows], units[val_rows])
                f1s.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
            if float(np.mean(f1s)) > best_score:
                best, best_score = (set_name, model_name), float(np.mean(f1s))
    return best, best_score


def sensor_fair_test(table, feature_sets, labels, folds, inner_group):
    """Main result: model chosen inside each training set, then tested once per round."""
    y, units = table["class"].to_numpy(), table["unit"].to_numpy()
    actual, predicted, chosen = [], [], []
    control_actual, control_predicted = [], []
    first_set = table[next(iter(feature_sets.values()))].to_numpy(dtype=np.float64)
    for test in folds:
        check_no_overlap(units, test)
        train_idx = np.flatnonzero(~test)
        (set_name, model_name), score = choose_model(table, train_idx, feature_sets, labels,
                                                     inner_group)
        x = table[feature_sets[set_name]].to_numpy(dtype=np.float64)
        model = sensor_candidates()[model_name]().fit(x[train_idx], y[train_idx])
        a, p = predict_recordings(model.predict_proba(x[test]), model.classes_, y[test], units[test])
        actual += a
        predicted += p
        chosen.append({"feature_set": set_name, "model": model_name,
                       "validation_macro_f1": round(score, 4)})
        control = majority_control().fit(first_set[train_idx], y[train_idx])
        a, p = predict_recordings(control.predict_proba(first_set[test]), control.classes_,
                                  y[test], units[test])
        control_actual += a
        control_predicted += p
    return {
        "score": classification_metrics(actual, predicted, labels),
        "chosen_per_round": chosen,
        "majority_control": classification_metrics(control_actual, control_predicted, labels),
    }


def sensor_time_split_test(table, feature_names, labels):
    """Secondary result: early part of every recording trains, late part tests; fixed random forest."""
    order = table["window"] if "window" in table else table.groupby("unit").cumcount()
    keep, test, units = time_split(table["unit"].to_numpy(), order.to_numpy())
    x = table[feature_names].to_numpy(dtype=np.float64)[keep]
    y = table["class"].to_numpy()[keep]
    check_no_overlap(units, test)
    result = {}
    for name, factory in (("score", sensor_candidates()["Random forest"]),
                          ("majority_control", majority_control)):
        model = factory().fit(x[~test], y[~test])
        a, p = predict_recordings(model.predict_proba(x[test]), model.classes_, y[test], units[test])
        result[name] = classification_metrics(a, p, labels)
    return result


def fit_final_sensor_model(table, feature_sets, labels, inner_group):
    """For predictions on new files: choose on all data (inner groups), then fit on all data."""
    all_idx = np.arange(len(table))
    (set_name, model_name), score = choose_model(table, all_idx, feature_sets, labels, inner_group)
    names = feature_sets[set_name]
    model = sensor_candidates()[model_name]().fit(table[names].to_numpy(dtype=np.float64),
                                           table["class"].to_numpy())
    return {"model": model, "features": names, "model_name": model_name, "feature_set": set_name,
            "validation_macro_f1": round(score, 4)}


# ==================================================================================================
# 3c. Image model: ImageNet-pretrained ResNet-18, fine-tuned
# Epochs are chosen on validation data inside the training data, then the model is retrained and tested
# once. Each test recording or gear gets the mean of its images' probabilities.
# ==================================================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_images(folder, rows, normalise="per_image"):
    """Read PNGs as RGB arrays (N, 3, H, W).

    per_image: each image scaled to mean 0, std 1 on its own (no information from other images).
    imagenet: pixel / 255, then the ImageNet mean and std (used for the rendered gears).
    """
    import cv2

    images = []
    for row in rows:
        image = cv2.imread(str(folder / row["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing image {row['file']}: run export-images first")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if normalise == "imagenet":
            image = (image / 255.0).astype(np.float32)
            images.append(((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        else:
            image = image.astype(np.float32).transpose(2, 0, 1)
            images.append((image - image.mean()) / (image.std() + 1e-6))
    return np.stack(images).astype(np.float32)


def augment_frames(batch, generator):
    """Training-only changes for camera frames: flip, shift up to 8 px, brightness x0.8-1.2, noise."""
    import torch

    n = batch.shape[0]
    flip = torch.rand(n, generator=generator) < 0.5
    batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
    shifted = []
    for image in batch:
        dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=generator))
        shifted.append(torch.roll(image, shifts=(dy, dx), dims=(-2, -1)))
    batch = torch.stack(shifted)
    scale = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
    return batch * scale + 0.05 * torch.randn(batch.shape, generator=generator)


def augment_gears(batch, generator):
    """Training-only changes for gear renders: a gear looks the same turned or mirrored."""
    import torch

    k = int(torch.randint(0, 4, (1,), generator=generator))
    batch = torch.rot90(batch, k, dims=(-2, -1))
    if torch.rand(1, generator=generator) < 0.5:
        batch = batch.flip(-1)
    dy, dx = (int(v) for v in torch.randint(-6, 7, (2,), generator=generator))
    return torch.roll(batch, shifts=(dy, dx), dims=(-2, -1))


def device():
    """The GPU if PyTorch can use one, else the CPU (PDM_DEVICE=cpu forces the CPU).

    GPU and CPU results can differ slightly, so each image result records its device.
    """
    import os

    import torch

    if os.environ.get("PDM_DEVICE", "").lower() != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True  # repeatable GPU runs
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    if os.environ.get("PDM_DEVICE", "").lower() == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def new_resnet18(n_classes):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def image_probabilities(model, x, batch_size=128):
    import torch

    where = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.softmax(model(torch.from_numpy(x[i : i + batch_size]).to(where)), dim=1).cpu().numpy()
            for i in range(0, len(x), batch_size)
        ])


def fit_resnet(x, y, epochs, labels, settings, val=None):
    """Fine-tune a fresh ResNet-18; with val=(x, class names, units) also return macro F1 per epoch."""
    import torch
    from sklearn.metrics import f1_score
    from torch import nn

    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    where = device()
    model = new_resnet18(len(labels)).to(where)
    weight = None
    if settings["class_weights"]:
        counts = np.bincount(y, minlength=len(labels)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1))).to(where)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    xs, ys = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xs), generator=generator)
        for start in range(0, len(xs), 32):
            batch = order[start : start + 32]
            if len(batch) < 2:
                continue
            optimiser.zero_grad()
            # augmentation runs on the CPU with a seeded generator, then the batch moves to the device
            images = settings["augment"](xs[batch], generator).to(where)
            loss_fn(model(images), ys[batch].to(where)).backward()
            optimiser.step()
        if val is not None:
            a, p = predict_recordings(image_probabilities(model, val[0]), np.asarray(labels), val[1], val[2])
            history.append(f1_score(a, p, labels=labels, average="macro", zero_division=0))
    return model, history


def one_recording_per_class(seed=42):
    """Validation for the epoch choice: one random training recording of each class, per round."""
    rng = np.random.default_rng(seed)

    def choose(names, units, train, labels, round_number):
        picked = [rng.choice(sorted(set(units[train & (names == label)]))) for label in labels]
        return train & np.isin(units, picked)

    return choose


def evaluate_images(folder, rows, labels, folds, settings, choose_validation, train_mask=None, segments=10):
    """Fair test (or time split, when the folds and units come from time_split)."""
    names = np.asarray([row["class"] for row in rows])
    units = np.asarray([row["unit"] for row in rows])
    y = np.asarray([labels.index(c) for c in names])
    if settings.get("load_per_round"):  # large 224 px images: read only the rows a step needs
        class LazyImages:
            def __getitem__(self, mask):
                return load_images(folder, [rows[i] for i in np.flatnonzero(mask)], settings["normalise"])
        x = LazyImages()
    else:
        x = load_images(folder, rows, settings["normalise"])
    train_mask = np.ones(len(rows), bool) if train_mask is None else train_mask
    unit_actual, unit_predicted, image_actual, image_predicted = [], [], [], []
    segment_actual, segment_predicted, rounds = [], [], []
    for number, test in enumerate(folds):
        check_no_overlap(units, test)
        train = ~test & train_mask
        val = choose_validation(names, units, train, labels, number)
        fit_part = train & ~val
        _, history = fit_resnet(x[fit_part], y[fit_part], settings["max_epochs"], labels, settings,
                         (x[val], names[val], units[val]))
        epochs = int(np.argmax(history)) + 1
        model, _ = fit_resnet(x[train], y[train], epochs, labels, settings)
        p = image_probabilities(model, x[test])
        a, q = predict_recordings(p, np.asarray(labels), names[test], units[test])
        unit_actual += a
        unit_predicted += q
        image_actual += list(names[test])
        image_predicted += [labels[i] for i in p.argmax(axis=1)]
        if segments:
            # extra, related test units: each test recording cut in time order into `segments` parts
            for unit in sorted(set(units[test])):
                pu, true = p[units[test] == unit], names[test][units[test] == unit][0]
                for part in np.array_split(np.arange(len(pu)), min(segments, len(pu))):
                    segment_actual.append(true)
                    segment_predicted.append(labels[int(pu[part].mean(axis=0).argmax())])
        rounds.append({"epochs": epochs, "validation_macro_f1_by_epoch": np.round(history, 4).tolist(),
                       "test_images": int(test.sum())})
        print(f"round {number + 1}: validation chose {epochs} epochs; tested {int(test.sum())} images",
              flush=True)
    result = {
        "unit_level": classification_metrics(unit_actual, unit_predicted, labels),
        "image_level": classification_metrics(image_actual, image_predicted, labels),
        "rounds": rounds,
        "device": str(device()),
    }
    if segments:
        result["segment_level"] = classification_metrics(segment_actual, segment_predicted, labels)
    return result


def resnet18_features(crops, batch_size=64):
    """Frozen ImageNet ResNet-18: 512 numbers per BGR crop (the network is not trained)."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(*list(network.children())[:-1]).eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[start:start + batch_size][..., ::-1].copy()).permute(0, 3, 1, 2)
            out.append(encoder((batch.float() / 255 - mean) / std).flatten(1).numpy())
    return np.concatenate(out)


def evaluate_tooth_photos(seed=613):
    """Grader images: good vs damaged tooth crops, each video or photo set held out in turn.

    Main result: frozen ResNet-18 features + balanced logistic regression (C 0.1, crop margin 0.15),
    fixed in advance. Check: margin and C chosen inside the training sources only. Control: colour and
    brightness statistics only. 94% of crops are good teeth, so balanced accuracy is the headline number.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(x, y, c):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=c,
                                                                 class_weight="balanced")).fit(x, y)

    def score(actual, predicted):
        block = classification_metrics(list(actual), list(predicted), TOOTH_LABELS)
        block["damaged_tooth_recall"] = round(float(recall_score(actual, predicted, pos_label="damaged_tooth")), 4)
        return block

    crops, rows = tooth_crops(0.15)
    labels = np.asarray([r["class"] for r in rows])
    sources = np.asarray([r["unit"] for r in rows])
    wide, wide_rows = tooth_crops(0.5)
    if [r["frame"] for r in wide_rows] != [r["frame"] for r in rows]:
        raise ValueError("Crop order differs between margins")
    views = {0.15: resnet18_features(crops), 0.5: resnet18_features(wide)}
    x = crops.reshape(len(crops), -1, 3).astype(np.float32) / 255
    colour = np.concatenate([x.mean(1), x.std(1),
                             np.percentile(x, [5, 50, 95], axis=1).transpose(1, 0, 2).reshape(len(x), -1)], 1)
    settings = [(margin, c) for margin in views for c in (0.01, 0.1, 1.0)]
    fixed, control, selected = (np.empty(len(rows), dtype=object) for _ in range(3))
    choices = []
    for source in sorted(set(sources)):
        test = sources == source
        check_no_overlap(sources, test)
        train = np.flatnonzero(~test)
        fixed[test] = fit(views[0.15][train], labels[train], 0.1).predict(views[0.15][test])
        control[test] = fit(colour[train], labels[train], 0.1).predict(colour[test])
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        splits = [(train[a], train[b]) for a, b in inner.split(train, labels[train], sources[train])
                  if len(set(labels[train[b]])) == 2]
        best = max(settings, key=lambda s: np.mean([
            balanced_accuracy_score(labels[b], fit(views[s[0]][a], labels[a], s[1]).predict(views[s[0]][b]))
            for a, b in splits]))
        choices.append(best)
        selected[test] = fit(views[best[0]][train], labels[train], best[1]).predict(views[best[0]][test])
    counts = {label: int((labels == label).sum()) for label in TOOTH_LABELS}
    print(f"{len(rows)} tooth crops {counts} from {len(set(sources))} videos or photo sets", flush=True)
    return {"unit_level": score(labels, fixed), "settings_chosen_inside": {
                **score(labels, selected),
                "choices": {f"margin {m}, C {c}": choices.count((m, c)) for m, c in sorted(set(choices))}},
            "colour_statistics_control": score(labels, control), "crops": len(rows), "crops_by_class": counts,
            "sources": len(set(sources)), "headline_metric": "balanced_accuracy"}


# ==================================================================================================
# 4. The eight machines: data sources, classes and fair-test hold-outs
# All data is laboratory data, not mining equipment. A machine's sensor and image data come from
# different rigs, so its two scores are not a matched pair.
# ==================================================================================================

CORA_SPEEDS = [["F5"], ["F15"], ["F50"], ["F60"]]
# Camera frames: epochs chosen on one random training recording per class; up to 15 epochs.
FRAME_IMAGES = {"normalise": "per_image", "augment": augment_frames, "class_weights": True,
                "max_epochs": 15, "seed": 42, "validation": "one_recording_per_class",
                "segments": 10}
# Gear renders: epochs chosen on the next round's three designs (inside the training data); up to
# 6 epochs; at most 50 training gears per design and class to keep CPU time manageable.
GEAR_IMAGES = {"normalise": "imagenet", "augment": augment_gears, "class_weights": False,
               "max_epochs": 6, "seed": 613, "validation": "next_round_groups",
               "train_per_group_and_class": 50, "segments": None, "load_per_round": True}


@dataclass
class Machine:
    title: str
    component: str
    labels: list
    sensor_source: str
    load_sensor: callable           # () -> (feature table, {feature set name: columns})
    sensor_group: str               # column the fair test holds out
    sensor_held_out: list           # one list of group values per test round
    inner_group: str                # groups used for validation inside the training data
    image_source: str
    export_images: callable         # (output folder) -> writes PNGs and index.csv
    image_held_out: list
    image_settings: dict = field(default_factory=lambda: dict(FRAME_IMAGES))
    time_split: bool = True         # also report the easier time split
    notes: list = field(default_factory=list)
    image_labels: list = None       # image classes when they differ from the sensor classes (Grader)


MACHINES = {
    "pump": Machine(
        title="Pump", component="support bearing",
        labels=["healthy", "inner", "outer", "ball"],
        sensor_source="University of Ottawa bearing rig, Huang and Baddour, Mendeley Data v43hmbwxpm v2, CC BY 4.0",
        load_sensor=ottawa_bearing,
        sensor_group="speed_profile", sensor_held_out=[["C", "D"]], inner_group="unit",
        image_source="XJTU-DV-Pump event camera, Zenodo 17447083, CC BY 4.0",
        export_images=lambda out: event_camera("pump", out),
        image_held_out=[["30Hz"]],
        time_split=False,
        notes=["sensor: train on speed profiles A/B, test 24 separate C/D recordings",
               "images: train on 20 Hz recordings, test 24 separate 30 Hz recordings"],
    ),
    "haul_truck": Machine(
        title="Haul Truck", component="wheel-motor bearing",
        labels=["healthy", "inner", "outer", "ball"],
        sensor_source="UODS-VAFDC bearing rig (vibration and microphone), CC BY 4.0",
        load_sensor=uods_bearing,
        sensor_group="bearing", sensor_held_out=[[k, k + 5, k + 10] for k in range(1, 6)],
        inner_group="bearing",
        image_source="XJTU-DV-Rotor event camera, Zenodo 17447083, CC BY 4.0",
        export_images=lambda out: event_camera("haul_truck", out),
        image_held_out=[["2000rpm"]],
        notes=["sensor: five rounds, each holding out three whole bearings",
               "sensor: ball faults were recorded without load, so ball results may partly reflect load",
               "images: train on 12 recordings at 1000 rpm, test 12 separate recordings at 2000 rpm"],
    ),
    "belt_conveyor": Machine(
        title="Belt Conveyor", component="geared drive",
        labels=["healthy", "wear25", "wear50", "wear75"],
        sensor_source="CORA motor/gearbox rig, gear-wear runs (Vx vibration, T1 temperature), CC BY 4.0",
        load_sensor=lambda: cora_rig(
            {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}, temperature=True),
        sensor_group="speed", sensor_held_out=CORA_SPEEDS, inner_group="speed",
        image_source="CORA infrared camera, same gear-wear runs, all 450 frames per run",
        export_images=lambda out: cora_infrared(
            "belt_conveyor", {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}, out),
        image_held_out=CORA_SPEEDS,
        notes=["four rounds, each holding out one whole speed; one run per class and speed"],
    ),
    "excavator": Machine(
        title="Excavator", component="drive train",
        labels=["healthy", "bearing", "unbalance", "misalignment"],
        sensor_source="CORA motor/gearbox rig, drive-fault runs (Vx vibration), CC BY 4.0",
        load_sensor=lambda: cora_rig(
            {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"}),
        sensor_group="speed", sensor_held_out=CORA_SPEEDS, inner_group="speed",
        image_source="CORA infrared camera, same runs, all 450 frames per run",
        export_images=lambda out: cora_infrared(
            "excavator", {"H": "healthy", "BD": "bearing", "U": "unbalance", "M": "misalignment"}, out),
        image_held_out=CORA_SPEEDS,
        notes=["four rounds, each holding out one whole speed; one run per class and speed"],
    ),
    "dozer": Machine(
        title="Dozer", component="final drive",
        labels=["healthy", "wear25", "wear50", "wear75"],
        sensor_source="CORA gear-wear runs: the SAME data as the Belt Conveyor (no public dozer data)",
        load_sensor=lambda: cora_rig(
            {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}, temperature=True),
        sensor_group="speed", sensor_held_out=CORA_SPEEDS, inner_group="speed",
        image_source="CORA infrared camera, same runs as the Belt Conveyor, every 3rd frame (150 per run)",
        export_images=lambda out: cora_infrared(
            "dozer", {"H": "healthy", "W25": "wear25", "W50": "wear50", "W75": "wear75"}, out, step=3),
        image_held_out=CORA_SPEEDS,
        notes=["reuses the Belt Conveyor's data, so it is not independent evidence"],
    ),
    "wheel_loader": Machine(
        title="Wheel Loader", component="electric drive",
        labels=["healthy", "bearing", "half_broken_bar", "broken_bar"],
        sensor_source="CORA motor/gearbox rig, rotor-bar and bearing runs (Vx vibration), CC BY 4.0",
        load_sensor=lambda: cora_rig(
            {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}),
        sensor_group="speed", sensor_held_out=CORA_SPEEDS, inner_group="speed",
        image_source="CORA infrared camera, same runs, all 450 frames per run",
        export_images=lambda out: cora_infrared(
            "wheel_loader", {"H": "healthy", "BD": "bearing", "HB": "half_broken_bar", "OB": "broken_bar"}, out),
        image_held_out=CORA_SPEEDS,
        notes=["four rounds, each holding out one whole speed; healthy and bearing runs are shared with the Excavator"],
    ),
    "drill_rig": Machine(
        title="Drill Rig", component="rotary-head reduction gear",
        labels=["healthy", "gear_wear", "gear_pitting", "tooth_crack"],
        sensor_source="MCC5-THU gearbox, Mendeley Data p92gj2732w v2, CC BY 4.0",
        load_sensor=mcc5_gearbox,
        sensor_group="rpm", sensor_held_out=[[1000], [2000], [3000]], inner_group="rpm",
        image_source="MFGNet-Gear synthetic CAD meshes (doi 10.7302/qrdj-n812, CC BY 4.0), rendered; not photographs",
        export_images=gear_renders,
        image_held_out=[["T20ID10", "T30ID20", "T40ID30"], ["T20ID15", "T30ID30", "T40ID40"],
                        ["T20ID20", "T30ID40", "T40ID50"], ["T20ID25", "T30ID50", "T40ID60"]],
        image_settings=dict(GEAR_IMAGES),
        time_split=False,
        notes=["sensor: leave one speed out (1000 / 2000 / 3000 rpm)",
               "images: four rounds, each holding out three whole gear designs; one answer per gear",
               "images are noise-free renders, so the image score is an upper bound, not a field result"],
    ),
    "grader": Machine(
        title="Grader", component="circle-drive slewing bearing",
        labels=["healthy", "inner_ring", "outer_ring", "rolling_element"],
        sensor_source="JUST Slewing bearing datasets-1, Xiaodie Ren, Mendeley Data hwg8v5j8t6 v1 "
                      "(doi 10.17632/hwg8v5j8t6.1), CC BY 4.0",
        load_sensor=just_slewing,
        sensor_group="condition", sensor_held_out=[[f"condition{n}"] for n in range(1, 7)],
        inner_group="condition",
        image_source="GearBox_adddingImg v1 by Vjs, Roboflow Universe "
                     "(https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg), CC BY 4.0; real photos",
        export_images=None,
        image_held_out=None,        # each video or photo set held out in turn
        image_settings={"method": "frozen_resnet18_features", "seed": 613},
        image_labels=TOOTH_LABELS,
        time_split=False,
        notes=["sensor: six rounds, each holding out one working condition (2 or 6 rpm, 0/30/60 N); "
               "one slewing support per class, so the model may partly learn the individual ring",
               "images: a SEPARATE two-class task (good vs damaged tooth crops of small gears), not the "
               "slewing ring; scored by balanced accuracy because 94% of crops are good teeth",
               "images: frozen ResNet-18 features + logistic regression, settings fixed in advance; "
               "choosing them inside the training sources is saved as a check"],
    ),
}


# ==================================================================================================
# 5. Run the workflow for one machine: load -> split -> train -> score -> save
#     run_sensor(machine[, "time"])    Results/<machine>/sensor[_timesplit].json
#     run_export_images(machine)       Images/<machine>/
#     run_images(machine[, "time"])    Results/<machine>/images[_timesplit].json
#     run_train_final(machine), run_train_final_images(machine)   models for predict()
# ==================================================================================================

def save_result(machine, name, result, panels, title, labels=None):
    m = MACHINES[machine]
    path = RESULTS / machine / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_confusion_figure(panels, labels or m.labels, title, FIGURES / machine / f"{name}.png")
    print(f"Saved {path.relative_to(RESULTS.parent)}")


def load_sensor_table(machine):
    """Build the feature table from the raw files; a copy is saved as CSV for inspection."""
    table, feature_sets = MACHINES[machine].load_sensor()
    PROCESSED.mkdir(exist_ok=True)
    table.to_csv(PROCESSED / f"{machine}_sensor_features.csv", index=False)
    return table, feature_sets


def run_sensor(machine, split="fair"):
    m = MACHINES[machine]
    table, feature_sets = load_sensor_table(machine)
    common = {"machine": machine, "source": m.sensor_source, "classes": m.labels,
              "rows": int(len(table)), "recordings": int(table["unit"].nunique()), "notes": m.notes}
    if split == "time":
        first_set = next(iter(feature_sets.values()))
        scores = sensor_time_split_test(table, first_set, m.labels)
        result = {**common, "test": "time split (secondary, easier): first 60% of every recording trains, "
                  "last 30% tests, 10% gap; fixed random forest", "features": first_set, **scores}
        name = "sensor_timesplit"
    else:
        folds = fair_test_folds(table[m.sensor_group], m.sensor_held_out)
        scores = sensor_fair_test(table, feature_sets, m.labels, folds, m.inner_group)
        result = {**common, "test": f"fair test: hold out {m.sensor_group} {m.sensor_held_out}, one round "
                  "each; model and feature set chosen on validation groups inside the training data",
                  "feature_sets": feature_sets, **scores}
        name = "sensor"
    s = result["score"]
    print(f"{m.title} sensor ({split}): accuracy {s['accuracy']:.3f}, macro F1 {s['macro_f1']:.3f} "
          f"on {s['test_units']} test recordings (majority control {result['majority_control']['accuracy']:.3f})")
    save_result(machine, name, result, [("per test recording", s)],
         f"{m.title}: sensor model, {'time split' if split == 'time' else 'fair test'}")
    return result


def run_export_images(machine):
    if MACHINES[machine].export_images is None:
        print(f"{machine}: nothing to export; its images are read straight from the raw zip")
        return
    MACHINES[machine].export_images(IMAGES / machine)


def run_images(machine, split="fair"):
    m = MACHINES[machine]
    settings = m.image_settings
    if settings.get("method") == "frozen_resnet18_features":  # Grader: tooth crops, own test
        scores = evaluate_tooth_photos(settings["seed"])
        result = {"machine": machine, "source": m.image_source, "classes": m.image_labels, "notes": m.notes,
                  "test": "fair test: hold out each video or photo set in turn (frames of one video never on "
                          "both sides); frozen ImageNet ResNet-18 features + logistic regression", **scores}
        for title, block in (("fixed settings (main)", scores["unit_level"]),
                             ("settings chosen inside (check)", scores["settings_chosen_inside"]),
                             ("colour statistics (control)", scores["colour_statistics_control"])):
            print(f"{m.title} images, {title}: balanced accuracy {block['balanced_accuracy']:.3f}, "
                  f"damaged-tooth recall {block['damaged_tooth_recall']:.3f}")
        save_result(machine, "images", result,
                    [("ResNet-18 features, fixed settings", scores["unit_level"]),
                     ("settings chosen inside", scores["settings_chosen_inside"]),
                     ("colour statistics (control)", scores["colour_statistics_control"])],
                    f"{m.title}: tooth crops of real gear photos, unseen video or photo set", m.image_labels)
        return result
    folder = IMAGES / machine
    rows = read_index(folder)
    groups = np.asarray([r["group"] for r in rows])
    train_mask = None
    if split == "time":
        keep, test, units = time_split([r["unit"] for r in rows], [int(r["order"]) for r in rows])
        rows = [dict(r, unit=u) for r, u in zip((r for r, k in zip(rows, keep) if k), units)]
        folds = [test]
        description = "time split (secondary, easier): first 60% of every recording's frames train, last 30% test"
    else:
        folds = fair_test_folds(groups, m.image_held_out)
        description = f"fair test: hold out {m.image_held_out}, one round each"
    if settings["validation"] == "next_round_groups":
        choose = lambda names, units, train, labels, n: train & np.isin(groups, m.image_held_out[(n + 1) % len(folds)])
        limit = settings["train_per_group_and_class"]
        rank = {}
        for r in rows:
            rank.setdefault((r["group"], r["class"]), set()).add(int(r["order"]))
        rank = {key: sorted(ids) for key, ids in rank.items()}
        train_mask = np.asarray([rank[(r["group"], r["class"])].index(int(r["order"])) < limit for r in rows])
    else:
        choose = one_recording_per_class(settings["seed"])
    scores = evaluate_images(folder, rows, m.labels, folds, settings, choose, train_mask,
                                  settings["segments"])
    result = {"machine": machine, "source": m.image_source, "classes": m.labels, "images": len(rows),
              "test": description, "notes": m.notes,
              "settings": {k: (v.__name__ if callable(v) else v) for k, v in settings.items()}, **scores}
    u = result["unit_level"]
    print(f"{m.title} images ({split}): accuracy {u['accuracy']:.3f} on {u['test_units']} test recordings; "
          f"{result['image_level']['accuracy']:.3f} on {result['image_level']['test_units']} images")
    name = "images_timesplit" if split == "time" else "images"
    save_result(machine, name, result, [("every test image", result["image_level"]),
                                 ("per test recording" if machine != "drill_rig" else "per test gear", u)],
         f"{m.title}: ResNet-18 on images, {'time split' if split == 'time' else 'fair test'}")
    return result


def deployment_metadata(machine, modality):
    config = MACHINES[machine]
    return {
        "version": 1,
        "machine": machine,
        "modality": modality,
        "component": config.component,
        "source": config.sensor_source if modality == "sensor" else config.image_source,
        "training_scope": "Final fit on all available training data; separate from held-out evaluation",
    }


def run_train_final(machine, random_forest=False):
    """Sensor model for new files: chosen and trained on all of this machine's data."""
    import joblib

    m = MACHINES[machine]
    table, feature_sets = load_sensor_table(machine)
    # a single new file has no run-level T1 temperature sample, so only vibration sets are offered
    feature_sets = {k: v for k, v in feature_sets.items() if not set(v) & set(TEMPERATURE)}
    if random_forest:
        feature_set, features = next(iter(feature_sets.items()))
        model_name = "Random forest"
        model = sensor_candidates()[model_name]().fit(
            table[features].to_numpy(dtype=np.float64), table["class"].to_numpy())
        final = {"model": model, "features": features, "model_name": model_name, "feature_set": feature_set}
    else:
        final = fit_final_sensor_model(table, feature_sets, m.labels, m.inner_group)
    final["metadata"] = {
        **deployment_metadata(machine, "sensor"),
        "classes": final["model"].classes_.tolist(),
        "preprocessing": {"extractor": "pdm.sensor_features", "features": final["features"],
                          "feature_set": final["feature_set"], "aggregation": "mean window probability"},
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS / f"{machine}_sensor.joblib")
    print(f"{m.title}: saved {final['model_name']} on '{final['feature_set']}' to {MODELS / f'{machine}_sensor.joblib'}")


def run_train_final_images(machine):
    """Image model for new files: all images, for the median number of epochs the fair test chose."""
    import torch

    m = MACHINES[machine]
    if m.image_settings.get("method"):
        raise ValueError(f"{machine}: no image model for new files (its image task is a separate check)")
    rounds = json.loads((RESULTS / machine / "images.json").read_text(encoding="utf-8"))["rounds"]
    epochs = int(np.median([r["epochs"] for r in rounds]))
    folder = IMAGES / machine
    dense = IMAGES / f"{machine}_denseframes"
    if machine in {"pump", "haul_truck"} and (dense / "index.csv").is_file():
        folder = dense
    rows = read_index(folder)
    if set(m.labels) != {row["class"] for row in rows} or epochs < 1:
        raise ValueError("Deployment training requires all expected classes and at least one epoch.")
    limit = m.image_settings.get("train_per_group_and_class")
    if limit:  # Drill Rig: the same cap as in the fair test (first `limit` gears per design and class)
        rank = {}
        for r in rows:
            rank.setdefault((r["group"], r["class"]), set()).add(int(r["order"]))
        rank = {key: sorted(ids)[:limit] for key, ids in rank.items()}
        rows = [r for r in rows if int(r["order"]) in rank[(r["group"], r["class"])]]
    x = load_images(folder, rows, m.image_settings["normalise"])
    y = np.asarray([m.labels.index(r["class"]) for r in rows])
    model, _ = fit_resnet(x, y, epochs, m.labels, m.image_settings)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            **deployment_metadata(machine, "image"),
            "model_name": "ResNet18 CNN",
            "classes": list(m.labels),
            "preprocessing": {"loader": "pdm.load_images", "normalise": m.image_settings["normalise"]},
            "epochs": epochs,
            "training_images": len(rows),
            "image_export": folder.name,
        },
    }, MODELS / f"{machine}_images.pt")
    print(f"{m.title}: saved ResNet-18 trained for {epochs} epochs on {len(rows)} images")


# ==================================================================================================
# 6. Predict a new file
# 1. Sensor recording or image?  2. Which dataset format (a format check, not machine learning)?
# Machines mapped onto one rig share a format; then --machine picks one.  3. Run that machine's model
# and average over windows. Unknown formats are rejected. Models only know their laboratory data.
# ==================================================================================================

CORA = ["belt_conveyor", "excavator", "dozer", "wheel_loader"]


@lru_cache(maxsize=3)
def indexed_image_hashes(folder):
    return {hashlib.sha256((folder / row["file"]).read_bytes()).digest() for row in read_index(folder)}


def identify(path):
    """(dataset format, machines that use it, details) from the file's contents."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        from scipy.io import whosmat

        variables = {name: shape for name, shape, _ in whosmat(path)}
        if "Channel_1" in variables:
            return "Ottawa bearing rig recording", ["pump"], {}
        if path.stem in variables and len(variables[path.stem]) == 2 and variables[path.stem][1] >= 2:
            return "UODS bearing rig recording", ["haul_truck"], {}
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            header = next(csv.reader([stream.readline(8192)]), [])
        schemas = [
            ("Ottawa bearing CSV", "pump", [OTTAWA_CSV_CHANNEL]),
            ("UODS bearing CSV", "haul_truck", UODS_CSV_CHANNELS),
            ("MCC5-THU gearbox", "drill_rig", MCC5_CHANNELS),
            ("JUST slewing bearing", "grader", JUST_CHANNELS),
        ]
        matches = [(source, machine) for source, machine, channels in schemas
                   if set(channels).issubset(header)]
        if matches:
            return " / ".join(source for source, _ in matches), [key for _, key in matches], {
                "mixed_schema": len(matches) > 1,
            }
        speed = re.fullmatch(r"[A-Za-z0-9]+_(F5|F15|F50|F60)_S_Vx\.csv", path.name, re.IGNORECASE)
        if speed and len(header) == 1:
            try:
                value = float(header[0])
            except ValueError:
                pass
            else:
                if np.isfinite(value):
                    return "CORA rig vibration (Vx)", CORA, {"speed": speed.group(1).upper()}
        raise ValueError("Unrecognized CSV schema")
    elif suffix in (".png", ".jpg", ".jpeg"):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            if not np.any(image != image[0, 0]):
                raise ValueError("Image contains no spatial detail.")
            if image.shape[:2] == (224, 224):
                try:
                    known_render = hashlib.sha256(path.read_bytes()).digest() in indexed_image_hashes(IMAGES / "drill_rig")
                except OSError:
                    known_render = False
                if known_render:
                    return "rendered gear image (MFGNet-Gear style)", ["drill_rig"], {"indexed_image": True}
                raise ValueError("Unknown Equipment: image provenance could not be verified.")
            if image.shape[:2] == (96, 128) and not image[:, :, 0].any():
                source = "event-camera frame (XJTU-DV style)"
                machines = ["pump", "haul_truck"]
                folders = {key: IMAGES / f"{key}_denseframes" for key in machines}
                digest = hashlib.sha256(path.read_bytes()).digest()
                try:
                    matches = [key for key, folder in folders.items() if digest in indexed_image_hashes(folder)]
                except OSError:
                    matches = []
                if len(matches) == 1:
                    return source, matches, {"indexed_image": True}
                return source, machines, {}
            if image.shape[:2] == (96, 128) and (image[:, :, 0] == image[:, :, 2]).all():
                return "infrared frame (CORA style)", CORA, {}
    raise ValueError(f"{path.name} matches none of the known dataset formats; no prediction made")


OTTAWA_CSV_CHANNEL = "ottawa_accel_200khz"
UODS_CSV_CHANNELS = ["uods_accel_42khz", "uods_mic_42khz"]


def sensor_features(path, source, details):
    if source.startswith("Ottawa"):
        if Path(path).suffix.lower() == ".csv":
            signal = pd.read_csv(path, usecols=[OTTAWA_CSV_CHANNEL], dtype=np.float64)[OTTAWA_CSV_CHANNEL].to_numpy()
            if len(signal) < 20_000:
                raise ValueError("The recording is too short. Ottawa CSVs need at least 20,000 samples.")
            return ottawa_signal_features(signal)
        return ottawa_recording(path)
    if source.startswith("UODS"):
        if Path(path).suffix.lower() == ".csv":
            signals = pd.read_csv(path, usecols=UODS_CSV_CHANNELS, dtype=np.float64)
            length = len(signals) // 21_000 * 21_000
            if not length:
                raise ValueError("The recording is too short. UODS CSVs need at least 21,000 samples.")
            signal, microphone = signals[UODS_CSV_CHANNELS].to_numpy()[:length].T
            if not np.isfinite(signal).all() or not np.isfinite(microphone).all():
                raise ValueError("The recording contains invalid UODS signal values.")
            return uods_signal_features(signal, microphone)
        return uods_recording(path)
    if source.startswith("MCC5"):
        with open(path, "rb") as handle:
            return mcc5_recording(handle)
    if source.startswith("JUST"):
        return just_recording(path)
    values = pd.read_csv(path, header=None, dtype=np.float64).iloc[:, 0].to_numpy()
    return cora_windows(values, details["speed"])


def predict_one(path, machine, source, details):
    """Step 3: run one machine's trained model on the file; class probabilities averaged over its windows."""
    if "image" in source or "frame" in source:
        return image_probability(path, machine)
    import joblib

    model_file = MODELS / f"{machine}_sensor.joblib"
    if not model_file.is_file():
        raise FileNotFoundError(f"No trained sensor model: run  python Notebook/pdm.py {machine} train-final")
    final = joblib.load(model_file)
    features = sensor_features(path, source, details)[final["features"]].to_numpy(dtype=np.float64)
    return dict(zip(final["model"].classes_, final["model"].predict_proba(features).mean(axis=0)))


def predict(path, machine=None):
    """Return one condition prediction when the source mapping is known."""
    source, machines, details = identify(path)
    if machine is not None and machine not in machines:
        raise ValueError(f"{source} does not belong to {machine} (it matches {', '.join(machines)})")
    if details.get("mixed_schema") or (machine is None and len(machines) > 1):
        return {"file format": source, "status": "Ambiguous Equipment",
                "candidates": [MACHINES[name].title for name in machines]}
    machine = machine or machines[0]
    probability = predict_one(path, machine, source, details)
    best = max(probability, key=probability.get)
    return {"file format": source, "machine": MACHINES[machine].title, "condition": best,
            "model probability": round(float(probability[best]), 3),
            "all classes": {label: round(float(probability[label]), 3) for label in MACHINES[machine].labels}}


def image_probability(path, machine):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    model_file = MODELS / f"{machine}_images.pt"
    if not model_file.is_file():
        raise FileNotFoundError(f"No trained image model: run  python Notebook/pdm.py {machine} train-final-images")
    config = MACHINES[machine]
    saved = torch.load(model_file, map_location="cpu", weights_only=True)
    # Existing exports contain just the state dict; newer bundles also record preprocessing.
    metadata = saved.get("metadata", {})
    if metadata and (metadata["machine"] != machine or metadata["modality"] != "image"):
        raise ValueError("Image checkpoint belongs to a different equipment or modality.")
    labels = metadata.get("classes", config.labels)
    normalise = metadata.get("preprocessing", {}).get("normalise", config.image_settings["normalise"])
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(saved.get("state_dict", saved))
    x = load_images(Path(path).parent, [{"file": Path(path).name}], normalise)
    return dict(zip(labels, image_probabilities(model, x)[0]))


# ==================================================================================================
# 6b. Upload page: predict() behind a web form, this computer only (http://localhost:8613)
# ==================================================================================================

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Condition check</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#222}
table{border-collapse:collapse;margin-top:16px}td{border:1px solid #ccc;padding:6px 10px;vertical-align:top}
.note{color:#666;font-size:14px}.error{color:#b00020}</style></head><body>
<h1>Mining equipment condition check</h1>
<p class="note">Upload one sensor recording (.mat or .csv) or one camera image (.png or .jpg). A recognized format can narrow the laboratory mapping. Shared formats require a confirmed mapping. Laboratory models only:
not tested on real mining machines.</p>
<form method="post" enctype="multipart/form-data">
<p><input type="file" name="file" required></p>
<p>Machine (only needed when several machines share a format): <select name="machine">
<option value="">find it from the file</option>{options}</select></p>
<p><button type="submit">Check condition</button></p></form>{result}</body></html>"""


def serve(port=8613):
    """Start the upload page; stop it with Ctrl-C."""
    import email
    import html
    import tempfile
    from http.server import BaseHTTPRequestHandler, HTTPServer

    options = "".join(f'<option value="{k}">{m.title}</option>' for k, m in MACHINES.items())

    def page(result=""):
        return PAGE.replace("{options}", options).replace("{result}", result).encode("utf-8")

    def table(answer):
        rows = []
        for key, value in answer.items():
            if isinstance(value, dict):
                value = "<br>".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in value.items())
            elif isinstance(value, list):
                value = "<br>".join(html.escape(str(v)) for v in value)
            else:
                value = html.escape(str(value))
            rows.append(f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>")
        return "<table>" + "".join(rows) + "</table>"

    class Handler(BaseHTTPRequestHandler):
        def send(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(page())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length > 300_000_000:
                return self.send(page('<p class="error">File too large (limit 300 MB).</p>'))
            body = self.rfile.read(length)
            message = email.message_from_bytes(
                f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body)
            fields = {part.get_param("name", header="content-disposition"): part for part in message.get_payload()}
            upload, choice = fields.get("file"), fields.get("machine")
            machine = (choice.get_payload(decode=True) or b"").decode().strip() if choice else ""
            name = Path(upload.get_filename() or "upload").name if upload else ""
            try:
                if not name:
                    raise ValueError("no file was uploaded")
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / name  # keep the name: CORA files carry their speed in it
                    path.write_bytes(upload.get_payload(decode=True))
                    answer = predict(path, machine or None)
                result = f"<h2>{html.escape(name)}</h2>" + table(answer)
            except (ValueError, FileNotFoundError) as problem:
                result = f'<p class="error">No prediction for {html.escape(name)}: {html.escape(str(problem))}</p>'
            self.send(page(result))

        def log_message(self, *args):
            pass

    print(f"Upload page: http://localhost:{port}  (this computer only; Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ==================================================================================================
# 7. Tables and figures for the notebook (read from Results/; nothing is trained here)
# ==================================================================================================

def load_result(machine, name):
    path = RESULTS / machine / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def results_overview():
    """One row per machine: accuracy per test recording (or gear) on each test."""
    rows = []
    for machine, m in MACHINES.items():
        row = {"Machine": m.title, "Component": m.component}
        for column, name, key in (("Sensor, fair test", "sensor", "score"),
                                  ("Images, fair test", "images", "unit_level"),
                                  ("Sensor, time split", "sensor_timesplit", "score"),
                                  ("Images, time split", "images_timesplit", "unit_level")):
            result = load_result(machine, name)
            row[column] = result[key][result.get("headline_metric", "accuracy")] if result else None
            if name == "sensor" and result:
                row["Majority control"] = result["majority_control"]["accuracy"]
                row["Test recordings (sensor)"] = result[key]["test_units"]
            if name == "images" and result:
                row["Test recordings/gears (images)"] = result[key]["test_units"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Machine")
    order = ["Component", "Sensor, fair test", "Majority control", "Test recordings (sensor)",
             "Images, fair test", "Test recordings/gears (images)", "Sensor, time split", "Images, time split"]
    return table[order]


def show_machine(machine):
    """Print the sources and notes, then show the saved confusion-matrix figures."""
    from IPython.display import Image, display

    m = MACHINES[machine]
    print(f"{m.title} ({m.component})")
    print(f"  sensor data: {m.sensor_source}")
    print(f"  image data:  {m.image_source}")
    for note in m.notes:
        print(f"  - {note}")
    sensor = load_result(machine, "sensor")
    if sensor:
        chosen = ", ".join(f"{c['model']} ({c['feature_set']})" for c in sensor["chosen_per_round"])
        print(f"  sensor model chosen in each round: {chosen}")
    # fair-test figures only, as small JPEGs (time-split figures are in Figures/<machine>/)
    import cv2

    for name in ("sensor", "images"):
        path = FIGURES / machine / f"{name}.png"
        if path.is_file():
            picture = cv2.imread(str(path))
            scale = min(1.0, (380 if name == "sensor" else 560) / picture.shape[1])
            picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 45])
            display(Image(data=jpeg.tobytes(), format="jpeg"))


# ==================================================================================================
# 8. Command line
# ==================================================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="a machine name, 'predict' or 'data-check'")
    parser.add_argument("command", nargs="?", help="sensor, export-images, images, train-final, "
                        "train-final-images; or the file to predict")
    parser.add_argument("--time-split", action="store_true", help="the easier secondary test")
    parser.add_argument("--machine", help="predict only: which machine the file belongs to")
    parser.add_argument("--random-forest", action="store_true", help="train-final: fit Random Forest on the first compatible feature set")
    args = parser.parse_args()
    if args.target == "data-check":
        raise SystemExit(check_data())
    if args.target == "serve":
        return serve()
    if args.target == "predict":
        if not args.command:
            parser.error("give the file to predict")
        try:
            answer = predict(args.command, args.machine)
        except (ValueError, FileNotFoundError) as problem:
            parser.exit(1, f"No prediction: {problem}\n")
        for key, value in answer.items():
            print(f"{key}: {value}")
        return
    if args.target not in MACHINES:
        parser.error(f"unknown machine {args.target}; choose from {', '.join(MACHINES)}")
    split = "time" if args.time_split else "fair"
    if split == "time" and not MACHINES[args.target].time_split:
        parser.error(f"{args.target} has no time split")
    steps = {
        "sensor": lambda: run_sensor(args.target, split),
        "export-images": lambda: run_export_images(args.target),
        "images": lambda: run_images(args.target, split),
        "train-final": lambda: run_train_final(args.target, random_forest=args.random_forest),
        "train-final-images": lambda: run_train_final_images(args.target),
    }
    if args.command not in steps:
        parser.error(f"unknown command {args.command}; choose from {', '.join(steps)}")
    steps[args.command]()


if __name__ == "__main__":
    main()
