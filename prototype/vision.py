"""Image features and visual explanations shared by train_image_models.py and app.py.

Wear features (tool / chip / work) match multimodal-pdm-from-scratch notebook 02.
Bearing features and heuristic scores match notebooks/OpenCV_Photo_Dataset_One_File.ipynb.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------- common
WEAR_KINDS = {
    "tool": "Milling tool edge (microscope)",
    "chip": "Metal chips (microscope)",
    "work": "Machined surface (microscope)",
}
# Training image sizes (width, height). Uploads are resized to these so that scale-dependent
# features such as edge density are comparable with the training photos.
WEAR_SIZE = {"tool": (1550, 500), "chip": (1024, 1024), "work": (512, 512)}
WEAR_CLASSES = ["sharp", "used", "dulled"]


MIN_SIDE = 16


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise ValueError("Could not read this file as an image.")
    if min(image.shape[:2]) < MIN_SIDE:
        raise ValueError(f"Image is too small ({image.shape[1]}×{image.shape[0]} px); "
                         f"it needs at least {MIN_SIDE} px on each side.")
    return image


def to_max_side(image: np.ndarray, side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- wear (Nonastreda)
def prepare_wear(image: np.ndarray, kind: str) -> np.ndarray:
    size = WEAR_SIZE[kind]
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def wear_features(image: np.ndarray, kind: str) -> dict[str, float]:
    """Pixel-only features for one wear image (same definitions as notebook 02)."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f: dict[str, float] = {}
    p = f"{kind}_"
    f[p + "g_mean"], f[p + "g_std"] = g.mean(), g.std()
    for q in (10, 50, 90):
        f[p + f"g_p{q}"] = np.percentile(g, q)
    f[p + "lap_var"] = cv2.Laplacian(g, cv2.CV_64F).var()
    f[p + "edge_density"] = (cv2.Canny(g, 50, 150) > 0).mean()
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    f[p + "grad_mean"], f[p + "grad_std"] = mag.mean(), mag.std()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i, ch in enumerate("hsv"):
        f[p + f"{ch}_mean"], f[p + f"{ch}_std"] = hsv[..., i].mean(), hsv[..., i].std()
    hist = cv2.calcHist([g], [0], None, [8], [0, 256]).ravel()
    for i, v in enumerate(hist / hist.sum()):
        f[p + f"hist{i}"] = v
    if kind == "tool":
        prof = g.mean(axis=1)
        for i, chunk in enumerate(np.array_split(prof, 16)):
            f[p + f"row{i:02d}"] = chunk.mean()
    return {k: float(v) for k, v in f.items()}


# --------------------------------------------------------------------------- bearing photos
BEARING_WORKING_SIDE = 512
BEARING_FEATURE_SIZE = 64
HOG = cv2.HOGDescriptor((BEARING_FEATURE_SIZE, BEARING_FEATURE_SIZE), (16, 16), (8, 8), (8, 8), 9)
HSV_BINS = 16
BACKGROUND_HUE = (91, 123)  # blue table behind the bearings
RUST_HUE = (6, 28)
BEARING_CATEGORIES = ["contamination", "corrosion", "discoloration", "surface_damage"]


def bearing_feature_vector(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, (BEARING_FEATURE_SIZE, BEARING_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hists = []
    for channel, upper in enumerate((180, 256, 256)):
        counts = cv2.calcHist([hsv], [channel], None, [HSV_BINS], [0, upper]).ravel()
        hists.append(counts / counts.sum())
    edge_density = float((cv2.Canny(gray, 80, 160) > 0).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    pixels = small.reshape(-1, 3).astype(np.float64)
    return np.concatenate([HOG.compute(gray).ravel(), *hists, [edge_density, sharpness],
                           pixels.mean(axis=0), pixels.std(axis=0)]).astype(np.float32)


def bearing_similarity_vector(full: np.ndarray) -> np.ndarray:
    """Colour histograms, edge density, sharpness and colour stats (HOG left out).

    Used for similar-photo search and the familiarity check: in the 1,820-dimensional
    vector with HOG, random noise looked as close to the references as a real bearing photo.
    """
    return full[..., HOG.getDescriptorSize():]


def bearing_masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[..., i].astype(np.float32) for i in range(3))
    background = (hue > BACKGROUND_HUE[0]) & (hue < BACKGROUND_HUE[1]) & (sat > 60) & (val > 90)
    fg = ~background
    rust = fg & (hue >= RUST_HUE[0]) & (hue <= RUST_HUE[1]) & (sat > 60)
    return {"foreground": fg, "corrosion": rust, "discoloration": fg & (sat > 90) & ~rust,
            "contamination": fg & (val < 70)}


def bearing_scores(bgr: np.ndarray) -> dict[str, float]:
    m = bearing_masks(bgr)
    fg = m["foreground"]
    if not fg.any():
        return dict(foreground_ratio=0.0, contamination=0.0, corrosion=0.0, discoloration=0.0, surface_damage=0.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)[fg]
    n = fg.sum()
    return dict(foreground_ratio=float(fg.mean()), contamination=float(m["contamination"].sum() / n),
                corrosion=float(m["corrosion"].sum() / n), discoloration=float(m["discoloration"].sum() / n),
                surface_damage=float(grad.mean()))


def bearing_pseudo_label(scores: dict[str, float], thresholds: dict[str, float]) -> str:
    rel = {c: scores[c] / thresholds[c] for c in BEARING_CATEGORIES if thresholds.get(c)}
    best = max(rel, key=rel.get)
    return best if rel[best] >= 1.0 else "clean_or_unclear"


def bearing_overlay(bgr: np.ndarray) -> np.ndarray:
    """Dim the background and colour pixels that drive each heuristic score."""
    m = bearing_masks(bgr)
    out = bgr.copy().astype(np.float32)
    out[~m["foreground"]] *= 0.25
    colours = {"contamination": (200, 60, 160), "discoloration": (0, 220, 255), "corrosion": (0, 0, 255)}
    for name, colour in colours.items():  # later colours win where masks overlap
        out[m[name]] = 0.35 * out[m[name]] + 0.65 * np.array(colour, np.float32)
    return out.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- visual steps
def clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def edges(bgr: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), lo, hi)


def texture_heatmap(bgr: np.ndarray) -> np.ndarray:
    """Local gradient energy (where the surface is rough or has edges), blended over the image."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    k = max(3, (min(g.shape) // 25) | 1)
    energy = cv2.GaussianBlur(mag, (k, k), 0)
    norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(bgr, 0.45, heat, 0.55, 0)


def row_profile(bgr: np.ndarray) -> np.ndarray:
    """Mean brightness of each image row; on tool-edge photos a worn flank shows as a bright band."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean(axis=1)
