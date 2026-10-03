"""Export out-of-fold Nonastreda wear predictions for the demo's inspection tab.

Run from the repository root:  python prototype/export_nonastreda.py
Needs the sibling project multimodal-pdm-from-scratch (or set NONASTREDA_PROJECT),
after its notebooks 01 and 02 have been run.

Same protocol as that project's notebook 02: leave one tool out (10 folds), fixed
random forests, late fusion = mean of the signal and image class probabilities.
Every prediction shown in the demo comes from a model that never saw that tool.
Data: Nonastreda, Truchan and Ahmadi (2025), DOI 10.17632/m892d2wtzh.1, CC BY 4.0.
Milling tool wear, not mining equipment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.ensemble import RandomForestClassifier

SEED = 613
CLASSES = ["sharp", "used", "dulled"]
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("NONASTREDA_PROJECT", ROOT.parent / "multimodal-pdm-from-scratch"))
OUT = ROOT / "data" / "processed" / "nonastreda_demo_predictions.parquet"
TRACE_POINTS = 600


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced", n_jobs=-1, random_state=SEED)


def main() -> None:
    paired = pd.read_parquet(PROJECT / "data" / "processed" / "nonastreda_paired.parquet")
    imgf = pd.read_parquet(PROJECT / "data" / "processed" / "nonastreda_image_features.parquet")
    if not (imgf["id"].values == paired["id"].values).all() or "tool_label" in paired.columns:
        raise ValueError("Nonastreda inputs are misaligned or contain tool_label.")
    sig_cols = [c for c in paired.columns if c.startswith(("fx_", "fy_", "fz_")) or c == "sig_len"]
    X_sig = paired[sig_cols].to_numpy()
    X_img = imgf.drop(columns="id").to_numpy()
    y = paired["image_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tools = paired["tool"].to_numpy()

    P_sig, P_img = np.zeros((len(y), 3)), np.zeros((len(y), 3))
    for t in sorted(set(tools)):
        tr, te = tools != t, tools == t
        P_sig[te] = rf().fit(X_sig[tr], y[tr]).predict_proba(X_sig[te])
        P_img[te] = rf().fit(X_img[tr], y[tr]).predict_proba(X_img[te])
    P = (P_sig + P_img) / 2

    # Downsampled force traces so the demo can "replay" the recorded sensor signal.
    mat = scipy.io.loadmat(PROJECT / "data" / "raw" / "nonastreda" / "forces_xyz_raw.mat")["baseDatastore"]
    traces = {ax: [] for ax in ("fx", "fy", "fz")}
    for i in range(mat.shape[0]):
        F = np.asarray(mat[i, 3], dtype=float)
        idx = np.linspace(0, F.shape[1] - 1, TRACE_POINTS).astype(int)
        for k, ax in enumerate(traces):
            traces[ax].append(F[k, idx].astype("float32"))

    out = paired[["id", "tool", "run", "blade", "image_label", "flank_wear",
                  "tool_thumb", "chip_thumb", "work_thumb", "sig_len"]].copy()
    for name, M in [("fusion", P), ("signal", P_sig), ("image", P_img)]:
        out[f"pred_{name}"] = [CLASSES[i] for i in M.argmax(1)]
        for i, c in enumerate(CLASSES):
            out[f"p_{name}_{c}"] = M[:, i].astype("float32")
    for ax, v in traces.items():
        out[f"trace_{ax}"] = v
    out.to_parquet(OUT, index=False)
    acc = (out["pred_fusion"] == out["image_label"]).mean()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} samples, out-of-fold fusion accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
