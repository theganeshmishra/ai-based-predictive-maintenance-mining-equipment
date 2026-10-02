"""Train the models behind the demo's "Check a photo" tab.

Run from the repository root:  python prototype/train_image_models.py

Wear models (Nonastreda, milling tools): one random forest per photo type (tool edge,
chips, machined surface), using only that photo's pixels. Scored with leave-one-tool-out
(10 folds, seed 613); the out-of-fold probabilities are kept so the demo's built-in samples
are always judged by a model that never saw that tool. The final model is refit on all
512 samples for uploaded photos.

Bearing photos: no verified labels exist, so no classifier is trained. The demo applies the
OpenCV heuristic and finds visually similar reference photos.

For both, a nearest-neighbour index gives "similar reference images" and an
out-of-distribution check: if an uploaded photo is further from every reference photo than
99% of reference photos are from their nearest neighbour in another tool or session, the
demo warns that the prediction is unreliable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import vision

SEED = 613
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("NONASTREDA_PROJECT", ROOT.parent / "multimodal-pdm-from-scratch"))
RAW = PROJECT / "data" / "raw" / "nonastreda"
EXT = {"tool": ".jpg", "chip": ".jpg", "work": ".png"}
OUT = Path(__file__).resolve().parent / "models" / "image_models.joblib"


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced", n_jobs=-1, random_state=SEED)


def ood_threshold(Z: np.ndarray, groups: np.ndarray, q: float = 0.99) -> float:
    """99th percentile of each reference photo's distance to its nearest photo from another group."""
    nn = NearestNeighbors(n_neighbors=min(60, len(Z))).fit(Z)
    dist, idx = nn.kneighbors(Z)
    nearest = [d[groups[i] != groups[i_]][0] if (groups[i_] != groups[i]).any() else d[-1]
               for d, i_, i in zip(dist, idx, range(len(Z)))]
    return float(np.quantile(nearest, q))


def wear_models() -> tuple[dict, pd.DataFrame]:
    labels = pd.read_csv(RAW / "labels.csv")
    reg = pd.read_csv(RAW / "labels_reg.csv")
    ids = labels["id"].to_numpy()
    tools = labels["id"].str.extract(r"^T(\d+)", expand=False).astype(int).to_numpy()
    y = labels["image_label"].map({c: i for i, c in enumerate(vision.WEAR_CLASSES)}).to_numpy()
    out, rows = {}, []
    for kind, ext in EXT.items():
        feats = []
        for i in ids:
            img = cv2.imread(str(RAW / kind / f"{i}{ext}"), cv2.IMREAD_COLOR)
            feats.append(vision.wear_features(vision.prepare_wear(img, kind), kind))
        F = pd.DataFrame(feats)
        X = F.to_numpy()
        oof = np.zeros((len(y), 3))
        per_tool = []
        for t in sorted(set(tools)):
            tr, te = tools != t, tools == t
            oof[te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])
            yp = oof[te].argmax(1)
            per_tool.append({"kind": kind, "tool": t,
                             "macro_f1": f1_score(y[te], yp, labels=[0, 1, 2], average="macro", zero_division=0),
                             "balanced_accuracy": balanced_accuracy_score(y[te], yp)})
        pt = pd.DataFrame(per_tool)
        rows.append({"photo_type": kind, "macro_f1_mean": pt["macro_f1"].mean(), "macro_f1_std": pt["macro_f1"].std(),
                     "balanced_acc_mean": pt["balanced_accuracy"].mean(), "balanced_acc_std": pt["balanced_accuracy"].std()})
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        out[kind] = {
            "feature_names": list(F.columns),
            "model": rf().fit(X, y),
            "scaler": scaler,
            "nn": NearestNeighbors(n_neighbors=5).fit(Z),
            "ood_threshold": ood_threshold(Z, tools),
            "ref_ids": ids, "ref_labels": labels["image_label"].to_numpy(),
            "ref_flank_wear": reg["flank_wear"].to_numpy(), "ref_tools": tools,
            "oof_proba": oof.astype(np.float32),
            "feature_ranges": {c: (float(F[c].quantile(0.01)), float(F[c].quantile(0.99))) for c in F.columns},
            "cv": rows[-1],
        }
        print(f"{kind}: macro-F1 {pt['macro_f1'].mean():.3f} ± {pt['macro_f1'].std():.3f} (leave one tool out)")
    return out, pd.DataFrame(rows).round(4)


def bearing_reference() -> dict:
    path = ROOT / "data" / "processed" / "opencv_photos_all_in_one.parquet"
    table = pq.read_table(path, columns=["image_id", "source_set", "group_id", "pseudo_label", "features",
                                         *vision.BEARING_CATEGORIES, "foreground_ratio"])
    meta = table.drop_columns(["features"]).to_pandas()
    X = table["features"].combine_chunks().values.to_numpy().reshape(table.num_rows, -1)
    X = vision.bearing_similarity_vector(X)
    keep = (meta["source_set"] == "original").to_numpy()
    meta, X = meta[keep].reset_index(drop=True), X[keep]
    thresholds = json.loads(pq.read_schema(path).metadata[b"pseudo_label_thresholds"])
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    groups = meta["group_id"].astype(str).to_numpy()
    print(f"bearing: {len(meta)} reference photos, {len(set(groups))} capture sessions")
    return {
        "scaler": scaler, "nn": NearestNeighbors(n_neighbors=5).fit(Z), "ood_threshold": ood_threshold(Z, groups),
        "ref_ids": meta["image_id"].to_numpy(), "ref_labels": meta["pseudo_label"].astype(str).to_numpy(),
        "thresholds": thresholds,
        "score_reference": meta[vision.BEARING_CATEGORIES].to_numpy(dtype=np.float32),
    }


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    wear, cv = wear_models()
    bundle = {"wear": wear, "bearing": bearing_reference(), "seed": SEED, "opencv": cv2.__version__}
    joblib.dump(bundle, OUT, compress=3)
    (ROOT / "results").mkdir(exist_ok=True)
    cv.to_csv(ROOT / "results" / "prototype_image_models_cv.csv", index=False)
    print(cv.to_string(index=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
