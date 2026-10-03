"""Train and evaluate the 8-class TiNA machine-state model used by the demo.

Run from the repository root after build_master.py:  python prototype/train_8state.py

Two evaluations, both split by whole days (never by random minute):
1. Block cross-validation: the 25 days are cut into 5 consecutive blocks of 5 days;
   each block is held out once.
2. Chronological hold-out for the demo: train on 2017-09-06..22, test on
   2017-09-23..30. The app replays these unseen test days.
A random-minute split is also run, only to show how much a leaky split inflates scores.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from build_master import STATES

SEED = 613
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "tina_minutes_8state.parquet"
RESULTS = ROOT / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
DEMO_OUT = ROOT / "data" / "processed" / "tina_8state_demo_predictions.parquet"
TEST_FROM = "2017-09-23"
N_BLOCKS = 5


def model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def scores(y_true, y_pred) -> dict[str, float]:
    present = [s for s in STATES if s in set(y_true)]
    return {
        "macro_f1_present_classes": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
    }


def per_class(y_true, y_pred, episodes) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=STATES, zero_division=0)
    ep = pd.Series(episodes[episodes > 0]).groupby(y_true[episodes > 0]).nunique()
    return pd.DataFrame({"state": STATES, "test_minutes": n, "test_episodes": [int(ep.get(s, 0)) for s in STATES],
                         "precision": p, "recall": r, "f1": f}).round(3)


def episode_hits(y_true, y_pred, episodes) -> pd.DataFrame:
    """Per maintenance episode: was the most common predicted state the true one?"""
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "ep": episodes})
    df = df[df["ep"] > 0]
    g = df.groupby("ep").agg(true=("true", "first"), pred_mode=("pred", lambda s: s.mode().iloc[0]),
                             any_maintenance_predicted=("pred", lambda s: bool(s.isin(STATES[2:]).any())))
    g["type_correct"] = g["true"] == g["pred_mode"]
    return g


def save_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=STATES)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(STATES)), STATES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(STATES)), STATES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85})
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.endswith(("_mean", "_std"))]
    X = df[feats].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)  # std of a single-value window; no label information involved
    y = df["state"].to_numpy()
    ep = df["episode"].to_numpy()
    days = sorted(df["day"].unique())

    # 1. block cross-validation by consecutive days
    blocks = np.array_split(np.array(days), N_BLOCKS)
    cv_pred = np.empty(len(y), dtype=object)
    rows = []
    for k, block in enumerate(blocks):
        te = df["day"].isin(block).to_numpy()
        cv_pred[te] = model().fit(X[~te], y[~te]).predict(X[te])
        majority = pd.Series(y[~te]).mode().iloc[0]
        rows.append({"block": k + 1, "days": f"{block[0]}..{block[-1]}", "test_minutes": int(te.sum()),
                     "model": "random_forest", **scores(y[te], cv_pred[te])})
        rows.append({"block": k + 1, "days": f"{block[0]}..{block[-1]}", "test_minutes": int(te.sum()),
                     "model": "majority_class", **scores(y[te], np.full(te.sum(), majority))})
    cv = pd.DataFrame(rows).round(4)
    cv.to_csv(RESULTS / "prototype_8state_cv_by_block.csv", index=False)
    cv_class = per_class(y, cv_pred, ep)
    cv_class.to_csv(RESULTS / "prototype_8state_cv_per_class.csv", index=False)
    cv_eps = episode_hits(y, cv_pred, ep)

    # For comparison only: a random-minute split puts neighbouring minutes of the same
    # day in train and test. It is leaky and must never be reported as the model score.
    leaky_pred = cross_val_predict(model(), X, y, cv=StratifiedKFold(N_BLOCKS, shuffle=True, random_state=SEED))
    save_confusion(y, cv_pred, "TiNA 8-state, 5-block day CV (counts; colour = row share)",
                   FIGURES / "confusion_8state_cv.jpg")

    # 2. chronological hold-out for the demo replay
    te = (df["day"] >= TEST_FROM).to_numpy()
    clf = model().fit(X[~te], y[~te])
    proba = clf.predict_proba(X[te])
    pred = clf.classes_[proba.argmax(1)]
    ho_class = per_class(y[te], pred, ep[te])
    ho_class.to_csv(RESULTS / "prototype_8state_holdout_per_class.csv", index=False)
    ho_eps = episode_hits(y[te], pred, ep[te])
    save_confusion(y[te], pred, f"TiNA 8-state, train < {TEST_FROM}, test {TEST_FROM}..09-30",
                   FIGURES / "confusion_8state_holdout.jpg")
    imp = pd.Series(clf.feature_importances_, index=feats).nlargest(15).round(4)
    imp.rename("importance").to_csv(RESULTS / "prototype_8state_top_features.csv")

    demo = df.loc[te, ["minute_end", "day", "state", "episode", "n_seconds", "maintenance_seconds",
                       "alarm_seconds"] + list(imp.index[:6])].reset_index(drop=True)
    demo["predicted"] = pred
    for i, c in enumerate(clf.classes_):
        demo[f"p_{c}"] = proba[:, i].astype("float32")
    demo.to_parquet(DEMO_OUT, index=False)

    summary = pd.DataFrame([
        {"evaluation": "5-block day CV", **scores(y, cv_pred),
         "maint_episodes_type_correct": f"{int(cv_eps['type_correct'].sum())}/{len(cv_eps)}",
         "maint_episodes_any_maint_flag": f"{int(cv_eps['any_maintenance_predicted'].sum())}/{len(cv_eps)}"},
        {"evaluation": f"hold-out {TEST_FROM}..09-30", **scores(y[te], pred),
         "maint_episodes_type_correct": f"{int(ho_eps['type_correct'].sum())}/{len(ho_eps)}",
         "maint_episodes_any_maint_flag": f"{int(ho_eps['any_maintenance_predicted'].sum())}/{len(ho_eps)}"},
        {"evaluation": "LEAKY random-minute CV (comparison only)", **scores(y, leaky_pred),
         "maint_episodes_type_correct": "n/a", "maint_episodes_any_maint_flag": "n/a"},
    ]).round(4)
    summary.to_csv(RESULTS / "prototype_8state_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print("\nCV by block:\n", cv.to_string(index=False))
    print("\nCV per class:\n", cv_class.to_string(index=False))
    print("\nHold-out per class:\n", ho_class.to_string(index=False))


if __name__ == "__main__":
    main()
