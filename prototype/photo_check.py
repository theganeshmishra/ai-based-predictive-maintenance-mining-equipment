"""'Check a photo' tab: upload or capture an image and run it through the inspection pipeline."""

from __future__ import annotations

import io
import os
from pathlib import Path

import altair as alt
import cv2
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

import vision

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(__file__).resolve().parent / "models" / "image_models.joblib"
PROJECT = Path(os.environ.get("NONASTREDA_PROJECT", ROOT.parent / "multimodal-pdm-from-scratch"))
NONASTREDA_RAW = PROJECT / "data" / "raw" / "nonastreda"
BEARING_RAW = ROOT / "data" / "raw" / "opencv" / "photos"
EXT = {"tool": ".jpg", "chip": ".jpg", "work": ".png"}
TYPES = {**vision.WEAR_KINDS, "bearing": "Bearing (phone photo)"}
CLASS_COLOURS = {"sharp": "#2e9e5b", "used": "#e8a33d", "dulled": "#d9534f"}
SCORE_HELP = {
    "corrosion": "share of bearing pixels with rust-coloured hue",
    "discoloration": "share of strongly coloured, non-rust pixels",
    "contamination": "share of very dark pixels (dirt, grease, deposits)",
    "surface_damage": "mean brightness gradient on the bearing (scratches, pitting, edges)",
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODELS) if MODELS.exists() else None


@st.cache_data
def wear_thumbs() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "nonastreda_demo_predictions.parquet"
    return pd.read_parquet(p, columns=["id", "tool_thumb", "chip_thumb", "work_thumb"]).set_index("id")


@st.cache_data
def bearing_thumbs() -> pd.Series:
    p = ROOT / "data" / "processed" / "opencv_photos_all_in_one.parquet"
    t = pq.read_table(p, columns=["image_id", "thumb_jpeg"]).to_pandas()
    return t.set_index("image_id")["thumb_jpeg"]


# --------------------------------------------------------------------------- analysis
def familiarity(z: np.ndarray, part: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """Ratio of nearest-reference distance to the 99% threshold (<1 = looks like the references)."""
    dist, idx = part["nn"].kneighbors(z)
    return float(dist[0, 0] / part["ood_threshold"]), dist[0], idx[0]


def analyse_wear(img: np.ndarray, kind: str, bundle: dict, sample_id: str | None = None) -> dict:
    part = bundle["wear"][kind]
    prepared = vision.prepare_wear(img, kind)
    feats = vision.wear_features(prepared, kind)
    x = np.array([[feats[c] for c in part["feature_names"]]])
    z = part["scaler"].transform(x)
    known = False
    if sample_id is None:
        d0, i0 = part["nn"].kneighbors(z, n_neighbors=1)
        if d0[0, 0] < 1e-6:  # an uploaded copy of a training photo: judge it like a built-in sample
            sample_id, known = part["ref_ids"][i0[0, 0]], True
    if sample_id is not None:
        # Built-in sample: use its leave-one-tool-out prediction, and only show neighbours
        # from other tools, so nothing the "model" shows comes from the sample's own tool.
        pos = int(np.flatnonzero(part["ref_ids"] == sample_id)[0])
        proba = part["oof_proba"][pos]
        dist, idx = part["nn"].kneighbors(z, n_neighbors=60)
        other = part["ref_tools"][idx[0]] != part["ref_tools"][pos]
        idx, dist = idx[0][other][:5], dist[0][other][:5]
        ratio = float(dist[0] / part["ood_threshold"])
    else:
        proba = part["model"].predict_proba(x)[0]
        ratio, dist, idx = familiarity(z, part)
    outside = [c for c in part["feature_names"]
               if not part["feature_ranges"][c][0] <= feats[c] <= part["feature_ranges"][c][1]]
    return {"kind": kind, "prepared": prepared, "proba": proba, "ratio": ratio, "nn_idx": idx, "nn_dist": dist,
            "outside": outside, "n_features": len(part["feature_names"]), "cv": part["cv"],
            "sample_id": sample_id, "known": known}


def analyse_bearing(img: np.ndarray, bundle: dict) -> dict:
    part = bundle["bearing"]
    working = vision.to_max_side(img, vision.BEARING_WORKING_SIDE)
    scores = vision.bearing_scores(working)
    z = part["scaler"].transform(vision.bearing_similarity_vector(vision.bearing_feature_vector(working))[None, :])
    ratio, dist, idx = familiarity(z, part)
    ref = part["score_reference"]
    pct = {c: float((ref[:, i] <= scores[c]).mean() * 100) for i, c in enumerate(vision.BEARING_CATEGORIES)}
    return {"kind": "bearing", "prepared": working, "scores": scores, "percentile": pct,
            "label": vision.bearing_pseudo_label(scores, part["thresholds"]),
            "ratio": ratio, "nn_idx": idx, "nn_dist": dist}


def detect_type(img: np.ndarray, bundle: dict) -> tuple[str, dict[str, float]]:
    """Pick the photo type whose reference photos the image is closest to (relative to each threshold)."""
    ratios = {}
    for kind in vision.WEAR_KINDS:
        part = bundle["wear"][kind]
        f = vision.wear_features(vision.prepare_wear(img, kind), kind)
        ratios[kind] = familiarity(part["scaler"].transform([[f[c] for c in part["feature_names"]]]), part)[0]
    working = vision.to_max_side(img, vision.BEARING_WORKING_SIDE)
    ratios["bearing"] = familiarity(
        bundle["bearing"]["scaler"].transform(vision.bearing_similarity_vector(vision.bearing_feature_vector(working))[None, :]), bundle["bearing"])[0]
    return min(ratios, key=ratios.get), ratios


# --------------------------------------------------------------------------- rendering
def show_steps(res: dict) -> None:
    img = res["prepared"]
    view = vision.to_max_side(img, 900)
    if res["kind"] == "bearing":
        steps = [("Input", view), ("Damage-cue overlay", vision.bearing_overlay(view)),
                 ("Texture heat map", vision.texture_heatmap(view)), ("Edges (Canny)", vision.edges(view, 80, 160))]
    else:
        steps = [("Input (resized to training size)", view), ("Contrast enhanced (CLAHE)", vision.clahe(view)),
                 ("Edges (Canny)", vision.edges(view)), ("Texture heat map", vision.texture_heatmap(view))]
    cols = st.columns(4)
    for c, (cap, im) in zip(cols, steps):
        c.image(im if im.ndim == 2 else vision.rgb(im), caption=cap, width="stretch")
    if res["kind"] == "bearing":
        st.caption("Overlay colours: red = rust hue (corrosion), yellow = strong colour (discolouration), "
                   "purple = very dark (contamination); the blue table background is dimmed.")


def familiarity_badge(ratio: float) -> None:
    sim = max(0.0, 1.0 - ratio / 2) * 100
    if ratio <= 1.0:
        st.success(f"**Looks like the reference photos** (similarity {sim:.0f}%). The result below is meaningful.")
    else:
        st.error(f"**Unfamiliar photo** (similarity {sim:.0f}%). It is further from every reference photo than "
                 "99% of reference photos are from each other, so treat the result as unreliable.")


def show_wear(res: dict, bundle: dict) -> dict:
    part = bundle["wear"][res["kind"]]
    proba = res["proba"]
    pred = vision.WEAR_CLASSES[int(np.argmax(proba))]
    familiarity_badge(res["ratio"])
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### Predicted wear: :{'green' if pred == 'sharp' else 'orange' if pred == 'used' else 'red'}[{pred}]")
        st.metric("Model score for this class", f"{proba.max():.0%}")
        cv = res["cv"]
        st.caption(f"How reliable is this photo type? Tested on tools the model never saw: macro-F1 "
                   f"{cv['macro_f1_mean']:.2f} ± {cv['macro_f1_std']:.2f} (chance about 0.33).")
        if res["kind"] == "tool":
            st.warning("Tool-edge photos are the weakest type for this simple model. Chip and machined-surface "
                       "photos give far better results.")
        if res["outside"]:
            st.caption(f"{len(res['outside'])} of {res['n_features']} image measurements fall outside the range "
                       f"seen in training: {', '.join(c.split('_', 1)[1] for c in res['outside'][:6])}"
                       f"{'…' if len(res['outside']) > 6 else ''}")
    with right:
        df = pd.DataFrame({"class": vision.WEAR_CLASSES, "score": proba})
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="model score"),
            y=alt.Y("class:N", sort=vision.WEAR_CLASSES, title=None),
            color=alt.Color("class:N", scale=alt.Scale(domain=list(CLASS_COLOURS), range=list(CLASS_COLOURS.values())),
                            legend=None),
            tooltip=["class", alt.Tooltip("score:Q", format=".2f")]).properties(height=140), width="stretch")
        if res["kind"] == "tool":
            prof = vision.row_profile(res["prepared"])
            st.altair_chart(alt.Chart(pd.DataFrame({"row": np.arange(len(prof)), "brightness": prof})).mark_line().encode(
                x=alt.X("row:Q", title="image row (top to bottom)"), y="brightness:Q").properties(
                height=120, title="Row brightness profile (a worn flank shows as a bright band)"), width="stretch")

    st.markdown("**Most similar reference photos** (with their measured wear)")
    thumbs = wear_thumbs()
    cols = st.columns(5)
    for c, i, d in zip(cols, res["nn_idx"], res["nn_dist"]):
        rid = part["ref_ids"][i]
        c.image(bytes(thumbs.at[rid, f"{res['kind']}_thumb"]), width="stretch")
        c.caption(f"{rid} · **{part['ref_labels'][i]}** · {part['ref_flank_wear'][i]:.0f} µm · distance {d:.1f}")
    votes = pd.Series(part["ref_labels"][res["nn_idx"]]).value_counts()
    st.caption("Neighbour vote: " + ", ".join(f"{k} {v}/5" for k, v in votes.items()))
    return {"prediction": pred, "score": round(float(proba.max()), 3),
            **{f"score_{c}": round(float(p), 3) for c, p in zip(vision.WEAR_CLASSES, proba)}}


def show_bearing(res: dict, bundle: dict) -> dict:
    part = bundle["bearing"]
    familiarity_badge(res["ratio"])
    st.info("Bearing photos have **no verified damage labels**, so this is a rule-based screening (OpenCV "
            "colour and texture cues), **not** a trained damage classifier.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### Screening flag: `{res['label']}`")
        st.caption("A cue is flagged when it is above the 70th percentile of the 1,841 reference photos; "
                   "the strongest flagged cue wins.")
        st.metric("Bearing area in photo", f"{res['scores']['foreground_ratio']:.0%}")
    with right:
        df = pd.DataFrame({"cue": vision.BEARING_CATEGORIES,
                           "percentile": [res["percentile"][c] for c in vision.BEARING_CATEGORIES],
                           "meaning": [SCORE_HELP[c] for c in vision.BEARING_CATEGORIES]})
        base = alt.Chart(df).encode(y=alt.Y("cue:N", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=140)))
        bars = base.mark_bar().encode(
            x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="percentile vs reference photos"),
            color=alt.condition("datum.percentile >= 70", alt.value("#d9534f"), alt.value("#5fa8e8")),
            tooltip=["cue", alt.Tooltip("percentile:Q", format=".0f"), "meaning"])
        rule = alt.Chart(pd.DataFrame({"x": [70]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=170), width="stretch")
    st.markdown("**Most similar reference photos** (heuristic group in bold)")
    thumbs = bearing_thumbs()
    cols = st.columns(5)
    for c, i, d in zip(cols, res["nn_idx"], res["nn_dist"]):
        rid = part["ref_ids"][i]
        c.image(bytes(thumbs.at[rid]), width="stretch")
        c.caption(f"{rid.replace('orig_', '')} · **{part['ref_labels'][i]}** · distance {d:.0f}")
    return {"prediction": res["label"], "score": None,
            **{f"pct_{c}": round(res["percentile"][c], 1) for c in vision.BEARING_CATEGORIES}}


# --------------------------------------------------------------------------- inputs
def sample_files(kind: str, n: int = 6) -> list[tuple[str, bytes, str | None]]:
    rng = np.random.default_rng(613 + st.session_state.get("sample_round", 0))
    if kind == "bearing":
        files = sorted(BEARING_RAW.glob("*.jpg"))
        pick = rng.choice(len(files), min(n, len(files)), replace=False) if files else []
        return [(files[i].name, files[i].read_bytes(), None) for i in pick]
    files = sorted((NONASTREDA_RAW / kind).glob(f"*{EXT[kind]}"))
    pick = rng.choice(len(files), min(n, len(files)), replace=False) if files else []
    return [(files[i].name, files[i].read_bytes(), files[i].stem) for i in pick]


def render() -> None:
    st.subheader("Check a photo")
    st.caption("Upload photos from your device or take one with the camera. Each photo goes through the same "
               "pipeline: OpenCV preprocessing → image measurements → model or screening rule → "
               "similar reference photos → reliability check.")
    bundle = load_bundle()
    if bundle is None:
        st.warning("Models not built yet. Run: `python prototype/train_image_models.py`")
        return

    c1, c2 = st.columns([2, 1])
    source = c1.radio("Photo source", ["Upload from device", "Use camera", "Try built-in samples"], horizontal=True)
    choice = c2.selectbox("What is in the photo?", ["Auto-detect", *TYPES.values()])
    kind_choice = None if choice == "Auto-detect" else next(k for k, v in TYPES.items() if v == choice)

    items: list[tuple[str, bytes, str | None]] = []
    if source == "Upload from device":
        ups = st.file_uploader("JPG or PNG, one or more", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
        items = [(u.name, u.getvalue(), None) for u in ups or []]
    elif source == "Use camera":
        shot = st.camera_input("Take a photo")
        if shot is not None:
            items = [("camera.jpg", shot.getvalue(), None)]
    else:
        sk = st.selectbox("Sample type", list(TYPES), format_func=TYPES.get,
                          index=list(TYPES).index(kind_choice) if kind_choice else 1)
        if st.button("Pick new samples"):
            st.session_state["sample_round"] = st.session_state.get("sample_round", 0) + 1
        items = sample_files(sk)
        kind_choice = sk
        st.caption("Wear samples are judged by a model trained **without that tool** (leave-one-tool-out), "
                   "so their results are honest test results. Bearing samples come from the reference set itself.")

    if not items:
        st.info("Add a photo to start, or choose **Try built-in samples**.")
        return

    rows = []
    for name, data, sample_id in items:
        try:
            img = vision.decode(data)
        except ValueError as e:
            st.error(f"{name}: {e}")
            continue
        kind, ratios = (kind_choice, None) if kind_choice else detect_type(img, bundle)
        res = analyse_bearing(img, bundle) if kind == "bearing" else analyse_wear(img, kind, bundle, sample_id)
        with st.expander(f"**{name}** · {TYPES[kind]}", expanded=len(items) <= 2):
            if ratios and min(ratios.values()) > 1.0:
                st.warning("This photo does not match any supported photo type (microscope images of milling "
                           "tools, chips or machined surfaces, or bearings on the reference table). It is shown "
                           "against the closest type, but the result is not meaningful.")
            if ratios:
                st.caption("Auto-detected type (closest reference set): " + ", ".join(
                    f"{TYPES[k].split(' (')[0]} {v:.2f}" for k, v in sorted(ratios.items(), key=lambda kv: kv[1]))
                    + " — lower is closer; above 1 means unfamiliar.")
            show_steps(res)
            if res.get("known"):
                st.info(f"This is reference photo **{res['sample_id']}** from the dataset, so it is judged by the "
                        "model trained without its tool, and similar photos come from other tools only.")
                sample_id = res["sample_id"]
            out = show_bearing(res, bundle) if kind == "bearing" else show_wear(res, bundle)
            if sample_id and kind != "bearing":
                truth = bundle["wear"][kind]["ref_labels"][bundle["wear"][kind]["ref_ids"] == sample_id][0]
                ok = truth == out["prediction"]
                st.markdown(f"True label for this sample: **{truth}** — prediction {'✅ correct' if ok else '❌ wrong'}")
                out["true_label"] = truth
        rows.append({"file": name, "photo_type": TYPES[kind], "familiar": res["ratio"] <= 1.0,
                     "familiarity_ratio": round(res["ratio"], 2), **out})

    if rows:
        st.markdown("#### Summary")
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, width="stretch")
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        st.download_button("Download report (CSV)", buf.getvalue(), "photo_check_report.csv", "text/csv")
