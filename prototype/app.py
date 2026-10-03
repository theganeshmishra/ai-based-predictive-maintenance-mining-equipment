"""Prototype demo: predictive-maintenance pipeline on recorded data.

Run from the repository root:  streamlit run prototype/app.py
No physical sensors are needed: recorded data is replayed as a "virtual sensor".
Build the inputs first (see prototype/README.md).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import photo_check

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = Path(__file__).resolve().parent / "figures"
STATES = ["normal", "alarm", "M4/MS3", "M4/MS5", "M9/MS1", "M9/MS6", "M2", "other_maintenance"]
STATE_COLOURS = ["#9aa5b1", "#e8a33d", "#2f6fbf", "#5fa8e8", "#2e9e5b", "#8fd19e", "#b0479a", "#d9534f"]
WEAR = ["sharp", "used", "dulled"]

st.set_page_config(page_title="PdM prototype", page_icon="🛠️", layout="wide")


def need(path: Path, how: str) -> bool:
    if not path.exists():
        st.warning(f"Missing `{path.relative_to(ROOT)}`. Build it with: `{how}`")
        return False
    return True


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_parquet(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


with st.sidebar:
    st.header("PdM prototype")
    st.caption("PROJ 613 capstone, Group 9. Academic prototype, not production-ready.")
    st.markdown(
        "- **Sensor module:** TiNA mining machine, 8 machine states per minute\n"
        "- **Inspection module:** Nonastreda milling-tool wear, signals + photos (3 classes)\n"
        "- **Photo module:** bearing photos, heuristic labels only\n\n"
        "The datasets come from different equipment, so they are **not merged**; "
        "each module is trained and tested on its own data.\n\n"
        "No physical sensors are used: recorded data is replayed as a *virtual sensor*."
    )

st.title("Predictive maintenance prototype")
st.caption("Sensor alert → visual inspection, on recorded public data. Every result shown was tested on data the "
           "model never saw; weak results are shown as weak.")

tab_photo, tab_over, tab_tina, tab_wear, tab_bear = st.tabs(
    ["📷 Check a photo", "Overview", "Sensor states (TiNA, 8 classes)", "Wear inspection (Nonastreda)",
     "Bearing photos"])

with tab_photo:
    photo_check.render()

# --------------------------------------------------------------------------- overview
with tab_over:
    st.subheader("Master catalogue: every sample, one table")
    cat_path = PROC / "master_catalog.csv"
    if need(cat_path, "python prototype/build_master.py"):
        cat = load_csv(cat_path)
        cols = st.columns(4)
        cols[0].metric("Catalogue rows", f"{len(cat):,}")
        for c, (name, g) in zip(cols[1:], cat.groupby("dataset")):
            c.metric(f"{name} ({g['label'].nunique()} labels)", f"{len(g):,}")
        counts = cat.groupby(["dataset", "label"]).size().reset_index(name="samples")
        chart_cols = st.columns(3)
        for c, (name, g) in zip(chart_cols, counts.groupby("dataset")):
            c.markdown(f"**{name}**")
            c.altair_chart(alt.Chart(g).mark_bar().encode(
                x=alt.X("samples:Q", title="samples"),
                y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)),
                tooltip=["label", "samples"],
            ).properties(height=alt.Step(24)), width="stretch")
        st.caption("Columns in the catalogue: " + ", ".join(cat.columns))

    st.subheader("Honest results so far")
    st.markdown(
        "| Module | Classes | Test protocol | Result |\n|---|---|---|---|\n"
        "| TiNA machine state | 8 | whole days held out | **no better than always predicting 'normal'** |\n"
        "| Nonastreda tool wear | 3 | leave one tool out (10 folds) | macro-F1 about 0.84 (signals + photos) |\n"
        "| Bearing photos | 5 heuristic | none: no verified labels | no model claimed |"
    )

# --------------------------------------------------------------------------- TiNA
with tab_tina:
    st.subheader("Machine-state replay: TiNA mining machine, 8 classes")
    st.caption("Each point is one minute of sensor data (mean and std of 103 anonymised sensors). "
               "The model was trained on 6 to 22 Sep 2017 and is replayed on the unseen days 23 to 30 Sep.")
    demo_path = PROC / "tina_8state_demo_predictions.parquet"
    if need(demo_path, "python prototype/build_master.py; python prototype/train_8state.py"):
        demo = load_parquet(demo_path)
        demo["time"] = pd.to_datetime(demo["minute_end"], unit="s")
        st.error("**Honest result:** on days it has not seen, the model predicts *normal* for almost every "
                 "minute and recognises none of the maintenance episodes. With a leaky random-minute split "
                 "the same model looks about 95% accurate. See the evaluation below.")

        day = st.selectbox("Day to replay", sorted(demo["day"].unique()), index=0)
        d = demo[demo["day"] == day].reset_index(drop=True)
        i = st.slider("Minute", 0, len(d) - 1, value=int(np.argmax(d["state"] != "normal")) if (d["state"] != "normal").any() else 0,
                      format="%d")
        row = d.iloc[i]
        c1, c2, c3 = st.columns(3)
        c1.metric("Time (UTC)", row["time"].strftime("%H:%M"))
        c2.metric("True state", row["state"])
        c3.metric("Predicted state", row["predicted"],
                  "correct" if row["predicted"] == row["state"] else "wrong",
                  delta_color="normal" if row["predicted"] == row["state"] else "inverse")

        scale = alt.Scale(domain=STATES, range=STATE_COLOURS)
        long = pd.concat([d.assign(track="true", s=d["state"]), d.assign(track="predicted", s=d["predicted"])])
        timeline = alt.Chart(long).mark_tick(thickness=3, size=28).encode(
            x=alt.X("time:T", title=None), y=alt.Y("track:N", title=None, sort=["true", "predicted"]),
            color=alt.Color("s:N", scale=scale, title=None, legend=alt.Legend(orient="bottom", columns=8)),
            tooltip=["time:T", "track", "s"])
        now = alt.Chart(pd.DataFrame({"time": [row["time"]]})).mark_rule(color="black").encode(x="time:T")
        st.altair_chart((timeline + now).properties(height=150), width="stretch")

        left, right = st.columns([2, 1])
        sensors = [c for c in d.columns if c.startswith("FEATURE")][:3]
        trace = d.melt(id_vars="time", value_vars=sensors, var_name="sensor", value_name="value")
        left.markdown("**Top sensors by model importance** (anonymised; physical meaning unknown)")
        left.altair_chart((alt.Chart(trace).mark_line().encode(
            x=alt.X("time:T", title=None), y="value:Q", color="sensor:N") + now).properties(height=220),
            width="stretch")
        probs = pd.DataFrame({"state": STATES, "probability": [row.get(f"p_{s}", 0.0) for s in STATES]})
        right.markdown("**Model probabilities, this minute**")
        right.altair_chart(alt.Chart(probs).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("state:N", sort=STATES, title=None),
            color=alt.Color("state:N", scale=scale, legend=None)).properties(height=220), width="stretch")

        st.markdown("#### Evaluation")
        if need(RESULTS / "prototype_8state_summary.csv", "python prototype/train_8state.py"):
            st.dataframe(load_csv(RESULTS / "prototype_8state_summary.csv"), hide_index=True)
            st.markdown(
                "- **Day CV / hold-out** keep whole days apart, as a real deployment would. "
                "Balanced accuracy near 0.125 is chance for 8 classes.\n"
                "- **Leaky random-minute CV** mixes neighbouring minutes of the same day into training and testing. "
                "Its high score shows the model learns *which day/time* a minute is from, not the machine state.\n"
                "- A drift-corrected version (each minute compared with the previous 2 hours) was also tried and "
                "stayed at chance; it is not used here.")
            st.dataframe(load_csv(RESULTS / "prototype_8state_cv_per_class.csv"), hide_index=True)
        c1, c2 = st.columns(2)
        for c, name in zip((c1, c2), ("confusion_8state_cv.jpg", "confusion_8state_holdout.jpg")):
            if (FIG / name).exists():
                c.image(str(FIG / name))

# --------------------------------------------------------------------------- Nonastreda
with tab_wear:
    st.subheader("Wear inspection: force signals + photos (Nonastreda)")
    st.caption("Milling-tool wear from one industrial machine, 10 tools, 512 samples. Representative data for "
               "the inspection step, **not mining equipment**. Truchan and Ahmadi (2025), "
               "DOI 10.17632/m892d2wtzh.1, CC BY 4.0.")
    wear_path = PROC / "nonastreda_demo_predictions.parquet"
    if need(wear_path, "python prototype/export_nonastreda.py"):
        w = load_parquet(wear_path)
        c1, c2 = st.columns(2)
        tool = c1.selectbox("Tool (each prediction comes from a model that never saw this tool)", sorted(w["tool"].unique()))
        wt = w[w["tool"] == tool].sort_values(["run", "blade"]).reset_index(drop=True)
        sid = c2.selectbox("Sample (T=tool, R=run, B=blade)", wt["id"])
        r = wt[wt["id"] == sid].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True wear class", r["image_label"])
        m2.metric("Predicted (signals + photos)", r["pred_fusion"],
                  "correct" if r["pred_fusion"] == r["image_label"] else "wrong",
                  delta_color="normal" if r["pred_fusion"] == r["image_label"] else "inverse")
        m3.metric("Measured flank wear", f"{r['flank_wear']:.0f} µm")
        m4.metric("Recording length", f"{r['sig_len'] / 1000:.0f} s")

        imgs = st.columns(3)
        for c, kind, cap in zip(imgs, ("tool", "chip", "work"), ("tool edge", "chips", "machined surface")):
            c.image(bytes(r[f"{kind}_thumb"]), caption=cap, width="stretch")

        left, right = st.columns([2, 1])
        n = len(r["trace_fx"])
        tr = pd.DataFrame({"t (s)": np.linspace(0, r["sig_len"] / 1000, n),
                           "Fx": r["trace_fx"], "Fy": r["trace_fy"], "Fz": r["trace_fz"]})
        left.markdown("**Recorded cutting forces** (downsampled for display)")
        left.line_chart(tr, x="t (s)", y=["Fx", "Fy", "Fz"], height=240)
        pr = pd.DataFrame([{"model": m, "class": c, "probability": float(r[f"p_{m}_{c}"])}
                           for m in ("signal", "image", "fusion") for c in WEAR])
        right.markdown("**Class probabilities**")
        right.altair_chart(alt.Chart(pr).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("class:N", sort=WEAR, title=None),
            color=alt.Color("class:N", sort=WEAR, legend=None), row=alt.Row("model:N", sort=["signal", "image", "fusion"], title=None),
        ).properties(height=60, width=220), width="content")

        st.markdown("#### This tool, all samples")
        acc = (wt["pred_fusion"] == wt["image_label"]).mean()
        st.caption(f"Tool {tool}: {acc:.0%} of {len(wt)} samples correct (signals + photos).")
        grid = wt.assign(correct=wt["pred_fusion"] == wt["image_label"])
        st.altair_chart(alt.Chart(grid).mark_rect().encode(
            x=alt.X("run:O"), y=alt.Y("blade:O"), color=alt.Color("image_label:N", sort=WEAR, title="true class"),
            opacity=alt.condition("datum.correct", alt.value(1.0), alt.value(0.35)),
            tooltip=["id", "image_label", "pred_fusion", "flank_wear"]).properties(height=140),
            width="stretch")
        st.caption("Faded cells are misclassified. Most errors sit where the class ranges of flank wear overlap.")

# --------------------------------------------------------------------------- bearings
with tab_bear:
    st.subheader("Bearing photos: heuristic labels only")
    st.warning("These 1,841 phone photos have **no verified damage labels**, and their mining provenance and "
               "permission to share are unconfirmed. The groups below come from an OpenCV colour/texture "
               "heuristic and are **not** a trained or validated model.")
    bp = PROC / "opencv_photos_all_in_one.parquet"
    if need(bp, "run notebooks/OpenCV_Photo_Dataset_One_File.ipynb"):
        b = load_parquet(bp, ("image_id", "source_set", "pseudo_label", "thumb_jpeg"))
        b = b[b["source_set"] == "original"]
        lab = st.radio("Heuristic group", sorted(b["pseudo_label"].unique()), horizontal=True)
        pick = b[b["pseudo_label"] == lab].sample(min(12, (b["pseudo_label"] == lab).sum()), random_state=613)
        cols = st.columns(6)
        for k, (_, p) in enumerate(pick.iterrows()):
            cols[k % 6].image(bytes(p["thumb_jpeg"]), caption=p["image_id"].replace("orig_", ""), width="stretch")
        st.caption(f"{(b['pseudo_label'] == lab).sum()} of {len(b)} original photos in this group; 12 shown (seed 613).")
