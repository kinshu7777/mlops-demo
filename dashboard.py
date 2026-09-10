"""
STEP 5: A simple screen to SEE what's going on, instead of reading JSON files.

Run it with:  python3 control.py dashboard
It opens in your browser automatically.
"""

import os
import json
import glob
import streamlit as st
import pandas as pd

from config import CONFIG

MODELS_DIR = CONFIG["models_dir_path"]

st.set_page_config(
    page_title="Delivery Model Monitor",
    page_icon="📦",
    layout="centered",
)

st.title("📦 Delivery Time Predictor")
st.caption("MLOps monitoring dashboard — model performance & data drift, at a glance")

meta_files = sorted(glob.glob(os.path.join(MODELS_DIR, "model_v*.json")))
drift_path = CONFIG["drift_report_path"]

# --- Top-line summary metrics ---
col1, col2, col3 = st.columns(3)

if meta_files:
    with open(meta_files[-1]) as f:
        latest = json.load(f)
    col1.metric("Current model version", f"v{latest['version']}")
    col2.metric("Accuracy (avg error)", f"{latest['mae_minutes']:.2f} min")
else:
    col1.metric("Current model version", "—")
    col2.metric("Accuracy (avg error)", "—")

if os.path.exists(drift_path):
    with open(drift_path) as f:
        drift = json.load(f)
    drift_label = "⚠️ Drifted" if drift.get("any_drift") else "✅ Stable"
    col3.metric("Data status", drift_label)
else:
    col3.metric("Data status", "—")

st.divider()

# --- Model version history ---
st.subheader("Model version history")

if not meta_files:
    st.info("No trained models yet. Run `python3 control.py train` first.")
else:
    rows = []
    for f in meta_files:
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows).sort_values("version")

    left, right = st.columns([2, 1])
    with left:
        st.line_chart(df.set_index("version")["mae_minutes"], height=250)
        st.caption("Lower is better — average minutes the model's prediction is off by.")
    with right:
        st.dataframe(
            df[["version", "mae_minutes", "training_rows"]].rename(
                columns={"mae_minutes": "MAE (min)", "training_rows": "rows"}
            ),
            hide_index=True,
            use_container_width=True,
        )

st.divider()

# --- Drift report ---
st.subheader("Latest drift check")

if not os.path.exists(drift_path):
    st.info("No drift report yet. Run `python3 control.py check-drift` first.")
else:
    with open(drift_path) as f:
        drift = json.load(f)

    if drift.get("any_drift"):
        st.error("⚠️ Drift detected in one or more features — a retrain is recommended.")
    else:
        st.success("✅ No significant drift — the model's assumptions still hold.")

    drift_rows = [
        {"feature": k, "p-value": v["p_value"], "drifted": "🔴 yes" if v["drifted"] else "🟢 no"}
        for k, v in drift.items() if k != "any_drift"
    ]
    st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)
    st.caption("A p-value below 0.05 means the feature's distribution has meaningfully shifted.")

st.divider()
st.caption("Built with FastAPI, scikit-learn, and Streamlit · Run `python3 control.py --help` for all commands")
