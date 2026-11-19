"""Streamlit interface for the Credit Scoring Model.

Run with:  streamlit run app.py
(Train first with `python credit_scoring.py` so models/credit_model.joblib exists.)
"""

import joblib
import pandas as pd
import streamlit as st

from credit_scoring import MODEL_PATH, engineer_features, ensure_dataset

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference() -> pd.DataFrame:
    return pd.read_csv(ensure_dataset())


st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha ML Internship - Task 1 | German Credit data (UCI Statlog)")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python credit_scoring.py` first to train and save the model.")
    st.stop()

bundle = load_model()
model = bundle["model"]
ref = load_reference().drop(columns="credit_risk")

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {bundle['model_name']}")
    for k in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        st.write(f"{k}: **{bundle['metrics'][k]:.3f}**")
    st.caption("Metrics are for the bad-credit class on the hold-out test set.")

st.subheader("Applicant details")
cat_cols = ref.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in ref.columns if c not in cat_cols]

values = {}
cols = st.columns(3)
for i, col in enumerate(ref.columns):
    target = cols[i % 3]
    label = col.replace("_", " ").title()
    if col in cat_cols:
        options = sorted(ref[col].unique().tolist())
        default = ref[col].mode()[0]
        values[col] = target.selectbox(label, options, index=options.index(default))
    else:
        lo, hi = int(ref[col].min()), int(ref[col].max())
        values[col] = target.number_input(label, min_value=lo, max_value=hi, value=int(ref[col].median()))

if st.button("Predict creditworthiness", type="primary"):
    row = engineer_features(pd.DataFrame([values]).assign(default=0)).drop(columns="default")
    p_bad = float(model.predict_proba(row)[0, 1])
    pred_bad = int(model.predict(row)[0])

    c1, c2 = st.columns(2)
    c1.metric("Probability of bad credit", f"{p_bad:.1%}")
    c2.metric("Probability of good credit", f"{1 - p_bad:.1%}")
    st.progress(min(max(p_bad, 0.0), 1.0))
    if pred_bad:
        st.error("⚠️ High risk: the applicant is predicted to have BAD credit.")
    else:
        st.success("✅ Low risk: the applicant is predicted to have GOOD credit.")
    st.caption("Educational demo built on a 1000-row dataset. Not for real lending decisions.")
