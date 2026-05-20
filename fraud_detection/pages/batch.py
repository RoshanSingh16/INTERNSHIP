import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES   = ["amount","hour","day_of_week","merchant_cat",
               "distance_km","prev_txn_hrs","v1","v2","v3","v4","v5"]

@st.cache_resource
def load_models():
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    le  = joblib.load(MODELS_DIR / "label_encoder.pkl")
    sc  = joblib.load(MODELS_DIR / "scaler.pkl")
    return rf, xgb, le, sc

def show():
    st.markdown('<p class="hero-title">Batch Fraud Scorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Upload a CSV of transactions and download results with fraud scores</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not (MODELS_DIR / "random_forest.pkl").exists():
        st.error("❌  Models not found. Run `python train_model.py` first."); return

    rf, xgb, le, sc = load_models()

    # ── expected format ────────────────────────────────────────────────────────
    with st.expander("📋 Expected CSV Format"):
        sample = pd.DataFrame([{
            "amount": 150.0, "hour": 14, "day_of_week": 2,
            "merchant_cat": "grocery", "distance_km": 8.5,
            "prev_txn_hrs": 20.0, "v1": 0.1, "v2": -0.2,
            "v3": 0.3, "v4": -0.1, "v5": 0.05,
        }])
        st.dataframe(sample, use_container_width=True)
        csv_sample = sample.to_csv(index=False)
        st.download_button("⬇️ Download Sample CSV", csv_sample, "sample_transactions.csv", "text/csv")

    # ── uploader ───────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload your transactions CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} transactions")
        st.dataframe(df.head(5), use_container_width=True)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}"); return

        with st.spinner("🔍 Scoring transactions …"):
            df2 = df.copy()
            df2["merchant_cat"] = le.transform(df2["merchant_cat"].astype(str))
            X = sc.transform(df2[FEATURES].values)

            rf_prob  = rf.predict_proba(X)[:, 1]
            xgb_prob = xgb.predict_proba(X)[:, 1]
            avg_prob = (rf_prob + xgb_prob) / 2

            df["rf_score"]      = (rf_prob  * 100).round(1)
            df["xgb_score"]     = (xgb_prob * 100).round(1)
            df["fraud_score"]   = (avg_prob * 100).round(1)
            df["prediction"]    = np.where(avg_prob >= 0.5, "🚨 FRAUD", "✅ LEGIT")

        # ── summary ────────────────────────────────────────────────────────────
        n_fraud = (avg_prob >= 0.5).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("🚨 Flagged as Fraud",  f"{n_fraud:,}",  delta=f"{n_fraud/len(df)*100:.1f}%")
        c3.metric("✅ Cleared",           f"{len(df)-n_fraud:,}")

        st.dataframe(df.sort_values("fraud_score", ascending=False), use_container_width=True)

        out = io.StringIO()
        df.to_csv(out, index=False)
        st.download_button(
            "⬇️ Download Scored CSV", out.getvalue(),
            "scored_transactions.csv", "text/csv")
