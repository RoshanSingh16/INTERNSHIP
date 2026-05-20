import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
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
    st.markdown('<p class="hero-title">Transaction Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter transaction details to get an instant fraud risk assessment</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not (MODELS_DIR / "random_forest.pkl").exists():
        st.error("❌  Models not found. Please run `python train_model.py` first.")
        return

    rf, xgb, le, sc = load_models()

    # ── input form ─────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("📋 Transaction Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            amount       = st.number_input("💰 Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=120.0, step=0.01)
            hour         = st.slider("🕐 Hour of Day (24h)", 0, 23, 14)
            day_of_week  = st.selectbox("📅 Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        with c2:
            merchant_cat = st.selectbox("🏪 Merchant Category", ["grocery","entertainment","travel","retail","dining"])
            distance_km  = st.number_input("📍 Distance from Home (km)", min_value=0.0, max_value=5000.0, value=12.0)
            prev_txn_hrs = st.number_input("⏱️ Hours Since Last Transaction", min_value=0.0, max_value=720.0, value=18.0)

        with c3:
            st.markdown("**🔢 PCA-derived Features** *(auto-generated in production)*")
            v1 = st.slider("V1", -5.0, 5.0, 0.0, 0.1)
            v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.1)
            v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.1)
            v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.1)
            v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.1)

        submitted = st.form_submit_button("🔍 Analyse Transaction")

    if submitted:
        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
        cat_enc = le.transform([merchant_cat])[0]

        raw = np.array([[amount, hour, day_map[day_of_week], cat_enc,
                         distance_km, prev_txn_hrs, v1, v2, v3, v4, v5]])
        X   = sc.transform(raw)

        rf_prob  = rf.predict_proba(X)[0][1]
        xgb_prob = xgb.predict_proba(X)[0][1]
        avg_prob = (rf_prob + xgb_prob) / 2
        is_fraud = avg_prob >= 0.5

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        r1, r2 = st.columns([1, 1.4])

        with r1:
            if is_fraud:
                st.markdown(f"""<div class="fraud-badge">⚠️ FRAUD DETECTED<br>
                    <span style='font-size:1rem;font-weight:400'>Confidence: {avg_prob*100:.1f}%</span></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="legit-badge">✅ LEGITIMATE<br>
                    <span style='font-size:1rem;font-weight:400'>Confidence: {(1-avg_prob)*100:.1f}%</span></div>""",
                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Random Forest",  f"{rf_prob*100:.1f}%")
            st.metric("XGBoost",        f"{xgb_prob*100:.1f}%")
            st.metric("Ensemble Avg",   f"{avg_prob*100:.1f}%")

        with r2:
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number+delta",
                value  = avg_prob * 100,
                title  = {"text": "Fraud Risk Score", "font": {"color": "#ccd6f6", "size": 16}},
                number = {"suffix": "%", "font": {"color": "#ccd6f6", "size": 40}},
                delta  = {"reference": 50, "increasing": {"color": "#ff4757"}, "decreasing": {"color": "#00b894"}},
                gauge  = {
                    "axis":  {"range": [0, 100], "tickcolor": "#8892b0"},
                    "bar":   {"color": "#ff4757" if is_fraud else "#00b894"},
                    "bgcolor": "#111827",
                    "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0,  40],  "color": "#0d2e1a"},
                        {"range": [40, 70],  "color": "#2e2200"},
                        {"range": [70, 100], "color": "#2e0d0d"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": 50},
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#ccd6f6",
                height=280, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        # ── risk factors ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Risk Factors")
        risk_flags = []
        if amount > 500:            risk_flags.append(("💸 High Amount",     f"${amount:.2f} — transactions >$500 are higher risk"))
        if hour < 6 or hour >= 22:  risk_flags.append(("🌙 Odd Hour",        f"Transaction at {hour:02d}:00 — unusual time"))
        if distance_km > 100:       risk_flags.append(("📍 Large Distance",  f"{distance_km} km from home"))
        if prev_txn_hrs < 2:        risk_flags.append(("⚡ Rapid Succession",f"Only {prev_txn_hrs:.1f}h since last transaction"))
        if merchant_cat == "travel": risk_flags.append(("✈️ Travel Merchant", "Travel purchases are common fraud targets"))

        if risk_flags:
            for name, desc in risk_flags:
                st.warning(f"**{name}** — {desc}")
        else:
            st.success("✅ No notable risk factors detected in this transaction.")
