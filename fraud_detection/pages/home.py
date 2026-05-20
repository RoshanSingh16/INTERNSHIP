import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json

def show():
    st.markdown('<p class="hero-title">FraudShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time Credit Card Fraud Detection · Powered by Ensemble ML</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────────
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        rf = m["random_forest"]
        xg = m["xgboost"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{rf['roc_auc']:.3f}</div>
                <div class="metric-lbl">RF · ROC-AUC</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{xg['roc_auc']:.3f}</div>
                <div class="metric-lbl">XGB · ROC-AUC</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{rf['recall']:.3f}</div>
                <div class="metric-lbl">RF · Recall</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{xg['f1']:.3f}</div>
                <div class="metric-lbl">XGB · F1 Score</div></div>""", unsafe_allow_html=True)
    else:
        st.info("⚠️  Models not trained yet. Run `python train_model.py` to generate models & metrics.")

    st.markdown("---")

    # ── feature overview ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("📌 How It Works")
        st.markdown("""
**FraudShield** uses a two-model ensemble to flag suspicious transactions:

| Step | Detail |
|------|--------|
| 1. Input | Transaction details (amount, time, location, merchant…) |
| 2. Encode | Label-encode categorical features |
| 3. Scale | StandardScaler normalization |
| 4. Predict | Random Forest + XGBoost inference |
| 5. Score | Confidence score (0–100%) |
| 6. Alert | Real-time FRAUD / LEGIT verdict |

> **SMOTE** is used during training to handle the class imbalance common in fraud datasets (typically <1% fraud).
        """)

    with col_right:
        st.subheader("🗂️ Dataset Stats")
        data_path = Path("data/creditcard.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            total   = len(df)
            fraud   = df.is_fraud.sum()
            legit   = total - fraud
            pct     = fraud / total * 100

            fig = go.Figure(go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[legit, fraud],
                hole=0.6,
                marker_colors=["#00b894", "#ff4757"],
                textfont_size=13,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                annotations=[dict(text=f"{pct:.1f}%<br>Fraud", x=0.5, y=0.5,
                                  font_size=16, showarrow=False, font_color="#ff4757")],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Total: {total:,} transactions · Fraud: {fraud:,} · Legit: {legit:,}")
        else:
            st.info("Run `python generate_dataset.py` to create sample data.")
