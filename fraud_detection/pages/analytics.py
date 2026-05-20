import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
DATA_PATH  = Path("data/creditcard.csv")

@st.cache_resource
def load_models():
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    sc  = joblib.load(MODELS_DIR / "scaler.pkl")
    le  = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return rf, sc, le

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

FEATURES = ["amount","hour","day_of_week","merchant_cat",
            "distance_km","prev_txn_hrs","v1","v2","v3","v4","v5"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font_color   ="#ccd6f6",
    margin       =dict(t=40, b=30, l=20, r=20),
)

def show():
    st.markdown('<p class="hero-title">Model Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Performance metrics, confusion matrix, and feature importance</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not (MODELS_DIR / "metrics.json").exists():
        st.error("Run `python train_model.py` first."); return
    if not DATA_PATH.exists():
        st.error("Run `python generate_dataset.py` first."); return

    with open(MODELS_DIR / "metrics.json") as f:
        m = json.load(f)

    rf_m  = m["random_forest"]
    xgb_m = m["xgboost"]

    # ── metrics table ──────────────────────────────────────────────────────────
    st.subheader("📊 Model Comparison")
    comp = pd.DataFrame({
        "Metric":    ["ROC-AUC","Avg Precision","Precision","Recall","F1"],
        "Random Forest": [rf_m["roc_auc"], rf_m["avg_prec"], rf_m["precision"], rf_m["recall"], rf_m["f1"]],
        "XGBoost":       [xgb_m["roc_auc"],xgb_m["avg_prec"],xgb_m["precision"],xgb_m["recall"],xgb_m["f1"]],
    })
    st.dataframe(comp.set_index("Metric").style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)

    # ── confusion matrices ─────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrices")
    col1, col2 = st.columns(2)
    for col, name, model_m in [(col1,"Random Forest",rf_m),(col2,"XGBoost",xgb_m)]:
        with col:
            cm = np.array(model_m["conf_matrix"])
            fig = ff.create_annotated_heatmap(
                cm, x=["Pred Legit","Pred Fraud"], y=["Act Legit","Act Fraud"],
                colorscale="Blues", showscale=False)
            fig.update_layout(title=name, **PLOTLY_THEME, height=280)
            st.plotly_chart(fig, use_container_width=True)

    # ── feature importance ─────────────────────────────────────────────────────
    st.subheader("📌 Feature Importance (Random Forest)")
    rf, sc, le = load_models()
    fi = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}) \
           .sort_values("Importance", ascending=True)

    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(**PLOTLY_THEME, height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── data distributions ─────────────────────────────────────────────────────
    st.subheader("📈 Data Distributions")
    df = load_data()
    df["label"] = df["is_fraud"].map({0:"Legit", 1:"Fraud"})

    tab1, tab2, tab3 = st.tabs(["Amount", "Hour", "Distance"])
    with tab1:
        fig = px.histogram(df[df.amount < df.amount.quantile(0.99)],
                           x="amount", color="label", barmode="overlay",
                           color_discrete_map={"Legit":"#00b894","Fraud":"#ff4757"},
                           nbins=60, opacity=0.75)
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.histogram(df, x="hour", color="label", barmode="overlay",
                           color_discrete_map={"Legit":"#00b894","Fraud":"#ff4757"},
                           nbins=24, opacity=0.75)
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = px.histogram(df[df.distance_km < 500], x="distance_km", color="label",
                           barmode="overlay",
                           color_discrete_map={"Legit":"#00b894","Fraud":"#ff4757"},
                           nbins=50, opacity=0.75)
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)
