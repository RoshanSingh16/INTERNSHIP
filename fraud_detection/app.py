"""
app.py  —  Credit Card Fraud Detection Dashboard
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FraudShield | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global style ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0d1224 !important; border-right: 1px solid #1e2a4a; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.hero-sub {
    color: #8892b0;
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 300;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
}
.metric-val { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #00d4ff; }
.metric-lbl { font-size: 0.78rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.fraud-badge {
    background: linear-gradient(135deg, #ff4757, #c0392b);
    color: white; font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    border-radius: 12px; padding: 18px 28px; text-align: center;
    box-shadow: 0 0 30px rgba(255,71,87,0.4);
}
.legit-badge {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white; font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    border-radius: 12px; padding: 18px 28px; text-align: center;
    box-shadow: 0 0 30px rgba(0,184,148,0.4);
}
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important; padding: 12px 28px !important;
    width: 100%; font-weight: 700 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: #ccd6f6 !important; }

div[data-testid="stMetric"] { background: #111827; border-radius: 12px; padding: 16px; border: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── sidebar nav ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🔍 Predict", "📊 Model Analytics", "📁 Batch Upload", "ℹ️ About"])
    st.markdown("---")
    st.markdown("<small style='color:#4a5568'>v1.0 · Built with Streamlit</small>", unsafe_allow_html=True)

# ── route ────────────────────────────────────────────────────────────────────────
if   page == "🏠 Home":            from pages import home;     home.show()
elif page == "🔍 Predict":         from pages import predict;  predict.show()
elif page == "📊 Model Analytics": from pages import analytics; analytics.show()
elif page == "📁 Batch Upload":    from pages import batch;    batch.show()
elif page == "ℹ️ About":           from pages import about;    about.show()
